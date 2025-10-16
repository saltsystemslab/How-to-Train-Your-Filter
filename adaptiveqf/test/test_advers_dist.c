#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <openssl/rand.h>

#include "include/hashutil.h"
#include "include/rand_util.h"
#include "include/exp_utility.h"

#define MAX_DATA_LINES 10000000
#define MAX_LINE_LENGTH 2048

uint64_t hash_str(char *str) {
	uint64_t hash = 5381;
	int c;
	while ((c = *str++)) {
		hash = ((hash << 5) + hash) + c;
	}
	return hash;
}

int main(int argc, char **argv)
{
	int num_trials = 3;
	int verbose = 1;
	char indexfilename[512];
	// int usingIndex = 0;
	if (argc < 6) {
		fprintf(stderr, "Please specify \nthe file path [eg. datasets/Malware_data.csv]\nthe index file path [eg. datasets/hashed_unif_10M_url.csv]\nthe number of queries [eg. 100000000]\nthe log of the number of slots in the QF [eg. 20]\nthe number of remainder bits in the QF [eg. 9]\n");
		exit(1);
	}
	
	size_t qbits = atoi(argv[4]);
	size_t rbits = atoi(argv[5]);
	size_t num_queries = strtoull(argv[3], NULL, 10);
    char filename[100];
	strcpy(filename, argv[1]);
	strcpy(indexfilename, argv[2]);
	int obj_index = get_obj_index(filename);
	int label_index = get_label_index(filename);
	char* dataset = get_dataset_name(filename);
	char* dist = get_dist_name(indexfilename);

	FILE * file_ptr;
	file_ptr = fopen(filename, "r");
	long *offsets = malloc(sizeof(long) * MAX_DATA_LINES);

	if (NULL == file_ptr) {
		printf("file can't be opened \n");
		return EXIT_FAILURE;
	}

	// First we go into the file and collect all of the insertions from the dataset.
	int num_inserts = 0;
	int current_index = 0;
	uint64_t *insert_set = malloc(MAX_DATA_LINES * sizeof(uint64_t));
	if (!insert_set) {
		printf("malloc insert_set failed");
		return EXIT_FAILURE;
	}
	if (RAND_bytes((unsigned char*)insert_set, MAX_DATA_LINES * sizeof(uint64_t)) != 1) {
		printf("RAND_bytes failed\n");
		return EXIT_FAILURE;
	}
	if (verbose) printf("Processing insertions\n");
	char buffer[256];
	fgets(buffer, 256, file_ptr); // get rid of the first row giving column names
	offsets[current_index] = ftell(file_ptr);
	while (fgets(buffer, 256, file_ptr)) {
		// first, obtain the label and item for the current row of the dataset
		current_index += 1;
		offsets[current_index] = ftell(file_ptr);
		char *label = NULL;
		char *item = NULL;
		char *token = strtok(buffer, ",");
		int count = 0;
		while (token != NULL) {
			count++;
			if (count == label_index) {
				label = token;
			} else if (count == obj_index) {
				item = token;
			}
			token = strtok(NULL, ",");
		}
		// now, depending on the label, determine if it should be inserted or not
		if (strcmp(filename,"../learned/data/malicious_url_scores.csv") == 0) {
			// here they're labeled as benign/malicious instead of 0 or 1. We say malicious == 1.
			if (label != NULL && strcmp(label,"malicious") == 0) {
				char *item_copy = strdup(item);
				insert_set[num_inserts] = hash_str(item_copy);
				num_inserts++;
			}
		} else {
			if (label != NULL && atoi(label) == 1) {
				char *item_copy = strdup(item);
				insert_set[num_inserts] = hash_str(item_copy);
				num_inserts++;
			}
		}
	}
	if (verbose) fprintf(stderr, "finished reading insertions\n");

	// At this point, we now have a large list of insertions,
	// and a list of file offsets corresponding to indices.

	// now, we want to read through the index file,
	// and fseek to whatever the query is to create a query set.
	
	uint64_t *query_set = malloc(num_queries * sizeof(uint64_t));
	uint64_t *query_labels = malloc(num_queries * sizeof(uint64_t));
	if (!query_set) {
		fprintf(stderr, "malloc failed for query_set\n");
		return 1;
	}
	if (!query_labels) {
		fprintf(stderr, "malloc failed for query_labels\n");
		return 1;
	}
	if (verbose) fprintf(stderr, "initialized query set\n");

	FILE * index_file_ptr = fopen(indexfilename, "r");
	if (!index_file_ptr) {
		printf("couldn't open index file\n");
		return EXIT_FAILURE;
	}
	fgets(buffer, 256, index_file_ptr); // get rid of the first row giving column names
	int current_query = 0;
	int pos_count = 0;
	while (fgets(buffer, 256, index_file_ptr)) {
		// first, obtain the index from the current row
		int index = atoi(strtok(buffer, ","));
		// now, use the array of offsets to go to the current element
		if (fseek(file_ptr, offsets[index], SEEK_SET) != 0) {
			fprintf(stderr, "fseek failed");
			fclose(index_file_ptr);
			return EXIT_FAILURE;
		}
		// finally, obtain the object that we want to insert
		if(!fgets(buffer, 256, file_ptr)) {
			fprintf(stderr, "failed to get stuff from file");
		}
		char *label = NULL;
		char *item = NULL;
		char *token = strtok(buffer, ",");
		int count = 0;
		while (token != NULL) {
			count++;
			if (count == label_index) {
				label = token;
			} else if (count == obj_index) {
				item = token;
			}
			token = strtok(NULL, ",");
		}
		// insert all corresponding indexes into the query set.
		char *item_copy = strdup(item);
		query_set[current_query] = hash_str(item_copy);
		// also track the corresponding label for the query...
		if (strcmp(filename,"../learned/data/malicious_url_scores.csv") == 0) {
			// here they're labeled as benign/malicious instead of 0 or 1. We say malicious urls are positive keys.
			// printf("label: %s\n", label);
			if (label != NULL && strcmp(label,"malicious") == 0) {
				query_labels[current_query] = 1;
				pos_count++;
			} else {
				query_labels[current_query] = 0;
			}
		} else {
			if (label != NULL) {
				query_labels[current_query] = atoi(label);
				if (query_labels[current_query] == 1) {
					pos_count++;
				}
			}
		}
		current_query++;
	}
	if (verbose) fprintf(stderr, "finished processing index file\n");
	fclose(index_file_ptr);

	// create a timer
	struct timeval timecheck;
	
	for (int i = 0; i < num_trials; i++) {
		for (double freq = 0; freq < 1; freq += 0.2) {
			time_t seed = time(NULL);
			srand(seed);
			if (verbose) printf("Running trial %d on seed %ld\n", i, seed);
			int murmur_seed = rand();

			// create the filter
			uint64_t nhashbits = qbits + rbits;
			uint64_t nslots = (1ULL << qbits);
			QF qf;
			
			gettimeofday(&timecheck, NULL);
			uint64_t start_time = timecheck.tv_sec * 1000000 + timecheck.tv_usec, end_time;
			if (!qf_malloc(&qf, nslots, nhashbits, 0, QF_HASH_INVERTIBLE, 0)) {
				fprintf(stderr, "Can't allocate CQF.\n");
				abort();
			}
			qf_set_auto_resize(&qf, false);
			gettimeofday(&timecheck, NULL);
			end_time = timecheck.tv_sec * 1000000 + timecheck.tv_usec;
			uint64_t filter_alloc_time = end_time - start_time;

			// create the reverse map
			gettimeofday(&timecheck, NULL);
			start_time = timecheck.tv_sec * 1000000 + timecheck.tv_usec;
			set_node *set = calloc(num_inserts, sizeof(set_node));
			gettimeofday(&timecheck, NULL);
			end_time = timecheck.tv_sec * 1000000 + timecheck.tv_usec;
			uint64_t set_alloc_time = end_time - start_time;

			// perform inserts
			uint64_t total_filter_insert_time = 0;
			uint64_t total_set_insert_time = 0;
			uint64_t filter_insert_time, set_insert_time;
			if (verbose) fprintf(stderr, "starting %d insertions\n", num_inserts);
			for (int j = 0; j < num_inserts; j++) {
				insert_set[j] = MurmurHash64A(&insert_set[j], sizeof(insert_set[j]), murmur_seed);
				int result = insert_key(&qf, set, num_inserts, insert_set[j], 1, &timecheck, &filter_insert_time, &set_insert_time);
				total_filter_insert_time += filter_insert_time;
				total_set_insert_time += set_insert_time;
				if (!result) {
					fprintf(stderr, "insertion %d failed\n", j);
					exit(1);
				}
			}
			if (verbose) fprintf(stderr, "finished %d insertions\n", num_inserts);

			// perform queries
			uint64_t ret_index, ret_hash, result;
			int ret_hash_len;
			int still_have_space = 1;
			int fp_count = 0;
			uint64_t total_query_time = 0;
			uint64_t total_adapt_time = 0;
			int num_learn = num_queries / 2; // how many queries to use for learning false positives
			int num_adv_learned = 0; // how many false positives were learned
			uint64_t *adv_list = malloc(num_queries * sizeof(uint64_t)); //the list of learned false positives
			int num_between_adv = (int)(num_queries/2)/(freq*num_queries);
			int current_adv_check = 0;
			int current_adv_count = 0;
			uint64_t query;
			fprintf(stderr, "started %ld queries with freq %.1f\n", num_queries, freq);
			for (int j = 0; j < num_queries; j++) {
				query_set[j] = MurmurHash64A(&query_set[j], sizeof(query_set[j]), murmur_seed);
				// start with a regular query
				query = query_set[j];
				current_adv_check++;
				if (j > num_learn) {
					if (current_adv_check == num_between_adv) {
						// replace the current query with an adversarial query
						query = adv_list[current_adv_count % num_adv_learned];
						current_adv_count++;
						// reset the count
						current_adv_check = 0;
					}
				}
				gettimeofday(&timecheck, NULL);
				start_time = timecheck.tv_sec * 1000000 + timecheck.tv_usec;
				result = qf_query(&qf, query, &ret_index, &ret_hash, &ret_hash_len, QF_KEY_IS_HASH);
				gettimeofday(&timecheck, NULL);
				end_time = timecheck.tv_sec * 1000000 + timecheck.tv_usec;
				total_query_time += end_time - start_time;
				if (result) {
					uint64_t temp = ret_hash | (1ull << ret_hash_len), orig_key = 0;
					set_query(set, num_inserts, temp, &orig_key);

					if (query != orig_key) {
						fp_count++;
						if (j < num_learn) {
							// if we're still in the learning phase, add to the list of false positives
							adv_list[num_adv_learned] = query;
							num_adv_learned++;
						}
						if (still_have_space) {
							gettimeofday(&timecheck, NULL);
							start_time = timecheck.tv_sec * 1000000 + timecheck.tv_usec;
							ret_hash_len = qf_adapt(&qf, ret_index, orig_key, query, &ret_hash, QF_KEY_IS_HASH | QF_NO_LOCK);
							if (ret_hash_len > 0) {
								int ret = set_delete(set, num_inserts, temp);
								if (ret == 0) {
									printf("%d\n", i);
									abort();
								}
								set_insert(set, num_inserts, ret_hash | (1ull << ret_hash_len), orig_key);
								gettimeofday(&timecheck, NULL);
								end_time = timecheck.tv_sec * 1000000 + timecheck.tv_usec;
								total_adapt_time += end_time - start_time;
							}
							else if (ret_hash_len == QF_NO_SPACE) {
								still_have_space = 0;
								fprintf(stderr, "\rfilter is full after %d queries\n", i);
								continue;
							}
						}
					}
				}
			}

			// calculate the false positive rate
			int total_negatives = num_queries - pos_count;
			double fpr = (double)fp_count / (fp_count + total_negatives);
			if (verbose) printf("False positive rate: %f\n", fpr);


			FILE * outputptr;
			outputptr = fopen("results/aqf_advers_results.csv", "a");
			fseek(outputptr, 0, SEEK_END);
			long filesize = ftell(outputptr);
			if (filesize == 0) {
				// File is empty, write header
				fprintf(outputptr, "dataset,query_dist,num_queries,freq,q,r,size,fpr,insert_time,map_time,amortized_query,amortized_adapt\n");
			}

			char new_result_row[512];
			snprintf(new_result_row, sizeof(new_result_row), "%s,%s,%ld,%.1f,%ld,%ld,%ld,%.14f,%ld,%ld,%.14f,%.14f\n",
					dataset, dist, num_queries, freq, qbits, rbits, qf.metadata->total_size_in_bytes, fpr,
					total_filter_insert_time + filter_alloc_time, total_set_insert_time + set_alloc_time,
					(double)total_query_time/num_queries,(double)total_adapt_time/num_queries);

			fprintf(outputptr, new_result_row);
			fclose(outputptr);
			
			set_free(set, num_inserts);
			qf_free(&qf);
			free(adv_list);
		}
	}
	free(query_set);
	free(query_labels);
	free(insert_set);
	return 0;
}