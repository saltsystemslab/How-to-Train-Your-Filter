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

	// First we go into the file and collect all of the insertions from the dataset.
	
	if (verbose) printf("Processing insertions\n");

	uint64_t *insert_set = calloc(MAX_DATA_LINES, sizeof(uint64_t));
	if (!insert_set) {
		printf("calloc insert_set failed\n");
		return EXIT_FAILURE;
	}
	long *offsets = calloc(sizeof(long), MAX_DATA_LINES);
	if (!offsets) {
		printf("calloc offsets failed\n");
		return EXIT_FAILURE;
	}
	char *buffer = malloc(4096);
	if (!buffer) {
		printf("malloc buffer failed\n");
		return EXIT_FAILURE;
	}

	int num_inserts;
	int read_result = read_file(filename, obj_index, label_index, 
								buffer, offsets, insert_set, 
								&num_inserts);
	if (!read_result) {
		printf("Insertion reading failed\n");
		return EXIT_FAILURE;
	}

	if (verbose) fprintf(stderr, "finished reading %d insertions\n", num_inserts);

	// At this point, we now have a large list of insertions,
	// and a list of file offsets corresponding to indices.

	// now, we want to read through the index file,
	// and fseek to whatever the query is to create a query set.
	
	uint64_t *query_set = calloc(num_queries, sizeof(uint64_t));
	if (!query_set) {
		fprintf(stderr, "calloc failed for query_set\n");
		return EXIT_FAILURE;
	}
	uint64_t *query_labels = calloc(num_queries, sizeof(uint64_t));
	if (!query_labels) {
		fprintf(stderr, "calloc failed for query_labels\n");
		return EXIT_FAILURE;
	}
	
	int total_queries = 0;
	int pos_count = 0;
	if (verbose) fprintf(stderr, "Processing queries\n");

	int query_read_result = read_queries(indexfilename, filename, obj_index, label_index, 
										buffer, offsets, query_set, query_labels, 
										&pos_count, &total_queries);
	if (!query_read_result) {
		printf("Query reading failed\n");
		return EXIT_FAILURE;
	}
	if (verbose) fprintf(stderr, "finished reading %d queries with %d positives\n", total_queries, pos_count);

	// Now insert all elements into the filter and perform the adversarial test

	QF qf;
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
			outputptr = fopen("../results/aqf/aqf_advers_results.csv", "a");
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
	free(buffer);
	return 0;
}