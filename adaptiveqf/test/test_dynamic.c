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
// #include "include/splinter_util.h"
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
	int num_trials = 1;
	int verbose = 1;
	char indexfilename[512];
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
	int num_negative = 0;
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
	uint64_t *negative_set = malloc(MAX_DATA_LINES * sizeof(uint64_t));
	if (!negative_set) {
		printf("malloc insert_set failed");
		return EXIT_FAILURE;
	}
	if (RAND_bytes((unsigned char*)negative_set, MAX_DATA_LINES * sizeof(uint64_t)) != 1) {
		printf("RAND_bytes failed\n");
		return EXIT_FAILURE;
	}
	if (verbose) fprintf(stderr, "Processing insertions\n");
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
		if (strcmp(filename,"../data/malicious_url_scores.csv") == 0) {
			// here they're labeled as benign/malicious instead of 0 or 1. We say malicious == 1.
			if (label != NULL) {
				char *item_copy = strdup(item);
				if (strcmp(label,"malicious") == 0) {
					insert_set[num_inserts] = hash_str(item_copy);
					num_inserts++;
				} else {
					negative_set[num_negative] = hash_str(item_copy);
					num_negative++;
				}
			}
		} else {
			if (label != NULL) {
				char *item_copy = strdup(item);
				if (atoi(label) == 1) {
					insert_set[num_inserts] = hash_str(item_copy);
					num_inserts++;
				} else {
					negative_set[num_negative] = hash_str(item_copy);
					num_negative++;
				}
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
	int total_queries = 0;
	int pos_count = 0;
	while (fgets(buffer, 256, index_file_ptr)) {
		total_queries++;
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
		if (strcmp(filename,"../data/malicious_url_scores.csv") == 0) {
			// here they're labeled as benign/malicious instead of 0 or 1. We say malicious urls are positive keys.
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
		time_t seed = time(NULL);
		srand(seed);
		if (verbose) printf("Running trial %d on seed %ld\n", i, seed);
		int murmur_seed = rand();

		// create the filter
		uint64_t nhashbits = qbits + rbits;
		uint64_t nslots = (1ULL << qbits);
		QF qf;
		
        if (!qf_malloc(&qf, nslots, nhashbits, 0, QF_HASH_INVERTIBLE, 0)) {
			fprintf(stderr, "Can't allocate CQF.\n");
			abort();
        }
        qf_set_auto_resize(&qf, false);

		// create the reverse map
		set_node *set = calloc(num_inserts, sizeof(set_node));

		// perform inserts
		
		set_node *positives = calloc(num_inserts, sizeof(set_node));
		uint64_t filter_insert_time, set_insert_time;
		for (int j = 0; j < num_inserts; j++) {
			insert_set[j] = MurmurHash64A(&insert_set[j], sizeof(insert_set[j]), murmur_seed);
			query_set[j] = MurmurHash64A(&query_set[j], sizeof(query_set[j]), murmur_seed);
			int result = insert_key(&qf, set, num_inserts, insert_set[j], 1, &timecheck, &filter_insert_time, &set_insert_time);
			set_insert(positives, num_inserts, insert_set[j], insert_set[j]);
			if (!result) {
				fprintf(stderr, "insertion %d failed\n", j);
				exit(1);
			}
		}

		// perform queries
		uint64_t ret_index, ret_hash, result;
		int ret_hash_len;
		int still_have_space = 1;
		int fp_count = 0;
        double churn_rate = 0.1;
        double inst_rate = 0.01;
        double churn_prop = 0.2;
        int num_replace = (int)(churn_prop * num_inserts);
        uint64_t temp;
        int churn_count = 0;
        int inst_count = 0;
        int space_between_churn = (int)(churn_rate * num_queries);
        int space_between_inst = (int)(inst_rate * num_queries);
		int start_index, end_index;
		int num_churns = 0;
		int inst_fp_count;
		int num_negative;
		uint64_t dummy_return;
		double inst_fpr;
		for (int j = 0; j < num_queries; j++) {
            if (churn_count == space_between_churn) {
				if (verbose) fprintf(stderr, "performing churn %d\n", num_churns);
				churn_count = 0;
                // perform a churn by replacing elements from the insert and the deletion set
				start_index = (num_churns % 5) * num_replace;
				end_index = start_index + num_replace;
				for (int k = start_index; k < end_index; k++) {
					temp = insert_set[k];
					insert_set[k] = negative_set[k];
					negative_set[k] = temp;
				}
                // will need to delete the filter and reverse map and remake with the new contents
				qf_free(&qf);
				set_free(set, num_inserts);
				set_free(positives, num_inserts);
				if (!qf_malloc(&qf, nslots, nhashbits, 0, QF_HASH_INVERTIBLE, 0)) {
					fprintf(stderr, "Can't allocate CQF.\n");
					abort();
				}
				qf_set_auto_resize(&qf, false);
				set = calloc(num_inserts, sizeof(set_node));
				positives = calloc(num_inserts, sizeof(set_node));
				for (int k = 0; k < num_inserts; k++) {
					if(!insert_key(&qf, set, num_inserts, insert_set[k], 1, &timecheck, &filter_insert_time, &set_insert_time)) {
						fprintf(stderr, "insertion %d failed\n", j);
						exit(1);
					}
					set_insert(positives, num_inserts, insert_set[k], insert_set[k]);
				}
				num_churns++;
            }
            if (inst_count == space_between_inst) {
				inst_count = 0;
				inst_fp_count = 0;
				num_negative = 0;
                // perform an instantaneous fpr check by querying all elements without adaptation
                // record the fpr in the file here.
				for (int k = 0; k < num_queries; k++) {
					if (set_query(positives, num_inserts, query_set[k], &dummy_return) == 0) {
						num_negative++;
					}
					result = qf_query(&qf, query_set[k], &ret_index, &ret_hash, &ret_hash_len, QF_KEY_IS_HASH);
					if (result) {
						uint64_t temp = ret_hash | (1ull << ret_hash_len), orig_key = 0;
						set_query(set, num_inserts, temp, &orig_key);
						if (query_set[k] != orig_key) {
							inst_fp_count++;
						}
					}
				}
				inst_fpr = (double)inst_fp_count / (inst_fp_count + num_negative);
				// if (verbose) printf("False positive rate: %f\n", inst_fpr);
				// write the instantaneous fpr to file
				FILE * outputptr;
				outputptr = fopen("../results/aqf/aqf_results_dynamic.csv", "a");
				fseek(outputptr, 0, SEEK_END);
				long filesize = ftell(outputptr);
				if (filesize == 0) {
					// File is empty, write header
					fprintf(outputptr, "dataset,query_dist,num_queries,curr_query,q,r,size,fpr\n");
				}

				char new_result_row[512];
				snprintf(new_result_row, sizeof(new_result_row), "%s,%s,%ld,%d,%ld,%ld,%ld,%.14f\n",
						dataset, dist, num_queries, j, qbits, rbits, qf.metadata->total_size_in_bytes, inst_fpr);

				fprintf(outputptr, new_result_row);
				fclose(outputptr);
            }
			result = qf_query(&qf, query_set[j], &ret_index, &ret_hash, &ret_hash_len, QF_KEY_IS_HASH);
			if (result) {
				uint64_t temp = ret_hash | (1ull << ret_hash_len), orig_key = 0;
				set_query(set, num_inserts, temp, &orig_key);

				if (query_set[j] != orig_key) {
					fp_count++;
					if (still_have_space) {
						ret_hash_len = qf_adapt(&qf, ret_index, orig_key, query_set[j], &ret_hash, QF_KEY_IS_HASH | QF_NO_LOCK);
						if (ret_hash_len > 0) {
							int ret = set_delete(set, num_inserts, temp);
							if (ret == 0) {
								printf("%d\n", j);
								abort();
							}
							set_insert(set, num_inserts, ret_hash | (1ull << ret_hash_len), orig_key);
						}
						else if (ret_hash_len == QF_NO_SPACE) {
							still_have_space = 0;
							fprintf(stderr, "\rfilter is full after %d queries\n", j);
							continue;
						}
					}
				}
			}
            churn_count++;
            inst_count++;
		}
		
		set_free(set, num_inserts);
		set_free(positives, num_inserts);
		qf_free(&qf);
	}
	free(query_set);
	free(query_labels);
	free(insert_set);
	free(buffer);
	return 0;
}