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
#define READ_SIZE 4096
#define URL_PATH "../data/malicious_url_scores.csv"

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

	if (verbose) printf("Processing insertions\n");

	uint64_t *insert_set = calloc(MAX_DATA_LINES, sizeof(uint64_t));
	if (!insert_set) {
		printf("calloc insert_set failed\n");
		return EXIT_FAILURE;
	}
	uint64_t *negative_set = calloc(MAX_DATA_LINES, sizeof(uint64_t));
	if (!negative_set) {
		printf("calloc insert_set failed\n");
		return EXIT_FAILURE;
	}
	long *offsets = calloc(sizeof(long), MAX_DATA_LINES);
	if (!offsets) {
		printf("calloc offsets failed\n");
		return EXIT_FAILURE;
	}
	char *buffer = malloc(READ_SIZE);
	if (!buffer) {
		printf("malloc buffer failed\n");
		return EXIT_FAILURE;
	}

	FILE * file_ptr;
	file_ptr = fopen(filename, "r");
	if (NULL == file_ptr) {
		printf("file can't be opened \n");
		return 0;
	}
	int curr_inserts = 0;
	int curr_neg = 0;
	int curr_index = 0;
	fgets(buffer, READ_SIZE, file_ptr); // get rid of the first header row
	offsets[curr_index] = ftell(file_ptr);
	while (fgets(buffer, READ_SIZE, file_ptr)) {
		if (strlen(buffer) == 0) {
			continue; // skip blank lines
		}
		if (strcmp(filename, URL_PATH) == 0) {
			char * malicious = strstr(buffer, ",malicious,");
			char *url;
			if (buffer[0] == "\"") {
				// if the first character is a ", it means that the element is a weirdly-formatted url
				int url_index = 1;
				while (buffer[url_index] != "\"") url_index++;
				// the url will be the substring from index 1 to url_index
				strncpy(url, buffer + 1, url_index-1);
			} else {
				// otherwise, simply dividing the str with "," gives the url
				url = strtok(buffer, ",");
			}
			if (url != NULL) {
				if (malicious) {
					insert_set[curr_inserts++] = hash_str(url);
				}  else {
					negative_set[curr_neg++] = hash_str(url);
				}
			}
		} else {
			// normally formatted with 1 as a positive indicator
			// read label and item then insert as necessary
			char *label = NULL;
			char *item = NULL;
			char *token = strtok(buffer, ",");
			int count = 0;
			while (token != NULL) {
				if (count == label_index) {
					label = token;
				} else if (count == obj_index) {
					item = token;
				}
				token = strtok(NULL, ",");
				count++;
			}
			char *item_copy = strdup(item);
			if (label != NULL && atoi(label) == 1) {
				insert_set[curr_inserts++] = hash_str(item_copy);
			} else {
				negative_set[curr_neg++] = hash_str(item_copy);
			}
		}
		curr_index++;
		offsets[curr_index] = ftell(file_ptr);
	}
	fclose(file_ptr);

	if (verbose) fprintf(stderr, "finished reading %d insertions and %d negative values\n", curr_inserts, curr_neg);
	
	if (curr_inserts > curr_neg) {
		fprintf(stderr, "not enough negative values in dataset to perform dynamic test\n");
		return EXIT_FAILURE;
	}
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
		set_node *set = calloc(curr_inserts, sizeof(set_node));

		// perform inserts
		

		// perform queries
		uint64_t ret_index, ret_hash, result;
		int ret_hash_len;
		int still_have_space = 1;
		int fp_count = 0;
        double churn_rate = 0.1;
        double inst_rate = 0.01;
        double churn_prop = 0.2;
        int num_replace = (int)(churn_prop * curr_inserts);
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

		set_node *positives = calloc(curr_inserts, sizeof(set_node));
		uint64_t filter_insert_time, set_insert_time;
		for (int j = 0; j < curr_inserts; j++) {
			insert_set[j] = MurmurHash64A(&insert_set[j], sizeof(insert_set[j]), murmur_seed);
			query_set[j] = MurmurHash64A(&query_set[j], sizeof(query_set[j]), murmur_seed);
			int result = insert_key(&qf, set, curr_inserts, insert_set[j], 1, &timecheck, &filter_insert_time, &set_insert_time);
			set_insert(positives, curr_inserts, insert_set[j], insert_set[j]);
			if (!result) {
				fprintf(stderr, "insertion %d failed\n", j);
				exit(1);
			}
		}

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
				set_free(set, curr_inserts);
				set_free(positives, curr_inserts);
				qf_free(&qf);
				if (!qf_malloc(&qf, nslots, nhashbits, 0, QF_HASH_INVERTIBLE, 0)) {
					fprintf(stderr, "Can't allocate AQF.\n");
					abort();
				}
				qf_set_auto_resize(&qf, false);
				set = calloc(curr_inserts, sizeof(set_node));
				positives = calloc(curr_inserts, sizeof(set_node));
				
				for (int k = 0; k < curr_inserts; k++) {
					int result = insert_key(&qf, set, curr_inserts, insert_set[j], 1, &timecheck, &filter_insert_time, &set_insert_time);
					set_insert(positives, curr_inserts, insert_set[j], insert_set[j]);
					if (!result) {
						fprintf(stderr, "insertion %d failed\n", k);
						exit(1);
					}
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
					if (set_query(positives, curr_inserts, query_set[k], &dummy_return) == 0) {
						num_negative++;
					}
					result = qf_query(&qf, query_set[k], &ret_index, &ret_hash, &ret_hash_len, QF_KEY_IS_HASH);
					if (result) {
						uint64_t temp = ret_hash, orig_key = 0;
						set_query(set, curr_inserts, temp, &orig_key);
						if (query_set[k] != orig_key) {
							inst_fp_count++;
						}
					}
				}
				inst_fpr = (double)inst_fp_count / (inst_fp_count + num_negative);
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
				uint64_t temp = ret_hash, orig_key = 0;
				set_query(set, curr_inserts, temp, &orig_key);

				if (query_set[j] != orig_key) {
					fp_count++;
					if (still_have_space) {
						ret_hash_len = qf_adapt(&qf, ret_index, orig_key, query_set[j], &ret_hash, QF_KEY_IS_HASH | QF_NO_LOCK);
						if (ret_hash_len > 0) {
							int ret = set_delete(set, curr_inserts, temp);
							if (ret == 0) {
								printf("%d\n", j);
								abort();
							}
							set_insert(set, curr_inserts, ret_hash, orig_key);
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
		
		set_free(set, curr_inserts);
		set_free(positives, curr_inserts);
		qf_free(&qf);
	}
	free(query_set);
	free(query_labels);
	free(insert_set);
	free(buffer);
	return 0;
}