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
	int num_trials = 3;
	int verbose = 1;
	if (argc < 4) {
		fprintf(stderr, "Please specify \nthe file path [eg. datasets/Malware_data.csv]\nthe log of the number of slots in the QF [eg. 20]\nthe number of remainder bits in the QF [eg. 9]\n");
		exit(1);
	}
	
	size_t qbits = atoi(argv[2]);
	size_t rbits = atoi(argv[3]);
    char filename[100];
	strcpy(filename, argv[1]);
	int obj_index = get_obj_index(filename);
	int label_index = get_label_index(filename);
	char* dataset = get_dataset_name(filename);
	char* dist = "onepass";

	// First we go into the file and collect all of the insertions from the dataset,
	// building a query set at the same time.

	FILE * file_ptr;
	file_ptr = fopen(filename, "r");
	long *offsets = malloc(sizeof(long) * MAX_DATA_LINES);
	if (NULL == file_ptr) {
		printf("file can't be opened \n");
		return EXIT_FAILURE;
	}
	uint64_t *insert_set = malloc(MAX_DATA_LINES * sizeof(uint64_t));
	if (!insert_set) {
		printf("malloc insert_set failed");
		return EXIT_FAILURE;
	}
	uint64_t *query_set = malloc(MAX_DATA_LINES * sizeof(uint64_t));
	if (!query_set) {
		fprintf(stderr, "malloc failed for query_set\n");
		return 1;
	}
	uint64_t *query_labels = malloc(MAX_DATA_LINES * sizeof(uint64_t));
	if (!query_labels) {
		fprintf(stderr, "malloc failed for query_labels\n");
		return 1;
	}
	if (RAND_bytes((unsigned char*)insert_set, MAX_DATA_LINES * sizeof(uint64_t)) != 1) {
		printf("RAND_bytes failed\n");
		return EXIT_FAILURE;
	}
	if (verbose) printf("Processing insertions\n");
	char* buffer = malloc(READ_SIZE);
	int curr_inserts = 0;
	int curr_index = 0;
	fgets(buffer, READ_SIZE, file_ptr); // get rid of the first header row
	offsets[curr_index] = ftell(file_ptr);
	while (fgets(buffer, READ_SIZE, file_ptr)) {
		if (strlen(buffer) == 0) {
			continue; // skip blank lines
		}
		if (strcmp(filename, URL_PATH) == 0) {
			query_labels[curr_index] = strstr(buffer, ",malicious,") ? 1 : 0;
			char *url;
			if (strstr(buffer, ",malicious,")) {
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
					insert_set[curr_inserts++] = hash_str(url);
				}
			} else {
				url = strtok(buffer, ",");
			}
			query_set[curr_index] = hash_str(url);
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
				query_labels[curr_index] = 1;
			} else {
				query_labels[curr_index] = 0;
			}
			query_set[curr_index] = hash_str(item_copy);
		}
		curr_index++;
		offsets[curr_index] = ftell(file_ptr);
	}
	fclose(file_ptr);
	if (verbose) fprintf(stderr, "finished reading %d insertions and %d queries\n", curr_inserts, curr_index);

	// At this point, we now have a large list of insertions,
	// and a list of file offsets corresponding to indices.

	// current_index describes the number of lines in the file,
	// which is also the number of queries for a one-pass test

	QF qf;
	struct timeval timecheck;
	
	for (int i = 0; i < num_trials; i++) {
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
		set_node *set = calloc(MAX_DATA_LINES, sizeof(set_node));
		gettimeofday(&timecheck, NULL);
		end_time = timecheck.tv_sec * 1000000 + timecheck.tv_usec;
		uint64_t set_alloc_time = end_time - start_time;

		// perform inserts
		uint64_t total_filter_insert_time = 0;
		uint64_t total_set_insert_time = 0;
		uint64_t filter_insert_time, set_insert_time;
		for (int j = 0; j < curr_inserts; j++) {
			insert_set[j] = MurmurHash64A(&insert_set[j], sizeof(insert_set[j]), murmur_seed);
			int result = insert_key(&qf, set, curr_inserts, insert_set[j], 1, &timecheck, &filter_insert_time, &set_insert_time);
			total_filter_insert_time += filter_insert_time;
			total_set_insert_time += set_insert_time;
			if (!result) {
				fprintf(stderr, "insertion %d failed\n", j);
				exit(1);
			}
		}
		if (verbose) fprintf(stderr, "finished %d insertions\n", curr_inserts);

		// perform queries
		uint64_t ret_index, ret_hash, result;
		int ret_hash_len;
		int still_have_space = 1;
		int fp_count = 0;
		uint64_t total_query_time = 0;
		uint64_t total_adapt_time = 0;
		if (verbose) printf("Performing %d queries\n", curr_index);
		for (int j = 0; j < curr_index; j++) {
			query_set[j] = MurmurHash64A(&query_set[j], sizeof(query_set[j]), murmur_seed);
			gettimeofday(&timecheck, NULL);
			start_time = timecheck.tv_sec * 1000000 + timecheck.tv_usec;
			result = qf_query(&qf, query_set[j], &ret_index, &ret_hash, &ret_hash_len, QF_KEY_IS_HASH);
			gettimeofday(&timecheck, NULL);
			end_time = timecheck.tv_sec * 1000000 + timecheck.tv_usec;
			total_query_time += end_time - start_time;
			if (result) {
				uint64_t temp = ret_hash, orig_key = 0;
				set_query(set, curr_inserts, ret_hash, &orig_key);

				if (query_labels[j] == 0) {
					fp_count++;
					if (still_have_space) {
						gettimeofday(&timecheck, NULL);
						start_time = timecheck.tv_sec * 1000000 + timecheck.tv_usec;
						ret_hash_len = qf_adapt(&qf, ret_index, orig_key, query_set[j], &ret_hash, QF_KEY_IS_HASH | QF_NO_LOCK);
						if (ret_hash_len > 0) {
							int ret = set_delete(set, curr_inserts, temp);
							if (ret == 0) {
								printf("%d\n", j);
								abort();
							}
							set_insert(set, curr_inserts, ret_hash, orig_key);
							gettimeofday(&timecheck, NULL);
							end_time = timecheck.tv_sec * 1000000 + timecheck.tv_usec;
							total_adapt_time += end_time - start_time;
						}
						else if (ret_hash_len == QF_NO_SPACE) {
							still_have_space = 0;
							fprintf(stderr, "\rfilter is full after %d queries\n", j);
							continue;
						}
					}
				}
			}
		}

		// calculate the false positive rate
		int total_negatives = curr_index - curr_inserts;
		if (verbose) printf("Total false positives in query set: %d\n", fp_count);
		double fpr = (double)fp_count / (fp_count + total_negatives);
		if (verbose) printf("False positive rate: %f\n", fpr);


		FILE * outputptr;
		outputptr = fopen("../results/aqf/aqf_results.csv", "a");
		fseek(outputptr, 0, SEEK_END);
		long filesize = ftell(outputptr);
		if (filesize == 0) {
			// File is empty, write header
			fprintf(outputptr, "dataset,query_dist,num_queries,q,r,size,fpr,insert_time,map_time,amortized_query,amortized_adapt\n");
		}

		char new_result_row[512];
		snprintf(new_result_row, sizeof(new_result_row), "%s,%s,%d,%ld,%ld,%ld,%.14f,%ld,%ld,%.14f,%.14f\n",
				dataset, dist, curr_index, qbits, rbits, qf.metadata->total_size_in_bytes, fpr,
				total_filter_insert_time + filter_alloc_time, total_set_insert_time + set_alloc_time,
				(double)total_query_time/curr_index,(double)total_adapt_time/curr_index);

		fprintf(outputptr, new_result_row);
		fclose(outputptr);
		
		set_free(set, curr_inserts);
		qf_free(&qf);
	}
	free(query_set);
	free(query_labels);
	free(insert_set);
	return 0;
}