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

	// now we insert all elements into the filter and perform the queries.
	
	QF qf;
	struct timeval timecheck;
	for (int i = 0; i < num_trials; i++) {
		time_t seed = time(NULL);
		// srand(seed);
		srand(i);
		if (verbose) printf("Running trial %d on seed %ld\n", i, seed);
		int murmur_seed = rand();

		// create the filter
		uint64_t nhashbits = qbits + rbits;
		uint64_t nslots = (1ULL << qbits);
		
		gettimeofday(&timecheck, NULL);
		uint64_t start_time = timecheck.tv_sec * 1000000 + timecheck.tv_usec, end_time;
        if (!qf_malloc(&qf, nslots, nhashbits, 0, QF_HASH_INVERTIBLE, 0)) {
			fprintf(stderr, "Can't allocate QF.\n");
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

		int num_success = 0;
		fprintf(stderr, "started %d insertions\n", num_inserts);
		for (int j = 0; j < num_inserts; j++) {
			insert_set[j] = MurmurHash64A(&insert_set[j], sizeof(insert_set[j]), murmur_seed);
			int result = insert_key(&qf, set, num_inserts, insert_set[j], 1, &timecheck, &filter_insert_time, &set_insert_time);
			if (!result) {
				fprintf(stderr, "failed to insert\n");
				abort();
			}
			total_filter_insert_time += filter_insert_time;
			total_set_insert_time += set_insert_time;
			if (result) num_success++;
			if (!result) {
				fprintf(stderr, "insertion %d failed\n", j);
				for (int k = 0; k < j; k++) {
					if (insert_set[k] == insert_set[j]) {
						fprintf(stderr, "duplicate key found at %d\n", k);
					}
				}
				// exit(1);
			}
		}
		if (verbose) fprintf(stderr, "finished %d insertions with %d successes\n", num_inserts, num_success);

		if (verbose) fprintf(stderr, "started queries\n");
		// perform queries
		uint64_t ret_index, ret_hash, result;
		int ret_hash_len;
		int still_have_space = 1;
		int fp_count = 0;
		uint64_t total_query_time = 0;
		uint64_t total_adapt_time = 0;
		int successful_deletes = 0;
		for (int j = 0; j < num_queries; j++) {
			query_set[j] = MurmurHash64A(&query_set[j], sizeof(query_set[j]), murmur_seed);
			gettimeofday(&timecheck, NULL);
			start_time = timecheck.tv_sec * 1000000 + timecheck.tv_usec;
			result = qf_query(&qf, query_set[j], &ret_index, &ret_hash, &ret_hash_len, QF_KEY_IS_HASH | QF_NO_LOCK);
			gettimeofday(&timecheck, NULL);
			end_time = timecheck.tv_sec * 1000000 + timecheck.tv_usec;
			total_query_time += end_time - start_time;
			if (result) {
				// element was found in the filter, now check if it's a false positive
				uint64_t temp = ret_hash | (1ull << ret_hash_len), orig_key = 0;
				uint64_t old_ret_hash = ret_hash;
				// int set_result = set_query(set, num_inserts, temp, &orig_key);
				int set_result = set_query(set, num_inserts, ret_hash, &orig_key);

				if (query_labels[j] == 0) {
					// negative query label but it was found in the filter, so it's a false positive
					fp_count++;
					if (still_have_space) {
						gettimeofday(&timecheck, NULL);
							start_time = timecheck.tv_sec * 1000000 + timecheck.tv_usec;
							ret_hash_len = qf_adapt(&qf, ret_index, orig_key, query_set[j], &ret_hash, QF_KEY_IS_HASH | QF_NO_LOCK);
							if (ret_hash_len > 0) {
								int ret = set_delete(set, num_inserts, old_ret_hash);
								// int ret = set_delete(set, num_inserts, temp);
								if (ret == 0) {
									printf("%d\n", j);
									abort();
								}
								set_insert(set, num_inserts, ret_hash, orig_key);
								// set_insert(set, num_inserts, ret_hash | (1ull << ret_hash_len), orig_key);
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
		snprintf(new_result_row, sizeof(new_result_row), "%s,%s,%ld,%ld,%ld,%ld,%.14f,%ld,%ld,%.14f,%.14f\n",
				dataset, dist, num_queries, qbits, rbits, qf.metadata->total_size_in_bytes, fpr,
				total_filter_insert_time + filter_alloc_time, total_set_insert_time + set_alloc_time,
				(double)total_query_time/num_queries,(double)total_adapt_time/num_queries);

		fprintf(outputptr, new_result_row);
		fclose(outputptr);
		
		set_free(set, num_inserts);
		qf_free(&qf);
	}
	free(query_set);
	free(query_labels);
	free(insert_set);
	free(buffer);
	return 0;
}