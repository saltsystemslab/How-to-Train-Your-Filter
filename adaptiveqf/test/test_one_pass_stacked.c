/*
For the given dataset, performs the one-pass experiment on the stacked filter,
querying every positive and negative element exactly once.
Reports the FPR for the filter on the dataset.
*/
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
#include "stacked_wrapper.h"

#define MAX_DATA_LINES 10000000
#define MAX_LINE_LENGTH 2048
#define READ_SIZE 4096
#define URL_PATH "../data/malicious_url_scores.csv"

int main(int argc, char **argv)
{
	int num_trials = 3;
	int verbose = 1;
	if (argc < 5) {
		fprintf(stderr, "Please specify \nthe file path [eg. datasets/Malware_data.csv]\nthe total size in bytes of the filter [eg. 1024]\nthe proportion of queries to track for negatives [eg. 0.25]\nthe index filename\n");
		exit(1);
	}
	
	char* end;
	size_t total_size = atoi(argv[2]);
	double negative_proportion = strtod(argv[3], &end);
    char filename[100];
	char indexfilename[512];
	strcpy(indexfilename, argv[4]);
	strcpy(filename, argv[1]);
	int obj_index = get_obj_index(filename);
	int label_index = get_label_index(filename);
	char* dataset = get_dataset_name(filename);
	char* dist = "onepass";

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
	char *buffer = malloc(READ_SIZE * 100);
	int num_inserts;
	int read_result = read_file(filename, obj_index, label_index, 
								buffer, offsets, insert_set, 
								&num_inserts);

	if (!read_result) {
		printf("Insertion reading failed\n");
		return EXIT_FAILURE;
	}

	if (verbose) fprintf(stderr, "finished reading %d insertions\n", num_inserts);


	uint64_t *query_set = calloc(MAX_DATA_LINES, sizeof(uint64_t));
	if (!query_set) {
		fprintf(stderr, "calloc failed for query_set\n");
		return EXIT_FAILURE;
	}
	uint64_t *query_labels = calloc(MAX_DATA_LINES, sizeof(uint64_t));
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

	fprintf(stderr, "finished reading %d insertions and %d queries\n", num_inserts, total_queries);

	// create a map of key to count, which we later convert to an array for sorting.
    set_node *count_map = calloc(num_inserts, sizeof(set_node));
    if (!count_map) {
        fprintf(stderr, "calloc failed for count_map\n");
        return EXIT_FAILURE;
    }


    // first, track all the counts of the negative queries in the first 25% of queries
    uint64_t *value;
    uint64_t num_check = (uint64_t)(total_queries * negative_proportion); // adjust this later?

	int num_negative = 0;
    for (uint64_t i = 0; i < num_check; i++) {
        if (query_labels[i] == 0) {
            value = 0;
            if (set_query(count_map, num_inserts, query_set[i], &value)) {
                // key already exists, so just increment count
                set_update(count_map, num_inserts, query_set[i], value + 1);
            } else {
                // key doesn't exist, start a new count
                set_insert(count_map, num_inserts, query_set[i], 1);
                num_negative++;
            }
        }
    }
    fprintf(stderr, "found %d unique negative queries in first %ld queries\n", num_negative, num_check);

    // now, convert the map to an array for sorting
    struct key_count *count_array = malloc(num_negative * sizeof(struct key_count));
	if (!count_array) {
        fprintf(stderr, "malloc failed for count_array\n");
        return EXIT_FAILURE;
    }
    int index = 0;
    for (int i = 0; i < num_inserts; i++) {
        // at the current index in the count map, check if all keys are non-zero
        set_node *ptr = &count_map[i];
        while  (ptr) {
            if (ptr->key != 0) {
                count_array[index].key = ptr->key;
                count_array[index].count = ptr->value;
                index++;
            }
            ptr = ptr->next;
        }
    }
    free(count_map);

    // now, sort the array in descending order by count
    qsort(count_array, num_negative, sizeof(struct key_count), key_count_comp);

    // create an array of sorted negative keys
    uint64_t *sorted_negatives = malloc(num_negative * sizeof(uint64_t));
    if (!sorted_negatives) {
        fprintf(stderr, "malloc failed for sorted_negatives\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < num_negative; i++) {
        sorted_negatives[i] = count_array[i].key;
    }

    // create an array of cdf values for the negatives
    double *cdf = malloc(num_negative * sizeof(double));
    if (!cdf) {
        fprintf(stderr, "malloc failed for cdf\n");
        return EXIT_FAILURE;
    }
    uint64_t total_count = 0;
    for (int i = 0; i < num_negative; i++) {
        total_count += count_array[i].count;
    }
	double sum = 0;
    for (int i = 0; i < num_negative; i++) {
		sum += count_array[i].count / total_count;
        cdf[i] = sum;
    }
    free(count_array);

    WrappedStackedFilter *filter;

	// current_index describes the number of lines in the file,
	// which is also the number of queries for a one-pass test

	struct timeval timecheck;
	
	for (int i = 0; i < num_trials; i++) {
		time_t seed = time(NULL);
		srand(seed);
		if (verbose) printf("Running trial %d on seed %ld\n", i, seed);
		int murmur_seed = rand();

		// create the filter
		
		gettimeofday(&timecheck, NULL);
		uint64_t start_time = timecheck.tv_sec * 1000000 + timecheck.tv_usec, end_time;
        WrappedStackedFilter *stacked_filter = StackedFilterCreate(total_size*8, insert_set, num_inserts, sorted_negatives, num_negative, cdf, num_negative, 0);
		gettimeofday(&timecheck, NULL);
		end_time = timecheck.tv_sec * 1000000 + timecheck.tv_usec;
		uint64_t filter_alloc_time = end_time - start_time;

		// perform queries
		int ret_hash_len;
		int still_have_space = 1;
		int fp_count = 0;
		int result;
		uint64_t total_query_time = 0;
		for (int j = 0; j < total_queries; j++) {
			gettimeofday(&timecheck, NULL);
			start_time = timecheck.tv_sec * 1000000 + timecheck.tv_usec;
			result = StackedFilterLookupElement(stacked_filter, &query_set[j]);
			gettimeofday(&timecheck, NULL);
			end_time = timecheck.tv_sec * 1000000 + timecheck.tv_usec;
			total_query_time += end_time - start_time;
			if (result) {
				if (query_labels[j] == 0) {
					fp_count++;
				}
			}
		}

		// calculate the false positive rate
		int total_negatives = total_queries- num_inserts;
		fprintf(stderr, "total negatives in query set: %d\n", total_negatives);
		if (verbose) printf("Total false positives in query set: %d\n", fp_count);
		double fpr = (double)fp_count / (fp_count + total_negatives);
		if (verbose) printf("False positive rate: %f\n", fpr);


		FILE * outputptr;
		outputptr = fopen("../results/stacked/stacked_results.csv", "a");
		fseek(outputptr, 0, SEEK_END);
		long filesize = ftell(outputptr);
		if (filesize == 0) {
			// File is empty, write header
			fprintf(outputptr, "dataset,query_dist,num_queries,prop_train,size,fpr,insert_time,amortized_query\n");
		}

		char new_result_row[512];
		snprintf(new_result_row, sizeof(new_result_row), "%s,%s,%ld,%.14f,%ld,%.14f,%ld,%.14f\n",
				dataset, dist, total_queries, negative_proportion, total_size, fpr,
				filter_alloc_time,
				(double)total_query_time/total_queries);

		fprintf(outputptr, new_result_row);
		fclose(outputptr);
		StackedFilterDestroy(stacked_filter);
	}
	free(query_set);
	free(query_labels);
	free(insert_set);
	free(buffer);
	return 0;
}