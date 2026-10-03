/*
For the given query set following some static distribution (i.e. uniform or Zipfian),
performs all queries on the chosen filter and reports FPR and construction / query times.

The filter is selected by name (see src/filter_registry.c), so this one harness replaces
test_distribution.c and test_distribution_stacked.c.
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
#include "include/filter_iface.h"

#define MAX_DATA_LINES 10000000
#define OUTPUT_PATH "../results/distribution_results.csv"

int main(int argc, char **argv)
{
	int num_trials = 3;
	int verbose = 1;
	char indexfilename[512];
	const filter_ops *ops = (argc > 1) ? find_filter(argv[1]) : NULL;
	if (argc < 5 || !ops || argc < 5 + ops->num_args) {
		fprintf(stderr, "Please specify \nthe filter name [eg. aqf, stacked]\nthe file path [eg. datasets/Malware_data.csv]\nthe index file path [eg. datasets/hashed_unif_10M_url.csv]\nthe number of queries [eg. 100000000]\nthe filter-specific arguments\n");
		if (ops) fprintf(stderr, "arguments for %s: %s\n", ops->name, ops->usage);
		exit(1);
	}

	size_t num_queries = strtoull(argv[4], NULL, 10);
	char filename[100];
	strcpy(filename, argv[2]);
	strcpy(indexfilename, argv[3]);
	int filter_argc = argc - 5;
	char **filter_argv = argv + 5;
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

	// now we build the filter and perform the queries.

	struct timeval timecheck;
	for (int i = 0; i < num_trials; i++) {
		time_t seed = time(NULL);
		srand(i);
		if (verbose) printf("Running trial %d on seed %ld\n", i, seed);
		int murmur_seed = rand();

		if (ops->rehash_per_trial) {
			for (int j = 0; j < num_inserts; j++) {
				insert_set[j] = MurmurHash64A(&insert_set[j], sizeof(insert_set[j]), murmur_seed);
			}
			for (size_t j = 0; j < num_queries; j++) {
				query_set[j] = MurmurHash64A(&query_set[j], sizeof(query_set[j]), murmur_seed);
			}
		}

		build_ctx ctx = {
			.positives = insert_set,
			.num_positives = num_inserts,
			.argc = filter_argc,
			.argv = filter_argv,
		};
		uint64_t *train_negatives = NULL;
		double *train_cdf = NULL;
		if (ops->train_proportion) {
			int num_check = (int)(num_queries * ops->train_proportion(filter_argc, filter_argv));
			ctx.num_train_negatives = build_training_negatives(query_set, query_labels, num_queries, num_check,
															   &train_negatives, &train_cdf);
			ctx.train_negatives = train_negatives;
			ctx.train_cdf = train_cdf;
		}

		// build the filter
		gettimeofday(&timecheck, NULL);
		uint64_t start_time = timecheck.tv_sec * 1000000 + timecheck.tv_usec, end_time;
		void *filter = ops->create(&ctx);
		gettimeofday(&timecheck, NULL);
		end_time = timecheck.tv_sec * 1000000 + timecheck.tv_usec;
		uint64_t build_time = end_time - start_time;

		// perform queries
		int fp_count = 0;
		uint64_t total_query_time = 0;
		uint64_t total_adapt_time = 0;
		for (size_t j = 0; j < num_queries; j++) {
			gettimeofday(&timecheck, NULL);
			start_time = timecheck.tv_sec * 1000000 + timecheck.tv_usec;
			int result = ops->query(filter, query_set[j]);
			gettimeofday(&timecheck, NULL);
			end_time = timecheck.tv_sec * 1000000 + timecheck.tv_usec;
			total_query_time += end_time - start_time;
			if (result && query_labels[j] == 0) {
				// negative query label but it was found in the filter, so it's a false positive
				fp_count++;
				if (ops->on_false_positive) {
					gettimeofday(&timecheck, NULL);
					start_time = timecheck.tv_sec * 1000000 + timecheck.tv_usec;
					ops->on_false_positive(filter, query_set[j]);
					gettimeofday(&timecheck, NULL);
					end_time = timecheck.tv_sec * 1000000 + timecheck.tv_usec;
					total_adapt_time += end_time - start_time;
				}
			}
		}

		// calculate the false positive rate
		int total_negatives = num_queries - pos_count;
		if (verbose) printf("Total false positives in query set: %d\n", fp_count);
		double fpr = (double)fp_count / total_negatives;
		if (verbose) printf("False positive rate: %f\n", fpr);

		char params[128];
		ops->params_str(filter, params, sizeof(params));
		uint64_t aux_time = ops->aux_build_time ? ops->aux_build_time(filter) : 0;

		FILE * outputptr;
		outputptr = fopen(OUTPUT_PATH, "a");
		fseek(outputptr, 0, SEEK_END);
		long filesize = ftell(outputptr);
		if (filesize == 0) {
			// File is empty, write header
			fprintf(outputptr, "dataset,query_dist,num_queries,filter,params,size,fpr,build_time,aux_build_time,amortized_query,amortized_adapt\n");
		}
		fprintf(outputptr, "%s,%s,%ld,%s,%s,%ld,%.14f,%ld,%ld,%.14f,%.14f\n",
				dataset, dist, num_queries, ops->name, params, ops->size_bytes(filter), fpr,
				build_time, aux_time,
				(double)total_query_time/num_queries, (double)total_adapt_time/num_queries);
		fclose(outputptr);

		ops->destroy(filter);
		free(train_negatives);
		free(train_cdf);
	}
	free(query_set);
	free(query_labels);
	free(insert_set);
	free(offsets);
	free(buffer);
	return 0;
}
