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
#include "include/harness_output.h"

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

	// row index of each insertion and query, used to look up features for filters that use them
	int *insert_rows = ops->uses_features ? malloc(MAX_DATA_LINES * sizeof(int)) : NULL;
	int *query_rows = ops->uses_features ? malloc(num_queries * sizeof(int)) : NULL;

	int num_inserts;
	int read_result = read_file(filename, obj_index, label_index,
								buffer, offsets, insert_set,
								&num_inserts, insert_rows);
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
										buffer, offsets, query_set, query_labels, num_queries,
										&pos_count, &total_queries, query_rows);
	if (!query_read_result) {
		printf("Query reading failed\n");
		return EXIT_FAILURE;
	}
	if (verbose) fprintf(stderr, "finished reading %d queries with %d positives\n", total_queries, pos_count);
	if ((size_t)total_queries < num_queries) {
		// the index file is shorter than requested: only query what was read
		fprintf(stderr, "index file only has %d queries, running %d instead of %zu\n", total_queries, total_queries, num_queries);
		num_queries = total_queries;
	}

	// now we build the filter and perform the queries.

	// load the feature vector of every row for filters that use them, checking that the harness's
	// rows line up with the exported features
	float *row_features = NULL, *positive_features = NULL;
	size_t num_features = 0, num_feature_rows = 0;
	char feature_dir[256] = "";
	if (ops->uses_features) {
		snprintf(feature_dir, sizeof(feature_dir), "../data/cascaded/%s", dataset);
		char path[512];
		size_t num_rows = 0, label_rows = 0, label_cols = 0;
		snprintf(path, sizeof(path), "%s/features.bin", feature_dir);
		row_features = map_float_matrix(path, &num_rows, &num_features);
		num_feature_rows = num_rows;
		snprintf(path, sizeof(path), "%s/labels.bin", feature_dir);
		float *row_labels = load_float_matrix(path, &label_rows, &label_cols);
		if (!row_features || !row_labels || label_rows != num_rows
			|| !check_row_labels(row_labels, num_rows, insert_rows, NULL, num_inserts, "insertion")
			|| !check_row_labels(row_labels, num_rows, query_rows, query_labels, num_queries, "query")) {
			fprintf(stderr, "couldn't load features for %s from %s (see learned/utils/export_cascaded_inputs.py)\n", ops->name, feature_dir);
			return EXIT_FAILURE;
		}
		free(row_labels);
		positive_features = malloc((size_t)num_inserts * num_features * sizeof(float));
		gather_rows(row_features, num_features, insert_rows, num_inserts, positive_features);
		if (verbose) fprintf(stderr, "loaded %zu features for each of %zu rows from %s\n", num_features, num_rows, feature_dir);
	}

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
			.positive_features = positive_features,
			.num_features = (int)num_features,
			.feature_dir = feature_dir,
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
		uint64_t start_time = TIMED_NOW(), end_time;
		void *filter = ops->create(&ctx);
		end_time = TIMED_NOW();
		uint64_t config_time = ops->config_time ? ops->config_time(filter) : 0;
		uint64_t sizing_time = ops->sizing_time ? ops->sizing_time(filter) : 0;
		uint64_t build_time = end_time - start_time - config_time - sizing_time;

		// perform queries
		int fp_count = 0;
		uint64_t fp_from_model = 0, fp_from_backup = 0;
		uint64_t total_query_time = 0;
		uint64_t total_adapt_time = 0;
		for (size_t j = 0; j < num_queries; j++) {
			start_time = TIMED_NOW();
			int result = ops->query(filter, query_set[j], row_features ? row_features + (size_t)query_rows[j] * num_features : NULL);
			end_time = TIMED_NOW();
			total_query_time += end_time - start_time;
			if (result && query_labels[j] == 0) {
				// negative query label but it was found in the filter, so it's a false positive
				fp_count++;
				count_fp_source(ops, filter, &fp_from_model, &fp_from_backup);
				if (ops->on_false_positive) {
					start_time = TIMED_NOW();
					ops->on_false_positive(filter, query_set[j]);
					end_time = TIMED_NOW();
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
		outputptr = open_results_file(OUTPUT_PATH);
		long filesize = ftell(outputptr);
		if (filesize == 0) {
			// File is empty, write header
			fprintf(outputptr, "dataset,query_dist,num_queries,filter,params,size,fpr," MEASUREMENT_COLUMNS "\n");
		}
		char measurements[512];
			format_measurements(measurements, sizeof(measurements), ops, filter, build_time, aux_time,
				total_query_time, total_adapt_time, num_queries, config_time, fp_from_model, fp_from_backup);
			fprintf(outputptr, "%s,%s,%ld,%s,%s,%ld,%.14f,%s\n",
				dataset, dist, num_queries, ops->name, params, ops->size_bytes(filter), fpr,
				measurements);
		fclose(outputptr);

		ops->destroy(filter);
		free(train_negatives);
		free(train_cdf);
	}
	free(query_set);
	free(query_labels);
	free(insert_set);
	free(insert_rows);
	free(query_rows);
	unmap_float_matrix(row_features, num_feature_rows, num_features);
	free(positive_features);
	free(offsets);
	free(buffer);
	return 0;
}
