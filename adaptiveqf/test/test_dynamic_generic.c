/*
Performs a dynamic experiment on the chosen filter: every churn_rate of the queries, a window of
the positive set is swapped with negatives and the filter is rebuilt from the new positive set.
Every inst_rate of the queries, an instantaneous FPR is measured over the whole query set
(without adaptation) and written to the results file.

The filter is selected by name (see src/filter_registry.c), so this one harness replaces
test_dynamic.c and test_dynamic_stacked.c.
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
#define READ_SIZE 4096
#define OUTPUT_PATH "../results/dynamic_results.csv"

// Builds a membership set for the current positives, used as ground truth since query labels
// no longer reflect membership once churns have swapped elements between the sets.
static set_node *build_membership(uint64_t *insert_set, int num_inserts) {
	set_node *positives = calloc(num_inserts, sizeof(set_node));
	if (!positives) {
		fprintf(stderr, "calloc failed for positives\n");
		exit(EXIT_FAILURE);
	}
	for (int k = 0; k < num_inserts; k++) {
		set_insert(positives, num_inserts, insert_set[k], insert_set[k]);
	}
	return positives;
}

int main(int argc, char **argv)
{
	int num_trials = 1;
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

	if (verbose) printf("Processing insertions\n");

	uint64_t *insert_set = calloc(MAX_DATA_LINES, sizeof(uint64_t));
	uint64_t *negative_set = calloc(MAX_DATA_LINES, sizeof(uint64_t));
	long *offsets = calloc(sizeof(long), MAX_DATA_LINES);
	char *buffer = malloc(READ_SIZE);
	if (!insert_set || !negative_set || !offsets || !buffer) {
		printf("allocation failed\n");
		return EXIT_FAILURE;
	}

	int curr_inserts, curr_neg;
	if (!read_file_with_negatives(filename, obj_index, label_index, buffer, offsets,
								  insert_set, &curr_inserts, negative_set, &curr_neg)) {
		printf("Insertion reading failed\n");
		return EXIT_FAILURE;
	}
	if (verbose) fprintf(stderr, "finished reading %d insertions and %d negative values\n", curr_inserts, curr_neg);
	if (curr_inserts > curr_neg) {
		fprintf(stderr, "not enough negative values in dataset to perform dynamic test\n");
		return EXIT_FAILURE;
	}

	uint64_t *query_set = calloc(num_queries, sizeof(uint64_t));
	uint64_t *query_labels = calloc(num_queries, sizeof(uint64_t));
	if (!query_set || !query_labels) {
		fprintf(stderr, "calloc failed for query_set or query_labels\n");
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

	for (int i = 0; i < num_trials; i++) {
		time_t seed = time(NULL);
		srand(i);
		if (verbose) printf("Running trial %d on seed %ld\n", i, seed);
		int murmur_seed = rand();

		if (ops->rehash_per_trial) {
			for (int j = 0; j < curr_inserts; j++) {
				insert_set[j] = MurmurHash64A(&insert_set[j], sizeof(insert_set[j]), murmur_seed);
			}
			for (size_t j = 0; j < num_queries; j++) {
				query_set[j] = MurmurHash64A(&query_set[j], sizeof(query_set[j]), murmur_seed);
			}
			// hash the slice of negative_set that churns swap into insert_set, so
			// swapped-in elements are keyed the same way as the rest of insert_set.
			for (int j = 0; j < curr_inserts; j++) {
				negative_set[j] = MurmurHash64A(&negative_set[j], sizeof(negative_set[j]), murmur_seed);
			}
		}

		build_ctx ctx = {
			.positives = insert_set,
			.num_positives = curr_inserts,
			.argc = filter_argc,
			.argv = filter_argv,
		};
		uint64_t *train_negatives = NULL;
		double *train_cdf = NULL;
		if (ops->train_proportion) {
			// the query stream does not change across churns, so the training negatives are learned once
			int num_check = (int)(num_queries * ops->train_proportion(filter_argc, filter_argv));
			ctx.num_train_negatives = build_training_negatives(query_set, query_labels, num_queries, num_check,
															   &train_negatives, &train_cdf);
			ctx.train_negatives = train_negatives;
			ctx.train_cdf = train_cdf;
		}

		void *filter = ops->create(&ctx);
		set_node *positives = build_membership(insert_set, curr_inserts);

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
		int num_churns = 0;
		uint64_t dummy_return;

		for (size_t j = 0; j < num_queries; j++) {
			if (churn_count == space_between_churn) {
				if (verbose) fprintf(stderr, "performing churn %d\n", num_churns);
				churn_count = 0;
				// perform a churn by replacing elements from the insert and the deletion set
				int start_index = (num_churns % 5) * num_replace;
				int end_index = start_index + num_replace;
				for (int k = start_index; k < end_index; k++) {
					temp = insert_set[k];
					insert_set[k] = negative_set[k];
					negative_set[k] = temp;
				}
				// rebuild the filter and membership set from the new positives
				ops->destroy(filter);
				set_free(positives, curr_inserts);
				filter = ops->create(&ctx);
				positives = build_membership(insert_set, curr_inserts);
				num_churns++;
			}
			if (inst_count == space_between_inst) {
				inst_count = 0;
				int inst_fp_count = 0;
				int num_negative = 0;
				// perform an instantaneous fpr check by querying all elements without adaptation
				for (size_t k = 0; k < num_queries; k++) {
					int is_member = set_query(positives, curr_inserts, query_set[k], &dummy_return);
					if (!is_member) {
						num_negative++;
					}
					if (ops->query(filter, query_set[k]) && !is_member) {
						inst_fp_count++;
					}
				}
				double inst_fpr = (double)inst_fp_count / num_negative;
				if (verbose) fprintf(stderr, "instantaneous FPR check: %d false positives out of %d negatives\n", inst_fp_count, num_negative);

				char params[128];
				ops->params_str(filter, params, sizeof(params));
				FILE * outputptr;
				outputptr = fopen(OUTPUT_PATH, "a");
				fseek(outputptr, 0, SEEK_END);
				long filesize = ftell(outputptr);
				if (filesize == 0) {
					// File is empty, write header
					fprintf(outputptr, "dataset,query_dist,num_queries,curr_query,filter,params,size,fpr\n");
				}
				fprintf(outputptr, "%s,%s,%ld,%ld,%s,%s,%ld,%.14f\n",
						dataset, dist, num_queries, j, ops->name, params, ops->size_bytes(filter), inst_fpr);
				fclose(outputptr);
			}
			if (ops->query(filter, query_set[j]) && !set_query(positives, curr_inserts, query_set[j], &dummy_return)) {
				fp_count++;
				if (ops->on_false_positive) {
					ops->on_false_positive(filter, query_set[j]);
				}
			}
			churn_count++;
			inst_count++;
		}

		ops->destroy(filter);
		set_free(positives, curr_inserts);
		free(train_negatives);
		free(train_cdf);
	}
	free(query_set);
	free(query_labels);
	free(insert_set);
	free(negative_set);
	free(offsets);
	free(buffer);
	return 0;
}
