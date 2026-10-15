/*
Performs a dynamic experiment on the chosen filter, in one of two workloads:
- swapping (test_dynamic_generic): the queries come from the given index file over the whole dataset, and each
  churn swaps a window of the positives (a fifth of the smaller of the positive and negative sets) with negatives.
- shifting (test_shifting_generic, this file built with SHIFTING_WORKLOAD): the positives and the negatives are
  each split in half into an active set and a replacement set. Each window's queries are sampled uniformly from
  the active sets, and each churn swaps a fifth of each active set with its replacement set.

The queries run in NUM_WINDOWS windows of equal length. The first is a warm-up: the filters are built from it and
the AQF adapts during it, but nothing is recorded. The second runs without a churn, and a churn starts each later
window. At each churn the filters are rebuilt from the new positives; filters that learn from query labels (the
stacked filter) and the learned filters (which retrain) learn only from the previous window's queries, labeled
with the new positives, so they never learn from the queries they are measured on. Every inst_rate of the queries
in the recorded windows, the instantaneous FPR is measured over the current window's queries (without adaptation)
and written to the results file.

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
#include "include/harness_output.h"

#define MAX_DATA_LINES 10000000
#define READ_SIZE 4096
#define NUM_WINDOWS 10
#ifdef SHIFTING_WORKLOAD
#define OUTPUT_PATH "../results/shifting_results.csv"
#define WORKLOAD "shifting"
#define FIRST_FILTER_ARG 4
#else
#define OUTPUT_PATH "../results/dynamic_results.csv"
#define WORKLOAD "swapping"
#define FIRST_FILTER_ARG 5
#endif

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

// Labels n queries with the current positives.
static void label_queries(set_node *positives, int num_positives, const uint64_t *keys, int n, uint64_t *labels) {
	uint64_t dummy;
	for (int k = 0; k < n; k++) {
		labels[k] = set_query(positives, num_positives, keys[k], &dummy) ? 1 : 0;
	}
}

// Writes the rows of the distinct negatives among n labeled queries to out and returns how many there are.
static int distinct_negative_rows(const uint64_t *keys, const int *rows, const uint64_t *labels, int n, int *out) {
	set_node *seen = calloc(n, sizeof(set_node));
	if (!seen) {
		fprintf(stderr, "calloc failed for seen\n");
		exit(EXIT_FAILURE);
	}
	int count = 0;
	uint64_t dummy;
	for (int k = 0; k < n; k++) {
		if (labels[k] == 0 && !set_query(seen, n, keys[k], &dummy)) {
			set_insert(seen, n, keys[k], 1);
			out[count++] = rows[k];
		}
	}
	set_free(seen, n); // also frees seen itself
	return count;
}

#ifdef SHIFTING_WORKLOAD
// splitmix64, so the shifting workload's split and queries are the same for every filter and trial
static uint64_t next_random(uint64_t *state) {
	uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
	z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
	return z ^ (z >> 31);
}

// Shuffles keys (and their rows, if any) together.
static void shuffle(uint64_t *keys, int *rows, int n, uint64_t *state) {
	for (int i = n - 1; i > 0; i--) {
		int j = (int)(next_random(state) % (uint64_t)(i + 1));
		uint64_t temp = keys[i];
		keys[i] = keys[j];
		keys[j] = temp;
		if (rows) {
			int temp_row = rows[i];
			rows[i] = rows[j];
			rows[j] = temp_row;
		}
	}
}

// Samples n queries uniformly from the active positives (the first num_pos of insert_set) and the active
// negatives (the first num_neg of negative_set).
static void sample_window(const uint64_t *insert_set, const int *insert_rows, int num_pos,
						  const uint64_t *negative_set, const int *negative_rows, int num_neg,
						  int n, uint64_t *keys, int *rows, uint64_t *state) {
	for (int k = 0; k < n; k++) {
		uint64_t r = next_random(state) % (uint64_t)(num_pos + num_neg);
		if (r < (uint64_t)num_pos) {
			keys[k] = insert_set[r];
			if (rows) rows[k] = insert_rows[r];
		} else {
			keys[k] = negative_set[r - num_pos];
			if (rows) rows[k] = negative_rows[r - num_pos];
		}
	}
}
#endif

// FNV-1a over a list of row indices, to check that runs with different filters see the same rows.
static uint64_t rows_checksum(const int *rows, size_t n) {
	uint64_t hash = 0xcbf29ce484222325ULL;
	for (size_t k = 0; k < n; k++) {
		hash = (hash ^ (uint32_t)rows[k]) * 0x100000001b3ULL;
	}
	return hash;
}

static void swap_range(uint64_t *a, uint64_t *b, int *a_rows, int *b_rows, int start, int count) {
	for (int k = start; k < start + count; k++) {
		uint64_t temp = a[k];
		a[k] = b[k];
		b[k] = temp;
		if (a_rows) {
			int temp_row = a_rows[k];
			a_rows[k] = b_rows[k];
			b_rows[k] = temp_row;
		}
	}
}

int main(int argc, char **argv)
{
	int num_trials = 3;
	int verbose = 1;
	const filter_ops *ops = (argc > 1) ? find_filter(argv[1]) : NULL;
	if (argc < FIRST_FILTER_ARG || !ops || argc < FIRST_FILTER_ARG + ops->num_args) {
#ifdef SHIFTING_WORKLOAD
		fprintf(stderr, "Please specify \nthe filter name [eg. aqf, stacked]\nthe file path [eg. datasets/Malware_data.csv]\nthe number of queries [eg. 10000000]\nthe filter-specific arguments\n");
#else
		fprintf(stderr, "Please specify \nthe filter name [eg. aqf, stacked]\nthe file path [eg. datasets/Malware_data.csv]\nthe index file path [eg. datasets/hashed_unif_10M_url.csv]\nthe number of queries [eg. 10000000]\nthe filter-specific arguments\n");
#endif
		if (ops) fprintf(stderr, "arguments for %s: %s\n", ops->name, ops->usage);
		exit(1);
	}

	char filename[100];
	strcpy(filename, argv[2]);
#ifdef SHIFTING_WORKLOAD
	size_t num_queries = strtoull(argv[3], NULL, 10);
	char *dist = "unif";
#else
	char indexfilename[512];
	strcpy(indexfilename, argv[3]);
	size_t num_queries = strtoull(argv[4], NULL, 10);
	char *dist = get_dist_name(indexfilename);
#endif
	int filter_argc = argc - FIRST_FILTER_ARG;
	char **filter_argv = argv + FIRST_FILTER_ARG;
	int obj_index = get_obj_index(filename);
	int label_index = get_label_index(filename);
	char* dataset = get_dataset_name(filename);

	if (verbose) printf("Processing insertions\n");

	uint64_t *original_inserts = calloc(MAX_DATA_LINES, sizeof(uint64_t));
	uint64_t *original_negatives = calloc(MAX_DATA_LINES, sizeof(uint64_t));
	long *offsets = calloc(sizeof(long), MAX_DATA_LINES);
	char *buffer = malloc(READ_SIZE);
	if (!original_inserts || !original_negatives || !offsets || !buffer) {
		printf("allocation failed\n");
		return EXIT_FAILURE;
	}

	// row index of each positive and negative, used to look up features for filters that use them (and to check
	// that every filter sees the same queries and churns)
	int *original_insert_rows = malloc(MAX_DATA_LINES * sizeof(int));
	int *original_negative_rows = malloc(MAX_DATA_LINES * sizeof(int));

	int total_inserts, total_negatives;
	if (!read_file_with_negatives(filename, obj_index, label_index, buffer, offsets,
								  original_inserts, &total_inserts, original_negatives, &total_negatives,
								  original_insert_rows, original_negative_rows)) {
		printf("Insertion reading failed\n");
		return EXIT_FAILURE;
	}
	if (verbose) fprintf(stderr, "finished reading %d insertions and %d negative values\n", total_inserts, total_negatives);

#ifdef SHIFTING_WORKLOAD
	// the active and replacement halves of each set
	int num_pos = total_inserts / 2, num_neg = total_negatives / 2;
	int pos_replace = (int)(0.2 * num_pos), neg_replace = (int)(0.2 * num_neg);
	if (verbose) fprintf(stderr, "%d active positives and %d active negatives; each churn replaces %d and %d\n",
						 num_pos, num_neg, pos_replace, neg_replace);
#else
	// each churn swaps a fifth of the smaller of the two sets, so datasets with fewer negatives than positives
	// (e.g. shalla) only ever replace part of their positives
	int num_pos = total_inserts, num_neg = total_negatives;
	int churn_base = num_pos < num_neg ? num_pos : num_neg;
	int pos_replace = (int)(0.2 * churn_base);
	if (verbose && churn_base < num_pos) {
		fprintf(stderr, "fewer negatives than positives: each churn replaces %d of %d positives\n", pos_replace, num_pos);
	}
#endif

	size_t window_len = num_queries / NUM_WINDOWS;
#ifdef SHIFTING_WORKLOAD
	// each window's queries are sampled when it starts; the previous window's are kept to learn from
	size_t num_query_slots = 2 * window_len;
#else
	size_t num_query_slots = num_queries;
#endif
	uint64_t *original_queries = calloc(num_query_slots, sizeof(uint64_t));
	uint64_t *query_set = calloc(num_query_slots, sizeof(uint64_t));
	uint64_t *query_labels = calloc(num_query_slots, sizeof(uint64_t));
	uint64_t *window_labels = calloc(window_len, sizeof(uint64_t));
	int *query_rows = malloc(num_query_slots * sizeof(int));
	int *train_negative_rows = ops->uses_features ? malloc(window_len * sizeof(int)) : NULL;
	if (!original_queries || !query_set || !query_labels || !window_labels) {
		fprintf(stderr, "calloc failed for the queries\n");
		return EXIT_FAILURE;
	}

#ifndef SHIFTING_WORKLOAD
	int total_queries = 0;
	int pos_count = 0;
	if (verbose) fprintf(stderr, "Processing queries\n");
	int query_read_result = read_queries(indexfilename, filename, obj_index, label_index,
										buffer, offsets, original_queries, query_labels, num_queries,
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
		window_len = num_queries / NUM_WINDOWS;
	}
#endif

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
		uint64_t *negative_labels = calloc(total_negatives, sizeof(uint64_t));
		if (!row_features || !row_labels || label_rows != num_rows
			|| !check_row_labels(row_labels, num_rows, original_insert_rows, NULL, total_inserts, "positive")
			|| !check_row_labels(row_labels, num_rows, original_negative_rows, negative_labels, total_negatives, "negative")
#ifndef SHIFTING_WORKLOAD
			|| !check_row_labels(row_labels, num_rows, query_rows, query_labels, num_queries, "query")
#endif
			) {
			fprintf(stderr, "couldn't load features for %s from %s (see learned/utils/export_cascaded_inputs.py)\n", ops->name, feature_dir);
			return EXIT_FAILURE;
		}
		free(row_labels);
		free(negative_labels);
		positive_features = malloc((size_t)num_pos * num_features * sizeof(float));
		if (verbose) fprintf(stderr, "loaded %zu features for each of %zu rows from %s\n", num_features, num_rows, feature_dir);
	}

	// working copies of the sets, restored from the originals at the start of each trial
	uint64_t *insert_set = malloc(total_inserts * sizeof(uint64_t));
	uint64_t *negative_set = malloc(total_negatives * sizeof(uint64_t));
	int *insert_rows = malloc(total_inserts * sizeof(int));
	int *negative_rows = malloc(total_negatives * sizeof(int));

	// the query positions where the instantaneous FPR is measured, relative to each recorded window
	double inst_rate = 0.01;
	size_t space_between_inst = (size_t)(inst_rate * num_queries);

	for (int i = 0; i < num_trials; i++) {
		time_t seed = time(NULL);
		srand(i);
		if (verbose) printf("Running trial %d on seed %ld\n", i, seed);
		int murmur_seed = rand();

		memcpy(insert_set, original_inserts, total_inserts * sizeof(uint64_t));
		memcpy(negative_set, original_negatives, total_negatives * sizeof(uint64_t));
		memcpy(insert_rows, original_insert_rows, total_inserts * sizeof(int));
		memcpy(negative_rows, original_negative_rows, total_negatives * sizeof(int));
#ifndef SHIFTING_WORKLOAD
		memcpy(query_set, original_queries, num_queries * sizeof(uint64_t));
#endif
		if (ops->rehash_per_trial) {
			for (int j = 0; j < total_inserts; j++) {
				insert_set[j] = MurmurHash64A(&insert_set[j], sizeof(insert_set[j]), murmur_seed);
			}
			for (int j = 0; j < total_negatives; j++) {
				negative_set[j] = MurmurHash64A(&negative_set[j], sizeof(negative_set[j]), murmur_seed);
			}
#ifndef SHIFTING_WORKLOAD
			for (size_t j = 0; j < num_queries; j++) {
				query_set[j] = MurmurHash64A(&query_set[j], sizeof(query_set[j]), murmur_seed);
			}
#endif
		}
#ifdef SHIFTING_WORKLOAD
		// split each set into its active half (the first half) and replacement half (the second half); the seed is the
		// same in every trial, so every trial (and filter) gets the same split, churns and queries, as in the
		// swapping workload, and trials only differ in the filters' own randomness
		uint64_t random_state = 0x5eed0000ULL;
		shuffle(insert_set, insert_rows, total_inserts, &random_state);
		shuffle(negative_set, negative_rows, total_negatives, &random_state);
#endif

		set_node *positives = build_membership(insert_set, num_pos);
		build_ctx ctx = {
			.positives = insert_set,
			.num_positives = num_pos,
			.argc = filter_argc,
			.argv = filter_argv,
			.positive_features = positive_features,
			.num_features = (int)num_features,
			.feature_dir = feature_dir,
			// learned filters always train on the previous window's labels (the warm-up window for the first build)
			.retrain = 1,
			.row_features = row_features,
			.positive_rows = insert_rows,
			.negative_rows = train_negative_rows,
		};
		uint64_t *train_negatives = NULL;
		double *train_cdf = NULL;
		int num_check = 0;
		if (ops->train_proportion) {
			// the given proportion of all queries, at most one window, from the end of the window learned from
			num_check = (int)(num_queries * ops->train_proportion(filter_argc, filter_argv));
			if ((size_t)num_check > window_len) {
				fprintf(stderr, "training proportion %.3f is longer than a window; using one window\n",
						ops->train_proportion(filter_argc, filter_argv));
				num_check = (int)window_len;
			}
		}

		void *filter = NULL;
		uint64_t previous_adaptations = 0;	// adaptations made by filters since rebuilt
		int num_churns = 0;
		uint64_t dummy_return;
		for (int w = 0; w < NUM_WINDOWS; w++) {
			if (w >= 2) {
				// perform a churn by swapping windows of the active sets with their replacements
				if (verbose) fprintf(stderr, "performing churn %d\n", num_churns);
#ifdef SHIFTING_WORKLOAD
				swap_range(insert_set, insert_set + num_pos, insert_rows, insert_rows + num_pos,
						   (num_churns % 5) * pos_replace, pos_replace);
				swap_range(negative_set, negative_set + num_neg, negative_rows, negative_rows + num_neg,
						   (num_churns % 5) * neg_replace, neg_replace);
#else
				swap_range(insert_set, negative_set, insert_rows, negative_rows, (num_churns % 5) * pos_replace, pos_replace);
#endif
				set_free(positives, num_pos);
				positives = build_membership(insert_set, num_pos);
				num_churns++;
			}

			// the window's queries, and the window the filter learns from when it is (re)built
			size_t current = 0, previous = 0;
#ifdef SHIFTING_WORKLOAD
			current = (w % 2) * window_len;
			previous = ((w + 1) % 2) * window_len;
			sample_window(insert_set, insert_rows, num_pos, negative_set, negative_rows, num_neg, (int)window_len,
						  query_set + current, query_rows + current, &random_state);
#else
			current = w * window_len;
			previous = w > 0 ? (w - 1) * window_len : 0;
#endif

			// the dataset rows of the window's queries and of the current positives, which are the same for every filter
			if (verbose) fprintf(stderr, "trial %d window %d: query rows checksum %016" PRIx64 ", positive rows checksum %016" PRIx64 "\n",
								 i, w, rows_checksum(query_rows + current, window_len), rows_checksum(insert_rows, num_pos));

			if (w == 0 || w >= 2) {
				// (re)build the filter from the current positives, learning from the warm-up window for the
				// first build and from the previous window afterwards, labeled with the current positives
				size_t learn = w == 0 ? current : previous;
				label_queries(positives, num_pos, query_set + learn, (int)window_len, window_labels);
				if (ops->train_proportion) {
					free(train_negatives);
					free(train_cdf);
					size_t start = window_len - num_check;
					ctx.num_train_negatives = build_training_negatives(query_set + learn + start, window_labels + start,
																	   window_len, num_check, &train_negatives, &train_cdf);
					ctx.train_negatives = train_negatives;
					ctx.train_cdf = train_cdf;
				}
				if (train_negative_rows) {
					ctx.num_negative_rows = distinct_negative_rows(query_set + learn, query_rows + learn, window_labels,
																   (int)window_len, train_negative_rows);
				}
				if (positive_features) gather_rows(row_features, num_features, insert_rows, num_pos, positive_features);
				if (filter) {
					if (ops->num_adaptations) previous_adaptations += ops->num_adaptations(filter);
					ops->destroy(filter);
				}
				filter = ops->create(&ctx);
			}

			for (size_t q = 0; q < window_len; q++) {
				if (w >= 1 && q % space_between_inst == 0) {
					// perform an instantaneous fpr check over the window's queries without adaptation
					int inst_fp_count = 0;
					uint64_t inst_fp_from_model = 0, inst_fp_from_backup = 0;
					int num_negative = 0;
					for (size_t k = current; k < current + window_len; k++) {
						int is_member = set_query(positives, num_pos, query_set[k], &dummy_return);
						if (!is_member) {
							num_negative++;
						}
						if (ops->query(filter, query_set[k], row_features ? row_features + (size_t)query_rows[k] * num_features : NULL) && !is_member) {
							inst_fp_count++;
							count_fp_source(ops, filter, &inst_fp_from_model, &inst_fp_from_backup);
						}
					}
					double inst_fpr = (double)inst_fp_count / num_negative;
					if (verbose) fprintf(stderr, "instantaneous FPR check: %d false positives out of %d negatives\n", inst_fp_count, num_negative);

					char params[128];
					ops->params_str(filter, params, sizeof(params));
					FILE * outputptr;
					outputptr = open_results_file(OUTPUT_PATH);
					long filesize = ftell(outputptr);
					if (filesize == 0) {
						// File is empty, write header
						fprintf(outputptr, "dataset,query_dist,workload,trial,num_queries,curr_query,filter,params,size,fpr,expected_fpr,fp_from_model,fp_from_backup,model_size,num_adaptations\n");
					}
					char expected_fpr_str[32] = "";
					if (ops->expected_fpr) snprintf(expected_fpr_str, sizeof(expected_fpr_str), "%.10g", ops->expected_fpr(filter));
					char fp_sources[64] = ",";
#ifdef TRACK_FP_SOURCE
					if (ops->last_query_source) snprintf(fp_sources, sizeof(fp_sources), "%lu,%lu", inst_fp_from_model, inst_fp_from_backup);
#endif
					// adaptations so far in this run, including those of filters since rebuilt
					char model_and_adaptations[64];
					format_model_and_adaptations(model_and_adaptations, sizeof(model_and_adaptations), ops, filter, previous_adaptations);
					fprintf(outputptr, "%s,%s,%s,%d,%ld,%ld,%s,%s,%ld,%.14f,%s,%s%s\n",
							dataset, dist, WORKLOAD, i, num_queries, w * window_len + q, ops->name, params, ops->size_bytes(filter), inst_fpr,
							expected_fpr_str, fp_sources, model_and_adaptations);
					fclose(outputptr);
				}
				size_t j = current + q;
				if (ops->query(filter, query_set[j], row_features ? row_features + (size_t)query_rows[j] * num_features : NULL)
					&& !set_query(positives, num_pos, query_set[j], &dummy_return)) {
					if (ops->on_false_positive) {
						ops->on_false_positive(filter, query_set[j]);
					}
				}
			}
		}

		ops->destroy(filter);
		set_free(positives, num_pos);
		free(train_negatives);
		free(train_cdf);
	}
	free(original_queries);
	free(query_set);
	free(query_labels);
	free(window_labels);
	free(original_inserts);
	free(original_negatives);
	free(insert_set);
	free(negative_set);
	free(original_insert_rows);
	free(original_negative_rows);
	free(insert_rows);
	free(negative_rows);
	free(query_rows);
	free(train_negative_rows);
	unmap_float_matrix(row_features, num_feature_rows, num_features);
	free(positive_features);
	free(offsets);
	free(buffer);
	return 0;
}
