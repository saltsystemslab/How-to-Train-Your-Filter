/*
Common interface that lets a single experiment harness drive any filter.

Each filter provides a filter_ops table in its own adapter file (e.g. src/aqf_adapter.c).
Filters implemented in C++ expose an extern "C" wrapper (see
external/StackedFilters/wrapper/stacked_wrapper.cpp) that their adapter calls into.

The harness owns everything shared across filters: reading data and queries, hashing,
timing, FPR calculation and result output. Adapters only build, query, and react to
false positives.
*/
#ifndef FILTER_IFACE_H
#define FILTER_IFACE_H

#include <stdint.h>
#include <stddef.h>

// Everything a filter may need at construction time. The harness fills in the
// training negatives only when the filter asks for them (train_proportion != NULL).
typedef struct {
	const uint64_t *positives;
	int num_positives;
	const uint64_t *train_negatives;	// unique negative keys, sorted by descending frequency
	const double *train_cdf;			// cumulative frequency for train_negatives
	int num_train_negatives;
	int argc;							// filter-specific command line arguments
	char **argv;
	// Only filled in for filters that use features (uses_features): the feature vector of each
	// positive (row-major, num_features per positive, in the same order as positives), and the
	// folder holding the dataset's features and model files (e.g. ../data/cascaded/url).
	const float *positive_features;
	int num_features;
	const char *feature_dir;
	// Set in the dynamic experiments: learned filters then train a new model on the current positives
	// and the given negatives (the previous window's negative queries) instead of using their saved
	// model. The feature vector of every dataset row and the rows of the positives and negatives are
	// given for this.
	int retrain;
	const float *row_features;
	const int *positive_rows;
	const int *negative_rows;
	int num_negative_rows;
} build_ctx;

// Time spent (in nanoseconds) in a learned filter's model and in its backup Bloom filters. The build times
// cover inserting the positives; the query times cover timed_queries sampled queries.
typedef struct {
	uint64_t build_model_ns;
	uint64_t build_filter_ns;
	uint64_t query_model_ns;
	uint64_t query_filter_ns;
	uint64_t timed_queries;
} filter_timing;

// Where a learned filter's positive answer came from (see last_query_source).
#define FP_SOURCE_MODEL 1	// the model's decision was not checked by a backup filter
#define FP_SOURCE_BACKUP 2	// a backup Bloom filter accepted the key

typedef struct {
	const char *name;					// selects the filter on the command line, e.g. "aqf"
	const char *usage;					// description of the filter-specific arguments
	int num_args;						// number of filter-specific arguments required

	// If set, the harness re-hashes all keys in place with a per-trial seed before
	// building (this is the original AQF experiment behaviour).
	int rehash_per_trial;

	// If set, the harness loads the dataset's feature vectors (see learned/utils/export_cascaded_inputs.py)
	// and passes each key's features to create and query, e.g. for learned filters.
	int uses_features;

	// Proportion of the query stream used to learn the negative distribution, or NULL
	// if the filter does not use training negatives.
	double (*train_proportion)(int argc, char **argv);

	void *(*create)(const build_ctx *ctx);
	// features is the key's feature vector for filters that use features, and NULL otherwise.
	int (*query)(void *filter, uint64_t key, const float *features);
	// Called after a query returned a false positive. NULL if the filter does not adapt.
	void (*on_false_positive)(void *filter, uint64_t key);
	size_t (*size_bytes)(void *filter);
	// Writes a short description of the filter's parameters (e.g. "q=16;r=5") for output.
	void (*params_str)(void *filter, char *buf, size_t len);
	// Time spent building auxiliary structures (e.g. the AQF reverse map), already
	// included in the harness-measured build time. NULL if there are none.
	uint64_t (*aux_build_time)(void *filter);
	// The FPR the filter's parameters predict for a non-member query, before any adaptation
	// (e.g. the AQF's 1 - (1 - 2^-(q+r))^n). NULL if the filter has no such model.
	double (*expected_fpr)(void *filter);
	// Time spent inside create configuring the filter from its parameters (e.g. the stacked filter's optimizer,
	// or calibrating a learned filter's model). The harness excludes it from the build time. NULL if none.
	uint64_t (*config_time)(void *filter);
	// Time spent inside create only to match the filter to the experiment's size budget (e.g. searching for the
	// target FPR that fits it), which the filter would not need outside the experiments. The harness excludes it
	// from both the build time and config_time. NULL if none.
	uint64_t (*sizing_time)(void *filter);
	// Splits the filter's time into its model and its backup filters. NULL for filters without a model,
	// whose time is all filter time.
	void (*timing)(void *filter, filter_timing *out);
	// Where the most recent positive answer came from (FP_SOURCE_MODEL or FP_SOURCE_BACKUP). Only tracked
	// when built with TRACK_FP_SOURCE; NULL for filters without a model.
	int (*last_query_source)(void *filter);
	// Number of successful adaptations the filter has made since it was built. NULL if it does not adapt.
	uint64_t (*num_adaptations)(void *filter);
	// Size in bytes of the model the filter uses (included in size_bytes). NULL for filters without a model.
	size_t (*model_size_bytes)(void *filter);
	void (*destroy)(void *filter);
} filter_ops;

extern const filter_ops aqf_ops;
extern const filter_ops stacked_ops;
extern const filter_ops bloom_ops;
extern const filter_ops plbf_ops;
extern const filter_ops clbf_ops;

// Returns the ops table for the given filter name, or NULL if it is not registered.
const filter_ops *find_filter(const char *name);

#endif
