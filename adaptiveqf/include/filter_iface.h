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
} build_ctx;

typedef struct {
	const char *name;					// selects the filter on the command line, e.g. "aqf"
	const char *usage;					// description of the filter-specific arguments
	int num_args;						// number of filter-specific arguments required

	// If set, the harness re-hashes all keys in place with a per-trial seed before
	// building (this is the original AQF experiment behaviour).
	int rehash_per_trial;

	// Proportion of the query stream used to learn the negative distribution, or NULL
	// if the filter does not use training negatives.
	double (*train_proportion)(int argc, char **argv);

	void *(*create)(const build_ctx *ctx);
	int (*query)(void *filter, uint64_t key);
	// Called after a query returned a false positive. NULL if the filter does not adapt.
	void (*on_false_positive)(void *filter, uint64_t key);
	size_t (*size_bytes)(void *filter);
	// Writes a short description of the filter's parameters (e.g. "q=16;r=5") for output.
	void (*params_str)(void *filter, char *buf, size_t len);
	// Time spent building auxiliary structures (e.g. the AQF reverse map), already
	// included in the harness-measured build time. NULL if there are none.
	uint64_t (*aux_build_time)(void *filter);
	void (*destroy)(void *filter);
} filter_ops;

extern const filter_ops aqf_ops;
extern const filter_ops stacked_ops;

// Returns the ops table for the given filter name, or NULL if it is not registered.
const filter_ops *find_filter(const char *name);

#endif
