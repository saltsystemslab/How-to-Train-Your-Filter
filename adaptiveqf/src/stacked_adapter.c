/*
filter_ops adapter for the stacked filter, calling the extern "C" API in
external/StackedFilters/wrapper/stacked_wrapper.cpp. Mirrors test/test_distribution_stacked.c.

Filter-specific arguments: the total size in bytes, and the proportion of queries used to
learn the negative distribution.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "stacked_wrapper.h"
#include "include/filter_iface.h"

typedef struct {
	WrappedStackedFilter *filter;
	size_t total_size;
	double negative_proportion;
} stacked_state;

static double stacked_train_proportion(int argc, char **argv) {
	return strtod(argv[1], NULL);
}

static void *stacked_create(const build_ctx *ctx) {
	stacked_state *s = calloc(1, sizeof(stacked_state));
	if (!s) {
		fprintf(stderr, "calloc failed for stacked_state\n");
		abort();
	}
	s->total_size = strtoull(ctx->argv[0], NULL, 10);
	s->negative_proportion = strtod(ctx->argv[1], NULL);
	s->filter = StackedFilterCreate(s->total_size * 8, ctx->positives, ctx->num_positives,
									ctx->train_negatives, ctx->num_train_negatives,
									ctx->train_cdf, ctx->num_train_negatives, 0);
	return s;
}

static int stacked_query(void *filter, uint64_t key) {
	return StackedFilterLookupElement(((stacked_state *)filter)->filter, &key);
}

static size_t stacked_size_bytes(void *filter) {
	return ((stacked_state *)filter)->total_size;
}

static void stacked_params_str(void *filter, char *buf, size_t len) {
	snprintf(buf, len, "prop_train=%.4f", ((stacked_state *)filter)->negative_proportion);
}

static void stacked_destroy(void *filter) {
	stacked_state *s = filter;
	StackedFilterDestroy(s->filter);
	free(s);
}

const filter_ops stacked_ops = {
	.name = "stacked",
	.usage = "<total size in bytes [eg. 1024]> <proportion of queries to track for negatives [eg. 0.25]>",
	.num_args = 2,
	.rehash_per_trial = 0,
	.train_proportion = stacked_train_proportion,
	.create = stacked_create,
	.query = stacked_query,
	.on_false_positive = NULL,
	.size_bytes = stacked_size_bytes,
	.params_str = stacked_params_str,
	.aux_build_time = NULL,
	.destroy = stacked_destroy,
};
