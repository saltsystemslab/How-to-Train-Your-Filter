/*
filter_ops adapter for the stacked filter, calling the extern "C" API in
external/StackedFilters/wrapper/stacked_wrapper.cpp.

Filter-specific arguments: the total size in bytes (e.g. the AQF's size for some q and r), and the
proportion of queries used to learn the negative distribution.
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
	uint64_t config_time;	// time spent choosing the layers (the stacked filter's optimizer)
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
									ctx->train_cdf, ctx->num_train_negatives, 0, &s->config_time);
	return s;
}

static int stacked_query(void *filter, uint64_t key, const float *features) {
	return StackedFilterLookupElement(((stacked_state *)filter)->filter, &key);
}

// Reports the requested size, which the results are grouped by. The layers' actual total can differ
// slightly (each layer is sized by its own formula, with a 2000-bit floor), so it is recorded in params.
static size_t stacked_size_bytes(void *filter) {
	return ((stacked_state *)filter)->total_size;
}

static void stacked_params_str(void *filter, char *buf, size_t len) {
	stacked_state *s = filter;
	snprintf(buf, len, "prop_train=%.4f;actual_size=%zu", s->negative_proportion, (StackedFilterGetSize(s->filter) + 7) / 8);
}

static uint64_t stacked_config_time(void *filter) {
	return ((stacked_state *)filter)->config_time;
}

static void stacked_destroy(void *filter) {
	stacked_state *s = filter;
	StackedFilterDestroy(s->filter);
	free(s);
}

const filter_ops stacked_ops = {
	.name = "stacked",
	.usage = "<total size in bytes [eg. 69160]> <proportion of queries to track for negatives [eg. 0.25]>",
	.num_args = 2,
	.rehash_per_trial = 0,
	.uses_features = 0,
	.train_proportion = stacked_train_proportion,
	.create = stacked_create,
	.query = stacked_query,
	.on_false_positive = NULL,
	.size_bytes = stacked_size_bytes,
	.params_str = stacked_params_str,
	.aux_build_time = NULL,
	.expected_fpr = NULL,
	.config_time = stacked_config_time,
	.destroy = stacked_destroy,
};
