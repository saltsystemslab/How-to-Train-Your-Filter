/*
filter_ops adapter for the Bloom filter used by CascadedLBF, calling the extern "C" API in
external/bloom_wrapper/bloom_wrapper.cpp.

Filter-specific arguments: the total size in bytes, which is spread evenly over the positive keys.
*/
#include <stdio.h>
#include <stdlib.h>

#include "bloom_wrapper.h"
#include "include/filter_iface.h"

typedef struct {
	WrappedBloomFilter *filter;
	size_t total_size;
	double bits_per_key;
} bloom_state;

static void *bloom_create(const build_ctx *ctx) {
	bloom_state *s = calloc(1, sizeof(bloom_state));
	if (!s) {
		fprintf(stderr, "calloc failed for bloom_state\n");
		abort();
	}
	s->total_size = strtoull(ctx->argv[0], NULL, 10);
	// the filter rounds its bit array up to whole 64-bit words, so size it from the words that fit in the
	// budget (minus a fraction of a bit, so floating point error can't round it up a word)
	size_t budget_words = s->total_size / 8;
	s->bits_per_key = (budget_words * 64.0 - 0.5) / ctx->num_positives;
	// seed from rand() so the hash follows the harness's per-trial srand
	uint64_t seed = ((uint64_t)rand() << 32) | (uint64_t)rand();
	s->filter = BloomFilterCreate(ctx->num_positives, s->bits_per_key, seed);
	for (int j = 0; j < ctx->num_positives; j++) {
		BloomFilterInsertElement(s->filter, ctx->positives[j]);
	}
	return s;
}

static int bloom_query(void *filter, uint64_t key, const float *features) {
	return BloomFilterLookupElement(((bloom_state *)filter)->filter, key);
}

static size_t bloom_size_bytes(void *filter) {
	return BloomFilterSizeInBytes(((bloom_state *)filter)->filter);
}

static void bloom_params_str(void *filter, char *buf, size_t len) {
	snprintf(buf, len, "bits_per_key=%.4f", ((bloom_state *)filter)->bits_per_key);
}

static void bloom_destroy(void *filter) {
	bloom_state *s = filter;
	BloomFilterDestroy(s->filter);
	free(s);
}

const filter_ops bloom_ops = {
	.name = "bloom",
	.usage = "<total size in bytes [eg. 1024]>",
	.num_args = 1,
	.rehash_per_trial = 0,
	.train_proportion = NULL,
	.create = bloom_create,
	.query = bloom_query,
	.on_false_positive = NULL,
	.size_bytes = bloom_size_bytes,
	.params_str = bloom_params_str,
	.aux_build_time = NULL,
	.expected_fpr = NULL,
	.config_time = NULL,
	.destroy = bloom_destroy,
};
