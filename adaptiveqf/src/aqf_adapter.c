/*
filter_ops adapter for the adaptive quotient filter. Owns the filter and its reverse map,
and performs adaptations on false positives. Mirrors the logic in test/test_distribution.c.

Filter-specific arguments: the log of the number of slots, and the number of remainder bits.
*/
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "include/gqf.h"
#include "include/gqf_int.h"
#include "include/exp_utility.h"
#include "include/filter_iface.h"

typedef struct {
	QF qf;
	set_node *set;
	uint64_t set_len;
	size_t qbits, rbits;
	int still_have_space;
	// result of the most recent query, needed to adapt on a false positive
	uint64_t ret_index, ret_hash;
	int ret_hash_len;
	uint64_t map_time;
} aqf_state;

static void *aqf_create(const build_ctx *ctx) {
	aqf_state *s = calloc(1, sizeof(aqf_state));
	if (!s) {
		fprintf(stderr, "calloc failed for aqf_state\n");
		abort();
	}
	s->qbits = atoi(ctx->argv[0]);
	s->rbits = atoi(ctx->argv[1]);
	s->still_have_space = 1;

	struct timeval timecheck;
	uint64_t nhashbits = s->qbits + s->rbits;
	uint64_t nslots = (1ULL << s->qbits);
	if (!qf_malloc(&s->qf, nslots, nhashbits, 0, QF_HASH_INVERTIBLE, 0)) {
		fprintf(stderr, "Can't allocate QF.\n");
		abort();
	}
	qf_set_auto_resize(&s->qf, false);

	// create the reverse map
	gettimeofday(&timecheck, NULL);
	uint64_t start_time = timecheck.tv_sec * 1000000 + timecheck.tv_usec, end_time;
	s->set_len = ctx->num_positives;
	s->set = calloc(s->set_len, sizeof(set_node));
	gettimeofday(&timecheck, NULL);
	end_time = timecheck.tv_sec * 1000000 + timecheck.tv_usec;
	s->map_time = end_time - start_time;

	// perform inserts
	uint64_t filter_insert_time, set_insert_time;
	for (int j = 0; j < ctx->num_positives; j++) {
		int result = insert_key(&s->qf, s->set, s->set_len, ctx->positives[j], 1, &timecheck, &filter_insert_time, &set_insert_time);
		if (!result) {
			fprintf(stderr, "failed to insert\n");
			abort();
		}
		s->map_time += set_insert_time;
	}
	return s;
}

static int aqf_query(void *filter, uint64_t key) {
	aqf_state *s = filter;
	return qf_query(&s->qf, key, &s->ret_index, &s->ret_hash, &s->ret_hash_len, QF_KEY_IS_HASH | QF_NO_LOCK);
}

static void aqf_on_false_positive(void *filter, uint64_t key) {
	aqf_state *s = filter;
	uint64_t orig_key = 0;
	uint64_t old_ret_hash = s->ret_hash;
	set_query(s->set, s->set_len, s->ret_hash, &orig_key);
	if (!s->still_have_space) return;

	s->ret_hash_len = qf_adapt(&s->qf, s->ret_index, orig_key, key, &s->ret_hash, QF_KEY_IS_HASH | QF_NO_LOCK);
	if (s->ret_hash_len > 0) {
		if (set_delete(s->set, s->set_len, old_ret_hash) == 0) {
			fprintf(stderr, "reverse map missing fingerprint %lu during adapt\n", old_ret_hash);
			abort();
		}
		set_insert(s->set, s->set_len, s->ret_hash, orig_key);
	}
	else if (s->ret_hash_len == QF_NO_SPACE) {
		s->still_have_space = 0;
		fprintf(stderr, "filter is full, no further adaptations\n");
	}
}

static size_t aqf_size_bytes(void *filter) {
	return ((aqf_state *)filter)->qf.metadata->total_size_in_bytes;
}

static void aqf_params_str(void *filter, char *buf, size_t len) {
	aqf_state *s = filter;
	snprintf(buf, len, "q=%zu;r=%zu", s->qbits, s->rbits);
}

static uint64_t aqf_aux_build_time(void *filter) {
	return ((aqf_state *)filter)->map_time;
}

static void aqf_destroy(void *filter) {
	aqf_state *s = filter;
	set_free(s->set, s->set_len);
	qf_free(&s->qf);
	free(s);
}

const filter_ops aqf_ops = {
	.name = "aqf",
	.usage = "<log of the number of slots [eg. 20]> <number of remainder bits [eg. 9]>",
	.num_args = 2,
	.rehash_per_trial = 1,
	.train_proportion = NULL,
	.create = aqf_create,
	.query = aqf_query,
	.on_false_positive = aqf_on_false_positive,
	.size_bytes = aqf_size_bytes,
	.params_str = aqf_params_str,
	.aux_build_time = aqf_aux_build_time,
	.destroy = aqf_destroy,
};
