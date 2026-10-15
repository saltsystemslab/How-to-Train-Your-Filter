/*
filter_ops adapters for the learned Bloom filters from CascadedLBF (PLBF and CLBF), calling the
extern "C" APIs in external/cascaded_wrapper. Both use the dataset's exported features and models
(see learned/utils/export_cascaded_inputs.py).

Filter-specific arguments: the size budget in bytes, and the folder of the trained XGBoost model under
the dataset's feature folder (e.g. xgboost_d4_r100_e03). For the PLBF, which always keeps its whole model,
the model can be "auto": of the dataset's largest xgboost_* models of each depth no larger than
PLBF_MODEL_FRACTION of the budget, the one the PLBF can configure with the lowest target FPR (within PLBF_F_TIE, then the larger) is used (see
choose_plbf_model and learned/utils/train_cascaded_models.sh). The filters are configured by a target FPR F,
so the adapter searches for the smallest F whose configured size (model plus Bloom filters, measured as
CascadedLBF reports sizes) fits the budget, then builds the filter at that F. The CLBF is configured as in
CascadedLBF's experiments, with lambda = 1 (choose the configuration that uses the least memory) and mu = 0.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>

#include "lbf_wrapper.h"
#include "include/exp_utility.h"
#include "include/filter_iface.h"

#define CLBF_LAMBDA 1.0
#define CLBF_MU 0.0
// with model "auto", the PLBF chooses among the models no larger than this fraction of the budget
#define PLBF_MODEL_FRACTION 0.2
// and counts target FPRs within this fraction of the lowest as a tie, which the larger model wins
#define PLBF_F_TIE 0.25
// CascadedLBF's CLBF crashes on single-tree models (its own experiments use at least 2 rounds)
#define CLBF_MIN_ROUNDS 2
// every TIMING_SAMPLE-th query is timed in its model and backup filter parts (timing each part of every query
// would slow every query down; the other queries only pay for the harness's timing of the whole query)
#define TIMING_SAMPLE 16
// the CLBF's sampled queries are instead recorded and timed after the experiment (see CLBFTimeLookups), up to
// this many of them
#define CLBF_MAX_TIMED 100000
#define MIN_F 1e-9
#define MAX_F 0.999

typedef struct {
	int is_clbf;
	void *handle;			// WrappedPLBF * or WrappedCLBF *
	size_t budget;
	char model[256];
	double F;				// target FPR the filter was configured with
	uint64_t config_time;	// time spent configuring the filter: loading and calibrating the model, and optimizing at F
	uint64_t sizing_time;	// time spent finding the parameters that fit the size budget: the model choice, the
							// search for F and size checks, which neither configures nor builds the filter
	filter_timing timing;	// time spent in the model and in the backup filters
	uint64_t num_queries;
	int last_source;		// where the last positive answer came from (TRACK_FP_SOURCE builds only)
	const float **sampled_features;	// the CLBF's sampled queries, not yet timed
	uint64_t *sampled_keys;
	size_t num_sampled;
} lbf_state;

// The search depends only on these inputs, so its result is reused across rebuilds and trials.
static struct {
	int valid;
	int is_clbf;
	size_t budget;
	int num_positives;
	char requested_model[256];	// the model argument
	char model[256];			// the model used (the CLBF may fall back to a smaller one)
	double F;
} last_search;

static size_t configured_size(lbf_state *s, size_t num_positives, double F) {
	return s->is_clbf ? CLBFConfiguredSize(s->handle, num_positives, F, CLBF_LAMBDA, CLBF_MU)
					  : PLBFConfiguredSize(s->handle, num_positives, F);
}

static size_t checked_configured_size(lbf_state *s, size_t num_positives, double F) {
	size_t size = configured_size(s, num_positives, F);
	if (size == 0) {
		fprintf(stderr, "%s: couldn't configure the filter at F=%g\n", s->is_clbf ? "clbf" : "plbf", F);
		abort();
	}
	return size;
}

// Finds the smallest target FPR whose configured size fits the budget, or returns -1 if the filter
// does not fit even at the largest target FPR. Size decreases as F grows.
static double search_fpr_for_budget(lbf_state *s, size_t num_positives) {
	double lo = MIN_F, hi = MAX_F;
	size_t largest_F_size = checked_configured_size(s, num_positives, hi);
	if (largest_F_size > s->budget) {
		fprintf(stderr, "%s: model %s doesn't fit the budget of %zu bytes (needs %zu bytes even at F=%g)\n",
				s->is_clbf ? "clbf" : "plbf", s->model, s->budget, largest_F_size, hi);
		return -1;
	}
	if (checked_configured_size(s, num_positives, lo) <= s->budget) {
		return lo;
	}
	// invariant: lo exceeds the budget, hi fits it; search in log space
	while (hi / lo > 1.001) {
		double mid = sqrt(lo * hi);
		if (checked_configured_size(s, num_positives, mid) <= s->budget) {
			hi = mid;
		} else {
			lo = mid;
		}
	}
	return hi;
}

static long model_file_size(const char *feature_dir, const char *model) {
	char path[768];
	struct stat st;
	snprintf(path, sizeof(path), "%s/%s/xgboost_model.bin", feature_dir, model);
	return stat(path, &st) == 0 ? (long)st.st_size : -1;
}

// A numeric field (e.g. "max_depth") of the info.json train_xgboost writes next to a model, or -1.
// Reads a number from a flat json file (-1 if it is missing).
static double json_number(const char *path, const char *field_name) {
	char line[256], field[64];
	snprintf(field, sizeof(field), "\"%s\":", field_name);
	FILE *f = fopen(path, "r");
	if (!f) return -1;
	double value = -1;
	while (fgets(line, sizeof(line), f)) {
		char *found = strstr(line, field);
		if (found) value = atof(found + strlen(field));
	}
	fclose(f);
	return value;
}

static double model_info(const char *feature_dir, const char *model, const char *field_name) {
	char path[768];
	snprintf(path, sizeof(path), "%s/%s/info.json", feature_dir, model);
	return json_number(path, field_name);
}

static int model_num_rounds(const char *feature_dir, const char *model) {
	return (int)model_info(feature_dir, model, "num_boost_round");
}

// Picks the largest of the dataset's xgboost_* models with at least min_rounds trees that is smaller than below_bytes.
static int next_smaller_model(const char *feature_dir, long below_bytes, int min_rounds, char *model, size_t len) {
	DIR *dir = opendir(feature_dir);
	if (!dir) return 0;
	struct dirent *entry;
	long best_size = -1;
	while ((entry = readdir(dir)) != NULL) {
		if (strncmp(entry->d_name, "xgboost_", 8) != 0) continue;
		if (model_num_rounds(feature_dir, entry->d_name) < min_rounds) continue;
		long size = model_file_size(feature_dir, entry->d_name);
		if (size >= 0 && size < below_bytes && size > best_size) {
			best_size = size;
			snprintf(model, len, "%s", entry->d_name);
		}
	}
	closedir(dir);
	return best_size >= 0;
}

// The PLBF's candidate models for "auto": of each depth, the largest model whose saved size is at most cap_bytes
// (or, if no model fits under the cap, the smallest model). Returns how many candidates were found.
#define MAX_CANDIDATES 8
static int candidate_models_under_cap(const char *feature_dir, double cap_bytes, char candidates[][256]) {
	DIR *dir = opendir(feature_dir);
	if (!dir) return 0;
	struct dirent *entry;
	int depths[MAX_CANDIDATES];
	long sizes[MAX_CANDIDATES];
	int count = 0;
	long smallest = -1;
	char smallest_model[256] = "";
	while ((entry = readdir(dir)) != NULL) {
		if (strncmp(entry->d_name, "xgboost_", 8) != 0) continue;
		long size = model_file_size(feature_dir, entry->d_name);
		if (size < 0) continue;
		if (smallest < 0 || size < smallest) {
			smallest = size;
			snprintf(smallest_model, sizeof(smallest_model), "%s", entry->d_name);
		}
		if (size > cap_bytes) continue;
		int depth = (int)model_info(feature_dir, entry->d_name, "max_depth");
		int i = 0;
		while (i < count && depths[i] != depth) i++;
		if (i == count) {
			if (count == MAX_CANDIDATES) continue;
			depths[count] = depth;
			sizes[count] = -1;
			count++;
		}
		if (size > sizes[i]) {
			sizes[i] = size;
			snprintf(candidates[i], 256, "%s", entry->d_name);
		}
	}
	closedir(dir);
	if (count == 0 && smallest >= 0) {
		snprintf(candidates[0], 256, "%s", smallest_model);
		count = 1;
	}
	return count;
}

// Picks up to count of the rows (without replacement, using the harness's per-trial rand()), moving them to
// the front of rows. Returns how many were picked.
static int sample_rows(int *rows, int num_rows, int count) {
	if (count > num_rows) count = num_rows;
	for (int i = 0; i < count; i++) {
		int j = i + rand() % (num_rows - i);
		int temp = rows[i];
		rows[i] = rows[j];
		rows[j] = temp;
	}
	return count;
}

// Copies the features of the given rows into X and sets their labels in y.
static void add_rows(const build_ctx *ctx, const int *rows, int count, float label, float *X, float *y) {
	for (int i = 0; i < count; i++) {
		memcpy(X + (size_t)i * ctx->num_features, ctx->row_features + (size_t)rows[i] * ctx->num_features,
			   ctx->num_features * sizeof(float));
		y[i] = label;
	}
}

// Retrains the model on the current labels, as the dataset's saved models were trained (see
// learned/utils/export_cascaded_inputs.py): the training and calibration sets take the fractions of the given
// positives and negatives in the dataset's split.json. The model keeps the depth, rounds and eta of s->model.
// Returns 0 on failure.
static int retrain_model(lbf_state *s, const build_ctx *ctx) {
	int depth = (int)model_info(ctx->feature_dir, s->model, "max_depth");
	int rounds = model_num_rounds(ctx->feature_dir, s->model);
	double eta = model_info(ctx->feature_dir, s->model, "eta");
	if (depth <= 0 || rounds <= 0 || eta <= 0) {
		fprintf(stderr, "%s: couldn't read the training parameters of model %s\n", s->is_clbf ? "clbf" : "plbf", s->model);
		return 0;
	}
	// the same split of the positives and negatives into training and calibration sets as the saved models
	// (positives may be in both; the negatives are split into disjoint sets), see export_cascaded_inputs.py
	char split_path[512];
	snprintf(split_path, sizeof(split_path), "%s/split.json", ctx->feature_dir);
	double train_pos_fraction = json_number(split_path, "train_pos_fraction"), val_pos_fraction = json_number(split_path, "val_pos_fraction");
	double train_neg_fraction = json_number(split_path, "train_neg_fraction"), val_neg_fraction = json_number(split_path, "val_neg_fraction");
	if (train_pos_fraction <= 0 || val_pos_fraction <= 0 || train_neg_fraction <= 0 || val_neg_fraction <= 0
		|| train_neg_fraction + val_neg_fraction > 1) {
		fprintf(stderr, "%s: couldn't read the training split from %s\n", s->is_clbf ? "clbf" : "plbf", split_path);
		return 0;
	}
	int *pos = malloc(ctx->num_positives * sizeof(int));
	int *neg = malloc(ctx->num_negative_rows * sizeof(int));
	memcpy(pos, ctx->positive_rows, ctx->num_positives * sizeof(int));
	memcpy(neg, ctx->negative_rows, ctx->num_negative_rows * sizeof(int));
	sample_rows(pos, ctx->num_positives, ctx->num_positives);
	sample_rows(neg, ctx->num_negative_rows, ctx->num_negative_rows);
	int num_train_pos = (int)(train_pos_fraction * ctx->num_positives + 0.5), num_val_pos = (int)(val_pos_fraction * ctx->num_positives + 0.5);
	int num_train_neg = (int)(train_neg_fraction * ctx->num_negative_rows + 0.5), num_val_neg = (int)(val_neg_fraction * ctx->num_negative_rows + 0.5);
	size_t n_train = num_train_pos + num_train_neg, n_val = num_val_pos + num_val_neg;
	float *train_X = malloc(n_train * ctx->num_features * sizeof(float)), *train_y = malloc(n_train * sizeof(float));
	float *val_X = malloc(n_val * ctx->num_features * sizeof(float)), *val_y = malloc(n_val * sizeof(float));
	add_rows(ctx, pos, num_train_pos, 1, train_X, train_y);
	add_rows(ctx, neg, num_train_neg, 0, train_X + (size_t)num_train_pos * ctx->num_features, train_y + num_train_pos);
	add_rows(ctx, pos, num_val_pos, 1, val_X, val_y);
	add_rows(ctx, neg + num_train_neg, num_val_neg, 0, val_X + (size_t)num_val_pos * ctx->num_features, val_y + num_val_pos);

	char model_path[] = "/tmp/lbf_retrained_model_XXXXXX";
	int fd = mkstemp(model_path);
	int ok = fd >= 0;
	if (ok) {
		close(fd);
		ok = LBFTrainXGBoostModel(train_X, train_y, n_train, ctx->num_features, depth, rounds, eta, model_path) == 0;
	}
	if (ok) {
		s->handle = s->is_clbf ? (void *)CLBFPrepare(model_path, val_X, val_y, n_val, ctx->num_features)
							   : (void *)PLBFPrepare(model_path, val_X, val_y, n_val, ctx->num_features);
		ok = s->handle != NULL;
		unlink(model_path);
	}
	fprintf(stderr, "%s: retrained %s on %d positives and %d negatives, calibrated on %d positives and %d negatives (current labels)\n",
			s->is_clbf ? "clbf" : "plbf", s->model, num_train_pos, num_train_neg, num_val_pos, num_val_neg);
	free(pos);
	free(neg);
	free(train_X);
	free(train_y);
	free(val_X);
	free(val_y);
	return ok;
}

// Loads the model and calibrates on the dataset's validation set, or retrains it on the current labels
// when the filter is rebuilt after the positives changed. Returns 0 on failure.
static int prepare_model(lbf_state *s, const build_ctx *ctx) {
	if (ctx->retrain) {
		return retrain_model(s, ctx);
	}
	char model_path[768], val_X_path[512], val_y_path[512];
	snprintf(model_path, sizeof(model_path), "%s/%s/xgboost_model.bin", ctx->feature_dir, s->model);
	snprintf(val_X_path, sizeof(val_X_path), "%s/X_val.bin", ctx->feature_dir);
	snprintf(val_y_path, sizeof(val_y_path), "%s/y_val.bin", ctx->feature_dir);
	size_t num_val, val_cols, num_val_y, val_y_cols;
	float *val_X = load_float_matrix(val_X_path, &num_val, &val_cols);
	float *val_y = load_float_matrix(val_y_path, &num_val_y, &val_y_cols);
	if (!val_X || !val_y || num_val != num_val_y || val_cols != (size_t)ctx->num_features) {
		fprintf(stderr, "%s: couldn't load the validation set from %s\n", s->is_clbf ? "clbf" : "plbf", ctx->feature_dir);
		free(val_X);
		free(val_y);
		return 0;
	}
	s->handle = s->is_clbf ? (void *)CLBFPrepare(model_path, val_X, val_y, num_val, ctx->num_features)
						   : (void *)PLBFPrepare(model_path, val_X, val_y, num_val, ctx->num_features);
	free(val_X);
	free(val_y);
	return s->handle != NULL;
}

static void release_model(lbf_state *s) {
	if (s->is_clbf) {
		CLBFDestroy(s->handle);
	} else {
		PLBFDestroy(s->handle);
	}
	s->handle = NULL;
}

static size_t lbf_size_bytes(void *filter);

// The PLBF's model for "auto", chosen once per budget and reused across rebuilds and trials.
static struct {
	int valid;
	size_t budget;
	char model[256];
} plbf_choice;

// Chooses the PLBF's model for "auto" among the candidates under the cap (see candidate_models_under_cap): the one
// the PLBF can configure with the lowest target FPR F within the budget, using the saved models and the dataset's
// calibration set (not the queries), where an F within PLBF_F_TIE of the lowest counts as a tie, which the larger
// model wins. This only matches the filter to the budget, so its time counts as sizing time, not as configuring
// or building the filter. Writes the model to s->model.
static void choose_plbf_model(lbf_state *s, const build_ctx *ctx) {
	if (plbf_choice.valid && plbf_choice.budget == s->budget) {
		snprintf(s->model, sizeof(s->model), "%s", plbf_choice.model);
		return;
	}
	double cap = PLBF_MODEL_FRACTION * s->budget;
	char candidates[MAX_CANDIDATES][256];
	int count = candidate_models_under_cap(ctx->feature_dir, cap, candidates);
	if (count == 0) {
		fprintf(stderr, "plbf: no xgboost_* models found in %s (see learned/utils/train_cascaded_models.sh)\n", ctx->feature_dir);
		abort();
	}
	snprintf(s->model, sizeof(s->model), "%s", candidates[0]);
	if (count > 1) {
		build_ctx saved = *ctx;
		saved.retrain = 0;	// compare the saved models, calibrated on the dataset's calibration set
		double Fs[MAX_CANDIDATES], best_F = -1;
		for (int i = 0; i < count; i++) {
			snprintf(s->model, sizeof(s->model), "%s", candidates[i]);
			if (!prepare_model(s, &saved)) {
				abort();
			}
			Fs[i] = search_fpr_for_budget(s, ctx->num_positives);
			release_model(s);
			fprintf(stderr, "plbf: candidate model %s (%ld bytes) fits the budget at F=%.6g\n", candidates[i],
					model_file_size(ctx->feature_dir, candidates[i]), Fs[i]);
			if (Fs[i] > 0 && (best_F < 0 || Fs[i] < best_F)) {
				best_F = Fs[i];
			}
		}
		long chosen_size = -1;
		for (int i = 0; i < count; i++) {
			long size = model_file_size(ctx->feature_dir, candidates[i]);
			if (Fs[i] > 0 && Fs[i] <= best_F * (1 + PLBF_F_TIE) && size > chosen_size) {
				chosen_size = size;
				snprintf(s->model, sizeof(s->model), "%s", candidates[i]);
			}
		}
	}
	plbf_choice.valid = 1;
	plbf_choice.budget = s->budget;
	snprintf(plbf_choice.model, sizeof(plbf_choice.model), "%s", s->model);
	fprintf(stderr, "plbf: using model %s (of the largest models of each depth up to %.0f bytes, the one with the lowest F, ties going to the larger)\n",
			s->model, cap);
}


static void *lbf_create(const build_ctx *ctx, int is_clbf) {
	lbf_state *s = calloc(1, sizeof(lbf_state));
	if (!s) {
		fprintf(stderr, "calloc failed for lbf_state\n");
		abort();
	}
	s->is_clbf = is_clbf;
	s->budget = strtoull(ctx->argv[0], NULL, 10);
	snprintf(s->model, sizeof(s->model), "%s", ctx->argv[1]);
	uint64_t sizing_start = TIMED_NOW();
	if (!is_clbf && strcmp(s->model, "auto") == 0) {
		choose_plbf_model(s, ctx);
	}
	s->sizing_time = TIMED_NOW() - sizing_start;

	if (is_clbf && model_num_rounds(ctx->feature_dir, s->model) < CLBF_MIN_ROUNDS) {
		fprintf(stderr, "clbf: model %s has fewer than %d trees (or no info.json), which CascadedLBF's CLBF does not support\n",
				s->model, CLBF_MIN_ROUNDS);
		abort();
	}

	char requested_model[256];
	snprintf(requested_model, sizeof(requested_model), "%s", s->model);
	int cached = !ctx->retrain && last_search.valid && last_search.is_clbf == is_clbf && last_search.budget == s->budget
				 && last_search.num_positives == ctx->num_positives && strcmp(last_search.requested_model, requested_model) == 0;
	if (cached) {
		snprintf(s->model, sizeof(s->model), "%s", last_search.model);
	}

	// Build with the model, falling back for the CLBF to the next smaller model whenever the filter
	// does not fit the budget (no configuration fits, or the built filter turns out larger).
	for (;;) {
		uint64_t prepare_start = TIMED_NOW();
		if (!prepare_model(s, ctx)) {
			abort();
		}
		uint64_t prepare_time = TIMED_NOW() - prepare_start;
		s->F = cached ? last_search.F : search_fpr_for_budget(s, ctx->num_positives);
		uint64_t before_build = TIMED_NOW();
		s->sizing_time += before_build - prepare_start - prepare_time;	// the search for F
		if (s->F > 0) {
			uint64_t optimize_time = 0;
			int built = is_clbf ? CLBFBuild(s->handle, s->F, CLBF_LAMBDA, CLBF_MU, ctx->positive_features, ctx->positives, ctx->num_positives,
											&optimize_time, &s->timing.build_model_ns, &s->timing.build_filter_ns)
								: PLBFBuild(s->handle, s->F, ctx->positive_features, ctx->positives, ctx->num_positives,
											&optimize_time, &s->timing.build_model_ns, &s->timing.build_filter_ns);
			if (built != 0) {
				abort();
			}
			uint64_t built_time = TIMED_NOW();
			size_t size = lbf_size_bytes(s);
			s->sizing_time += TIMED_NOW() - built_time;	// checking the built filter's size
			if (size <= s->budget) {
				// configuring the filter: loading and calibrating the model, and CascadedLBF's optimization at F
				s->config_time = prepare_time + optimize_time;
				break;
			}
			// a model that doesn't fit was part of finding the parameters that fit the budget
			s->sizing_time += built_time - before_build;
			fprintf(stderr, "%s: model %s built to %zu bytes, over the budget of %zu bytes\n",
					is_clbf ? "clbf" : "plbf", s->model, size, s->budget);
		}
		if (!is_clbf) {
			fprintf(stderr, "plbf: the budget of %zu bytes is too small for model %s\n", s->budget, s->model);
			abort();
		}
		s->sizing_time += prepare_time;
		long current_size = model_file_size(ctx->feature_dir, s->model);
		release_model(s);
		cached = 0;
		if (!next_smaller_model(ctx->feature_dir, current_size, CLBF_MIN_ROUNDS, s->model, sizeof(s->model))) {
			fprintf(stderr, "clbf: no model in %s fits the budget of %zu bytes\n", ctx->feature_dir, s->budget);
			abort();
		}
		fprintf(stderr, "clbf: trying the next smaller model, %s\n", s->model);
	}

	if (!cached && !ctx->retrain) {
		last_search.valid = 1;
		last_search.is_clbf = is_clbf;
		last_search.budget = s->budget;
		last_search.num_positives = ctx->num_positives;
		snprintf(last_search.requested_model, sizeof(last_search.requested_model), "%s", requested_model);
		snprintf(last_search.model, sizeof(last_search.model), "%s", s->model);
		last_search.F = s->F;
	}
	fprintf(stderr, "%s: budget %zu bytes -> F=%.6g\n", is_clbf ? "clbf" : "plbf", s->budget, s->F);
	return s;
}

static void *plbf_create(const build_ctx *ctx) {
	return lbf_create(ctx, 0);
}

static void *clbf_create(const build_ctx *ctx) {
	return lbf_create(ctx, 1);
}

static int lbf_query(void *filter, uint64_t key, const float *features) {
	lbf_state *s = filter;
#ifdef TRACK_FP_SOURCE
	return s->is_clbf ? CLBFLookupElementSource(s->handle, features, key, &s->last_source)
					  : PLBFLookupElementSource(s->handle, features, key, &s->last_source);
#else
	if (++s->num_queries % TIMING_SAMPLE == 0) {
		if (!s->is_clbf) {
			s->timing.timed_queries++;
			return PLBFLookupElementTimed(s->handle, features, key, &s->timing.query_model_ns, &s->timing.query_filter_ns);
		}
		if (s->num_sampled < CLBF_MAX_TIMED) {
			if (!s->sampled_keys) {
				s->sampled_features = malloc(CLBF_MAX_TIMED * sizeof(*s->sampled_features));
				s->sampled_keys = malloc(CLBF_MAX_TIMED * sizeof(*s->sampled_keys));
			}
			s->sampled_features[s->num_sampled] = features;
			s->sampled_keys[s->num_sampled++] = key;
		}
	}
	return s->is_clbf ? CLBFLookupElement(s->handle, features, key) : PLBFLookupElement(s->handle, features, key);
#endif
}

static void lbf_timing(void *filter, filter_timing *out) {
	lbf_state *s = filter;
	if (s->num_sampled) {
		// time the CLBF's sampled queries now, outside the harness's timing of the queries
		CLBFTimeLookups(s->handle, s->sampled_features, s->sampled_keys, s->num_sampled,
						&s->timing.query_model_ns, &s->timing.query_filter_ns);
		s->timing.timed_queries += s->num_sampled;
		s->num_sampled = 0;
	}
	*out = s->timing;
}

static int lbf_last_query_source(void *filter) {
	return ((lbf_state *)filter)->last_source;
}

static size_t lbf_size_bytes(void *filter) {
	lbf_state *s = filter;
	return s->is_clbf ? CLBFSizeInBytes(s->handle) : PLBFSizeInBytes(s->handle);
}

static void lbf_params_str(void *filter, char *buf, size_t len) {
	lbf_state *s = filter;
	size_t trees = s->is_clbf ? CLBFNumTrees(s->handle) : PLBFNumTrees(s->handle);
	snprintf(buf, len, "budget=%zu;F=%.6g;model=%s;trees=%zu", s->budget, s->F, s->model, trees);
}

static size_t lbf_model_size_bytes(void *filter) {
	lbf_state *s = filter;
	return s->is_clbf ? CLBFModelSizeInBytes(s->handle) : PLBFModelSizeInBytes(s->handle);
}

static double lbf_expected_fpr(void *filter) {
	return ((lbf_state *)filter)->F;
}

static uint64_t lbf_config_time(void *filter) {
	return ((lbf_state *)filter)->config_time;
}

static uint64_t lbf_sizing_time(void *filter) {
	return ((lbf_state *)filter)->sizing_time;
}

static void lbf_destroy(void *filter) {
	lbf_state *s = filter;
	if (s->is_clbf) {
		CLBFDestroy(s->handle);
	} else {
		PLBFDestroy(s->handle);
	}
	free(s->sampled_features);
	free(s->sampled_keys);
	free(s);
}

const filter_ops plbf_ops = {
	.name = "plbf",
	.usage = "<size budget in bytes [eg. 69160]> <model folder under ../data/cascaded/<dataset>, or auto to pick among the largest ones of each depth up to 0.2x the budget>",
	.num_args = 2,
	.rehash_per_trial = 0,
	.uses_features = 1,
	.train_proportion = NULL,
	.create = plbf_create,
	.query = lbf_query,
	.on_false_positive = NULL,
	.size_bytes = lbf_size_bytes,
	.params_str = lbf_params_str,
	.aux_build_time = NULL,
	.expected_fpr = lbf_expected_fpr,
	.config_time = lbf_config_time,
	.sizing_time = lbf_sizing_time,
	.timing = lbf_timing,
	.last_query_source = lbf_last_query_source,
	.model_size_bytes = lbf_model_size_bytes,
	.destroy = lbf_destroy,
};

const filter_ops clbf_ops = {
	.name = "clbf",
	.usage = "<size budget in bytes [eg. 69160]> <model folder under ../data/cascaded/<dataset> [eg. xgboost_d4_r100_e03]>",
	.num_args = 2,
	.rehash_per_trial = 0,
	.uses_features = 1,
	.train_proportion = NULL,
	.create = clbf_create,
	.query = lbf_query,
	.on_false_positive = NULL,
	.size_bytes = lbf_size_bytes,
	.params_str = lbf_params_str,
	.aux_build_time = NULL,
	.expected_fpr = lbf_expected_fpr,
	.config_time = lbf_config_time,
	.sizing_time = lbf_sizing_time,
	.timing = lbf_timing,
	.last_query_source = lbf_last_query_source,
	.model_size_bytes = lbf_model_size_bytes,
	.destroy = lbf_destroy,
};
