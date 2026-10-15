#include "lbf_wrapper.h"
#include "lbf_common.h"
// Library headers first, so the access override below only applies to CascadedLBF's own classes.
#include <xgboost/c_api.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
// The lookup and insert loops below follow CLBF::contains_all and CLBF::add_all step for step, so the model and
// the backup Bloom filters can be timed separately and the source of a positive answer can be reported. They
// need the CLBF's private members (its trees, thresholds and Bloom filters).
#define private public
#include "clbf/calibration_manager.h"
#undef private

struct WrappedCLBF {
    CLBF base;                      // holds the loaded model
    Calibration_Manager manager;    // holds the calibration statistics
    CLBF filter;                    // the configured filter
    std::vector<char> branch_unchecked;  // whether each branch Bloom filter was configured to accept everything
    std::vector<char> final_unchecked;   // whether each final-region Bloom filter was configured to accept everything
    std::vector<float> row;         // reused feature vector for lookups
    size_t num_features;
    bool built = false;
};

// Where CLBF::contains_all would stop for x: the key is checked by the trunk filters of trees 0..last_tree,
// then (unless a trunk filter rejects it) by branch filter last_tree, or by final-region filter region if
// region >= 0. Computes only the model's part of the cascade, so it also gives the insertion path of a positive.
struct ClbfPath {
    size_t last_tree;
    int region;
};

static inline float clbf_score_step(WrappedCLBF *wrapped, size_t tree, float score) {
    return wrapped->filter.calculate_cumulative_score(wrapped->row, tree, score);
}

// Whether the model sends the key down a branch after tree j (only for j < d - 1).
static inline bool clbf_branches(WrappedCLBF *wrapped, size_t j, float score) {
    return wrapped->filter.th_b[j] < score;
}

static inline int clbf_final_region(WrappedCLBF *wrapped, float score) {
    CLBF &f = wrapped->filter;
    return std::lower_bound(f.th_f.begin(), f.th_f.end(), score) - f.th_f.begin();
}

// CLBF::contains_all for one key; reports where a positive answer came from in *source and the number of trees
// the key was scored by in *trees when they are not NULL.
static inline bool clbf_lookup(WrappedCLBF *wrapped, const float *x, uint64_t key, int *source, size_t *trees) {
    CLBF &f = wrapped->filter;
    if (f.d == 0) {
        // no model, only the single Bloom filter
        if (source) *source = LBF_SOURCE_BACKUP;
        if (trees) *trees = 0;
        return f.bf_t[0].Contain(key) == bloomfilter::Ok;
    }
    wrapped->row.assign(x, x + wrapped->num_features);
    float score = f.calculate_initial_score();
    for (size_t j = 0; j < f.d; ++j) {
        if (f.bf_t[j].Contain(key) == bloomfilter::Status::NotFound) {
            if (trees) *trees = j;
            return false;
        }
        score = clbf_score_step(wrapped, j, score);
        bool last = j == f.d - 1;
        bool branch = !last && clbf_branches(wrapped, j, score);
        if (last || branch) {
            int region = last ? clbf_final_region(wrapped, score) : -1;
            bool found = (last ? f.bf_f[region] : f.bf_b[j]).Contain(key) == bloomfilter::Ok;
            if (source) {
                bool unchecked = last ? wrapped->final_unchecked[region] : wrapped->branch_unchecked[j];
                *source = unchecked ? LBF_SOURCE_MODEL : LBF_SOURCE_BACKUP;
            }
            if (trees) *trees = j + 1;
            return found;
        }
    }
    if (trees) *trees = f.d;
    return true;
}

// Only the model's part of a lookup that scored the key with its first `trees` trees: the same feature copy,
// tree evaluations and threshold comparisons, without the Bloom filter checks.
static inline float clbf_model_part(WrappedCLBF *wrapped, const float *x, size_t trees) {
    CLBF &f = wrapped->filter;
    if (trees == 0) return 0;
    wrapped->row.assign(x, x + wrapped->num_features);
    float score = f.calculate_initial_score(), decisions = 0;
    for (size_t j = 0; j < trees; ++j) {
        score = clbf_score_step(wrapped, j, score);
        if (j == f.d - 1) {
            decisions += clbf_final_region(wrapped, score);
        } else {
            decisions += clbf_branches(wrapped, j, score);
        }
    }
    return score + decisions;
}

extern "C" {

WrappedCLBF *CLBFPrepare(const char *model_path, const float *val_X, const float *val_y, size_t n_val, size_t num_features) {
    CascadedTmpLock lock;
    WrappedCLBF *wrapped = new WrappedCLBF;
    wrapped->num_features = num_features;
    // XGBoost warns on every model save because CascadedLBF uses a .bin extension
    XGBSetGlobalConfig("{\"verbosity\": 0}");
    try {
        QuietCout quiet;
        wrapped->base.load_trained_ml_model(model_path, "xgboost");
        wrapped->manager.set_clbf(wrapped->base);
        wrapped->manager.calibrate(to_rows(val_X, n_val, num_features), to_rows(val_y, n_val, 1));
    } catch (const std::exception &e) {
        fprintf(stderr, "CLBF: failed to prepare: %s\n", e.what());
        delete wrapped;
        return NULL;
    }
    return wrapped;
}

size_t CLBFConfiguredSize(WrappedCLBF *wrapped, size_t num_positives, double F, double lambda, double mu) {
    CascadedTmpLock lock;
    try {
        CLBF filter;
        {
            QuietCout quiet;
            filter = wrapped->manager.configure_lbf_assembly(num_positives, F, lambda, mu);
        }
        return saved_size_bytes(filter);
    } catch (const std::exception &e) {
        fprintf(stderr, "CLBF: failed to configure at F=%g: %s\n", F, e.what());
        return 0;
    }
}

int CLBFBuild(WrappedCLBF *wrapped, double F, double lambda, double mu, const float *positive_X, const uint64_t *keys, size_t num_positives,
              uint64_t *optimize_ns, uint64_t *model_ns, uint64_t *filter_ns) {
    // time spent waiting for other processes is not part of building the filter, so it is reported with the
    // optimization time
    uint64_t lock_start = now_ns();
    CascadedTmpLock lock;
    uint64_t lock_wait = now_ns() - lock_start;
    try {
        QuietCout quiet;
        // Calibration_Manager::configure_lbf_assembly, split into choosing the parameters (find_optimal_parameters)
        // and building the filter from them
        Calibration_Manager &m = wrapped->manager;
        uint64_t start = now_ns();
        m.lbf_config = find_optimal_parameters(num_positives, F, lambda, mu, m.calibration_stats_pos, m.calibration_stats_neg,
                                               m.calibration_time_pos_us, m.calibration_time_neg_us, m.clbf.size_ml_bits,
                                               false, "", "", 0.0f, 0.0f, 0.0f);
        if (optimize_ns) *optimize_ns = now_ns() - start + lock_wait;
        wrapped->filter = m.clbf.configure_lbf_assembly(num_positives, m.lbf_config);
    } catch (const std::exception &e) {
        fprintf(stderr, "CLBF: failed to build at F=%g: %s\n", F, e.what());
        return -1;
    }
    CLBF &f = wrapped->filter;
    // same conditions configure_lbf_assembly uses to give a filter that accepts everything
    const LbfConfig config = wrapped->manager.get_lbf_config();
    wrapped->branch_unchecked.assign(f.d > 0 ? f.d - 1 : 0, 0);
    for (size_t i = 0; i + 1 < f.d; i++) wrapped->branch_unchecked[i] = config.b[i] >= 1 - 1e-9;
    wrapped->final_unchecked.assign(f.k, 0);
    for (size_t i = 0; i < f.k; i++) wrapped->final_unchecked[i] = config.f[i] >= 1 - 1e-9;

    // CLBF::add_all, split into finding every positive's path through the cascade with the model, then
    // inserting it into the Bloom filters along that path (the result is the same, since insertion order
    // does not matter)
    uint64_t start = now_ns();
    std::vector<ClbfPath> paths(num_positives);
    if (f.d > 0) {
        for (size_t i = 0; i < num_positives; i++) {
            wrapped->row.assign(positive_X + i * wrapped->num_features, positive_X + (i + 1) * wrapped->num_features);
            float score = f.calculate_initial_score();
            for (size_t j = 0; j < f.d; ++j) {
                score = clbf_score_step(wrapped, j, score);
                if (j == f.d - 1) {
                    paths[i] = {j, clbf_final_region(wrapped, score)};
                } else if (clbf_branches(wrapped, j, score)) {
                    paths[i] = {j, -1};
                    break;
                }
            }
        }
    }
    uint64_t scored = now_ns();
    for (size_t i = 0; i < num_positives; i++) {
        if (f.d == 0) {
            f.bf_t[0].Add(keys[i]);
            continue;
        }
        for (size_t j = 0; j <= paths[i].last_tree; j++) f.bf_t[j].Add(keys[i]);
        if (paths[i].region >= 0) {
            f.bf_f[paths[i].region].Add(keys[i]);
        } else {
            f.bf_b[paths[i].last_tree].Add(keys[i]);
        }
    }
    uint64_t inserted = now_ns();
    if (model_ns) *model_ns = scored - start;
    if (filter_ns) *filter_ns = inserted - scored;
    wrapped->built = true;
    return 0;
}

bool CLBFLookupElement(WrappedCLBF *wrapped, const float *x, uint64_t key) {
    return clbf_lookup(wrapped, x, key, NULL, NULL);
}

void CLBFTimeLookups(WrappedCLBF *wrapped, const float *const *xs, const uint64_t *keys, size_t n,
                     uint64_t *model_ns, uint64_t *filter_ns) {
    if (n == 0) return;
    volatile float sink = 0;
    std::vector<size_t> trees(n);
    // whole passes over the queries, each timed as one block: the lookups, then only their model parts. The
    // passes run lookup, model, model, lookup so that neither kind always follows the other (a pass leaves
    // features and trees in the cache for the next one), and each kind's time is the mean of its two passes.
    auto lookup_pass = [&] {
        uint64_t start = now_ns();
        for (size_t i = 0; i < n; ++i) sink = sink + clbf_lookup(wrapped, xs[i], keys[i], NULL, &trees[i]);
        return now_ns() - start;
    };
    auto model_pass = [&] {
        uint64_t start = now_ns();
        for (size_t i = 0; i < n; ++i) sink = sink + clbf_model_part(wrapped, xs[i], trees[i]);
        return now_ns() - start;
    };
    uint64_t total = lookup_pass();
    uint64_t model = model_pass();
    model += model_pass();
    total += lookup_pass();
    total /= 2;
    model = std::min(total, model / 2);
    *model_ns += model;
    *filter_ns += total - model;
    (void)sink;
}

bool CLBFLookupElementSource(WrappedCLBF *wrapped, const float *x, uint64_t key, int *source) {
    return clbf_lookup(wrapped, x, key, source, NULL);
}

// CascadedLBF's own lookup, used to check that the lookups above give the same answers.
LBF_API bool CLBFLookupElementReference(WrappedCLBF *wrapped, const float *x, uint64_t key) {
    std::vector<std::vector<float>> X(1, std::vector<float>(x, x + wrapped->num_features));
    std::vector<uint64_t> keys(1, key);
    return wrapped->filter.contains_all(X, keys)[0] == bloomfilter::Ok;
}

size_t CLBFSizeInBytes(WrappedCLBF *wrapped) {
    CascadedTmpLock lock;
    return wrapped->built ? saved_size_bytes(wrapped->filter) : 0;
}

size_t CLBFModelSizeInBytes(WrappedCLBF *wrapped) {
    if (!wrapped->built || wrapped->filter.d == 0) return 0;
    CascadedTmpLock lock;
    return (size_t)std::lround(get_xgboost_model_size_kb(wrapped->filter.booster) * 1000);
}

size_t CLBFNumTrees(WrappedCLBF *wrapped) {
    return wrapped->built ? wrapped->filter.d : 0;
}

void CLBFDestroy(WrappedCLBF *wrapped) {
    delete wrapped;
}

}
