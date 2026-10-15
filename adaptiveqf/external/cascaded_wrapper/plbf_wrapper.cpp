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
// The lookup and insert loops below follow PLBF::contains_all and PLBF::add_all step for step, so the model and
// the backup Bloom filters can be timed separately and the source of a positive answer can be reported. They
// need the PLBF's private members (its trees, thresholds and Bloom filters).
#define private public
#include "plbf/calibration_manager.h"
#undef private

struct WrappedPLBF {
    PLBF base;                      // holds the loaded model
    Calibration_Manager manager;    // holds the calibration statistics
    PLBF filter;                    // the configured filter
    std::vector<char> region_unchecked;  // whether each region's Bloom filter was configured to accept everything
    std::vector<float> row;         // reused feature vector for lookups
    size_t num_features;
    bool built = false;
};

// Scores x with the model and returns its region (the Bloom filter to check), as PLBF::contains_all does.
static inline int plbf_region(WrappedPLBF *wrapped, const float *x) {
    PLBF &f = wrapped->filter;
    wrapped->row.assign(x, x + wrapped->num_features);
    float score = f.calculate_initial_score();
    for (size_t j = 0; j < f.d; ++j) {
        score = f.calculate_cumulative_score(wrapped->row, j, score);
    }
    return std::lower_bound(f.th_f.begin(), f.th_f.end(), score) - f.th_f.begin();
}

extern "C" {

int LBFTrainXGBoostModel(const float *X, const float *y, size_t num_rows, size_t num_features,
                         int max_depth, int num_boost_round, double eta, const char *out_path) {
    XGBSetGlobalConfig("{\"verbosity\": 0}");
    DMatrixHandle dtrain;
    BoosterHandle booster;
    if (XGDMatrixCreateFromMat(X, num_rows, num_features, -1, &dtrain) != 0) return -1;
    XGDMatrixSetFloatInfo(dtrain, "label", y, num_rows);
    XGBoosterCreate(&dtrain, 1, &booster);
    XGBoosterSetParam(booster, "objective", "binary:logistic");
    XGBoosterSetParam(booster, "max_depth", std::to_string(max_depth).c_str());
    XGBoosterSetParam(booster, "eta", std::to_string((float)eta).c_str());
    XGBoosterSetParam(booster, "verbosity", "0");
    XGBoosterSetParam(booster, "seed", "42");
    for (int i = 0; i < num_boost_round; i++) {
        XGBoosterUpdateOneIter(booster, i, dtrain);
    }
    // saved as train_xgboost does: the model size, then the model in XGBoost's default (UBJSON) format
    bst_ulong out_len;
    const char *model_data;
    int result = XGBoosterSaveModelToBuffer(booster, "{\"format\": \"ubj\"}", &out_len, &model_data);
    if (result == 0) {
        FILE *f = fopen(out_path, "wb");
        if (f && fwrite(&out_len, sizeof(out_len), 1, f) == 1 && fwrite(model_data, 1, out_len, f) == out_len) {
            result = 0;
        } else {
            result = -1;
        }
        if (f) fclose(f);
    }
    XGBoosterFree(booster);
    XGDMatrixFree(dtrain);
    return result;
}

WrappedPLBF *PLBFPrepare(const char *model_path, const float *val_X, const float *val_y, size_t n_val, size_t num_features) {
    CascadedTmpLock lock;
    WrappedPLBF *wrapped = new WrappedPLBF;
    wrapped->num_features = num_features;
    // XGBoost warns on every model save because CascadedLBF uses a .bin extension
    XGBSetGlobalConfig("{\"verbosity\": 0}");
    try {
        QuietCout quiet;
        wrapped->base.load_trained_ml_model(model_path, "xgboost");
        wrapped->manager.set_plbf(wrapped->base);
        wrapped->manager.calibrate(to_rows(val_X, n_val, num_features), to_rows(val_y, n_val, 1));
    } catch (const std::exception &e) {
        fprintf(stderr, "PLBF: failed to prepare: %s\n", e.what());
        delete wrapped;
        return NULL;
    }
    return wrapped;
}

size_t PLBFConfiguredSize(WrappedPLBF *wrapped, size_t num_positives, double F) {
    CascadedTmpLock lock;
    try {
        PLBF filter;
        {
            QuietCout quiet;
            filter = wrapped->manager.configure_lbf_assembly(num_positives, F);
        }
        return saved_size_bytes(filter);
    } catch (const std::exception &e) {
        fprintf(stderr, "PLBF: failed to configure at F=%g: %s\n", F, e.what());
        return 0;
    }
}

int PLBFBuild(WrappedPLBF *wrapped, double F, const float *positive_X, const uint64_t *keys, size_t num_positives,
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
        m.lbf_config = find_optimal_parameters(num_positives, F, m.calibration_stats_pos, m.calibration_stats_neg,
                                               m.calibration_time_pos_us, m.calibration_time_neg_us, m.plbf.ml_model_size_kb);
        if (optimize_ns) *optimize_ns = now_ns() - start + lock_wait;
        wrapped->filter = m.plbf.configure_lbf_assembly(num_positives, m.lbf_config);
    } catch (const std::exception &e) {
        fprintf(stderr, "PLBF: failed to build at F=%g: %s\n", F, e.what());
        return -1;
    }
    // same condition configure_lbf_assembly uses to give a region a Bloom filter that accepts everything
    const LbfConfig config = wrapped->manager.get_lbf_config();
    wrapped->region_unchecked.assign(config.f.size(), 0);
    for (size_t i = 0; i < config.f.size(); i++) {
        wrapped->region_unchecked[i] = config.f[i] >= 1 - 1e-9;
    }

    // PLBF::add_all, split into scoring every positive with the model, then inserting them into the
    // backup Bloom filters (the result is the same, since insertion order does not matter)
    uint64_t start = now_ns();
    std::vector<int> regions(num_positives);
    for (size_t i = 0; i < num_positives; i++) {
        regions[i] = plbf_region(wrapped, positive_X + i * wrapped->num_features);
    }
    uint64_t scored = now_ns();
    for (size_t i = 0; i < num_positives; i++) {
        wrapped->filter.bf_f[regions[i]].Add(keys[i]);
    }
    uint64_t inserted = now_ns();
    if (model_ns) *model_ns = scored - start;
    if (filter_ns) *filter_ns = inserted - scored;
    wrapped->built = true;
    return 0;
}

bool PLBFLookupElement(WrappedPLBF *wrapped, const float *x, uint64_t key) {
    int region = plbf_region(wrapped, x);
    return wrapped->filter.bf_f[region].Contain(key) == bloomfilter::Ok;
}

bool PLBFLookupElementTimed(WrappedPLBF *wrapped, const float *x, uint64_t key, uint64_t *model_ns, uint64_t *filter_ns) {
    uint64_t start = now_ns();
    int region = plbf_region(wrapped, x);
    uint64_t scored = now_ns();
    bool found = wrapped->filter.bf_f[region].Contain(key) == bloomfilter::Ok;
    uint64_t checked = now_ns();
    *model_ns += segment_ns(start, scored);
    *filter_ns += segment_ns(scored, checked);
    return found;
}

bool PLBFLookupElementSource(WrappedPLBF *wrapped, const float *x, uint64_t key, int *source) {
    int region = plbf_region(wrapped, x);
    *source = wrapped->region_unchecked[region] ? LBF_SOURCE_MODEL : LBF_SOURCE_BACKUP;
    return wrapped->filter.bf_f[region].Contain(key) == bloomfilter::Ok;
}

// CascadedLBF's own lookup, used to check that the lookups above give the same answers.
LBF_API bool PLBFLookupElementReference(WrappedPLBF *wrapped, const float *x, uint64_t key) {
    std::vector<std::vector<float>> X(1, std::vector<float>(x, x + wrapped->num_features));
    std::vector<uint64_t> keys(1, key);
    return wrapped->filter.contains_all(X, keys)[0] == bloomfilter::Ok;
}

size_t PLBFSizeInBytes(WrappedPLBF *wrapped) {
    CascadedTmpLock lock;
    return wrapped->built ? saved_size_bytes(wrapped->filter) : 0;
}

size_t PLBFModelSizeInBytes(WrappedPLBF *wrapped) {
    if (!wrapped->built || wrapped->filter.d == 0) return 0;
    CascadedTmpLock lock;
    return (size_t)std::lround(get_xgboost_model_size_kb(wrapped->filter.booster) * 1000);
}

size_t PLBFNumTrees(WrappedPLBF *wrapped) {
    return wrapped->built ? wrapped->filter.d : 0;
}

void PLBFDestroy(WrappedPLBF *wrapped) {
    delete wrapped;
}

}
