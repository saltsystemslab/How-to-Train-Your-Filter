/**
 * Declares C API wrappers around the C++ learned Bloom filters from CascadedLBF:
 * the partitioned learned Bloom filter (PLBF, CascadedLBF/src/plbf) and the
 * cascaded learned Bloom filter (CLBF, CascadedLBF/src/clbf).
 *
 * Both filters take a feature vector alongside each key, use a trained XGBoost model,
 * and are configured by a target FPR F after calibrating on a labeled validation set.
 * Sizes are in bytes and measured as CascadedLBF reports them: the size of the saved model.
 */

#ifndef LBF_WRAPPER_H
#define LBF_WRAPPER_H
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// The wrappers are built as shared libraries with hidden visibility, so only this API is exported
// and each library keeps its own copy of the CascadedLBF code (the PLBF and CLBF headers define
// different classes and non-inline functions with the same names, which cannot be linked together).
#define LBF_API __attribute__((visibility("default")))

#ifdef __cplusplus
extern "C" {
#endif

typedef struct WrappedPLBF WrappedPLBF;
typedef struct WrappedCLBF WrappedCLBF;

// Where a positive answer came from (see the *LookupElementSource functions): the filter that accepted the key
// was one CascadedLBF configures with FPR >= 1, which accepts everything, so the model's decision was unchecked;
// or it was a real backup Bloom filter.
#define LBF_SOURCE_MODEL 1
#define LBF_SOURCE_BACKUP 2

// Trains an XGBoost model the same way as CascadedLBF's train_xgboost (binary:logistic, missing value -1,
// seed 42) on X (num_rows rows of num_features floats) and labels y, and saves it to out_path in the format
// PLBFPrepare/CLBFPrepare load. Returns 0 on success.
LBF_API int LBFTrainXGBoostModel(const float *X, const float *y, size_t num_rows, size_t num_features,
                                 int max_depth, int num_boost_round, double eta, const char *out_path);

// Loads the XGBoost model at model_path and calibrates on the validation set
// (val_X: n_val rows of num_features floats, val_y: n_val labels of 0 or 1). Returns NULL on failure.
LBF_API WrappedPLBF *PLBFPrepare(const char *model_path, const float *val_X, const float *val_y, size_t n_val, size_t num_features);
// Size of a PLBF configured for num_positives keys at target FPR F, without building it. 0 on failure.
LBF_API size_t PLBFConfiguredSize(WrappedPLBF *wrapped, size_t num_positives, double F);
// Configures the PLBF at target FPR F and inserts the positives (row-major features). Returns 0 on success.
// optimize_ns receives the time spent choosing the filter's parameters for F (CascadedLBF's find_optimal_parameters)
// and waiting for other processes using CascadedLBF's temporary files,
// and model_ns and filter_ns the time spent scoring the positives with the model and inserting them into the backup
// Bloom filters (each may be NULL). Building the filter from the chosen parameters is timed by none of them.
LBF_API int PLBFBuild(WrappedPLBF *wrapped, double F, const float *positive_X, const uint64_t *keys, size_t num_positives,
                      uint64_t *optimize_ns, uint64_t *model_ns, uint64_t *filter_ns);
LBF_API bool PLBFLookupElement(WrappedPLBF *wrapped, const float *x, uint64_t key);
// Lookup that adds the time spent in the model and in the backup Bloom filters to *model_ns and *filter_ns.
LBF_API bool PLBFLookupElementTimed(WrappedPLBF *wrapped, const float *x, uint64_t key, uint64_t *model_ns, uint64_t *filter_ns);
// Lookup that reports in *source where a positive answer came from (LBF_SOURCE_MODEL or LBF_SOURCE_BACKUP).
LBF_API bool PLBFLookupElementSource(WrappedPLBF *wrapped, const float *x, uint64_t key, int *source);
LBF_API size_t PLBFSizeInBytes(WrappedPLBF *wrapped);
// Size of the model the built filter uses, measured as CascadedLBF reports model sizes (the saved XGBoost model),
// and its number of trees (0 if the filter uses no model).
LBF_API size_t PLBFModelSizeInBytes(WrappedPLBF *wrapped);
LBF_API size_t PLBFNumTrees(WrappedPLBF *wrapped);
LBF_API void PLBFDestroy(WrappedPLBF *wrapped);

// Same interface for the CLBF; lambda weighs memory against query time when choosing its
// configuration (1 = memory only) and mu weighs positive against negative query time.
LBF_API WrappedCLBF *CLBFPrepare(const char *model_path, const float *val_X, const float *val_y, size_t n_val, size_t num_features);
LBF_API size_t CLBFConfiguredSize(WrappedCLBF *wrapped, size_t num_positives, double F, double lambda, double mu);
LBF_API int CLBFBuild(WrappedCLBF *wrapped, double F, double lambda, double mu, const float *positive_X, const uint64_t *keys, size_t num_positives,
                      uint64_t *optimize_ns, uint64_t *model_ns, uint64_t *filter_ns);
LBF_API bool CLBFLookupElement(WrappedCLBF *wrapped, const float *x, uint64_t key);
// The CLBF interleaves its trees with its trunk Bloom filters, so timing each part inside a lookup would take
// two clock reads per tree. Instead this repeats the n given lookups after the experiment: whole passes over
// them are timed, once doing the full lookups and once doing only their model parts (the trees each lookup went
// through), and the rest of the lookups' time is the filters' time. Adds the totals over the n lookups.
LBF_API void CLBFTimeLookups(WrappedCLBF *wrapped, const float *const *xs, const uint64_t *keys, size_t n,
                             uint64_t *model_ns, uint64_t *filter_ns);
LBF_API bool CLBFLookupElementSource(WrappedCLBF *wrapped, const float *x, uint64_t key, int *source);
LBF_API size_t CLBFSizeInBytes(WrappedCLBF *wrapped);
LBF_API size_t CLBFModelSizeInBytes(WrappedCLBF *wrapped);
LBF_API size_t CLBFNumTrees(WrappedCLBF *wrapped);
LBF_API void CLBFDestroy(WrappedCLBF *wrapped);

#ifdef __cplusplus
}
#endif

#endif
