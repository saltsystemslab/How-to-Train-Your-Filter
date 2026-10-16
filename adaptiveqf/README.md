 # AdaptiveQF and Filter Experiments

 This folder originates from the base implementation of the **[AdaptiveQF](https://github.com/splatlab/adaptiveqf)**.

 The main difference from the original code is that the `test` folder contains new scripts for various experiments, while `include/exp_utility.h` and `src/exp_utility.c` define utility functions used during those scripts.
 These scripts run all five filters we compare (`aqf`, `stacked`, `bloom`, `plbf`, `clbf`), with the **Stacked**, Bloom and **CascadedLBF** filters called through the wrappers in `external`.
 Our experiments use the `*_generic` scripts and `write_queries.c` described below. The other scripts in `test` and the filters in `other_filters` come with the original **AdaptiveQF** code and aren't used in our experiments.

 The `external` folder now includes the **[Stacked Filter](https://github.com/kylebd99/StackedFilters)** implementation.
 `StackedFilters/wrapper` contains a C wrapper (`extern "C"`) for the **Stacked Filter** allowing it to be called from the C experiment scripts.
 `bloom_wrapper` contains a C wrapper for the Bloom filter used by **[CascadedLBF](../CascadedLBF)** (`CascadedLBF/thirdparty/fastfilter_cpp/src/bloom/bloom.h`), which must be present to build.
 `cascaded_wrapper` contains C wrappers for CascadedLBF's learned Bloom filters (PLBF and CLBF). They are built as shared
 libraries in `obj/` because the two filters' headers define different classes with the same names.

 ### Learned filters (plbf, clbf)
 These are **CascadedLBF**'s C++ PLBF (`CascadedLBF/src/plbf`) and CLBF (`CascadedLBF/src/clbf`), the learned filters in our
 comparison. The CLBF's own optimizer also considers the case where the CLBF reduces to a PLBF
 (`CascadedLBF/src/clbf/find_optimal_parameters_plbf.h`), which is a different code path from the `plbf` filter.
 The wrappers in `external/cascaded_wrapper` call these filters directly: CascadedLBF's own experiments and scripts
 (`CascadedLBF/experiment`, `CascadedLBF/scripts`, including its PLBF experiments) aren't used, apart from its XGBoost trainer
 (`CascadedLBF/experiment/xgboost/train_xgboost.cpp`), and neither are its other filters.
 Both are also separate from the python **FastPLBF** (a different PLBF implementation) and **Ada-BF** versions in `../learned`,
 which are only used in optional supplementary experiments.

 These filters query with each key's feature vector and need a trained XGBoost model. `../prepare_learned_models.sh` prepares
 every dataset once; for a single dataset, its two steps are:
 1. Export the dataset's features (the same ones used by the python learned filters, defined in `../learned/updated_classifiers.py`), labels, and the XGBoost
    training and calibration sets: `cd ../learned && python3 utils/export_cascaded_inputs.py --dataset url`
    (writes to `../data/cascaded/url`).
 2. Train XGBoost models with CascadedLBF's trainer: `cd ../learned && ./utils/train_cascaded_models.sh url`
    (eta 0.3; depth 4 with 1-20 and 30-100 boosting rounds, and depth 8 with 10-100 rounds, written to
    `../data/cascaded/url/xgboost_d<depth>_r<rounds>_e03`).

 The model folder is the filter's second argument. Both filters are configured by a target FPR, so the adapter searches for
 the smallest target FPR whose size (model plus Bloom filters, measured as CascadedLBF reports it) fits the budget.
 - `clbf` is built as in CascadedLBF's experiments: the depth 4, 100 round model (`xgboost_d4_r100_e03`), lambda = 1 and mu = 0.
   It chooses how many of the model's trees to keep, so the model can be larger than the budget.
 - `plbf` always keeps its whole model. With `auto`, it takes the largest trained model of each depth no larger than 0.2x the
   budget, and uses the one that reaches the lowest target FPR (the larger one if they are within 25% of each other).

 Each experiment has a single script that can run any supported filter. The filter is chosen by name as the first argument
 (e.g. `./test_distribution_generic aqf ...` or `./test_distribution_generic stacked ...`), followed by the shared experiment arguments
 and then the filter-specific arguments (`<q> <r>` for `aqf`, `<size in bytes> <training proportion>` for `stacked`, `<size in bytes>` for `bloom`,
 `<size budget in bytes> <model folder>` for `plbf` and `clbf`).
 Running a script without arguments prints its usage.
 - `test_distribution_generic.c`: The distribution experiment, where 10 million queries from a specified query distribution are performed on the filter after performing all insertions. This test also records the construction and query times for the filters.
 - `test_one_pass_generic.c`: The one-pass experiment, where after performing all inserts, all elements from the dataset are queried exactly once, in the order given by a one-pass index file.
 - `test_advers_dist_generic.c`: The adversarial experiment, where 10 million queries from a specified query distribution are performed on the filter after performing all insertions. During the first half of the query set, false positives are recorded. During the second half of the query set, a proportion of the query set is replaced with the recorded false positives.
 - `test_dynamic_generic.c`: The dynamic experiments, in two workloads: the swapping workload (`test_dynamic_generic`), where the queries come from a query index file and each churn swaps part of the positive key set with negative keys (a fifth of the smaller of the positive and negative sets, so shalla, with more positives than negatives, only ever replaces a third of its positives), and the shifting workload (`test_shifting_generic`, the same file built with `SHIFTING_WORKLOAD`, which takes no index file), where the positive and the negative keys are each split in half into an active and a replacement set, each window's queries are sampled uniformly from the active sets, and each churn swaps a fifth of each active set with its replacement set. The queries run in 10 windows: a warm-up window (not recorded), a window without a churn, then a churn at the start of each of the other 8 windows, after which the filters are rebuilt from the new positives. Filters only learn from the previous window's queries, labeled with the current positives: the stacked filter learns its negatives from them (its training proportion of all queries, at most one window), and the learned filters retrain on them and the current positives (split into training and calibration sets in the same proportions as the saved models, see `learned/utils/export_cascaded_inputs.py`). The instantaneous FPR is measured every 1% of the queries over the current window's queries. Each workload runs 3 trials.

 Results are written to `../results/distribution_results.csv` (distribution and one-pass), `../results/advers_results.csv`, `../results/dynamic_results.csv` (swapping) and `../results/shifting_results.csv`,
 with a `filter` column naming the filter and a `params` column holding its parameters. `size` is the filter's total size in bytes,
 `model_size` the part of it taken by the model (0 for filters without one; learned filters also list their number of `trees` in
 `params`), and `num_adaptations` the AQF's successful adaptations (in the dynamic results, the total so far in the run, including
 those of filters since rebuilt; empty for filters that don't adapt).

 ### Timing
 All times are in nanoseconds (monotonic clock):
 - `build_time_ns`: building the filter from its chosen parameters. Choosing them is not included: the search for a learned filter's
   target FPR, CascadedLBF's and the stacked filter's parameter optimization, loading and calibrating (or retraining) models, and
   checking the built size are reported as `config_time_ns`.
 - `aux_build_time_ns`: the part of the build spent on auxiliary structures (the AQF's reverse map).
 - `build_model_ns` / `build_filter_ns`: scoring the positives with the model / inserting them into the backup filters (filters
   without a model spend all their build time in the filter).
 - `amortized_query_ns`: average time per query, timed around each query. `query_model_ns` / `query_filter_ns` split a learned filter's
   query into its model and backup filters, measured on every 16th query (with the timer's own cost subtracted), since timing the parts of
   every query would slow every query down.
 - `amortized_adapt_ns`: the AQF's adaptations.

 ### Where learned filters' false positives come from
 The `*_generic_fpsource` binaries are built with `TRACK_FP_SOURCE` (`make all` builds both sets). They record, for PLBF and CLBF, whether
 each false positive came from an unchecked model decision (the key was accepted by a filter CascadedLBF configured to accept everything)
 or from a backup Bloom filter error, in the `fp_from_model` and `fp_from_backup` columns. They do not time anything, so their timing
 columns are empty; set `TRACK_FP_SOURCE=1` in `../run_queries.sh` to run them.

 ### Adding a filter
 Filters are accessed through the `filter_ops` interface in `include/filter_iface.h`. To add one:
 1. Write an adapter in `src/` implementing `filter_ops` (see `src/aqf_adapter.c` and `src/stacked_adapter.c`). A filter written in C++ should expose
    an `extern "C"` wrapper like `external/StackedFilters/wrapper/stacked_wrapper.cpp` for its adapter to call.
 2. Declare its ops table in `include/filter_iface.h` and add it to `src/filter_registry.c`.
 3. Add its object files to the `test_*_generic` targets in the `Makefile`.

 The harness handles reading data and queries, hashing, timing, FPR calculation and output; adapters only build, query, and (optionally)
 react to false positives. Filters that use a sample of negative queries at construction time (like the stacked filter) receive them through `build_ctx`.

 Some utility functions are also included. In particular, `test/write_queries.c` describes how query indices for each dataset are drawn,
 and `src/exp_utility.c` includes helper functions for the test scripts.

For example usages, see `../run_queries.sh`