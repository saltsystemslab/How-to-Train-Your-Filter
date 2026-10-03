 # AdaptiveQF and Stacked Filter

 This folder originates from the base implementation of the **[AdaptiveQf](https://github.com/splatlab/adaptiveqf)**.

 The main difference from the original code is that the `test` folder contains new scripts for various experiments, while `include/exp_utility.h` and `src/exp_utility.c` define utility functions used during those scripts.

 The `external` folder now includes the **[Stacked Filter](https://github.com/kylebd99/StackedFilters)** implementation.
 `StackedFilters/wrapper` contains a C wrapper (`extern "C"`) for the **Stacked Filter** allowing it to be called from the C experiment scripts.

 Each experiment has a single script that can run any supported filter. The filter is chosen by name as the first argument
 (e.g. `./test_distribution_generic aqf ...` or `./test_distribution_generic stacked ...`), followed by the shared experiment arguments
 and then the filter-specific arguments (`<q> <r>` for `aqf`, `<size in bytes> <training proportion>` for `stacked`).
 Running a script without arguments prints its usage.
 - `test_distribution_generic.c`: The distribution experiment, where 10 million queries from a specified query distribution are performed on the filter after performing all insertions. This test also records the construction and query times for the filters.
 - `test_one_pass_generic.c`: The one-pass experiment, where after performing all inserts, all elements from the dataset are queried exactly once, in the order given by a one-pass index file.
 - `test_advers_dist_generic.c`: The adversarial experiment, where 10 million queries from a specified query distribution are performed on the filter after performing all insertions. During the first half of the query set, false positives are recorded. During the second half of the query set, a proportion of the query set is replaced with the recorded false positives.
 - `test_dynamic_generic.c`: The dynamic experiment, where 10 million queries from a specified query distribution are performed on the filter after performing all insertions, but throughout the course of the queries the positive key set is gradually replaced with negative keys then back to the original set. The filter is rebuilt from the new positive set after each change.

 Results are written to `../results/distribution_results.csv` (distribution and one-pass), `../results/advers_results.csv`, and `../results/dynamic_results.csv`,
 with a `filter` column naming the filter and a `params` column holding its parameters.

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