 # Adaptive Quotient Filter

 This folder originates from the base implementation of the **[AdaptiveQf](https://github.com/splatlab/adaptiveqf)**.

 The main difference from the original code is that the `test` folder contains new scripts for various experiments, while `include/exp_utility.h` and `src/exp_utility.c` define utility functions used during those scripts.

 The main scripts for adaptive filter experiments in the `test` folder include:
 - `test_distribution.c`: The distribution experiment, where 10 million queries from a specified query distribution are performed on the adaptive quotient filter after performing all insertions. This test also records the construction and query times for the filters.
 - `test_one_pass.c`: The one-pass distribution experiment, where after performing all inserts, all elements from the datasets are queried exactly once.
 - `test_advers_dist.c`: The adversarial experiment, where 10 million queries from a specified query distribution are performed on the adaptive quotient filter after performing all insertions. During the first half of the query set, false positives are recorded. During the second half of the query set, a proportion of the query set is replaced with the recorded false positives.
 - `test_dynamic.c`: The dynamic experiment, where 10 million queries from a specified query distribution are performed on the adaptive quotient filter after performing all insertions, but throughout the course of the queries the positive key set is gradually replaced with negative keys then back to the original set.

For example usages, see `../run_queries.sh`