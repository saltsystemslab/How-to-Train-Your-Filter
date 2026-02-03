# Learned Filters

This folder originates from the base implementation of the **[FastPLBF](https://github.com/atsukisato/FastPLBF)**.

`src/filters` contains the implementations of the learned filters used in the experiments. `ada_bf_index_query.py` and `ada_bf_model.py` are modified versions of the original **[Ada-BF](https://github.com/DAIZHENWEI/Ada-BF)** implementation, while `FastPLBF_M_dist.py` and `FastPLBF_M_model.py` are the modified versions of the original **[FastPLBF](https://github.com/atsukisato/FastPLBF)** implementation.

The main scripts for running learned filter experiments are in this folder. They include:
- `run_exp_dynamic_with_model_rebuild.py`: The dynamic experiment where filter contents change over the course of a set of queries and learned filters are allowed to retrain models in between churns.
- `run_exp_dynamic_with_model_prescores.py`: The dynamic experiment without model retraining.
- `run_exp_model_degrad.py`: The training proportion experiment where an increasing number of negative keys are included in the training set of the models.
- `run_exp_with_changing_model.py`: The model proportion experiment where the filter size is fixed but the internal trained model size increases.
- `run_exp_with_model_prescores.py`: The distribution experiment, where 10 million queries from a specified query distribution (including adversarial) are performed on a learned filter after a model precomputes scores for all elements in the dataset.
- `run_exp_with_model.py`: The timing experiment, where the filters perform queries following some distribution but the model computes scores during query time.
- `plot_results.py`: plots the results for all filters (including stacked and learned filters).

All experiments are attempted with all learned filters (plbf, adabf) using all model variants (RandomForest, DecisionTree, LogisticRegression). When plotting the results, FPR experiments will generally
only include all learned filters using the DecisionTree.

We also include utility functions in the `utils` folder, such as those used to obtain the sizes of AdaptiveQF filters across experiments or those used to graph score distributions of models across different datasets.

For example usages, see `../run_queries.sh`.