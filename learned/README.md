# Learned Filters

This folder originates from the base implementation of the **[FastPLBF](https://github.com/atsukisato/FastPLBF)**.

The learned filters in our main comparison are **[CascadedLBF](https://github.com/atsukisato/CascadedLBF)**'s C++ PLBF and CLBF, which run from the `../adaptiveqf` experiments (see `../adaptiveqf/README.md`). This folder holds the scripts that prepare them and plot every filter's results, as well as python versions of the **FastPLBF** and **Ada-BF** filters used in optional supplementary experiments. The python **FastPLBF** is a separate implementation of the PLBF from CascadedLBF's, with its own models (scikit-learn instead of XGBoost), so the python `plbf` results are not the same filter as the `plbf` in the main comparison.

## Scripts for the main comparison
- `updated_classifiers.py`: reads each dataset and defines its key features, which both the python learned filters and the CascadedLBF filters (through `utils/export_cascaded_inputs.py`) use. It also trains and loads the python filters' models, saved to `models/`.
- `read_pos_counts.py`: prints the AdaptiveQF q value for each dataset's number of positive keys, used by `../run_queries.sh`.
- `plot_generic_results.py`: plots the comparison of the five filters (`aqf`, `stacked`, `bloom`, `plbf`, `clbf`) from the `../adaptiveqf` experiments: the FPR for each workload, and the construction and query times.
- `plot_radar.py`: summarizes every experiment in one radar chart for the **AdaptiveQF**, **Stacked**, **PLBF** and **CLBF** filters, where each axis is normalized per dataset (see the comments at the top of the script). With `--split`, it saves the overview (with a legend) and one radar per filter as two figures.
- `utils/export_cascaded_inputs.py` and `utils/train_cascaded_models.sh`: export each dataset's features and train the XGBoost models the CascadedLBF filters use (both run by `../prepare_learned_models.sh`).
- `utils/time_model_training.sh`: times the training of those models, since the construction times include it.
- `utils/prepare_sweeps.py` and `utils/collect_sweeps.py`: set up and collect the learned filter sweeps (`../run_sweeps.sh`), which rerun the model proportion and training proportion experiments below with CascadedLBF's PLBF and XGBoost models.
- `utils/write_supplementary_tables.py`: writes the supplementary tables S1 and S2 (where false positives come from, and query repetition).

## Python learned filters (optional)
`src/filters` contains the python learned filter implementations. `ada_bf_index_query.py` and `ada_bf_model.py` are modified versions of the original **[Ada-BF](https://github.com/DAIZHENWEI/Ada-BF)** implementation, while `FastPLBF_M_dist.py` and `FastPLBF_M_model.py` are the modified versions of the original **[FastPLBF](https://github.com/atsukisato/FastPLBF)** implementation. These versions have a model compute key scores on-the-fly. The other files in `src/filters` (such as `PLBF.py`, `FastPLBF.py` and `FastPLBFpp.py`) come with the original **FastPLBF** code; only `PLBF_M.py` and `PLBF_M_model.py`, which the modified versions build on, are used.

The scripts for running the python learned filter experiments are in this folder. They are only run by `../run_queries.sh` with `RUN_PYTHON_LEARNED=1`. They include:
- `run_exp_dynamic_with_model_rebuild.py`: The dynamic experiment where filter contents change over the course of a set of queries and learned filters are allowed to retrain models in between churns.
- `run_exp_dynamic_with_prescores.py`: The dynamic experiment without model retraining.
- `run_exp_model_degrad.py`: The training proportion experiment where an increasing number of negative keys are included in the training set of the models.
- `run_exp_with_changing_model.py`: The model proportion experiment where the filter size is fixed but the internal trained model size increases.
- `run_exp_with_model_prescores.py`: The distribution experiment, where 10 million queries from a specified query distribution (including adversarial) are performed on a learned filter after a model precomputes scores for all elements in the dataset.
- `run_exp_with_model.py`: The timing experiment, where the filters perform queries following some distribution but the model computes scores during query time.
- `plot_results.py`: plots the results of these experiments (and of the learned filter sweeps).

All of these experiments are attempted with both python learned filters (plbf, adabf) using all model variants (RandomForest, DecisionTree, LogisticRegression). When plotting the results, FPR experiments will generally only include the learned filters using the DecisionTree. `parameter_search.py` helps choose the parameters of each model type.

We also include other utility functions in the `utils` folder, such as those used to obtain the sizes of AdaptiveQF filters across experiments or those used to graph score distributions of models across different datasets.

For example usages, see `../run_queries.sh` and `../graph_results.sh`.
