# How to Train Your Filter - When to Learn, Stack, or Adapt: [Experiments & Analysis]

Learned, stacked, and adaptive filters all represent promising advances over traditional filters for approximate membership testing, yet they have not been directly compared against each other. Learned filters achieve competitive space-accuracy trade-offs by leveraging machine learning models, stacked filters use workload knowledge to build cascading filters including negative values, while adaptive filters provide theoretically bounded false positive rates (regardless of query distribution) by requiring auxiliary reverse maps. This project presents the first comprehensive comparative evaluation of these three filter paradigms using real-world datasets and diverse query workloads including one-pass, uniform, Zipfian, and adversarial distributions.

This repository contains a compilation of implementations for the filters we compare: the **[AdaptiveQF](https://github.com/splatlab/adaptiveqf)**, the **[Stacked](https://github.com/kylebd99/StackedFilters)** filter, and the C++ **PLBF** and **CLBF** learned filters from **[CascadedLBF](https://github.com/atsukisato/CascadedLBF)**, along with the Bloom filter baseline that CascadedLBF uses. CascadedLBF also has its own experiments for these filters, which we don't use: our experiments call its filter implementations directly. The `adaptiveqf` folder originates from the source **AdaptiveQF** code, with the source **Stacked Filter** code moved within it to the `external/StackedFilters` folder. Its experiment harnesses run all five filters through a common interface, using the wrappers in `adaptiveqf/external` for the **Stacked**, Bloom and **CascadedLBF** filters. Minor changes to the **AdaptiveQF** were made to fix some bugs. The `learned` folder originates from the source **[FastPLBF](https://github.com/atsukisato/FastPLBF)** code, with the **[Ada-BF](https://github.com/DAIZHENWEI/Ada-BF)** implementation moved inside. It holds the plotting and utility scripts, as well as python versions of the **FastPLBF** (a separate implementation of the PLBF) and **Ada-BF** filters (which have a model compute key scores on-the-fly) used in optional supplementary experiments.

## Requirements
To prepare the **AdaptiveQF**, you first need to have [SplinterDB](https://splinterdb.org/) set up in the `adaptiveqf/external` folder.

The learned filters run from the `adaptiveqf` experiments (`plbf`, `clbf`) and the Bloom filter baseline use **[CascadedLBF](https://github.com/atsukisato/CascadedLBF)**, which is not included in this repository. Clone it into the root directory at the commit we used, along with its XGBoost submodule:
```
git clone https://github.com/atsukisato/CascadedLBF.git
cd CascadedLBF
git checkout 5927949a3df1789a6dead4adc6e33a844f1b94c9
git submodule update --init thirdparty/xgboost
```
Then, build XGBoost and run `make` in the `adaptiveqf` folder to create the necessary executables; `setup_server.sh` (see below) does both. `learned/utils/train_cascaded_models.sh` builds CascadedLBF's model trainer the first time it runs.

The datasets **can be directly found as compressed files in the `data/compressed/`** folder. When decompressed, they should be placed in the `data/` folder as:
- `url.tar.gz` => `malicious_url_scores.csv`
- `ember.tar.gz` => `combined_ember_metadata.csv`
- `caida.tar.gz` => `caida.csv`
- `shalla.tar.gz` => `shalla_combined.csv`

<details>
<summary>Manually downloading and processing the datasets can also be done by using the instructions in this section.</summary>

- [Malicious URLs](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset): This dataset just needs to be downloaded and stored in `data/` as `malicious_url_scores.csv`.
- [Ember](https://github.com/elastic/ember): First, the Ember project needs to be cloned into `learned/ember_import`. Then, unlabelled rows from the dataset should be removed using `learned/utils/eliminate_unlabelled_ember.py`. The cleaned dataset should be stored in `data/` as `combined_ember_metadata.csv`.
- [Shalla](data/compressed/): Since the dataset was discontinued, the link directs to a folder where the original data `shalla_original.tar.gz` can be found (`shalla.tar.gz` gives the direct data we used without needing any additional processing). After downloading the Shalla dataset, the [Cisco Top 1M Domain](https://s3-us-west-1.amazonaws.com/umbrella-static/index.html) dataset should be downloaded. Then, `learned/utils/process_shalla.py` will combine the two into a combined dataset of malicious and popular websites. The result should be stored in `data/` as `shalla_combined.csv`.
- [Caida](https://www.caida.org/catalog/datasets/passive_dataset/): The `learned/utils/caida_vectorizer.py` creates a simplified version of the Caida dataset. The result should be stored in `data/` as `caida.csv`.
</details>

## Running Experiments 

Once all the datasets are downloaded and stored in the correct location, we provide bash scripts to replicate the experiments we performed in the paper.

1. From the root directory, run `generate_queries.sh` to generate query index files.

2. To run the learned filters from the `adaptiveqf` experiments (`plbf`, `clbf`), run `prepare_learned_models.sh` from the root directory once. It exports the features used by the python learned filters, the XGBoost training and calibration sets (split as in the CLBF paper: for url, 80% of the negatives for training and another 10% for calibration, with all positives in both; for the other datasets, as for ember, 10% of the negatives for training and another 10% for calibration, with 10% of the positives for training and all of them for calibration), and trains the XGBoost models ahead of time, saving them under `data/cascaded/`; experiments then load the saved models. This needs the `CascadedLBF` folder.

3. *(Optional reproducibility helpers, not needed to produce the results.)* To check a machine before running the experiments, run `setup_server.sh` from the root directory: it builds XGBoost and the harnesses and checks the tools, datasets, query files and models. Then run `test_small.sh`, which runs a small version of all experiments (url only, two sizes, 1 million queries, about an hour) and checks their results, leaving `results/` as it was.

4. Once the query files are generated, use `run_queries.sh` in the root directory to run all experiments with the parameters described in the paper, for all five filters (`aqf`, `stacked`, `bloom`, `plbf`, `clbf`). The experiments run in two phases:
   - the uniform workload at the size construction and query times are reported for (`TIMING_R`, the adaptiveqf's r = 5 size), timed, one experiment at a time with nothing else running (the adaptiveqf runs first on each dataset, since the other filters are built to its sizes);
   - the uniform workload at the other sizes and the Zipfian, one-pass, adversarial and dynamic (swapping and shifting) workloads, untimed and in parallel (`NUM_PARALLEL`), with the harnesses that record where the learned filters' false positives came from.

   After the timed phase, the training of the timed learned filters' XGBoost models is timed too (`learned/utils/time_model_training.sh`, written to `results/model_train_times.csv`), since the construction times include it.

   Every experiment runs single-threaded (as in the CLBF paper) and runs 3 trials; the dynamic workloads give every filter and trial the same queries and churns.

   The optional python learned filter experiments (the **FastPLBF** and **Ada-BF** versions in `learned/`, run as `plbf` and `adabf` and separate from the CascadedLBF `plbf` above) are only run with `RUN_PYTHON_LEARNED=1`, in background processes after both phases.

5. To run the learned filter sweeps (the model's share of the space and the proportion of negative keys in the training set), run `run_sweeps.sh` from the root directory after `run_queries.sh`. Each sweep runs in its own copy of the experiment layout under `sweeps/`, so the main results are untouched, and its figures are saved to `results/figures/`.

6. When all experiments are finished running, from the root directory run `graph_results.sh` to plot the results: the comparison of the five filters and the radar chart summarizing every experiment (for the **AdaptiveQF**, **Stacked**, **PLBF** and **CLBF** filters) are saved to `results/figures/generic/`, and the supplementary tables S1 and S2 to `results/tables/`. To save the radar chart as two figures instead (the overview with a legend, and one radar per filter), run `python3 plot_radar.py --split` from the `learned` folder. If the python learned filter experiments were run, their figures are plotted too.

More information about filter implementations and utility scripts is found in the `adaptiveqf` and `learned` folders.

For any questions, please contact the authors of the paper.