# How to Train Your Filter: When to Adapt or Learn [Experiments & Analysis]

Learned filters and adaptive filters both represent promising advances over traditional filters for approximate membership testing, yet they have not been directly compared against each other. Learned filters achieve competitive space-accuracy trade-offs by leveraging machine learning models, while adaptive filters provide theoretically bounded false positive rates (regardless of query distribution) by requiring auxiliary reverse maps. This project presents the first comprehensive comparative evaluation of these two filter paradigms using real-world datasets and diverse query workloads including one-pass, uniform, Zipfian, and adversarial distributions.

This repository contains a compilation of implementations for **[Ada-BF](https://github.com/DAIZHENWEI/Ada-BF)**, **[FastPLBF](https://github.com/atsukisato/FastPLBF)**, and **[AdaptiveQF](https://github.com/splatlab/adaptiveqf)** filters. The `adaptiveqf` folder originates from the source **AdaptiveQF** code. The `learned` folder originates from the source **FastPLBF** code, with the **Ada-BF** implementation moved inside. Minor changes to the **AdaptiveQF** were made to fix some bugs, while the **FastPLBF** and **Ada-BF** feature new versions which have a model compute key scores on-the-fly.

## Requirements
To prepare the **AdaptiveQF**, you first need to have [SplinterDB](https://splinterdb.org/) set up in the `adaptiveqf/external` folder. Then, run `make` to create the necessary executables.

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

2. Once the queries are finished, use `run_queries.sh` in the root directory to run all experiments with the parameters described in the paper. Note that learned filter experiments will run in background processes.

3. When all experiments are finished running, from the root directory run `graph_results.sh` to plot the results.

More information about filter implementations and utility scripts are found in the `adaptiveqf` and `learned` folders.

For any questions, please contact the authors of the paper.