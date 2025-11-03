"""
Runs the changing model test, increasing the size of the internal model
while fixing the overall filter size.
"""
from src.filters.FastPLBF_M_dist import FastPLBF_M
from src.filters.ada_bf_index_query import Find_Optimal_Parameters
import argparse
import pandas as pd
from updated_classifiers import (create_model, obtain_raw_and_vectorized_keys, 
                                 EMBER_DATASET, URL_DATASET, SHALLA_DATASET, CAIDA_DATASET, POS_INDICATOR, CONFIG)
import os
import random
from sklearn.model_selection import train_test_split
from filelock import FileLock


FILTERS = ["plbf", "adabf"]
RESULTS_PATH = "../results/learned/changing_model_size.csv"
RESULTS_COLUMNS = ["dataset", "filter", "bytes", "model_bytes", "num_queries", "model_accuracy", "fpr"]

def write_results_safely(file_path, columns, row):
    lock = FileLock(file_path + ".lock")
    with lock:
        if os.path.exists(file_path):
            results_df = pd.read_csv(file_path)
        else:
            results_df = pd.DataFrame(columns=columns)
        results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
        results_df.to_csv(file_path, index=False)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', action="store", dest="dataset", type=str, required=True,
                    help="the list of datasets to run")
parser.add_argument('--query_path', action="store", dest="query_path", type=str, required=False,
                        help="path of the query indices")
parser.add_argument('--filters', nargs="+", action="store", dest="filters", type=str, required=True,
                    help="filters: which learned filters to use")
parser.add_argument('--trials', action="store", dest="trials", type=int, required=False,
                    help="k: the number of trials to run")

results = parser.parse_args()

dataset = results.dataset
QUERY_PATH = results.query_path
N = 2000
k = 5
num_trials = 3 if results.trials is None else results.trials
filters = results.filters

config = {EMBER_DATASET: {"leaves": [5, 10, 15, 20, 25, 30], "total_size": 500000},
          URL_DATASET: {"leaves": [2, 4, 6, 8, 10], "total_size": 70000},
          SHALLA_DATASET: {"leaves": [20, 40, 60, 80, 100, 120, 140], "total_size": 4000000},
          CAIDA_DATASET: {"leaves": [20, 40, 60, 80, 100], "total_size": 3000000}}
    
if dataset not in [EMBER_DATASET, URL_DATASET, SHALLA_DATASET, CAIDA_DATASET]:
    raise Exception(f"{dataset} not implemented yet.")
    
print(f"starting evaluation on {dataset} ==============================")
print("obtaining keys from dataset...")
keys, vectorized_keys, labels = obtain_raw_and_vectorized_keys(dataset)

for filter in filters:
    print(f"starting with {filter}")
    for i in range(num_trials):
        print(f"starting trial {i}:")
        for current_leaves in config[dataset]["leaves"]:
            rand_seed = random.randint(0, 4294967295)
            
            print(f"obtaining trained model with {current_leaves} leaves")
            clf, model_size, construct_time, train_time, accuracy = create_model(keys, vectorized_keys, labels, dataset, sample_random=rand_seed, train_random=rand_seed, max_leaves=current_leaves)

            model_size, construct_time, train_time, accuracy = int(model_size), float(construct_time), float(train_time), float(accuracy)
            scores = clf.predict_proba(vectorized_keys)[:, 1]

            print("setting up queries...")
            query_keys = []
            query_vec = []
            query_labels = []
            if results.query_path is not None:
                # Find all the rows in the data that correspond to the queries
                query_indices = pd.read_csv(QUERY_PATH)["index"]
                query_indices = query_indices[query_indices < len(keys)]
                query_keys = keys[query_indices]
                query_scores = scores[query_indices]
                query_labels = labels[query_indices]
            else:
                # If there is no query path defined, we assume we perform the one-pass test
                query_keys = keys
                query_scores = scores
                query_labels = labels
            print(f"ready to begin {len(query_keys)} queries...")

            remaining_bit_size = (config[dataset]["total_size"] - model_size) * 8

            # distinguish between the positive and negative keys
            pos_keys = keys[labels == CONFIG[dataset][POS_INDICATOR]]
            pos_scores = scores[labels == CONFIG[dataset][POS_INDICATOR]]
            neg_keys = keys[labels != CONFIG[dataset][POS_INDICATOR]]
            neg_scores = scores[labels != CONFIG[dataset][POS_INDICATOR]]

            # choose a subset of the negative keys for the filter to train on
            train_neg_keys, test_neg_keys, train_neg_scores, test_neg_scores = train_test_split(neg_keys, neg_scores, train_size=0.3)

            if filter not in FILTERS:
                print(f"{filter} is not implemented...")
                continue
            if filter == "plbf":
                print("Creating plbf")
                learned_filter = FastPLBF_M(pos_keys, pos_scores, train_neg_scores, float(remaining_bit_size), N, k)
            elif filter == "adabf":
                print("Creating adabf")
                learned_filter, thresholds_opt, k_max_opt = Find_Optimal_Parameters(2.1, 2.6, 8, 11, remaining_bit_size, 
                                                                            pos_keys, train_neg_keys, pos_scores, train_neg_scores)
                
            # now test on the actual query set
            fp_cnt = 0
            query_count = 0

            for key, score, label in zip(query_keys, query_scores, query_labels):
                found = learned_filter.contains(key, score)
                if label == CONFIG[dataset][POS_INDICATOR]:
                    assert(found)
                else:
                    if found:
                        fp_cnt += 1
            neg_queries = query_keys[query_labels != CONFIG[dataset][POS_INDICATOR]] 
            fpr = fp_cnt / (fp_cnt + len(neg_queries))
            print(f"False Positive Rate: {fpr} [{fp_cnt} / ({fp_cnt} + {len(neg_queries)})]")
            
            current_result = {"dataset": dataset, "filter": filter, "bytes": config[dataset]["total_size"], "model_bytes": model_size, 
                                "num_queries": len(query_keys), "model_accuracy": accuracy, "fpr": fpr}
            write_results_safely(RESULTS_PATH, RESULTS_COLUMNS, current_result)
print("finished")