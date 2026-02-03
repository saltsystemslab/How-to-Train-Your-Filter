"""
Runs the training-proportion test, increasing the number of negative keys included in the training
set to determine its effect on the filter's overall accuracy.
"""
from updated_classifiers import (create_model, obtain_raw_and_vectorized_keys, POS_INDICATOR, CONFIG)
from sklearn.metrics import accuracy_score
from src.filters.FastPLBF_M_dist import FastPLBF_M
from src.filters.ada_bf_index_query import Find_Optimal_Parameters
import argparse
import os
from filelock import FileLock
import pandas as pd
from sklearn.model_selection import train_test_split
FILTERS = ["plbf", "adabf"]
RESULTS_PATH = "../results/learned/degrad_results_with_model_scores.csv"
RESULTS_COLUMNS = ["dataset", "filter", "model_type", "bytes", "query_dist", "num_queries", 
                                         "train_set_size", "fpr"]


def write_results_safely(file_path, columns, row):
    lock = FileLock(file_path + ".lock")
    with lock:
        if os.path.exists(file_path):
            results_df = pd.read_csv(file_path)
        else:
            results_df = pd.DataFrame(columns=columns)
        results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
        results_df.to_csv(file_path, index=False)

models = ["random_forest", "decision_tree", "logistic_regression"]

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', nargs="+", action="store", dest="datasets", type=str, required=True,
                    help="the list of datasets to run")
parser.add_argument('--query_path', action="store", dest="query_path", type=str, required=False,
                        help="path of the query indices")
parser.add_argument('--N', action="store", dest="N", type=int, required=True,
                    help="N: the number of segments")
parser.add_argument('--k', action="store", dest="k", type=int, required=True,
                    help="k: the number of regions")
parser.add_argument('--M', nargs="+", action="store", dest="M", type=float, required=True,
                    help="M: list of target memory usages for backup Bloom filters (in bytes)")
parser.add_argument('--filters', nargs="+", action="store", dest="filters", type=str, required=True,
                    help="filters: which learned filters to use")
parser.add_argument('--trials', action="store", dest="trials", type=int, required=False,
                    help="k: the number of trials to run")
parser.add_argument('--adv', action='store_true', dest='adv',
                        help='adv: whether or not to run adversarial tests')
parser.add_argument('--new_model', action='store_true', dest='new_model',
                        help='adv: whether or not to create a new model')

results = parser.parse_args()

DATASETS = results.datasets
QUERY_PATH = results.query_path
N = results.N
k = results.k
M = results.M
new_model = results.new_model
num_trials = 3 if results.trials == "none" else results.trials
include_adversarial = results.adv
filters = results.filters
train_sizes = [0.2, 0.4, 0.6, 0.8]
for dataset in DATASETS:
    print(f"training on {dataset}")
    keys, vectorized_keys, labels = obtain_raw_and_vectorized_keys(dataset, create_data=False)
    for trial in range(num_trials):
        print(f"starting trial {trial}...")
        for train_size in train_sizes:
            print(f"evaluating train set size {train_size}")
            for model in models:
                clf, size_in_bytes, construct_time, train_time, accuracy = create_model(keys, vectorized_keys, labels,
                                                                            dataset, save_model=False, train_size=train_size, train_random=None, sample_random=None, model_type=model)
                print("model size: ", size_in_bytes)
                print("construct time: ", construct_time)
                print("train time: ", train_time)
                y_pred = clf.predict(vectorized_keys)
                accuracy = accuracy_score(labels, y_pred)
                print(f"overall accuracy: {accuracy:.4f}")
                scores = clf.predict_proba(vectorized_keys)[:, 1]
                assert len(keys) == len(scores) == len(labels)
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

                max_size_with_fp = -1
                for current_byte_size in M:
                    print(f"Starting tests for filters of size {current_byte_size} bytes -------------------")
                    remaining_bit_size = (current_byte_size - size_in_bytes) * 8

                    # distinguish between the positive and negative keys
                    print("labels: ", labels)
                    pos_keys = keys[labels == CONFIG[dataset][POS_INDICATOR]]
                    pos_scores = scores[labels == CONFIG[dataset][POS_INDICATOR]]
                    neg_keys = keys[labels != CONFIG[dataset][POS_INDICATOR]]
                    neg_scores = scores[labels != CONFIG[dataset][POS_INDICATOR]]

                    # choose a subset of the negative keys for the filter to train on
                    train_neg_keys, test_neg_keys, train_neg_scores, test_neg_scores = train_test_split(neg_keys, neg_scores, train_size=0.3)

                    for filter in filters:
                        if filter not in FILTERS:
                            print(f"{filter} is not implemented...")
                            continue
                        if filter == "plbf":
                            print("Creating plbf")
                            print("pos keys length: ", len(pos_keys))
                            learned_filter = FastPLBF_M(pos_keys, pos_scores, train_neg_scores, remaining_bit_size, N, k)
                        elif filter == "adabf":
                            print("Creating adabf")
                            learned_filter, thresholds_opt, k_max_opt = Find_Optimal_Parameters(2.1, 2.6, 8, 11, remaining_bit_size, 
                                                                                        pos_keys, train_neg_keys, pos_scores, train_neg_scores)
                        fp_cnt = 0
                        # now test on the actual query set
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

                        # now, we save the results to a file
                        if QUERY_PATH == "none":
                            dist = "none"
                        elif "unif" in QUERY_PATH:
                            dist = "unif"
                        elif "zipf" in QUERY_PATH:
                            dist = "zipf"
                        else:
                            dist = "other"
                        
                        current_result = {"dataset": dataset, "filter": filter, "model_type": model, "bytes": current_byte_size, "query_dist": dist, "num_queries": len(query_keys), 
                                        "train_set_size": train_size, "fpr": fpr}
                        write_results_safely(RESULTS_PATH, RESULTS_COLUMNS, current_result)
print("finished")