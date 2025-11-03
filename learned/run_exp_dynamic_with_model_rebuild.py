"""
Runs the dynamic test, occasionally replacing a proportion of the contents
of the filter, but allowing the filter to retrain a model after each churn.
"""
from src.filters.FastPLBF_M_dist import FastPLBF_M
from src.filters.ada_bf_index_query import Find_Optimal_Parameters
import argparse
import pandas as pd
from updated_classifiers import (create_model, obtain_raw_and_vectorized_keys, 
                                 EMBER_DATASET, URL_DATASET, SHALLA_DATASET, CAIDA_DATASET, POS_INDICATOR, CONFIG)
import os
import time
import random
import numpy as np
from sklearn.model_selection import train_test_split
from filelock import FileLock


FILTERS = ["plbf", "adabf"]
RESULTS_PATH = "../results/learned/dynamic_results_with_model_rebuild.csv"
RESULTS_COLUMNS = ["dataset", "filter", "bytes", "query_dist", "num_queries", 
                            "curr_query", "fpr"]

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

results = parser.parse_args()

DATASETS = results.datasets
QUERY_PATH = results.query_path
N = results.N
k = results.k
M = results.M
num_trials = 3 if results.trials == "none" else results.trials
filters = results.filters

for dataset in DATASETS:
    if dataset not in [EMBER_DATASET, URL_DATASET, SHALLA_DATASET, CAIDA_DATASET]:
        print(f"{dataset} not implemented yet.")
        continue

    rand_seed = random.randint(0, 4294967295)

    print(f"starting evaluation on {dataset} ==============================")
    print("obtaining keys from dataset...")
    keys, vectorized_keys, labels = obtain_raw_and_vectorized_keys(dataset)

    print("obtaining trained model")
    clf, model_size, construct_time, train_time, accuracy = create_model(keys, vectorized_keys, labels, dataset, sample_random=rand_seed, train_random=rand_seed)

    model_size, construct_time, train_time, accuracy = int(model_size), float(construct_time), float(train_time), float(accuracy)
    scores = clf.predict_proba(vectorized_keys)[:, 1]

    # set up the queries
    query_indices = pd.read_csv(QUERY_PATH)["index"]
    query_indices = query_indices[query_indices < len(keys)]
    query_keys = keys[query_indices]
    query_scores = scores[query_indices]
    query_labels = labels[query_indices]

    # distinguish the positive and negative keys/scores
    pos_keys = keys[labels == CONFIG[dataset][POS_INDICATOR]]
    pos_scores = scores[labels == CONFIG[dataset][POS_INDICATOR]]
    neg_keys = keys[labels != CONFIG[dataset][POS_INDICATOR]]
    neg_scores = scores[labels != CONFIG[dataset][POS_INDICATOR]]

    print(f"negative keys: {len(neg_keys)}, positive keys: {len(pos_keys)}")
    print(f"ready to begin {len(query_keys)} queries...")

    max_size_with_fp = -1
    for current_byte_size in M:
        print(f"Starting tests for filters of size {current_byte_size} bytes -------------------")
        remaining_bit_size = (current_byte_size - model_size) * 8
        print(f"backup filter size: {remaining_bit_size}")

        replacement_size = int(0.2 * len(pos_keys))
        if len(neg_keys) < len(pos_keys):
            raise Exception("not enough negative keys to do the dynamic test")
            
        space_between_replace = int(0.1 * len(query_keys))
        space_between_inst = int(0.01 * len(query_keys))

        # choose a subset of the negative keys for the filter to train on
        train_neg_keys, test_neg_keys, train_neg_scores, test_neg_scores = train_test_split(neg_keys, neg_scores, train_size=0.3)

        pos_key_set = set(pos_keys)
        for filter in filters:
            if filter not in FILTERS:
                print(f"{filter} is not implemented...")
                continue
            if filter == "plbf":
                print("Creating plbf")
                learned_filter = FastPLBF_M(pos_keys, pos_scores, train_neg_scores, remaining_bit_size, N, k)
            elif filter == "adabf":
                print("Creating adabf")
                learned_filter, thresholds_opt, k_max_opt = Find_Optimal_Parameters(2.1, 2.6, 8, 11, remaining_bit_size, 
                                                                            pos_keys, train_neg_keys, pos_scores, train_neg_scores)
                
            # now test on the actual query set
            for i in range(num_trials):
                print(f"starting trial {i}:")
                num_q_to_learn_adv = int(len(query_keys) / 2)
                fp_cnt = 0
                first_half_fp = list()
                first_half_fp_cnt = 0
                first_half_neg_cnt = 0
                query_count = 0

                replace_count = 0
                inst_count = 0
                num_replacements = 0
                start_query = time.time()
                for i, key in enumerate(query_keys):
                    query_count += 1
                    replace_count += 1
                    inst_count += 1
                    if replace_count == space_between_replace:
                        # do a churn of 20% of the items
                        print(f"before churn, negative keys: {len(neg_keys)}, positive keys: {len(pos_keys)}")
                        print("Duplicates in keys:", len(keys) - len(np.unique(keys)))
                        print("churning!")

                        replace_count = 0
                        start_replace = (num_replacements % 5) * replacement_size
                        rows_to_swap = np.arange(start_replace, min(start_replace + replacement_size, len(pos_keys)))

                        labels = np.array(labels)
                        keys = np.array(keys)

                        pos_indexes = np.where(labels == CONFIG[dataset][POS_INDICATOR])[0]
                        neg_indexes = np.where(labels != CONFIG[dataset][POS_INDICATOR])[0]

                        rows_to_swap = np.arange(start_replace, min(start_replace + replacement_size, len(pos_indexes)))

                        # pick the indices to swap directly
                        swap_pos = pos_indexes[rows_to_swap]
                        swap_neg = neg_indexes[rows_to_swap]

                        # swap their labels directly
                        labels[swap_pos], labels[swap_neg] = 0, 1

                        # recompute pos/neg keys
                        pos_keys = keys[labels == CONFIG[dataset][POS_INDICATOR]]
                        neg_keys = keys[labels != CONFIG[dataset][POS_INDICATOR]]

                        print(f"after swap/reindex, negative keys: {len(neg_keys)}, positive keys: {len(pos_keys)}")

                        # train a new model on the updated keys
                        print("retraining model...")
                        clf, model_size, construct_time, train_time, accuracy = create_model(keys, vectorized_keys, labels, dataset, sample_random=rand_seed, train_random=rand_seed)
                        
                        # redo the scores
                        scores = clf.predict_proba(vectorized_keys)[:, 1]

                        # at this point, we have row-aligned keys, labels, and scores
                        # now we just need to distinguish the positive and negative keys/scores
                        pos_scores = scores[labels == CONFIG[dataset][POS_INDICATOR]]
                        neg_scores = scores[labels != CONFIG[dataset][POS_INDICATOR]]

                        # the query keys are also aligned to the scores, so we need to update those as well
                        query_scores = scores[query_indices]
                        query_keys = keys[query_indices]
                        query_labels = labels[query_indices]

                        print(f"after relabel, negative keys: {len(neg_scores)}, positive keys: {len(pos_scores)}")

                        # choose a subset of the negative keys for the filter to train on
                        train_neg_keys, test_neg_keys, train_neg_scores, test_neg_scores = train_test_split(neg_keys, neg_scores, train_size=0.3)

                        # with the new model, remake the filter
                        if filter == "plbf":
                            print("Creating plbf")
                            learned_filter = FastPLBF_M(pos_keys, pos_scores, train_neg_scores, remaining_bit_size, N, k)
                        elif filter == "adabf":
                            print("Creating adabf")
                            learned_filter, thresholds_opt, k_max_opt = Find_Optimal_Parameters(2.1, 2.6, 8, 11, remaining_bit_size, 
                                                                                        pos_keys, train_neg_keys, pos_scores, train_neg_scores)
                        print("successfully completed a churn")
                        pos_key_set = set(pos_keys)
                        num_replacements += 1
                    if inst_count == space_between_inst:
                        # obtain the instantaneous false positive rate
                        print(f"trying to obtain inst fpr")
                        curr_inst_count = 0
                        inst_fp = 0
                        
                        start = time.time()
                        for inst_key, inst_score, inst_label in zip(query_keys, query_scores, query_labels):
                            curr_inst_count += 1
                            found = learned_filter.contains(inst_key, inst_score)
                            if inst_label == CONFIG[dataset][POS_INDICATOR]:
                                assert(found)
                            else:
                                if found:
                                    inst_fp += 1
                            
                            if curr_inst_count % int(len(query_keys) / 2) == 0:
                                end = time.time()
                                print(f"{curr_inst_count / (len(query_keys)) * 100}% finished with inst after {end - start} seconds")
                                start = time.time()
                        # get the instantaneous fpr
                        inst_count = 0
                        neg_queries = query_keys[query_labels != CONFIG[dataset][POS_INDICATOR]] 
                        fpr = inst_fp / (inst_fp + len(neg_queries))
                        # now, we save the results to a file
                        if QUERY_PATH == "none":
                            dist = "none"
                        elif "unif" in QUERY_PATH:
                            dist = "unif"
                        elif "zipf" in QUERY_PATH:
                            dist = "zipf"
                        else:
                            dist = "other"
                        print(f"False Positive Rate: {fpr} [{inst_fp} / ({inst_fp} + {len(neg_queries)})]")
                        current_result = {"dataset": dataset, "filter": filter, "bytes": current_byte_size, 
                                          "query_dist": dist, "num_queries": len(query_keys), 
                                            "curr_query": query_count, "fpr": fpr}
                        write_results_safely(RESULTS_PATH, RESULTS_COLUMNS, current_result)
                    score = query_scores[i]
                    found = learned_filter.contains(key, score)
                    if key in pos_key_set:
                        assert(found)
                    else:
                        if found:
                            fp_cnt += 1  
                    if query_count % int(len(query_keys) / 20) == 0:
                        end_query = time.time()
                        print(f"{query_count / (len(query_keys)) * 100}% finished overall after {end_query - start_query} seconds")
                        start_query = time.time()
print("finished")