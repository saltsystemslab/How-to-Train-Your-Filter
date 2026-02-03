"""
Runs the distribution test, training a model on a dataset, having the new model precompute scores for all keys,
then testing the resulting learned filter on the given query distribution. Can also additionally run an adversarial test
on the same given query set.
"""
from src.filters.FastPLBF_M_dist import FastPLBF_M
from src.filters.ada_bf_index_query import Find_Optimal_Parameters
import argparse
import pandas as pd
from updated_classifiers import (read_model, create_model, obtain_raw_and_vectorized_keys, 
                                 EMBER_DATASET, URL_DATASET, SHALLA_DATASET, CAIDA_DATASET, POS_INDICATOR, CONFIG, SUPPORTED_MODELS)
import os
import random
import time
from sklearn.model_selection import train_test_split
from filelock import FileLock


FILTERS = ["plbf", "adabf"]
RESULTS_PATH = "../results/learned/overall_results_with_model_scores.csv"
ADVERSARIAL_PATH = "../results/learned/overall_advers_with_model_scores.csv"
RESULTS_COLUMNS = ["dataset", "model_type", "filter", "bytes", "query_dist", "num_queries", 
                            "model_accuracy", "fpr", "amort_query"]
ADVERS_COLUMNS = ["dataset", "model_type", "filter", "bytes", "num_queries", "freq", "fpr"]

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

for dataset in DATASETS:
    for i in range(num_trials):
        if dataset not in [EMBER_DATASET, URL_DATASET, SHALLA_DATASET, CAIDA_DATASET]:
            print(f"{dataset} not implemented yet.")
            continue

        rand_seed = random.randint(0, 4294967295)
        
        print(f"starting evaluation on {dataset} ==============================")
        print("obtaining keys from dataset...")
        keys, vectorized_keys, labels = obtain_raw_and_vectorized_keys(dataset)

        for model_type in ["logistic_regression"]: # for testing
        # for model_type in SUPPORTED_MODELS:
            print(f"using model type: {model_type}...")
            print("obtaining trained model")
            if new_model:
                clf, model_size, construct_time, train_time, accuracy = create_model(keys, vectorized_keys, labels, dataset, model_type=model_type, sample_random=rand_seed, train_random=rand_seed)
            else:
                clf, model_size, construct_time, train_time, accuracy = read_model(dataset, model_type=model_type)

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

            max_size_with_fp = -1
            for current_byte_size in M:
                print(f"Starting tests for filters of size {current_byte_size} bytes -------------------")
                print(f"Model size: {model_size} bytes, Target filter size: {current_byte_size} bytes")
                remaining_bit_size = (current_byte_size - model_size) * 8

                # distinguish between the positive and negative keys
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
                        learned_filter = FastPLBF_M(pos_keys, pos_scores, train_neg_scores, remaining_bit_size, N, k)
                    elif filter == "adabf":
                        print("Creating adabf")
                        learned_filter, thresholds_opt, k_max_opt = Find_Optimal_Parameters(2.1, 2.6, 8, 11, remaining_bit_size, 
                                                                                    pos_keys, train_neg_keys, pos_scores, train_neg_scores)
                        
                    # now test on the actual query set
                    print(f"starting trial {i}:")
                    num_q_to_learn_adv = len(query_keys) / 2
                    fp_cnt = 0
                    first_half_fp = list()
                    first_half_fp_cnt = 0
                    first_half_neg_cnt = 0
                    query_count = 0

                    query_time_ns = 0
                    for key, score, label in zip(query_keys, query_scores, query_labels):
                        start = time.perf_counter_ns()
                        found = learned_filter.contains(key, score)
                        end = time.perf_counter_ns()
                        query_time_ns += end - start
                        if label == CONFIG[dataset][POS_INDICATOR]:
                            assert(found)
                        else:
                            if query_count < num_q_to_learn_adv:
                                first_half_neg_cnt += 1
                            if found:
                                fp_cnt += 1
                                if query_count < num_q_to_learn_adv:
                                    first_half_fp.append((key, score))
                                    first_half_fp_cnt += 1        
                        query_count += 1
                    neg_queries = query_keys[query_labels != CONFIG[dataset][POS_INDICATOR]] 
                    fpr = fp_cnt / (fp_cnt + len(neg_queries))
                    print(f"False Positive Rate: {fpr} [{fp_cnt} / ({fp_cnt} + {len(neg_queries)})]")

                    # now, we save the results to a file
                    if QUERY_PATH == None:
                        dist = "onepass"
                    elif "unif" in QUERY_PATH:
                        dist = "unif"
                    elif "zipf" in QUERY_PATH:
                        dist = "zipf"
                    else:
                        dist = "other"
                    
                    current_result = {"dataset": dataset, "model_type": model_type, "filter": filter, "bytes": current_byte_size, "query_dist": dist, "num_queries": len(query_keys), 
                                    "model_accuracy": accuracy, "fpr": fpr, "amort_query": query_time_ns / len(query_keys)}
                    write_results_safely(RESULTS_PATH, RESULTS_COLUMNS, current_result)

                    # now we have finished with the main test. We now start the optional adversarial test
                    first_half_labels = query_labels[:int(len(query_labels)/2)]
                    if include_adversarial and first_half_fp_cnt != 0:
                        advers_freqs = [0.02, 0.04, 0.06, 0.08, 0.1]
                        for freq in advers_freqs:
                            print(f"Starting test with {freq} adversarial queries")
                            # first, calculate how many adversarial queries we need to perform
                            num_queries_overall = len(query_keys)
                            num_queries_remaining = int(num_queries_overall / 2)
                            num_advers = int(num_queries_overall * freq)
                            space_between_advers = int(num_queries_remaining / num_advers)

                            # do the remaining queries
                            second_half_fp_cnt = 0
                            remaining_keys = query_keys[num_queries_remaining:]
                            remaining_scores = query_scores[num_queries_remaining:]
                            remaining_labels = query_labels[num_queries_remaining:]

                            # for the adversarial workload, loop through the false positives
                            first_half_fp = list(first_half_fp)
                            current_index = 0
                            current_query = 0
                            negative_count = 0
                            advers_count = 0
                            for key, score, label in zip(remaining_keys, remaining_scores, remaining_labels):
                                current_query += 1
                                was_pos = label == CONFIG[dataset][POS_INDICATOR]
                                if current_query == space_between_advers:
                                    # do an adversarial query instead
                                    key, score = first_half_fp[current_index]
                                    advers_count += 1
                                    was_pos = False
                                    current_query = 0
                                    if current_index == len(first_half_fp) - 1:
                                        current_index = 0
                                    else:
                                        current_index += 1
                                found = learned_filter.contains(key, score)
                                if was_pos:
                                    assert(found)
                                elif not was_pos:
                                    negative_count += 1
                                    if found:
                                        second_half_fp_cnt += 1
                            neg_rows_first = (first_half_labels != CONFIG[dataset][POS_INDICATOR])
                            num_neg = len(first_half_labels[neg_rows_first]) + negative_count
                            total_fp = first_half_fp_cnt + second_half_fp_cnt
                            adversarial_fpr = total_fp / (total_fp + num_neg) 
                            print(f"negative queries: {len(first_half_labels[neg_rows_first])} + {negative_count}")
                            print(f"False Positive with {freq} Adversarial: {adversarial_fpr} [ {total_fp} / ({total_fp} + {num_neg})]")
                            print(f"First half fp: {first_half_fp_cnt}, Second half fp: {second_half_fp_cnt}")
                            # now save the results to a file
                            adversarial_result = {"dataset": dataset, "model_type": model_type, "filter": filter, "bytes": current_byte_size, 
                                                "num_queries": len(query_keys), "freq": freq, "fpr": adversarial_fpr}
                            write_results_safely(ADVERSARIAL_PATH, ADVERS_COLUMNS, adversarial_result)
print("finished")