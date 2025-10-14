"""
Runs the distribution test, training a model on a dataset, then testing the resulting learned filter on the given
query set. On each query, the model must compute the key's score on-the-fly.
Mainly used to time construction and query times for learned filters.
"""
from src.filters.FastPLBF_M_model import FastPLBF_M_Model
from src.filters.ada_bf_model import Find_Optimal_Parameters
import argparse
import pandas as pd
from updated_classifiers import (read_model, create_model, obtain_raw_and_vectorized_keys, 
                                 EMBER_DATASET, URL_DATASET, SHALLA_DATASET, CAIDA_DATASET, POS_INDICATOR, CONFIG)
import os
import numpy as np
import random
from sklearn.model_selection import train_test_split
from statistics import median
from filelock import FileLock

def write_results_safely(file_path, columns, row):
    lock = FileLock(file_path + ".lock")
    with lock:
        if os.path.exists(file_path):
            results_df = pd.read_csv(file_path)
        else:
            results_df = pd.DataFrame(columns=columns)
        results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
        results_df.to_csv(file_path, index=False)

RESULTS_PATH = "results/results_with_model.csv"
ADVERSARIAL_PATH = "results/advers_with_model.csv"
RESULTS_COLUMNS = ["dataset", "filter", "bytes", "query_dist", "num_queries", 
                    "construct_time", "train_time", "model_accuracy",
                    "initial_scores", "segment_division", "t_f_finding", "insert_scores", 
                    "bloom_init", "region_finding", "filter_inserts",
                    "fpr", "throughput", "med_pos_time", "med_neg_time",
                    "amort_score_time", "amort_region_time", "amort_back_filter_time"]
ADVERS_COLUMNS = ["dataset", "filter", "bytes", "num_queries", "freq",
                  "fpr", "throughput"]

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', nargs="+", action="store", dest="datasets", type=str, required=True,
                    help="the list of datasets to run")
parser.add_argument('--query_path', action="store", dest="query_path", type=str, required=False,
                        help="path of the query indices")
parser.add_argument('--N', action="store", dest="N", type=int, required=True,
                    help="N: the number of segments")
parser.add_argument('--k', action="store", dest="k", type=int, required=True,
                    help="k: the number of regions")
parser.add_argument('--bytes', nargs="+", action="store", dest="bytes", type=float, required=True,
                    help="M: list of target memory usages for backup Bloom filters (in bytes)")
parser.add_argument('--filters', nargs="+", action="store", dest="filters", type=str, required=True,
                    help="filters: which learned filters to use")
parser.add_argument('--trials', action="store", dest="trials", type=int, required=False,
                    help="k: the number of trials to run")
parser.add_argument('--new_model', action='store_true', dest='new_model',
                        help='adv: whether or not to create a new model')

results = parser.parse_args()

DATASETS = results.datasets
QUERY_PATH = results.query_path
current_N = results.N
current_k = results.k
target_byte_size = results.bytes
new_model = results.new_model
num_trials = 3 if results.trials == "none" else results.trials
filters = results.filters

# set up the csvs we are saving
results_df = None
adversarial_df = None
if os.path.exists(RESULTS_PATH):
    results_df = pd.read_csv(RESULTS_PATH)
else:
    results_df = pd.DataFrame(columns=RESULTS_COLUMNS)

if os.path.exists(ADVERSARIAL_PATH):
    adversarial_df = pd.read_csv(ADVERSARIAL_PATH)
else:
    adversarial_df = pd.DataFrame(columns=ADVERS_COLUMNS)


for i in range(num_trials):
    rand_seed = random.randint(0, 4294967295)
    # now run the test for each dataset we're testing on
    for dataset in DATASETS:
        if dataset not in [EMBER_DATASET, URL_DATASET, SHALLA_DATASET, CAIDA_DATASET]:
            print(f"{dataset} not implemented yet.")
            continue
        print(f"starting evaluation on {dataset} ==============================")
        print("obtaining keys from dataset...")
        keys, vectorized_keys, labels = obtain_raw_and_vectorized_keys(dataset)
        hashed_keys = np.array(list(range(len(keys))))

        assert(len(keys) == len(hashed_keys))

        print("obtaining trained model")
        if new_model:
            keys, vectorized_keys, labels = obtain_raw_and_vectorized_keys(dataset)
            clf, size_in_bytes, construct_time, train_time, accuracy = create_model(keys, vectorized_keys, labels, dataset, sample_random=rand_seed, train_random=rand_seed)
        else:
            clf, size_in_bytes, construct_time, train_time, accuracy = read_model(dataset)

        size_in_bytes, construct_time, train_time, accuracy = int(size_in_bytes), float(construct_time), float(train_time), float(accuracy)
        print("setting up queries...")
        query_keys = []
        query_hashed_keys = []
        query_vec = []
        query_labels = []
        if results.query_path is not None:
            # Find all the rows in the data that correspond to the queries
            query_indices = pd.read_csv(QUERY_PATH)["index"]
            query_indices = query_indices[query_indices < len(keys)]
            query_keys = keys[query_indices]
            query_hashed_keys = keys[query_indices]
            query_vec = vectorized_keys[query_indices]
            query_labels = labels[query_indices]
        else:
            # If there is no query path defined, we assume we perform the one-pass test
            query_keys = keys
            query_hashed_keys = hashed_keys
            query_vec = vectorized_keys
            query_labels = labels
        print(f"ready to begin {len(query_keys)} queries...")

        max_size_with_fp = -1
        for current_byte_size in target_byte_size:
            print(f"Starting tests for filter of size {current_byte_size} bytes -------------------")
            remaining_bit_size = (current_byte_size - size_in_bytes) * 8

            # distinguish between the positive and negative keys
            pos_keys = keys[labels == CONFIG[dataset][POS_INDICATOR]]
            pos_hashed_keys = hashed_keys[labels == CONFIG[dataset][POS_INDICATOR]]
            pos_vec = vectorized_keys[labels == CONFIG[dataset][POS_INDICATOR]]
            neg_keys = keys[labels != CONFIG[dataset][POS_INDICATOR]]
            neg_vec = vectorized_keys[labels != CONFIG[dataset][POS_INDICATOR]]

            # define a subset of the negative keys to train the filter on
            # make sure to use the same random state that the model was trained on (so that the same negative keys are used)
            train_negative_vec, test_negative_vec = train_test_split(neg_vec, train_size=0.3, random_state = rand_seed)
            
            for current_filter in filters:
                if current_filter == "plbf":
                    filter = FastPLBF_M_Model(clf, pos_keys, pos_vec, train_negative_vec, remaining_bit_size, current_N, current_k)
                elif current_filter == "adabf":
                    filter, thresholds_opt, k_max_opt = Find_Optimal_Parameters(2.1, 2.6, 8, 11, remaining_bit_size, 
                                                                                clf, pos_keys, pos_vec, train_negative_vec)
                    
                # now test on the actual query set
                print(f"starting trial {i}:")
                fp_cnt = 0
                pos_times = []
                neg_times = []
                score_times = []
                search_times = []
                filter_times = []
                first_half_times = []
                first_half_fp = list()
                first_half_fp_cnt = 0
                first_half_neg_cnt = 0
                query_count = 0

                for key, vectorized_key, label in zip(query_keys, query_vec, query_labels):
                    found, score, score_time, search_time, filter_time = filter.contains(key, vectorized_key)
                    score_times.append(score_time)
                    search_times.append(search_time)
                    filter_times.append(filter_time)
                    total_query_time = score_time + search_time + filter_time
                    if label == CONFIG[dataset][POS_INDICATOR]:
                        pos_times.append(total_query_time)
                        assert(found)
                    else:
                        neg_times.append(total_query_time)
                        if found:
                            fp_cnt += 1
                throughput = len(query_keys) / (sum(neg_times) + sum(pos_times))
                print(f"Throughput: {throughput} queries/sec")            
                fpr = fp_cnt / (fp_cnt + len(neg_keys))
                print(f"False Positive Rate: {fpr} [{fp_cnt} / ({fp_cnt} + {len(neg_keys)})]")
                first_half_fp = list(set(first_half_fp))

                # now, we save the results to a file
                if QUERY_PATH == "none":
                    dist = "none"
                elif "unif" in QUERY_PATH:
                    dist = "unif"
                elif "zipf" in QUERY_PATH:
                    dist = "zipf"
                else:
                    dist = "other"
                
                if current_filter == "plbf":
                    current_result = {"dataset": dataset, "filter": "plbf", "bytes": current_byte_size, "query_dist": dist, "num_queries": len(query_keys), 
                                "construct_time": construct_time, "train_time": train_time, "model_accuracy": accuracy,
                                "initial_scores": filter.timing["initial_scores"], "segment_division": filter.timing["segment_division"], 
                                "t_f_finding": filter.timing["t_f_finding"], "insert_scores": sum(filter.timing["insert_scores"]), 
                                "bloom_init": filter.timing["bloom_init"], "region_finding": sum(filter.timing["region_finding"]), 
                                "filter_inserts": sum(filter.timing["filter_inserts"]),
                                    "fpr": fpr, "throughput": throughput, "med_pos_time": median(pos_times), "med_neg_time": median(neg_times),
                                    "amort_score_time": sum(score_times) / len(score_times), 
                                    "amort_region_time": sum(search_times) / len(search_times), "amort_back_filter_time": sum(filter_times) / len(filter_times)}
                elif current_filter == "adabf":
                    current_result = {"dataset": dataset, "filter": "adabf", "bytes": current_byte_size, "query_dist": dist, "num_queries": len(query_keys), 
                                "construct_time": construct_time, "train_time": train_time, "model_accuracy": accuracy,
                                "initial_scores": filter.timing["initial_scores"], 
                                "bloom_init": filter.timing["bloom_init"], "region_finding": filter.timing["threshold_finding"], 
                                "filter_inserts": filter.timing["filter_inserts"],
                                    "fpr": fpr, "throughput": throughput, "med_pos_time": median(pos_times), "med_neg_time": median(neg_times),
                                    "amort_score_time": sum(score_times) / len(score_times), 
                                    "amort_region_time": sum(search_times) / len(search_times), "amort_back_filter_time": sum(filter_times) / len(filter_times)}
                write_results_safely(RESULTS_PATH, RESULTS_COLUMNS, current_result)
print("finished")