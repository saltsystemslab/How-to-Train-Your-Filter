from .utils.ThresMaxDivDP import MaxDivDP, ThresMaxDiv
from .utils.OptimalFPR_M import OptimalFPR_M
from .utils.SpaceUsed import SpaceUsed
from .utils.ExpectedFPR import ExpectedFPR
from .utils.const import INF
from .PLBF_M import PLBF_M
from .utils.Classifier import train_url_classifier
from .utils.url_classifier import vectorize_url

import numpy as np
import os
import time
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data_configuration = {
    "data/fake_news_predictions.csv": {
        "dataset": 'news',
        "estimators": 15,
        "leaves": 10,
        "obj_name": "title",
        "score_name": "prediction_score",
        "label_name": "label",
        "pos_indicator": 1
    },
    "data/malicious_url_scores.csv": {
        "dataset": "url",
        "estimators": 50,
        "leaves": 20,
        "obj_name": "url",
        "score_name": "prediction_score",
        "label_name": "type",
        "pos_indicator": "malicious"
    },
    "data/combined_ember_metadata.csv": {
        "dataset": "ember",
        "estimators": 60,
        "leaves": 20,
        "obj_name": "sha256",
        "score_name": "score",
        "label_name": "label",
        "pos_indicator": 1
    }
}

results_filename = "results/results.csv"
results_columns = ['dataset', 'size', 'fpr', 'throughput', 'num_queries', 'query_dist']
adversarial_results_filename = "results/advers_results.csv"
adversarial_results_columns = ['dataset', 'freq', 'size', 'fpr', 'throughput', 'num_queries']
# TODO - decide what to do about throughput

class FastPLBF_M(PLBF_M):
    def __init__(self, pos_keys: list, pos_scores: list[float], neg_scores: list[float], M: float, N: int, k: int):
        """
        Args:
            pos_keys (list): keys
            pos_scores (list[float]): scores of keys
            neg_scores (list[float]): scores of non-keys
            M (float): the target memory usage for backup Bloom filters
            N (int): number of segments
            k (int): number of regions
        """

        # assert 
        # assert(isinstance(pos_keys, list))
        # assert(isinstance(pos_scores, list))
        assert(len(pos_keys) == len(pos_scores))
        # assert(isinstance(neg_scores, list))
        assert(isinstance(M, float))
        assert(0 < M)
        assert(isinstance(N, int))
        assert(isinstance(k, int))

        for score in pos_scores:
            assert(0 <= score <= 1)
        for score in neg_scores:
            assert(0 <= score <= 1)

        
        self.M = M
        self.N = N
        self.k = k
        self.n = len(pos_keys)

        
        segment_thre_list, g, h = self.divide_into_segments(pos_scores, neg_scores)
        self.find_best_t_and_f(segment_thre_list, g, h)
        self.insert_keys(pos_keys, pos_scores)
        
    def find_best_t_and_f(self, segment_thre_list, g, h):
        minExpectedFPR = INF
        t_best = None
        f_best = None

        DPKL, DPPre = MaxDivDP(g, h, self.N, self.k)
        for j in range(self.k, self.N+1):
            t = ThresMaxDiv(DPPre, j, self.k, segment_thre_list)
            if t is None:
                continue
            f = OptimalFPR_M(g, h, t, self.M, self.k, self.n)
            if minExpectedFPR > ExpectedFPR(g, h, t, f, self.n):
                minExpectedFPR = ExpectedFPR(g, h, t, f, self.n)
                t_best = t
                f_best = f

        self.t = t_best
        self.f = f_best
        print("t: ", self.t)
        print("f: ", self.f)
        self.memory_usage_of_backup_bf = SpaceUsed(g, h, t, f, self.n)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action="store", dest="data_path", type=str, required=True,
                        help="path of the dataset")
    parser.add_argument('--query_path', action="store", dest="query_path", type=str, required=False,
                        help="path of the query indices")
    parser.add_argument('--N', action="store", dest="N", type=int, required=True,
                        help="N: the number of segments")
    parser.add_argument('--k', action="store", dest="k", type=int, required=True,
                        help="k: the number of regions")
    parser.add_argument('--M', action="store", dest="M", type=float, required=True,
                        help="M: the target memory usage for backup Bloom filters")
    parser.add_argument('--t', action='store', dest='t', type=int, required=False,
                        help='t: the number of trials to run')
    parser.add_argument('--adv', action='store_true', dest='adv',
                        help='adv: whether or not to run adversarial tests')
    
    # TODO - figure out saving results
    results_df = None
    adversarial_df = None
    if os.path.exists(results_filename):
        results_df = pd.read_csv(results_filename)
    else:
        results_df = pd.DataFrame(columns=results_columns)

    if os.path.exists(adversarial_results_filename):
        results_df = pd.read_csv(adversarial_results_filename)
    else:
        results_df = pd.DataFrame(columns=adversarial_results_columns)

    results, leftovers = parser.parse_known_args()

    DATA_PATH = results.data_path
    QUERY_PATH = results.query_path
    N = results.N
    k = results.k
    M = results.M
    # the PLBF considers size in units of bits, but the size values we take from the AQF are in terms of bytes, so let's convert...
    M = results.M * 8
    num_trials = 1 if results.t == "none" else results.t # unsure if this is useful because the results are always the same
    include_adversarial = results.adv

    data = pd.read_csv(DATA_PATH)
    query_indices = pd.read_csv(QUERY_PATH) if QUERY_PATH != "none" else []

    all_queries = []
    if results.query_path is not None:
        all_queries = data.iloc[query_indices["index"]]
    else:
        all_queries = data
    print("num queries: ", len(all_queries))

    obj_name = data_configuration[DATA_PATH]["obj_name"]
    score_name = data_configuration[DATA_PATH]["score_name"]
    label_name = data_configuration[DATA_PATH]["label_name"]
    pos_indicator = data_configuration[DATA_PATH]["pos_indicator"]

    for i in range(num_trials):

        # Now use a subset of the original data to train the PLBF
        all_negative = data.loc[(data[label_name] != pos_indicator)]
        all_positive = data.loc[(data[label_name] == pos_indicator)]
        train_negative, test_negative = train_test_split(all_negative, test_size = 0.7, random_state = 0)
        
        # here, we want to turn this into a list following the query distribution

        # PLBF paper specifies that we must train on ALL positive keys
        train_pos_keys            = list(all_positive[obj_name])
        train_pos_scores          = list(all_positive[score_name])
        train_neg_keys      = list(train_negative[obj_name])
        train_neg_scores    = list(train_negative[score_name])

        negative_queries = all_queries.loc[(all_queries[label_name] != pos_indicator)]
        positive_queries = all_queries.loc[(all_queries[label_name] == pos_indicator)]
        # Now we define the test set by using the merged set
        # the test set could just be the original data

        print(f"True positives: {len(positive_queries)}")
        print(f"True negatives: {len(negative_queries)}")

        construct_start = time.time()
        plbf = FastPLBF_M(train_pos_keys, train_pos_scores, train_neg_scores, M, N, k)
        construct_end = time.time()
        
        # test on the actual mixed query set.
        fp_cnt = 0
        pos_times = []
        neg_times = []
        first_half_times = []
        false_positives = set()
        query_count = 0
        for key, score, label in zip(all_queries[obj_name], all_queries[score_name], all_queries[label_name]):
            query_start = time.time()
            found = plbf.contains(key, score)
            query_end = time.time()
            if query_count < len(all_queries) / 2:
                first_half_times.append(query_end - query_start)
            if label == pos_indicator:
                pos_times.append(query_end - query_start)
                # assert no false negatives
                assert(found)
            else:
                neg_times.append(query_end - query_start)
                if found:
                    fp_cnt += 1
                    if query_count < len(all_queries) / 2:
                        # use the first half of queries to collect false positives for the adversarial test
                        false_positives.add((key, score))
        first_half_fp_cnt = len(false_positives)
        false_positives = list(false_positives)
        print("avg time for neg query: ", sum(neg_times) / len(neg_times))
        print("avg time for pos query: ", sum(pos_times) / len(pos_times))
        throughput = len(all_queries) / (sum(neg_times) + sum(pos_times))
        print(f"Throughput: {throughput} queries/sec")
        # neg queries are expected to take more time because the score indicates that an additional step of checking
        # the internal filters needs to be done.
        print(f"Construction Time: {construct_end - construct_start}")
        print(f"Memory Usage of Backup BF: {plbf.memory_usage_of_backup_bf}")
        fpr = fp_cnt / (fp_cnt + len(negative_queries))
        print(f"False Positive Rate: {fpr} [{fp_cnt} / ({fp_cnt} + {len(negative_queries)})]")
        
        # now save the results to a file
        if QUERY_PATH == "none":
            dist = "none"
        elif "unif" in QUERY_PATH:
            dist = "unif"
        elif "zipf" in QUERY_PATH:
            dist = "zipf"
        else:
            dist = "other"
        initial_result = {"dataset": data_configuration[DATA_PATH]["dataset"], "size": int(M/8), "fpr": fpr, "throughput": throughput, "num_queries": len(all_queries), "query_dist": dist}
        results_df = pd.concat([results_df, pd.DataFrame([initial_result])], ignore_index=True)
        results_df.to_csv(results_filename, index=False)
        
        # optional adversarial test
        first_half_queries = all_queries[:int(len(all_queries)/2)]
        if include_adversarial and "unif" in QUERY_PATH and len(false_positives) != 0:
            # figure out the adversarial sizes
            advers_freqs = [0.02, 0.04, 0.06, 0.08, 0.1]
            for freq in advers_freqs:
                print(f"Starting test with {freq} adversarial queries")
                # first, calculate how many adversarial queries we need to perform
                num_queries_overall = len(all_queries)
                num_queries_remaining = int(num_queries_overall / 2)
                # num_queries = (1-advers_freq) * (num_queries + num_advers)
                num_advers = int(num_queries_overall / (1-freq) - num_queries_overall)
                num_queries_remaining -= num_advers

                # do the remaining queries
                second_half_fp_cnt = 0
                remaining_queries = all_queries[num_queries_remaining:]
                remaining_times = []
                for key, score, label in zip(remaining_queries[obj_name], remaining_queries[score_name], remaining_queries[label_name]):
                    query_start = time.time()
                    found = plbf.contains(key, score)
                    query_end = time.time()
                    was_pos = label == pos_indicator
                    remaining_times.append(query_end - query_start)
                    if was_pos:
                        assert(found)
                    elif not was_pos and found:
                        second_half_fp_cnt += 1
                
                # for the adversarial workload, loop through the false positives
                current_index = 0
                adversarial_times = []
                adversarial_fp_cnt = 0
                for i in range(num_advers):
                    key, score = false_positives[current_index]
                    query_start = time.time()
                    found = plbf.contains(key, score)
                    query_end = time.time()
                    adversarial_times.append(query_end - query_start)
                    if found:
                        adversarial_fp_cnt += 1
                    if current_index == len(false_positives) - 1:
                        current_index = 0
                    else:
                        current_index += 1

                neg_rows_first = (first_half_queries[label_name] != pos_indicator)
                neg_rows_second = (remaining_queries[label_name] != pos_indicator)
                num_neg = len(first_half_queries[neg_rows_first]) + len(remaining_queries[neg_rows_second]) + num_advers
                total_fp = first_half_fp_cnt + second_half_fp_cnt + adversarial_fp_cnt
                print(f"Average time for adversarial query: {sum(adversarial_times) / len(adversarial_times)}")
                adversarial_avg_time = (sum(first_half_times) + sum(remaining_times) + sum(adversarial_times)) / (len(first_half_times) + len(remaining_times) + len(adversarial_times))
                adversarial_throughput = 1 / adversarial_avg_time
                print(f"Average time with adversarial workload: ", adversarial_avg_time)
                adversarial_fpr = total_fp / (total_fp + num_neg)
                print(f"False Positive with {freq} Adversarial: {adversarial_fpr} [ {total_fp} / ({total_fp} + {num_neg})]")

                # now save the results to a file
                adversarial_result = {"dataset": data_configuration[DATA_PATH]["dataset"], "freq": freq, "size": int(M/8), "fpr": adversarial_fpr, "throughput": adversarial_throughput, "num_queries": len(all_queries)}
                adversarial_df = pd.concat([adversarial_df, pd.DataFrame([adversarial_result])], ignore_index=True)
                adversarial_df.to_csv(adversarial_results_filename, index=False)
