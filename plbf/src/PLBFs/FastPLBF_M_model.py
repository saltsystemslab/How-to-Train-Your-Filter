from .utils.ThresMaxDivDP import MaxDivDP, ThresMaxDiv
from .utils.OptimalFPR_M import OptimalFPR_M
from .utils.SpaceUsed import SpaceUsed
from .utils.ExpectedFPR import ExpectedFPR
from .utils.const import INF
from .PLBF_M_model import PLBF_M_Model
from .utils.Classifier import train_url_classifier
from .utils.url_classifier import vectorize_url

from sklearn.ensemble import RandomForestClassifier
import sys
import os
import time
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

training_configuration = {
    "news": {
        "dataset": 'data/fake_news_predictions.csv',
        "estimators": 15,
        "leaves": 10
    },
    "url": {
        "dataset": "data/malicious_url_scores.csv",
        "estimators": 50,
        "leaves": 20
    },
    "ember": {
        "dataset": "ember",
        "estimators": 60,
        "leaves": 20
    }
}

def obtain_normal_and_vectorized_data(dataset: str):
    """
    Returns keys, vectorized form, and labels for the specified dataset.

    Usually, keys are used to insert into the filter, but the vectorized keys
    are used for the model to assess the key.
    """
    if dataset == "url":
        data = pd.read_csv(training_configuration["url"]["dataset"])
        data['label'] = data['type'].apply(lambda x: 1 if x == 'malicious' else 0)
        keys = np.array(data['url'])
        vectorized_keys = np.array([vectorize_url(url) for url in keys])
        labels = np.array(data['label'])
        return keys, vectorized_keys, labels
        # pos_rows = (data['type'] == 'malicious')
        # neg_rows = (data['type'] != 'malicious')
        # now return key, vectorized form, and label for positive and negative keys
        return keys[pos_rows], vectorized_keys[pos_rows], labels[pos_rows], keys[neg_rows], vectorized_keys[neg_rows], labels[neg_rows]
    # elif dataset == "ember":
        # return get_ember_keys()
    else:
        print("TODO")

class FastPLBF_M_Model(PLBF_M_Model):
    def __init__(self, model: RandomForestClassifier, pos_keys, vectorized_pos, vectorized_neg, M: float, N: int, k: int):
        """
        Args:
            model (sklearn model): model to use for score calculations
            pos_keys (list): string positive keys to insert
            neg_keys (list): string negative keys to include in training
            M (float): the target memory usage for backup Bloom filters
            N (int): number of segments
            k (int): number of regions
        """

        # assert 
        # assert(isinstance(pos_keys, list))
        # assert(isinstance(neg_keys, list))
        assert(isinstance(M, float))
        assert(0 < M)
        assert(isinstance(N, int))
        assert(isinstance(k, int))

        self.clf = model
        
        # use the model to obtain scores
        training_score_start = time.time()
        pos_scores = self.clf.predict_proba(vectorized_pos)[:, 1]
        neg_scores = self.clf.predict_proba(vectorized_neg)[:, 1]
        training_score_end = time.time()

        for score in pos_scores:
            assert(0 <= score <= 1)
        for score in neg_scores:
            assert(0 <= score <= 1)

        self.M = M
        self.N = N
        self.k = k
        self.n = len(pos_keys)

        segment_start = time.time()
        segment_thre_list, g, h = self.divide_into_segments(pos_scores, neg_scores)
        segment_end = time.time()
        t_f_start = time.time()
        self.find_best_t_and_f(segment_thre_list, g, h)
        t_f_end = time.time()
        insert_scoring_times, init_bloom_time, insert_find_region_times, filter_insert_times = self.insert_keys(pos_keys, vectorized_pos)
        self.timing = {
            "initial_scores": training_score_end - training_score_start,
            "segment_division": segment_end - segment_start,
            "t_f_finding": t_f_end - t_f_start,
            "insert_scores": insert_scoring_times,
            "bloom_init": init_bloom_time,
            "region_finding": insert_find_region_times,
            "filter_inserts": filter_insert_times
        }
        
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
    parser.add_argument('--data', action="store", dest="data", type=str, required=True,
                        help="name of the dataset")
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

    results, leftovers = parser.parse_known_args()

    DATA = results.data
    QUERY_PATH = results.query_path
    N = results.N
    k = results.k
    M = results.M
    num_trials = 5 if results.t == "none" else results.t

    print("obtaining keys...")
    keys, vectorized_keys, labels = obtain_normal_and_vectorized_data(DATA)
    # data = pd.read_csv(DATA_PATH)
    query_indexes = pd.read_csv(QUERY_PATH) if QUERY_PATH != "none" else []
    queries = []
    vectorized_queries = []
    query_labels = []
    selected_data = []

    print("separating queries...")
    if results.query_path is not None:
        # Find all the rows in the data that correspond to the queries
        queries = keys[query_indexes["index"]]
        vectorized_queries = vectorized_keys[query_indexes["index"]]
        query_labels = labels[query_indexes["index"]]
    else:
        queries = keys
        vectorized_queries = vectorized_keys
        query_labels = labels
    num_fp = []
    
    # create boolean masks to find true positives in the data
    positive_rows = (labels == 1)
    negative_rows = (labels != 1)

    positive_keys = keys[positive_rows]
    positive_vec = vectorized_keys[positive_rows]
    positive_labels = labels[positive_rows]
    negative_keys = keys[negative_rows]
    negative_vec = vectorized_keys[negative_rows]
    negative_labels = labels[negative_rows]
    
    # create a training set of the negative labels from the overall data
    print("creating training set...")
    (train_negative_keys, test_negative_keys, 
     train_negative_vec, test_negative_vec, 
     train_negative_labels, test_negative_labels) = train_test_split(negative_keys, negative_vec, 
                                                                     negative_labels, test_size = 0.7, random_state = 0)

    print("constructing PLBF...")
    construct_start = time.time() 
    plbf = FastPLBF_M_Model("URL", positive_keys, positive_vec, train_negative_vec, M, N, k)
    construct_end = time.time()

    # now, use the query set
    actual_false_positives = []
    pos_times = []
    neg_times = []
    print("starting queries...")
    for key, vectorized_key, label in zip(queries, vectorized_queries, query_labels):
        query_start = time.time()
        found = plbf.contains(key, vectorized_key)
        query_end = time.time()
        if label == 1:
            assert(found)
            pos_times.append(query_end - query_start)
        else:
            if not found:
                actual_false_positives.append((key, vectorized_key, label))
            neg_times.append(query_end - query_start)
    print("avg time for pos query: ", sum(pos_times) / len(pos_times))
    print("avg time for neg query: ", sum(neg_times) / len(neg_times))
    fp_cnt = len(actual_false_positives)

    false_positives = list(set(actual_false_positives))
    
    # optional - perform additional adversarial queries
    num_neg_queries = np.sum(query_labels != 1)
    print(f"Construction Time: {construct_end - construct_start}")
    print(f"Memory Usage of Backup BF: {plbf.memory_usage_of_backup_bf}")
    print(f"False Positive Rate: {fp_cnt / num_neg_queries} [{fp_cnt} / {num_neg_queries}]")

    
    # optional adversarial test
    # TODO - make it so that each trial has the same number of queries...
    # rn we have performed 10M queries and collected data on false positives.
    # what we want to do is take the positive and negative keys and split them in half.
    # for the first 5M, just use the 
    include_adversarial = True
    if include_adversarial:
        # figure out the adversarial sizes
        advers_freqs = [0.02, 0.04, 0.06, 0.08, 0.1]
        for freq in advers_freqs:
            # first, calculate how many adversarial queries we need to perform
            num_queries = len(queries)
            # num_queries = (1-advers_freq) * (num_queries + num_advers)
            num_advers = int(num_queries / (1-freq) - num_queries)
            num_neg = num_neg_queries + num_advers
            current_index = 0
            fp_cnt = 0
            fp_times = []
            for i in range(num_advers):
                if len(false_positives) == 0:
                    fp_times.append(0)
                    break
                key = false_positives[current_index]
                vectorized_key = vectorize_url(key)
                query_start = time.time()
                found = plbf.contains(key, vectorized_key)
                query_end = time.time()
                fp_times.append(query_end - query_start)
                if found:
                    fp_cnt += 1
                if current_index == len(false_positives) - 1:
                    current_index = 0
                else:
                    current_index += 1
            print(f"Average time for adversarial query: {sum(fp_times) / len(fp_times)}")
            print(f"False Positive with {freq} Adversarial: {fp_cnt / (num_neg + len(false_positives))} [ {fp_cnt} / {(num_neg + len(false_positives))}]")
