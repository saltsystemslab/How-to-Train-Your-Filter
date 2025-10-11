from .utils.ThresMaxDivDP import ThresMaxDivDP
from .utils.OptimalFPR_M import OptimalFPR_M
from .utils.SpaceUsed import SpaceUsed
from .utils.ExpectedFPR import ExpectedFPR
from .utils.prList import prList
from .utils.const import INF, EPS
from .utils.url_classifier import vectorize_url
from .utils.Classifier import train_url_classifier

import time
import bisect
from bloom_filter import BloomFilter
import pandas as pd
from sklearn.model_selection import train_test_split

import argparse
import numpy as np

class PLBF_M_Model:
    def __init__(self, dataset: str, pos_keys, neg_keys, M: float, N: int, k: int):
        """
        Args:
            pos_keys (list): vectorized positive keys
            neg_keys (list): vectorized negative keys
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

        self.dataset = "URL"
        # first, take the input and train a model...
        # df_pos = pd.DataFrame({'key': pos_keys, 'label': 1})
        # df_neg = pd.DataFrame({'key': neg_keys, 'label': 0})
        # df = pd.concat([df_pos, df_neg], ignore_index=True)
        input_keys = np.vstack([pos_keys, neg_keys])
        input_labels = np.array([1] * len(pos_keys) + [0] * len(neg_keys))

        self.clf = None
        if dataset == "URL":
            self.clf = train_url_classifier(50, 20, keys=input_keys, labels=input_labels)
        else:
            raise Exception("Classifier for dataset not implemented yet")

        # use the model to obtain scores
        pos_scores = self.clf.predict_proba(pos_keys)[:, 1]
        neg_scores = self.clf.predict_proba(neg_keys)[:, 1]

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


    def divide_into_segments(self, pos_scores: list[float], neg_scores: list[float]):
        segment_thre_list = [i / self.N for i in range(self.N + 1)]
        g = prList(pos_scores, segment_thre_list)
        h = prList(neg_scores, segment_thre_list)
        return segment_thre_list, g, h

    def find_best_t_and_f(self, segment_thre_list, g, h):
        minExpectedFPR = INF
        t_best = None
        f_best = None

        for j in range(self.k, self.N+1):
            t = ThresMaxDivDP(g, h, j, self.k)
            if t is None:
                continue
            f = OptimalFPR_M(g, h, t, self.M, self.k, self.n)
            if minExpectedFPR > ExpectedFPR(g, h, t, f, self.n):
                minExpectedFPR = ExpectedFPR(g, h, t, f, self.n)
                t_best = t
                f_best = f

        self.t = t_best
        self.f = f_best
        self.memory_usage_of_backup_bf = SpaceUsed(g, h, t, f, self.n)
    
    def obtain_score(self, vectorized_key):
        """
        assumes vectorized key
        """
        return self.clf.predict_proba(vectorized_key.reshape(1, -1))[:, 1][0]

    def insert_keys(self, pos_keys: list, vectorized_pos):
        """
        Inserts keys into backup filters and returns (row-aligned) info about the time it took
        to perform each operation during the insertion process.

        Args
        ----
        pos_keys : list
            raw keys to insert into the filters
        vectorized_pos : np array
            vectorized keys for model to calculate scores from, row-aligned with pos_keys

        Returns
        -------
        scoring_times : list(float)
            for each insertion, the time it took for the model to calculate its score (seconds)
        bloom_time : float
            the time it takes to initialize the bloom filters (seconds)
        region_times : list(float)
            for each insertion, the time it takes to find the region it belongs to (seconds)
        insertion_times : list(float)
            for each successful insertion, the time it takes to insert the key into the backup filters
        """
        scoring_times = []
        pos_scores = []
        for key in vectorized_pos:
            score_start = time.time()
            score = self.obtain_score(key)
            score_end = time.time()
            pos_scores.append(score)
            scoring_times.append(score_end - score_start)
        pos_scores = [self.obtain_score(key) for key in vectorized_pos]
        pos_cnt_list = [0 for _ in range(self.k + 1)]
        for score in pos_scores:
            region_idx = self.get_region_idx(score)
            pos_cnt_list[region_idx] += 1
        
        bloom_start = time.time()
        self.backup_bloom_filters = [None for _ in range(self.k + 1)]
        for i in range(1, self.k + 1):
            if 0 < self.f[i] < 1:
                self.backup_bloom_filters[i] = BloomFilter(max_elements = pos_cnt_list[i], error_rate = self.f[i])
            elif self.f[i] == 0:
                assert(pos_cnt_list[i] == 0)
                self.backup_bloom_filters[i] = BloomFilter(max_elements = 1, error_rate = 1 - EPS)
        bloom_end = time.time()
        bloom_time = bloom_end - bloom_start
        
        region_times = []
        insertion_times = []
        for key, score in zip(pos_keys, pos_scores):
            region_start = time.time()
            region_idx = self.get_region_idx(score)
            region_end = time.time()
            if self.backup_bloom_filters[region_idx] is not None:
                insertion_start = time.time()
                self.backup_bloom_filters[region_idx].add(key)
                insertion_end = time.time()
                insertion_times.append(insertion_end - insertion_start)
            region_times.append(region_end - region_start)
        return scoring_times, bloom_time, region_times, insertion_times

    def get_region_idx(self, score):
        region_idx = bisect.bisect_left(self.t, score)
        if region_idx == 0:
            region_idx = 1
        return region_idx

    def contains(self, key, vectorized_key):
        """
        Assumes key is already vectorized

        Returns
        -------
        result (boolean) : whether or not the key is in the filter
        score (float) : the score for the key calculated by the model
        score_time (float) : the time it took to calculate the score (seconds)
        region_time (float) : the time it took to find the corresponding region (seconds)
        filter_search_time (float) : the time it took to query the corresponding backup filter (seconds)
        """
        # vectorized_key = encode_key(key)
        score_start = time.time()
        score = self.clf.predict_proba(vectorized_key.reshape(1, -1))[0, 1]
        score_end = time.time()

        assert(0 <= score <= 1)
        region_start = time.time()
        region_idx = self.get_region_idx(score)
        region_end = time.time()

        filter_search_start = time.time()
        if self.backup_bloom_filters[region_idx] is None:
            filter_search_end = time.time()
            return True, score, score_end - score_start, region_end - region_start, filter_search_end - filter_search_start
        result = (key in self.backup_bloom_filters[region_idx])
        filter_search_end = time.time()
        return result, score, score_end - score_start, region_end - region_start, filter_search_end - filter_search_start



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action="store", dest="data_path", type=str, required=True,
                        help="path of the dataset")
    parser.add_argument('--N', action="store", dest="N", type=int, required=True,
                        help="N: the number of segments")
    parser.add_argument('--k', action="store", dest="k", type=int, required=True,
                        help="k: the number of regions")
    parser.add_argument('--M', action="store", dest="M", type=float, required=True,
                        help="M: the target memory usage for backup Bloom filters")

    results = parser.parse_args()

    DATA_PATH = results.data_path
    N = results.N
    k = results.k
    M = results.M

    data = pd.read_csv(DATA_PATH)
    negative_sample = data.loc[(data['label'] != 1)]
    positive_sample = data.loc[(data['label'] == 1)]
    train_negative, test_negative = train_test_split(negative_sample, test_size = 0.7, random_state = 0)
    
    pos_keys            = list(positive_sample['key'])
    pos_scores          = list(positive_sample['score'])
    train_neg_keys      = list(train_negative['key'])
    train_neg_scores    = list(train_negative['score'])
    test_neg_keys       = list(test_negative['key'])
    test_neg_scores     = list(test_negative['score'])

    construct_start = time.time()
    plbf = PLBF_M_Model(pos_keys, pos_scores, train_neg_scores, M, N, k)
    construct_end = time.time()

    # assert : no false negative
    pos_times = []
    for key, score in zip(pos_keys, pos_scores):
        query_start = time.time()
        result = plbf.contains(key,score)
        query_end = time.time()
        pos_times.append(query_end - query_start)
        assert(result)
    print("avg time for pos query: ", sum(pos_times) / len(pos_times))
    
    # test
    fp_cnt = 0
    neg_times = []
    for key, score in zip(test_neg_keys, test_neg_scores):
        query_start = time.time()
        found = plbf.contains(key, score)
        query_end = time.time()
        neg_times.append(query_end - query_start)
        if found:
            fp_cnt += 1
    print("avg time for neg query: ", sum(neg_times) / len(neg_times))

    print(f"Construction Time: {construct_end - construct_start}")
    print(f"Memory Usage of Backup BF: {plbf.memory_usage_of_backup_bf}")
    print(f"False Positive Rate: {fp_cnt / len(test_neg_keys)} [{fp_cnt} / {len(test_neg_keys)}]")

