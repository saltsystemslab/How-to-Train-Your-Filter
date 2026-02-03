from .Bloom_filter import hashfunc
import numpy as np

class Ada_BloomFilter():
    def __init__(self, n, hash_len, k_max):
        self.n = n
        self.hash_len = int(hash_len)
        self.h = []
        self.thresholds_opt = None
        self.k_max_opt = None
        for i in range(int(k_max)):
            self.h.append(hashfunc(self.hash_len))
        self.table = np.zeros(self.hash_len, dtype=int)
    def insert(self, key, k):
        for j in range(int(k)):
            t = self.h[j](key)
            self.table[t] = 1
    def clear(self):
        self.table = np.zeros(self.hash_len, dtype=int)
    def test(self, key, k):
        test_result = 0
        match = 0
        for j in range(int(k)):
            t = self.h[j](key)
            match += 1*(self.table[t] == 1)
        if match == k:
            test_result = 1
        return test_result
    def contains(self, key, score):
        ix = min(np.where(score < self.thresholds_opt)[0])
        # thres = thresholds[ix]
        k = self.k_max_opt - ix
        return self.test(key, k)


def R_size(count_key, count_nonkey, R0):
    R = [0]*len(count_key)
    R[0] = R0
    for k in range(1, len(count_key)):
        R[k] = max(int(count_key[k] * (np.log(count_nonkey[0]/count_nonkey[k])/np.log(0.618) + R[0]/count_key[0])), 1)
    return R


def Find_Optimal_Parameters(c_min, c_max, num_group_min, num_group_max, R_sum, pos_keys, neg_keys, pos_scores, neg_scores, key_scores=None):
    c_set = np.arange(c_min, c_max+10**(-6), 0.1)

    FP_opt = neg_keys.shape[0]

    k_min = 0
    for k_max in range(num_group_min, num_group_max+1):
        for c in c_set:
            tau = sum(c ** np.arange(0, k_max - k_min + 1, 1))
            n = pos_keys.shape[0]
            hash_len = R_sum
            bloom_filter = Ada_BloomFilter(n, hash_len, k_max)
            thresholds = np.zeros(k_max - k_min + 1)
            thresholds[-1] = 1.1
            num_negative = sum(neg_scores <= thresholds[-1])
            num_piece = int(num_negative / tau) + 1
            scores = neg_scores[(neg_scores <= thresholds[-1])]
            scores = np.sort(scores)
            for k in range(k_min, k_max):
                i = k - k_min
                score_1 = scores[scores < thresholds[-(i + 1)]]
                if int(num_piece * c ** i) < len(score_1):
                    thresholds[-(i + 2)] = score_1[-int(num_piece * c ** i)]

            keys = pos_keys
            scores = pos_scores

            scores_for_insertions = pos_scores if key_scores is None else key_scores
            for score_s, key_s in zip(scores_for_insertions, keys):
                ix = min(np.where(score_s < thresholds)[0])
                k = k_max - ix
                bloom_filter.insert(key_s, k)
            ML_positive = neg_keys[(neg_scores >= thresholds[-2])]
            key_negative = neg_keys[(neg_scores < thresholds[-2])]
            score_negative = neg_scores[(neg_scores < thresholds[-2])]
            test_result = np.zeros(len(key_negative))
            ss = 0
            for score_s, key_s in zip(score_negative, key_negative):
                ix = min(np.where(score_s < thresholds)[0])
                # thres = thresholds[ix]
                k = k_max - ix
                test_result[ss] = bloom_filter.test(key_s, k)
                ss += 1
            FP_items = sum(test_result) + len(ML_positive)
            if FP_opt > FP_items:
                FP_opt = FP_items
                bloom_filter_opt = bloom_filter
                thresholds_opt = thresholds
                k_max_opt = k_max
    bloom_filter_opt.thresholds_opt = thresholds_opt
    bloom_filter_opt.k_max_opt = k_max_opt
    return bloom_filter_opt, thresholds_opt, k_max_opt