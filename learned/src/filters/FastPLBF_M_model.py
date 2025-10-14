"""
Extension of FastPLBF which stores a learned model and performs score inference
on queries during runtime.
"""

from .utils.ThresMaxDivDP import MaxDivDP, ThresMaxDiv
from .utils.OptimalFPR_M import OptimalFPR_M
from .utils.SpaceUsed import SpaceUsed
from .utils.ExpectedFPR import ExpectedFPR
from .utils.const import INF
from .PLBF_M_model import PLBF_M_Model
from sklearn.ensemble import RandomForestClassifier
import time


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
