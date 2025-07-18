from .Bloom_filter import hashfunc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import time
import os



# DATA_PATH = './URL_data.csv'
# num_group_min = 8
# num_group_max = 12
# R_sum = 200000
# c_min = 1.8
# c_max = 2.1


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


def Find_Optimal_Parameters(c_min, c_max, num_group_min, num_group_max, R_sum, pos_keys, neg_keys, pos_scores, neg_scores):
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
            # print("thresholds: ", thresholds)
            num_negative = sum(neg_scores <= thresholds[-1])
            # print("num_negative: ", num_negative)
            num_piece = int(num_negative / tau) + 1
            # print("num_piece: ", num_piece)
            scores = neg_scores[(neg_scores <= thresholds[-1])]
            scores = np.sort(scores)
            for k in range(k_min, k_max):
                i = k - k_min
                score_1 = scores[scores < thresholds[-(i + 1)]]
                if int(num_piece * c ** i) < len(score_1):
                    thresholds[-(i + 2)] = score_1[-int(num_piece * c ** i)]

            keys = pos_keys
            scores = pos_scores

            for score_s, key_s in zip(scores, keys):
                ix = min(np.where(score_s < thresholds)[0])
                k = k_max - ix
                bloom_filter.insert(key_s, k)
            # print("thresholds[-2]:", thresholds[-2])
            # print("min neg_score:", neg_scores.min())
            # print("max neg_score:", neg_scores.max())
            ML_positive = neg_keys[(neg_scores >= thresholds[-2])]
            key_negative = neg_keys[(neg_scores < thresholds[-2])]
            score_negative = neg_scores[(neg_scores < thresholds[-2])]
            # print(f"ML_positive: {len(ML_positive)}")
            test_result = np.zeros(len(key_negative))
            ss = 0
            for score_s, key_s in zip(score_negative, key_negative):
                ix = min(np.where(score_s < thresholds)[0])
                # thres = thresholds[ix]
                k = k_max - ix
                test_result[ss] = bloom_filter.test(key_s, k)
                ss += 1
            FP_items = sum(test_result) + len(ML_positive)
            # print('False positive items: %d, Number of groups: %d, c = %f' %(FP_items, k_max, round(c, 2)))
            # print(f"fp_opt: {FP_opt}")
            # print(f"fp_items: {FP_items}")
            if FP_opt > FP_items:
                FP_opt = FP_items
                bloom_filter_opt = bloom_filter
                thresholds_opt = thresholds
                k_max_opt = k_max

    # print('Optimal FPs: %f, Optimal c: %f, Optimal num_group: %d' % (FP_opt, c_opt, num_group_opt))
    # print(f"thresholds_opt: {thresholds_opt}")
    # print(f"k_max_op: {k_max_opt}")
    bloom_filter_opt.thresholds_opt = thresholds_opt
    bloom_filter_opt.k_max_opt = k_max_opt
    return bloom_filter_opt, thresholds_opt, k_max_opt

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

results_filename = "results/adabf_results.csv"
results_columns = ['dataset', 'size', 'fpr', 'throughput', 'num_queries', 'query_dist']
adversarial_results_filename = "results/adabf_advers_results.csv"
adversarial_results_columns = ['dataset', 'freq', 'size', 'fpr', 'throughput', 'num_queries']


'''
Implement Ada-BF
'''
if __name__ == '__main__':
    '''Stage 1: Find the hyper-parameters (spare 30% samples to find the parameters)'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action="store", dest="data_path", type=str, required=True,
                        help="path of the dataset")
    parser.add_argument('--query_path', action="store", dest="query_path", type=str, required=False,
                        help="path of queries")
    parser.add_argument('--size_of_Ada_BF', action="store", dest="R_sum", type=int, required=True,
                        help="size of the Ada-BF")
    parser.add_argument('--num_group_min', action="store", dest="min_group", type=int, required=True,
                        help="Minimum number of groups")
    parser.add_argument('--num_group_max', action="store", dest="max_group", type=int, required=True,
                        help="Maximum number of groups")
    parser.add_argument('--c_min', action="store", dest="c_min", type=float, required=True,
                        help="minimum ratio of the keys")
    parser.add_argument('--c_max', action="store", dest="c_max", type=float, required=True,
                        help="maximum ratio of the keys")
    parser.add_argument('--trials', action="store", dest="trials", type=int, required=False,
                        help="number of trials to run")
    parser.add_argument('--adv', action="store_true", dest="adv",
                        help="whether or not to include the adversarial test")

    results_df = None
    adversarial_df = None
    if os.path.exists(results_filename):
        results_df = pd.read_csv(results_filename)
    else:
        results_df = pd.DataFrame(columns=results_columns)

    if os.path.exists(adversarial_results_filename):
        adversarial_df = pd.read_csv(adversarial_results_filename)
    else:
        adversarial_df = pd.DataFrame(columns=adversarial_results_columns)

    results = parser.parse_args()
    DATA_PATH = results.data_path
    QUERY_PATH = results.query_path
    num_group_min = results.min_group
    num_group_max = results.max_group
    R_sum = results.R_sum
    # ADA-BF considers size in units of bits, but the size values we take from the AQF are in terms of bytes, so let's convert...
    R_sum = results.R_sum * 8
    c_min = results.c_min
    c_max = results.c_max
    num_trials = 1 if results.trials == "none" else results.trials
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

    negative_queries = all_queries.loc[(all_queries[label_name] != pos_indicator)]

    for i in range(num_trials):
        print(f"Trial {i} =========================================================")
        # Setup filter
        negative_sample = data.loc[(data[label_name] != pos_indicator)]
        positive_sample = data.loc[(data[label_name] == pos_indicator)]
        neg_keys = negative_sample.sample(frac = 0.3)
        bloom_filter_opt, thresholds_opt, k_max_opt = Find_Optimal_Parameters(c_min, c_max, num_group_min, num_group_max, R_sum, 
                                                                            neg_keys, positive_sample, score_name=score_name, obj_name=obj_name)
        
        fp_count = 1
        fp_tn_count = 1 #laplace smooth to avoid divide by 0 (only an issue in small test datasets)
        
        if QUERY_PATH == None:
            ML_positive = negative_sample.loc[(negative_sample[score_name] >= thresholds_opt[-2]), obj_name]
            url_negative = negative_sample.loc[(negative_sample[score_name] < thresholds_opt[-2]), obj_name]
            score_negative = negative_sample.loc[(negative_sample[score_name] < thresholds_opt[-2]), score_name]
            test_result = np.zeros(len(url_negative))
            ss = 0
            for score_s, url_s in zip(score_negative, url_negative):
                ix = min(np.where(score_s < thresholds_opt)[0])
                # thres = thresholds[ix]
                k = k_max_opt - ix
                test_result[ss] = bloom_filter_opt.test(url_s, k)
                ss += 1
            FP_items = sum(test_result) + len(ML_positive)
            print('False positive items: %d' % FP_items)
            
        else:
            fp_cnt = 0
            pos_times = []
            neg_times = []
            first_half_times = []
            false_positives = set()
            query_count = 0
            for key, score, label in zip(all_queries[obj_name], all_queries[score_name], all_queries[label_name]):
                query_start = time.time()
                ix = min(np.where(score < thresholds_opt)[0])
                # thres = thresholds[ix]
                k = k_max_opt - ix
                found = bloom_filter_opt.test(key, k)
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
            throughput = 1 / (sum(neg_times) + sum(pos_times))
            print(f"Throughput: {throughput} queries/sec")
            # neg queries are expected to take more time because the score indicates that an additional step of checking
            # the internal filters needs to be done.
            # print(f"Construction Time: {construct_end - construct_start}")
            # print(f"Memory Usage of Backup BF: {plbf.memory_usage_of_backup_bf}")
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
            initial_result = {"dataset": data_configuration[DATA_PATH]["dataset"], "size": int(R_sum/8), "fpr": fpr, "throughput": throughput, "num_queries": len(all_queries), "query_dist": dist}
            results_df = pd.concat([results_df, pd.DataFrame([initial_result])], ignore_index=True)
            results_df.to_csv(results_filename, index=False)

            # optional adversarial test
            include_adversarial = False
            if include_adversarial:
                # also save to adversarial as the 0-freq adversarial dataset
                adversarial_result = {"dataset": data_configuration[DATA_PATH]["dataset"], "freq": 0, "size": R_sum, "fpr": fpr, "throughput": throughput, "num_queries": len(all_queries)}
                adversarial_df = pd.concat([adversarial_df, pd.DataFrame([adversarial_result])], ignore_index=True)
        
            first_half_queries = all_queries[:int(len(all_queries)/2)]
            if include_adversarial and "unif" in QUERY_PATH:
                # figure out the adversarial sizes
                advers_freqs = [0.02, 0.04, 0.06, 0.08, 0.1]
                for freq in advers_freqs:
                    print(f"Starting test with {freq} adversarial queries")
                    # first, calculate how many adversarial queries we need to perform
                    num_queries_overall = len(all_queries)
                    num_queries_remaining = int(num_queries_overall / 2)
                    # for reference: num_queries = (1-advers_freq) * (num_queries + num_advers)
                    num_advers = int(num_queries_overall / (1-freq) - num_queries_overall)
                    num_queries_remaining -= num_advers

                    # do the remaining queries
                    second_half_fp_cnt = 0
                    remaining_queries = all_queries[num_queries_remaining:]
                    remaining_times = []
                    for key, score, label in zip(remaining_queries[obj_name], remaining_queries[score_name], remaining_queries[label_name]):
                        query_start = time.time()
                        ix = min(np.where(score < thresholds_opt)[0])
                        # thres = thresholds[ix]
                        k = k_max_opt - ix
                        found = bloom_filter_opt.test(key, k)
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
                        ix = min(np.where(score < thresholds_opt)[0])
                        # thres = thresholds[ix]
                        k = k_max_opt - ix
                        found = bloom_filter_opt.test(key, k)
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
                    # print(f"Average time for adversarial query: {sum(adversarial_times) / len(adversarial_times)}")
                    adversarial_avg_time = (sum(first_half_times) + sum(remaining_times) + sum(adversarial_times)) / (len(first_half_times) + len(remaining_times) + len(adversarial_times))
                    adversarial_throughput = 1 / adversarial_avg_time
                    # print(f"Average time with adversarial workload: ", adversarial_avg_time)
                    adversarial_fpr = total_fp / (total_fp + num_neg)
                    print(f"False Positive with {freq} Adversarial: {adversarial_fpr} [ {total_fp} / ({total_fp} + {num_neg})]")

                    # now save the results to a file
                    adversarial_result = {"dataset": data_configuration[DATA_PATH]["dataset"], "freq": freq, "size": int(R_sum/8), "fpr": adversarial_fpr, "throughput": adversarial_throughput, "num_queries": len(all_queries)}
                    adversarial_df = pd.concat([adversarial_df, pd.DataFrame([adversarial_result])], ignore_index=True)
                    adversarial_df.to_csv(adversarial_results_filename, index=False)