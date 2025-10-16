"""
Script to read the number of positive keys in each dataset and print the recommended q-value for an adaptive filter
"""
from updated_classifiers import (obtain_raw_and_vectorized_keys, 
                                 EMBER_DATASET, URL_DATASET, SHALLA_DATASET, CAIDA_DATASET, POS_INDICATOR, CONFIG)
from math import log2, ceil
import argparse

# DATASETS = [EMBER_DATASET, URL_DATASET, SHALLA_DATASET, CAIDA_DATASET]
DATASETS = [CAIDA_DATASET]

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', action="store", dest="dataset", type=str, required=True,
                    help="the dataset to check")
results = parser.parse_args()

dataset = results.dataset

keys, vectorized_keys, labels = obtain_raw_and_vectorized_keys(dataset, create_data=False)
pos_keys = keys[labels == CONFIG[dataset][POS_INDICATOR]]
<<<<<<< HEAD
# print(f"{dataset} positive keys: {len(pos_keys)}, total keys is: {len(keys)}, recommended q-bits is {ceil(log2(len(pos_keys)))}")
=======
# print(f"{dataset} positive keys: {len(pos_keys)}, recommended q-bits is {ceil(log2(len(pos_keys)))}")
>>>>>>> b3c79e3ff3c6270405208edebdf97ed34fb87616
print(ceil(log2(len(pos_keys))))
exit(0)