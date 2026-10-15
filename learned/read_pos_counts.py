"""
Script to read the number of positive keys in each dataset and print the recommended q-value for an adaptive filter
"""
from math import log2, ceil
import argparse
import os

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', action="store", dest="dataset", type=str, required=True,
                    help="the dataset to check")
parser.add_argument('--fraction', action="store", dest="fraction", type=float, default=1.0,
                    help="the fraction of the positive keys the filter holds (e.g. 0.5 for the shifting workload)")
results = parser.parse_args()

dataset = results.dataset

# the labels exported for the learned filters (see utils/export_cascaded_inputs.py) give the count without
# loading the dataset
labels_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "cascaded", dataset, "labels.bin")
if os.path.exists(labels_path):
    with open(labels_path, "rb") as f:
        rows, cols = np.fromfile(f, dtype=np.uint64, count=2)
        num_pos = int((np.fromfile(f, dtype=np.float32, count=int(rows * cols)) == 1).sum())
else:
    # loading the dataset needs the learned filters' code (and data, for ember)
    from updated_classifiers import obtain_raw_and_vectorized_keys, POS_INDICATOR, CONFIG
    keys, vectorized_keys, labels = obtain_raw_and_vectorized_keys(dataset, create_data=False)
    num_pos = len(keys[labels == CONFIG[dataset][POS_INDICATOR]])
print(ceil(log2(int(num_pos * results.fraction))))
exit(0)