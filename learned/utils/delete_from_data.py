"""
Helper script which removes all rows corresponding to a specific dataset and/or filter
from a (results) CSV file.
"""
import pandas as pd
import argparse
import os
from filelock import FileLock

parser = argparse.ArgumentParser()
parser.add_argument("--filepath", action="store", dest="filepath", type=str, required=True,
                    help="file to delete from")
parser.add_argument("--dataset", action="store", dest="dataset", type=str, required=False,
                    help="dataset rows to delete")
parser.add_argument("--filter", action="store", dest="filter", type=str, required=False,
                    help="filter data to delete")

results = parser.parse_args()
filepath = results.filepath
dataset = results.dataset
filter = results.filter

lock = FileLock(filepath + ".lock")
with lock:
    if os.path.exists(filepath):
        results_df = pd.read_csv(filepath)
    else:
        print(f"File {filepath} doesn't exist")
        exit(1)
    if dataset is not None and dataset not in results_df['dataset'].values:
        print(f"Dataset {dataset} not found in {filepath}")
        exit(1)
    if filter is not None and filter not in results_df['filter'].values:
        print(f"Filter {filter} not found in {filepath}")
        exit(1)
    condition = pd.Series(True, index=results_df.index)
    if filter is not None:
        condition &= results_df['filter'] != filter
    if dataset is not None:
        condition &= results_df['dataset'] != dataset
    results_df = results_df[condition]
    results_df.to_csv(filepath, index=False)
