"""
Helper script which prints the size of a filter given an adaptiveqf's q and r values
"""
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', action="store", dest="dataset", type=str, required=True,
                    help="the dataset to find")
parser.add_argument('--path', action="store", dest="path", type=str, required=True,
                    help="path to relevant results file")
parser.add_argument('--q', action="store", dest="q", type=int, required=True,
                    help="the adaptiveqf q-value")
parser.add_argument('--r', action="store", dest="r", type=int, required=True,
                    help="the adaptiveqf r-value")

results = parser.parse_args()


# read the results file
adaptive_results = pd.read_csv(results.path)
# read the first row with the correct dataset, q, and r
matches = adaptive_results[(adaptive_results['dataset'] == results.dataset) &
                               (adaptive_results['q'] == results.q) &
                               (adaptive_results['r'] == results.r)]

# print the corresponding size
if not matches.empty:
    first_match = matches.iloc[0]
    print(first_match['size'])
else:
    print("no match found")