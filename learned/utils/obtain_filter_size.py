"""
Helper script which prints a value (by default the size) from the adaptiveqf results for the given q and r values,
e.g. --column expected_fpr gives the target FPR that the other filters are matched to.
"""
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', action="store", dest="dataset", type=str, required=True,
                    help="the dataset to find")
parser.add_argument('--path', action="store", dest="path", type=str, required=True,
                    help="path to the combined results file written by the generic harnesses")
parser.add_argument('--q', action="store", dest="q", type=int, required=True,
                    help="the adaptiveqf q-value")
parser.add_argument('--r', action="store", dest="r", type=int, required=True,
                    help="the adaptiveqf r-value")
parser.add_argument('--column', action="store", dest="column", type=str, default="size",
                    help="the results column to print (e.g. size or expected_fpr)")

results = parser.parse_args()


# read the results file, which holds rows for every filter, and keep the adaptiveqf rows
all_results = pd.read_csv(results.path)
adaptive_results = all_results[all_results['filter'] == 'aqf']
# read the first row with the correct dataset, q, and r (params are written as "q=16;r=5")
matches = adaptive_results[(adaptive_results['dataset'] == results.dataset) &
                           (adaptive_results['params'] == f"q={results.q};r={results.r}")]

# print the corresponding value
if not matches.empty:
    first_match = matches.iloc[0]
    print(first_match[results.column])
else:
    print("no match found")