"""
Script which converts a binary file of raw query index integers into a csv with the index on each row.
"""
import argparse

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--only', action="store", dest="only", type=str, default=None, choices=["zipf", "unif"],
                    help="only convert this kind of query file (default: both)")
args = parser.parse_args()

dataset_info = {
    "url": {
        "dataset": "url",
        "rows": 162796,
        "filenames": ["zipf_10M_url", "unif_10M_url"]
    },
    "ember": {
        "dataset": "ember",
        "rows": 800000,
        "filenames": ["zipf_10M_ember", "unif_10M_ember"]
    },
    "shalla": {
            "dataset": "shalla",
            "rows": 3905928,
            "filenames": ["zipf_10M_shalla", "unif_10M_shalla"]
    },
    "caida": {
        "dataset": "caida",
        "rows": 8493974,
        "filenames": ["zipf_10M_caida", "unif_10M_caida"]
    }
}

for dataset in dataset_info.values():
    for query_set in dataset["filenames"]:
        if args.only and not query_set.startswith(args.only):
            continue
        print("creating: ", query_set)
        binary_name = f"../data/{query_set}.bin"
        f = open(binary_name, "r")
        elements = np.fromfile(f, dtype=np.uint32)
        if "unif" in query_set:
            # the elements were a uniformly random sample of the indices
            df = pd.DataFrame({"index": elements})
            df.to_csv(f"../data/updated_query_indices/hashed_{query_set}.csv", index=False)
        else:
            # the elements were zipfian-distributed ranks from 1 to the number of rows, so the lowest ranks are the most
            # frequent. Each rank is mapped to a row by a random permutation of the rows (seeded, so the queries can be
            # regenerated), so the most frequent queries are random rows rather than the first rows of the file, and
            # no two ranks share a row.
            pd.DataFrame(elements).to_csv(f"../data/updated_query_indices/unhashed_{query_set}.csv", index=False)
            rows = np.random.default_rng(0).permutation(dataset["rows"])
            hashed_values = rows[elements.astype(np.int64) - 1]
            df = pd.DataFrame({"index": hashed_values})
            df.to_csv(f"../data/updated_query_indices/hashed_{query_set}.csv", index=False)
        f.close()
