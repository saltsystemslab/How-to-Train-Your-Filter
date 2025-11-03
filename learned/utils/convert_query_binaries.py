"""
Script which converts a binary file of raw query index integers into a csv with the index on each row.
"""
import numpy as np
import pandas as pd
import random

a = 584141
b = 108881
p = 479001599

def hash_to_range(elements, a, b, p, max):
    return (a * elements + b) % p % max

dataset_info = {
    "url": {
        "dataset": "url",
        "rows": 162798,
        "p": 193939,
        "filenames": ["zipf_10M_url", "unif_10M_url"]
    },
    "ember": {
        "dataset": "ember",
        "rows": 800000,
        "p": 993319,
        "filenames": ["zipf_10M_ember", "unif_10M_ember"]
    },
    "shalla": {
            "dataset": "shalla",
            "rows": 3905928,
            "p": 4477457,
            "filenames": ["zipf_10M_shalla", "unif_10M_shalla"]
    },
    "caida": {
        "dataset": "caida",
        "rows": 8493975,
        "p": 7774777,
        "filenames": ["zipf_10M_caida", "unif_10M_caida"]
    }
}

for dataset in dataset_info.values():
    for query_set in dataset["filenames"]:
        print("creating: ", query_set)
        binary_name = f"../data/{query_set}.bin"
        f = open(binary_name, "r")
        elements = np.fromfile(f, dtype=np.uint32)
        if "unif" in query_set:
            # the elements were a uniformly random sample of the indices
            df = pd.DataFrame({"index": elements})
            df.to_csv(f"../data/updated_query_indices/hashed_{query_set}.csv", index=False)
        else:
            # the elements were a zipfian-distributed sample of the indicies, biased towards lower indices.
            # we need to hash the elements so that random elements have higher frequencies rather than just the lower indices.
            pd.DataFrame(elements).to_csv(f"../data/updated_query_indices/unhashed_{query_set}.csv", index=False)
            hashed_values = hash_to_range(elements, a, b, dataset["p"], dataset["rows"])
            df = pd.DataFrame({"index": hashed_values})
            df.to_csv(f"../data/updated_query_indices/hashed_{query_set}.csv", index=False)
        f.close()
