# This file is used to convert a binary file with query indexes into a csv with the index on each row.

import numpy as np
import pandas as pd
import random

a = 584141
b = 108881
p = 479001599

def hash_to_range(elements, a, b, p, max):
    return (a * elements + b) % p % max

dataset_info = {
    "news": {
        "dataset": "news",
        "rows": 35919,
        "p": 37199,
        "filenames": ["zipf_10M_news", "zipf_100M_news", "unif_10M_news", "unif_100M_news"]
    },
    "url": {
        "dataset": "url",
        "rows": 162798,
        "p": 193939,
        "filenames": ["zipf_10M_url", "zipf_100M_url", "unif_10M_url", "unif_100M_url"]
    },
    "ember": {
        "dataset": "ember",
        "rows": 800000,
        "p": 993319,
        "filenames": ["zipf_10M_ember", "zipf_100M_ember", "unif_10M_ember", "unif_100M_ember"]
    }
}

querysets = ["zipf_10M_ember", "zipf_100M_ember", "unif_10M_ember", "unif_100M_ember",
             "zipf_10M_url", "zipf_100M_url", "unif_10M_url", "unif_100M_url",
             "zipf_10M_news", "zipf_100M_news", "unif_10M_news", "unif_100M_news"]

for dataset in dataset_info.values():
    for query_set in dataset["filenames"]:
        print("creating: ", query_set)
        binary_name = f"data/{query_set}.bin"
        f = open(binary_name, "r")
        elements = np.fromfile(f, dtype=np.uint32)
        if "unif" in query_set:
            # the elements were a uniformly random sample of the indices
            df = pd.DataFrame({"index": elements})
            df.to_csv(f"data/updated_query_indices/hashed_{query_set}.csv", index=False)
        else:
            # the elements were a zipfian-distributed sample of the indicies, biased towards lower indices.
            # we need to hash the elements so that random elements have higher frequencies rather than just the lower indices.
            pd.DataFrame(elements).to_csv(f"data/updated_query_indices/unhashed_{query_set}.csv", index=False)
            hashed_values = hash_to_range(elements, a, b, dataset["p"], dataset["rows"])
            df = pd.DataFrame({"index": hashed_values})
            df.to_csv(f"data/updated_query_indices/hashed_{query_set}.csv", index=False)
        f.close()


# for queryset in querysets:
#     binary_name = "data/" + queryset + ".bin"
#     f = open(binary_name, "r") 
#     elements = np.fromfile(f, dtype=np.uint32)
#     if "unif" in queryset:
#         df.to_csv("hashed_" + queryset + ".csv")
#     else:
#         pd.DataFrame(elements).to_csv("data/query_indices/unhashed_" + queryset + ".csv")
#         max_range = elements.shape[0]
#         if "news" in queryset:
#             max_range = 35919
#         elif "url" in queryset:
#             max_range = 162798
#         elif "ember" in queryset:
#             max_range = 800000
#         hashed_values = hash_to_range(elements, a, b, p, max_range)
#         df = pd.DataFrame(hashed_values)
#         df.sort_values(by=df.columns[0], inplace=True) # TODO - consider removing this since it makes cache effects a little weird
#         df.to_csv("data/query_indices/hashed_" + queryset + ".csv")
    # f.close()