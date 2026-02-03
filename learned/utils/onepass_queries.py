"Writes a CSV file with query indices for one-pass queries."

import numpy as np

datasets = {"url": 162798, "ember": 800000, "shalla": 3905928, "caida": 8493974}

header = "index"
for dataset in datasets:
    num_queries = datasets[dataset]
    # write a random permutation of the numbers from 0 to num_queries-1
    arr = np.arange(num_queries)
    np.random.shuffle(arr)
    with open(f"../../data/updated_query_indices/onepass_{dataset}.csv", "w") as f:
        f.write(header + "\n")
        for i in range(arr.shape[0]):
            f.write(f"{arr[i]}\n")