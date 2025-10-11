"""
Helper script which plots how the query distribution looks for specific query sets.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

unhashed_querysets = ["unhashed_zipf_10M_ember"]
hashed_querysets = ["hashed_zipf_10M_ember"]

for unhashed_queryset in unhashed_querysets:
    plt.figure(figsize=(3.5,3.5))
    df = pd.read_csv('data/updated_query_indices/' + unhashed_queryset + '.csv')
    # indexes = df.iloc[:, 1]
    plt.hist(df, bins=50, log=True)
    plt.xlabel('Index')
    plt.ylabel('Frequency')
    plt.xticks([0, 200000, 400000, 600000])
    plt.tight_layout()
    plt.savefig('figures/queries_' + unhashed_queryset + '.pdf')
    plt.clf()

for hashed_queryset in hashed_querysets:
    plt.figure(figsize=(3.5,3.5))
    df = pd.read_csv('data/updated_query_indices/' + hashed_queryset + '.csv')
    # indexes = df.iloc[:, 1]
    plt.hist(df, bins=50, log=True)
    plt.xlabel('Index')
    plt.ylabel('Frequency')
    plt.xticks([0, 200000, 400000, 600000])
    plt.tight_layout()
    plt.savefig('figures/queries_' + hashed_queryset + '.pdf')
    plt.clf()