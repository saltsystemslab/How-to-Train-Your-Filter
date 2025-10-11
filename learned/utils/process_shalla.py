"""
Script which takes the Shalla dataset and a dataset of top 1M websites and creates
a malicious/benign dataset.

The 1M websites are taken from Cisco Umbrella 
(https://s3-us-west-1.amazonaws.com/umbrella-static/index.html)
"""
import pandas as pd

SHALLA_PATH = "data/shalla.txt"
TOP_1M_PATH = "data/top-1m.csv"

malicious_urls = pd.read_csv(SHALLA_PATH, header=None, names=["url"])
top_1m_df = pd.read_csv(TOP_1M_PATH, header=None, names=["url"])

# filter out malicious urls from the top 1M sites, if any
benign_mask = ~top_1m_df["url"].isin(malicious_urls["url"])
benign_urls = top_1m_df[benign_mask]

malicious_urls.loc[:, "label"] = 1
benign_urls.loc[:, "label"] = 0

combined_data = pd.concat([benign_urls[["url", "label"]], malicious_urls[["url", "label"]]], ignore_index=True)

combined_data.to_csv("data/shalla_combined.csv", index=False)