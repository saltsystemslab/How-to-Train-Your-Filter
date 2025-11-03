# takes a csv and confirms that all the rows are not duplicates.
import pandas as pd
# df = pd.read_csv("../data/malicious_url_scores.csv")
df = pd.read_csv("../data/shalla_combined.csv")
urls = df["url"]
print("length of data: ", len(df))
print("number of unique urls: ", len(set(urls)))