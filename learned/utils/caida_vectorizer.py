"""
Script which takes the original dataset and converts it into (simple)
vectorized columns.
"""
import pandas as pd
import re
import ipaddress
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', action='store_true', dest='verbose',
                        help='verbose: whether or not to print progress')
results = parser.parse_args()
verbose = False if results.verbose is None else results.verbose

# Load data
if verbose:
    print("reading file")
    tqdm.pandas()
df = pd.read_csv("../data/20140619-140100.csv")

# Filter for TCP and UDP only
if verbose:
    print("filtering protocol")
df = df[df['Protocol'].isin(['TCP', 'UDP'])].copy()

# Rename and encode protocol as label
if verbose:
    print("assigning labels")
df.rename(columns={'Protocol': 'Label'}, inplace=True)
df['Label'] = (df['Label'] == 'UDP').astype(int)

# Convert IPs to integers
def ip_to_int(ip):
    try:
        return int(ipaddress.ip_address(ip))
    except:
        return 0

if verbose:
    print("converting ips")
    df['Source'] = df['Source'].progress_apply(ip_to_int)
    df['Destination'] = df['Destination'].progress_apply(ip_to_int)
else:
    df['Source'] = df['Source'].apply(ip_to_int)
    df['Destination'] = df['Destination'].apply(ip_to_int)

# TODO - consider parsing more information like so:
# Extract length from Info field
# def extract_length(info):
#     match = re.search(r'Len=(\d+)', info)
#     if match:
#         return int(match.group(1))
#     return 0
# if verbose:
#     print("parsing lengths")
#     df['Length'] = df['Info'].progress_apply(extract_length)
# else:
#     df['Length'] = df['Info'].apply(extract_length)

# Drop unused columns
if verbose:
    print("saving file")
df_final = df.drop(columns=['Info'])
df_final.to_csv('../data/caida.csv', index=False)