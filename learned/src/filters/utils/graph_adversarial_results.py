import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# for the adversarial set, what we want to do is take the largest filter allotted for that
# dataset, then compare adabf, plbf, and aqf

# get a list of colors
colors = plt.get_cmap("tab10").colors

# first, plot the line for the plbf
adversarial_filename = "results/advers_results.csv"

adversarial_df = pd.read_csv(adversarial_filename)
freq = adversarial_df['freq']
sizes = adversarial_df['size']
fpr = adversarial_df['fpr']

filter = "adabf" if "adabf" in adversarial_filename else "plbf"

color_index = 0
# sort by size, graph freq vs fpr
for size in np.unique(sizes):
    # find the freq/fpr values that correspond to these sizes
    size_matching_rows = (adversarial_df['size'] == size)
    current_freq = freq[size_matching_rows]
    current_fpr = fpr[size_matching_rows]
    # for each adversary freq, find the median, 25th percentile, and 75th percentile
    freqs = []
    medians = []
    percent_25 = []
    percent_75 = []
    for freq in np.unique(current_freq):
        freqs.append(freq)
        freq_matching_rows = (current_freq == freq)
        freq_matching_fpr = current_fpr[freq_matching_rows]
        medians.append(np.median(freq_matching_fpr))
        percent_25.append(np.percentile(freq_matching_fpr, 25))
        percent_75.append(np.percentile(freq_matching_fpr, 75))
    # graph the median, 25th percentile, and 75th percentile, then fill in between the percentiles
    plt.plot(freqs, medians, label=str(f"{size} {filter}"), color=colors[color_index])
    # plt.fill_between(freq, percent_25, percent_75, alpha=0.2, color=colors[color_index])
    if color_index == len(colors) - 1:
        color_index = 0
    else:
        color_index += 1

adversarial_df = pd.read_csv("results/adabf_advers_results.csv")
freq = adversarial_df['freq']
sizes = adversarial_df['size']
fpr = adversarial_df['fpr']

filter = "adabf"

color_index = 1
# sort by size, graph freq vs fpr
for size in np.unique(sizes):
    # find the freq/fpr values that correspond to these sizes
    size_matching_rows = (adversarial_df['size'] == size)
    current_freq = freq[size_matching_rows]
    current_fpr = fpr[size_matching_rows]
    # for each adversary freq, find the median, 25th percentile, and 75th percentile
    freqs = []
    medians = []
    percent_25 = []
    percent_75 = []
    for freq in np.unique(current_freq):
        freqs.append(freq)
        freq_matching_rows = (current_freq == freq)
        freq_matching_fpr = current_fpr[freq_matching_rows]
        medians.append(np.median(freq_matching_fpr))
        percent_25.append(np.percentile(freq_matching_fpr, 25))
        percent_75.append(np.percentile(freq_matching_fpr, 75))
    # graph the median, 25th percentile, and 75th percentile, then fill in between the percentiles
    plt.plot(freqs, medians, label=str(f"{size} {filter}"), color=colors[color_index])
    # plt.fill_between(freq, percent_25, percent_75, alpha=0.2, color=colors[color_index])
    if color_index == len(colors) - 1:
        color_index = 0
    else:
        color_index += 1

plt.xlabel('Adversarial Query Frequency')
plt.ylabel('False-positive Rate')
plt.title('Adversarial Frequency vs FPR (Ember, 10M Queries Total)')
plt.legend(title="Filter Sizes")
plt.savefig('ember_10M_combined.pdf', bbox_inches='tight')
plt.clf()