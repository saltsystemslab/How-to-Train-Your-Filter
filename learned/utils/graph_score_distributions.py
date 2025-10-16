"""
Helper script which plots how the key score distributions look for each query set.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

fig, axs = plt.subplots(1, 4, figsize=(12,2), layout='constrained')
axs[0].set_ylabel('Proportion of Keys')

filepaths = {'url': {'path': 'data/malicious_url_scores.csv', 'label_name': 'type', 'pos_indicator': 'malicious', 'score_name': 'prediction_score'}, 
             'ember': {'path': 'data/combined_ember_metadata.csv', 'label_name': 'label', 'pos_indicator': 1,'score_name': 'score'}, 
             'shalla': {'path': 'data/shalla_combined.csv', 'label_name': 'label', 'pos_indicator': 1, 'score_name': 'score'}, 
             'caida': {'path': 'data/backup/caida.csv', 'label_name': 'Label', 'pos_indicator': 1, 'score_name': 'score'}}

for i, dataset in enumerate(filepaths):
    df = pd.read_csv(filepaths[dataset]['path'])
    true_rows = df.loc[df[filepaths[dataset]['label_name']] == filepaths[dataset]['pos_indicator']]
    false_rows = df.loc[df[filepaths[dataset]['label_name']] != filepaths[dataset]['pos_indicator']]
    true_scores = true_rows[filepaths[dataset]['score_name']]
    false_scores = false_rows[filepaths[dataset]['score_name']]
    axs[i].hist([true_scores, false_scores], weights=[np.ones(len(true_scores)) / len(df), np.ones(len(false_scores)) / len(df)], bins=20, stacked=False, color=['blue', 'red'])
    axs[i].set_xlabel('Score')
    axs[i].set_title(dataset)
    axs[i].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[i].set_xticks([0.25, 0.5, 0.75, 1])
    
fig.legend(['Non-keys', 'Keys'], loc="outside center right")
plt.savefig('figures/combined_score_distributions.pdf')
plt.clf()