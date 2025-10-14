"""
Functions which plot the results from learned and adaptive filter experiments as they appear in the paper.
When run as a main script, generates all the plots.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statistics import median
import math
import textwrap
from matplotlib.ticker import FormatStrFormatter


FILTER_COLORS = {"plbf": "blue", "adabf": "green", "aqf": "orange"}
FILTER_MARKERS = {"plbf": "D", "adabf": "^", "aqf": "o"}
COLORS = plt.get_cmap('tab20').colors
TIMING_COMP = ['Score Inference', 'Filter Query', 'Reverse Map Adapt', 'Filter Inserts',
                    'Model Training', 'Threshold Finding', 'Reverse Map Updates']
COMP_COLORS = {comp: COLORS[i] for i, comp in enumerate(TIMING_COMP)}

dataset_sizes = {"shalla": [4280640, 4807488, 5334336, 5861184, 6388032, 6914880, 7441728], 
                 "ember": [539890, 606338, 672786, 739234, 805682, 872130, 938578], 
                 "url": [69160, 77672, 86184, 94696, 103208, 111720, 120232],
                 "caida": [2144675, 2408635, 2672595, 2936555, 3200515, 3464475, 3728435]}

one_pass_sizes = {"url": 162798, "shalla": 3905928, "ember": 800000, "caida": 8493974}

dataset_pos = {"ember": 800000, "url": 55681, "shalla": 2926705, "caida": 1196194}

plt.rcParams['text.usetex'] = False

def plot_fpr_space_tradeoff(learned_filename: str, adaptive_filename: str, num_queries=[10000000], learned_filters=None):
    # first, grab the data from the learned results, and separate the data based on the filter
    learned_df = pd.read_csv(learned_filename)
    adaptive_df = pd.read_csv(adaptive_filename)
    if learned_filters is None:
        learned_filters = {"plbf", "adabf"}
    unique_datasets = ["url", "ember", "shalla", "caida"]
    unique_query_dists = set(learned_df['query_dist'])
    
    
    for query_dist in unique_query_dists:
        # each type of query distribution has its own plot
        fig, axs = plt.subplots(1, 4, figsize=(12,2), layout='constrained')
        current_count = 0
        axs[0].set_ylabel("False Positive Rate")
        overall_max = None
        overall_min = None
        for dataset in unique_datasets:
            axs[current_count].set_title(dataset)
            # plt.figure(figsize=(3,3))
            unique_sizes = dataset_sizes[dataset]
            if query_dist == "onepass":
                num_queries = [one_pass_sizes[dataset]]
            else:
                num_queries = [10000000]
            for query_num in num_queries:
                data_for_dataset = adaptive_df[adaptive_df['dataset'] == dataset]
                data_for_num_queries = data_for_dataset[data_for_dataset['num_queries'] == query_num]
                adaptive_data_for_query_dist = data_for_num_queries[data_for_num_queries['query_dist'] == query_dist]
                # now for each filter size, get the median, min, and max
                for filter in learned_filters:
                    if filter not in FILTER_COLORS.keys():
                        print(f"{filter} not implemented...")
                        continue
                    data_for_dataset = learned_df[learned_df['dataset'] == dataset]
                    data_for_num_queries = data_for_dataset[data_for_dataset['num_queries'] == query_num]
                    data_for_query_dist = data_for_num_queries[data_for_num_queries['query_dist'] == query_dist]
                    data_for_filter = data_for_query_dist[data_for_query_dist['filter'] == filter]
                    if len(data_for_filter) == 0:
                        print(f"No data found for {filter}-{dataset}-{size}...")
                        continue
                    # now for each filter size, get the median, min, and max 
                    sizes = []
                    meds = []
                    mins = []
                    maxes = []
                    for size in unique_sizes:
                        current_size_data = data_for_filter[data_for_filter['bytes'] == size]
                        if len(data_for_filter) == 0:
                            print(f"No data found for {filter}-{dataset}-{size}...")
                            continue
                        fprs = current_size_data['fpr']
                        if len(fprs) == 0:
                            print(f"No data found for {filter}-{dataset}-{size}...")
                            continue
                        sizes.append(size / dataset_pos[dataset] * 8)
                        meds.append(median(fprs))
                        mins.append(min(fprs))
                        maxes.append(max(fprs))
                        overall_max = max(fprs) if (overall_max is None or overall_max < max(fprs)) else overall_max
                        overall_min = max(fprs) if min(fprs) != 0 and (overall_min is None or overall_min > min(fprs)) else overall_min
                    # now plot the data
                    axs[current_count].plot(sizes, meds, label=(filter if current_count == 0 else ""), color=FILTER_COLORS[filter], marker=FILTER_MARKERS[filter])
                    axs[current_count].fill_between(sizes, mins, maxes, alpha=0.2, color=FILTER_COLORS[filter])
                # now process the adaptive filter
                
                sizes = []
                meds = []
                mins = []
                maxes = []

                for size in unique_sizes:
                    current_size_data = adaptive_data_for_query_dist[adaptive_data_for_query_dist['size'] == size]
                    if current_size_data.empty:
                        print(f"current size of data: {len(current_size_data)}")
                        print(f"[EMPTY] {dataset}, {query_num}, {size}, {query_dist}")
                    fprs = current_size_data['fpr']
                    sizes.append(size / dataset_pos[dataset] * 8)
                    meds.append(median(fprs))
                    mins.append(min(fprs))
                    maxes.append(max(fprs))
                    overall_max = max(fprs) if (overall_max is None or overall_max < max(fprs)) else overall_max
                    overall_min = max(fprs) if min(fprs) != 0 and (overall_min is None or overall_min > min(fprs)) else overall_min
                
                # now plot the data
                axs[current_count].plot(sizes, meds, label=('adaptiveqf' if current_count == 0 else ""), color=FILTER_COLORS['aqf'], marker=FILTER_MARKERS['aqf'])
                axs[current_count].fill_between(sizes, mins, maxes, alpha=0.2, color=FILTER_COLORS['aqf'])
                axs[current_count].set_xlabel('Bits per Key')
                axs[current_count].set_yscale('log')
            # after processing a dataset, move on to the next subplot
            current_count += 1
        # go back through all the graphs and set their y-limits
        for ax in axs.flat:
            ax.set_ylim(math.pow(10, math.floor(math.log10(overall_min))), 
                        math.pow(10, math.ceil(math.log10(overall_max))))
        # label and save the data
        fig.legend(loc='outside center right')
        plt.savefig(f'figures/combined_fpr_{query_dist}_{query_num/1000000 if query_dist != "onepass" else ""}M.pdf')
        plt.clf()

def plot_model_degradation(learned_filename: str, size=None):
    learned_df = pd.read_csv(learned_filename)
    datasets = ["url", "ember", "shalla", "caida"]
    learned_filters = ["plbf", "adabf"]
    set_sizes = [0.2, 0.4, 0.6, 0.8] # TODO - wait for results to come in for 0.8
    fig, axs = plt.subplots(1, 4, figsize=(12,2), layout='constrained')
    current_count = 0
    axs[0].set_ylabel("False Positive Rate")
    for dataset in datasets:
        axs[current_count].set_title(dataset)
        size_to_process = min(dataset_sizes[dataset])
        for learned_filter in learned_filters:
            if learned_filter not in FILTER_COLORS.keys():
                print(f"{learned_filter} not implemented...")
                continue
            data_for_dataset = learned_df[learned_df['dataset'] == dataset]
            data_for_num_queries = data_for_dataset[data_for_dataset['num_queries'] == 10000000]
            data_for_query_dist = data_for_num_queries[data_for_num_queries['query_dist'] == "unif"]
            data_for_filter = data_for_query_dist[data_for_query_dist['filter'] == learned_filter]
            current_data = data_for_filter[data_for_filter['bytes'] == size_to_process]
            if len(current_data) == 0:
                print(f"No data found for {learned_filter}...")
                continue
            # now for each filter size, get the median, min, and max 
            meds = []
            mins = []
            maxes = []
            for train_size in set_sizes:
                current_size_data = data_for_filter[data_for_filter['train_set_size'] == train_size]
                fprs = current_size_data['fpr']
                if len(fprs) == 0:
                    print(f"Empty: {dataset}, {learned_filter}, {train_size}")
                meds.append(median(fprs))
                mins.append(min(fprs))
                maxes.append(max(fprs))
            # now plot the data
            axs[current_count].plot(set_sizes, meds, label=(learned_filter if current_count == 0 else ""), color=FILTER_COLORS[learned_filter], marker=FILTER_MARKERS[learned_filter])
            axs[current_count].fill_between(set_sizes, mins, maxes, alpha=0.2, color=FILTER_COLORS[learned_filter])
            axs[current_count].set_xlabel('Training Set Proportion')
            axs[current_count].set_yscale('log')
            axs[current_count].set_yticks([0.0001, 0.001, 0.01, 0.1])
        current_count += 1
    fig.legend(loc='outside center right')
    plt.savefig(f'figures/combined_degrad_10M_.pdf')
    plt.clf()

ADVERSARIAL_SIZES = {"url": 77672, "ember": 606338, "shalla": 4807488, "caida": 2408635}
# for this one, we want to try the best size for each filter that still results in a false-positive...
def plot_adversarial(learned_filename: str, adaptive_filename: str, learned_filters=None):
    learned_df = pd.read_csv(learned_filename)
    adaptive_df = pd.read_csv(adaptive_filename)
    if learned_filters is None:
        learned_filters = {"plbf", "adabf"}
    unique_datasets = ["url", "ember", "shalla", "caida"]
    unique_num_queries = [10000000]
    fig, axs = plt.subplots(1, 4, figsize=(12,2), layout='constrained')
    axs[0].set_ylabel('False Positive Rate')
    overall_max = None
    overall_min = None
    current_count = 0
    for dataset in unique_datasets:
        axs[current_count].set_title(dataset)
        for num_queries in unique_num_queries:
            current_size = ADVERSARIAL_SIZES[dataset]
            # find the maximum size such that there is no row where 'fpr' is 0
            unique_freqs = sorted(set(learned_df['freq']))
            for filter in learned_filters:
                current_filter_data = learned_df[
                    (learned_df['dataset'] == dataset) &
                    (learned_df['num_queries'] == num_queries) &
                    (learned_df['filter'] == filter)
                ]
                data_for_size = current_filter_data[current_filter_data['bytes'] == current_size]
                if data_for_size.empty:
                    print(f"[EMPTY] {dataset}, {filter}, {num_queries}, {current_size}")
                # now, create row-aligned arrays of freq and fpr
                freqs = []
                meds = []
                mins = []
                maxes = []
                for freq in unique_freqs:
                    data_for_freq = data_for_size[data_for_size['freq'] == freq]
                    fprs = data_for_freq['fpr']
                    freqs.append(freq)
                    meds.append(median(fprs))
                    mins.append(min(fprs))
                    maxes.append(max(fprs))
                    overall_max = max(fprs) if (overall_max is None or overall_max < max(fprs)) else overall_max
                    overall_min = max(fprs) if min(fprs) != 0 and (overall_min is None or overall_min > min(fprs)) else overall_min
                axs[current_count].plot(freqs, meds, label=(filter if current_count == 0 else ""), color=FILTER_COLORS[filter], marker=FILTER_MARKERS[filter])
                axs[current_count].fill_between(freqs, mins, maxes, alpha=0.2, color=FILTER_COLORS[filter])
            adaptive_data = adaptive_df[
                (adaptive_df['dataset'] == dataset) &
                (adaptive_df['num_queries'] == num_queries) &
                (adaptive_df['size'] == (304848 if dataset == "url" else current_size))
            ]
            freqs = []
            meds = []
            mins = []
            maxes = []
            for freq in unique_freqs:
                data_for_freq = adaptive_data[adaptive_data['freq'] == freq]
                if data_for_freq.empty:
                    print(f"[EMPTY] {dataset}, adaptiveqf, {num_queries}, {current_size}")
                    continue
                fprs = data_for_freq['fpr']
                freqs.append(freq)
                meds.append(median(fprs))
                mins.append(min(fprs))
                maxes.append(max(fprs))
                overall_max = max(fprs) if (overall_max is None or overall_max < max(fprs)) else overall_max
                overall_min = max(fprs) if min(fprs) != 0 and (overall_min is None or overall_min > min(fprs)) else overall_min
            axs[current_count].set_xlabel('Adversarial Frequency')
            axs[current_count].plot(freqs, meds, label=('adaptiveqf' if current_count == 0 else ""), color=FILTER_COLORS['aqf'], marker=FILTER_MARKERS['aqf'])
            axs[current_count].fill_between(freqs, mins, maxes, alpha=0.2, color=FILTER_COLORS['aqf'])
            axs[current_count].set_yscale('log')
        current_count += 1
    # label and save the plot
    for ax in axs.flat:
        ax.set_ylim(math.pow(10, math.floor(math.log10(overall_min))), 
                    math.pow(10, math.ceil(math.log10(overall_max))))
    # plt.title(f'Adversarial Query FPR-Space Tradeoff on {dataset} ({num_queries / 1000000}M Queries)')
    fig.legend(loc="outside center right")
    plt.savefig(f'figures/combined_fpr_adversarial_{num_queries/1000000}M.pdf')
    plt.clf()
    
construction_sizes = {"url": 69160, "ember": 539890, "shalla": 4280640, "caida": 2144675}
def plot_construction_times(learned_filename: str, adaptive_filename: str, learned_filters=None):
    # first establish the categories for the construction
    learned_df = pd.read_csv(learned_filename)
    adaptive_df = pd.read_csv(adaptive_filename)
    learned_categories = ['Model Training', 'Threshold Finding', 'Filter Inserts']
    adaptive_categories = ['Filter Inserts', 'Reverse Map Updates']
    all_categories = sorted(list(set(learned_categories + adaptive_categories)))
    if learned_filters is None:
        learned_filters = {"plbf", "adabf"}

    # now, for each dataset, track the different times...
    # these will map dataset => filter dict, which in turn stores filter => height/prop
    dataset_results = dict()

    # now go through each dataset and obtain the results for each kind of filter
    for dataset in construction_sizes:
        plbf_data = learned_df[
            (learned_df['dataset'] == dataset) &
            (learned_df['bytes'] == construction_sizes[dataset]) &
            (learned_df['filter'] == 'plbf')
        ]
    
        if len(plbf_data) != 0:
            bar = {'Model Training': np.median(plbf_data['construct_time']) + np.median(plbf_data['train_time']) + np.median(plbf_data['initial_scores']),
                   'Threshold Finding': np.median(plbf_data['segment_division']) + np.median(plbf_data['t_f_finding']),
                   'Filter Inserts': np.median(plbf_data['bloom_init']) + np.median(plbf_data['region_finding']) + np.median(plbf_data['filter_inserts'])}
            plbf_bar_heights = [bar.get(part, 0) for part in all_categories]
            plbf_total_height = np.sum(plbf_bar_heights)
            plbf_bar_prop = [part / plbf_total_height for part in plbf_bar_heights]
        else:
            print("couldn't find data for plbf on ", dataset)
            exit(1)
        adabf_data = learned_df[
                (learned_df['dataset'] == dataset) &
                (learned_df['bytes'] == construction_sizes[dataset]) &
                (learned_df['filter'] == 'adabf')
            ]
        if len(adabf_data) != 0:
            bar = {'Model Training': median(adabf_data['construct_time']) + median(adabf_data['train_time']) + median(adabf_data['initial_scores']),
                   'Threshold Finding': median(adabf_data['region_finding']),
                   'Filter Inserts': median(adabf_data['bloom_init']) + median(adabf_data['filter_inserts'])}
            adabf_bar_heights = [bar.get(part, 0) for part in all_categories]
            adabf_total_height = sum(adabf_bar_heights)
            adabf_bar_prop = [part / adabf_total_height for part in adabf_bar_heights]
        else:
            print("couldn't find data for adabf on ", dataset)
            exit(1)
        adaptive_data = adaptive_df[
                (adaptive_df['dataset'] == dataset) &
                (adaptive_df['size'] == construction_sizes[dataset])
        ]
        if len(adaptive_data) != 0:
            bar = {'Filter Inserts': median(adaptive_df['insert_time']) + median(adaptive_df['alloc_time']), 'Reverse Map Updates': median(adaptive_df['amortized_map_insert']) * dataset_pos[dataset]}
            adaptive_bar_heights = [bar.get(part, 0) for part in all_categories]  
            adaptive_total_height = sum(adaptive_bar_heights)
            adaptive_bar_prop = [part / adaptive_total_height for part in adaptive_bar_heights]
        else:
            print("couldn't find data for aqf on ", dataset)
            exit(1)
        dataset_results[dataset] = {
            'plbf': {'height': plbf_total_height, 'prop': plbf_bar_prop, 'bar': plbf_bar_heights},
            'adabf': {'height': adabf_total_height, 'prop': adabf_bar_prop, 'bar': adabf_bar_heights},
            'adaptiveqf': {'height': adaptive_total_height, 'prop': adaptive_bar_prop, 'bar': adaptive_bar_heights}}

    fig, axs = plt.subplots(1, 4, figsize=(5, 2), layout='constrained')
    fig.set_constrained_layout_pads(w_pad=0.001, h_pad=0.05, wspace=0.05, hspace=0.1)
    axs[0].set_ylabel('Construct Time (ks)')
    current_count = 0
    width = 0.6
    for dataset in construction_sizes:
        x = np.arange(3)
        axs[current_count].set_title(dataset, fontsize='large')
        axs[current_count].margins(y=0.1, x=0.1)
        bottom = np.zeros(3)
        plbf_bar_prop = dataset_results[dataset]['plbf']['bar']
        adabf_bar_prop = dataset_results[dataset]['adabf']['bar']
        adaptiveqf_bar_prop = dataset_results[dataset]['adaptiveqf']['bar']
        for (i, part) in enumerate(all_categories):
            values = [plbf_bar_prop[i] / 1000, adabf_bar_prop[i] / 1000, adaptiveqf_bar_prop[i] / 1000]
            axs[current_count].bar(x, values, edgecolor='black', bottom=bottom, label=(textwrap.fill(part, 12) if current_count == 0 else ""), color=COMP_COLORS[part], width=width)
            bottom += values
        for i, height in enumerate(bottom):
            axs[current_count].text(
                x[i],
                height,
                f"{height:.1e}" if height < 1 else f'{height:.1f}',
                ha='center',
                va='bottom',
                fontsize='xx-small',
            )
        axs[current_count].set_xticks(x, ['plbf', 'adabf', 'aqf'], fontsize='small')
        for label in axs[current_count].get_yticklabels():
            label.set_fontsize('x-small')
        current_count += 1
    fig.legend(loc="lower center", ncol=len(all_categories), bbox_to_anchor=(0.5, -0.25), fontsize='medium', handlelength=1)
    plt.savefig(f'figures/combined_const_with_prop.pdf', bbox_inches='tight')
    plt.clf()

    # now for the construction, need to keep the order of the datasets
    # and keep track of the bar heights across datasets
    x = np.arange(len(construction_sizes))
    plbf_heights = [dataset_results[dataset]['plbf']['height'] for dataset in construction_sizes]
    adabf_heights = [dataset_results[dataset]['adabf']['height'] for dataset in construction_sizes]
    adaptive_heights = [dataset_results[dataset]['adaptiveqf']['height'] for dataset in construction_sizes]
    width = 0.15
    plt.figure(figsize=(12,2))
    plt.bar(x-0.2, plbf_heights, width, color=FILTER_COLORS['plbf'], label='plbf')
    plt.bar(x, adabf_heights, width, color=FILTER_COLORS['adabf'], label='adabf')
    plt.bar(x+0.2, adaptive_heights, width, color=FILTER_COLORS['aqf'], label='adaptiveqf')
    plt.xticks(x, [dataset for dataset in construction_sizes])
    plt.ylabel('Construct Time (s)')
    plt.yscale('log')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig("figures/combined_overall_construction.pdf", bbox_inches='tight')
    plt.clf()

    x = np.arange(3) # set up a bar for each filter
    width = 0.6
    fig, axs = plt.subplots(1, 4, figsize=(12,2), layout='constrained')
    current_count = 0        
    axs[0].set_ylabel("Const Time Prop")
    for dataset in construction_sizes:
        axs[current_count].set_title(dataset)
        bottom = np.zeros(3)
        plbf_bar_prop = dataset_results[dataset]['plbf']['prop']
        adabf_bar_prop = dataset_results[dataset]['adabf']['prop']
        adaptiveqf_bar_prop = dataset_results[dataset]['adaptiveqf']['prop']
        for (i, part) in enumerate(all_categories):
            values = [plbf_bar_prop[i], adabf_bar_prop[i], adaptiveqf_bar_prop[i]]
            axs[current_count].bar(x, values, bottom=bottom, label=(part if current_count == 0 else ""), color=COMP_COLORS[part], width=width)
            bottom += values
        axs[current_count].set_xticks(x, ['plbf', 'ada-bf', 'adaptiveqf'])
        axs[current_count].set_xlim(-0.5, 2.5) # may need to get rid of this later?
        current_count += 1  
    fig.legend(loc='outside center right')
    plt.savefig(f'figures/combined_constructiontime_prop.pdf')  
    plt.clf() 

def plot_query_times(learned_filename: str, adaptive_filename: str):
    # first establish the categories for the queries
    plbf_df = pd.read_csv(learned_filename)
    adaptive_df = pd.read_csv(adaptive_filename)
    learned_categories = ['Score Inference', 'Filter Query']
    adaptiveqf_categories = ['Filter Query', 'Reverse Map Updates']
    all_categories = sorted(list(set(learned_categories + adaptiveqf_categories)))

    # now, for each dataset, track the different times...
    # these will map dataset => filter dict, which in turn stores filter => height/prop
    dataset_results = dict()
    # now go through each dataset and obtain the results for each kind of filter
    for dataset in construction_sizes:
        plbf_data = plbf_df[
                (plbf_df['dataset'] == dataset) &
                (plbf_df['bytes'] == construction_sizes[dataset]) & 
                (plbf_df['filter'] == 'plbf')
            ]
        if len(plbf_data) != 0:
            bar = {'Score Inference': plbf_data['amort_score_time'].iloc[0],
                    'Filter Query': plbf_data['amort_back_filter_time'].iloc[0] + plbf_data['amort_region_time'].iloc[0]}
            plbf_bar_heights = [bar.get(part, 0) for part in all_categories]
            plbf_total_height = sum(plbf_bar_heights)
            plbf_bar_prop = [part / plbf_total_height for part in plbf_bar_heights]
        else:
            print("couldn't find data for plbf on ", dataset)
            exit(1)
        adabf_data = plbf_df[
                (plbf_df['dataset'] == dataset) &
                (plbf_df['bytes'] == construction_sizes[dataset]) & 
                (plbf_df['filter'] == 'adabf')
            ]
        if len(adabf_data) != 0:
            bar = {'Score Inference': adabf_data['amort_score_time'].iloc[0], 
                'Filter Query': adabf_data['amort_back_filter_time'].iloc[0] + adabf_data['amort_region_time'].iloc[0]}
            adabf_bar_heights = [bar.get(part, 0) for part in all_categories]
            adabf_total_height = sum(adabf_bar_heights)
            adabf_bar_prop = [part / adabf_total_height for part in adabf_bar_heights]
        else:
            print("couldn't find data for adabf on ", dataset)
            exit(1)
        adaptive_data = adaptive_df[
                        (adaptive_df['dataset'] == dataset) &
                        (adaptive_df['size'] == construction_sizes[dataset])
                ]
        if len(adaptive_data) != 0:
            bar = {'Filter Query': adaptive_data['amortized_query'].iloc[0], 
                'Reverse Map Updates': adaptive_data['amortized_adapt'].iloc[0]}
            adaptiveqf_bar_heights = [bar.get(part, 0) for part in all_categories]
            adaptiveqf_total_height = sum(adaptiveqf_bar_heights)
            adaptiveqf_bar_prop = [part / adaptiveqf_total_height for part in adaptiveqf_bar_heights]
        else:
            print("couldn't find data for aqf on ", dataset)
            exit(1)
        dataset_results[dataset] = {
            'plbf': {'height': plbf_total_height, 'prop': plbf_bar_prop, 'bar': plbf_bar_heights},
            'adabf': {'height': adabf_total_height, 'prop': adabf_bar_prop, 'bar': adabf_bar_heights},
            'adaptiveqf': {'height': adaptiveqf_total_height, 'prop': adaptiveqf_bar_prop, 'bar': adaptiveqf_bar_heights}}


    # now for the overall query times, need to keep the order of the datasets
    # and keep track of the bar heights across datasets
    x = np.arange(len(construction_sizes))
    plbf_heights = [dataset_results[dataset]['plbf']['height'] for dataset in construction_sizes]
    adabf_heights = [dataset_results[dataset]['adabf']['height'] for dataset in construction_sizes]
    adaptive_heights = [dataset_results[dataset]['adaptiveqf']['height'] for dataset in construction_sizes]

    fig, axs = plt.subplots(1, 4, figsize=(5, 2), layout='constrained')
    fig.set_constrained_layout_pads(w_pad=0.001, h_pad=0.05, wspace=0.05, hspace=0.1)
    axs[0].set_ylabel('Amort. Query Time (ms)')
    current_count = 0
    width = 0.6
    for dataset in construction_sizes:
        x = np.arange(3)
        axs[current_count].set_title(dataset, fontsize='large')
        axs[current_count].margins(y=0.1, x=0.1)
        axs[current_count].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        bottom = np.zeros(3)
        plbf_bar_prop = dataset_results[dataset]['plbf']['bar']
        adabf_bar_prop = dataset_results[dataset]['adabf']['bar']
        adaptiveqf_bar_prop = dataset_results[dataset]['adaptiveqf']['bar']
        for (i, part) in enumerate(all_categories):
            values = [plbf_bar_prop[i] * 1000, adabf_bar_prop[i] * 1000, adaptiveqf_bar_prop[i] * 1000]
            axs[current_count].bar(x, values, edgecolor='black', bottom=bottom, label=(textwrap.fill(part, 12) if current_count == 0 else ""), color=COMP_COLORS[part], width=width)
            bottom += values
        for i, height in enumerate(bottom):
            axs[current_count].text(
                x[i],
                height,
                f"{height:.1e}" if height < 1 else f'{height:.1f}',
                ha='center',
                va='bottom',
                fontsize='x-small',
            )
        axs[current_count].set_xticks(x, ['plbf', 'adabf', 'aqf'], fontsize='small')
        for label in axs[current_count].get_yticklabels():
            label.set_fontsize('x-small')
        current_count += 1
    fig.legend(loc="lower center", ncol=len(all_categories), bbox_to_anchor=(0.5, -0.25), fontsize='medium', handlelength=1)
    plt.savefig(f'figures/combined_query_with_prop.pdf', bbox_inches='tight')
    plt.clf()

    x = np.arange(4)
    width = 0.15
    plt.figure(figsize=(12,2))
    plt.bar(x-0.2, plbf_heights, width, color=FILTER_COLORS['plbf'], label='plbf')
    plt.bar(x, adabf_heights, width, color=FILTER_COLORS['adabf'], label='adabf')
    plt.bar(x+0.2, adaptive_heights, width, color=FILTER_COLORS['aqf'], label='adaptiveqf')
    plt.xticks(x, [dataset for dataset in construction_sizes])
    # plt.xlabel('Datasets')
    plt.ylabel('Amort. Query Time (s)')
    plt.yscale('log')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig("figures/combined_overall_query.pdf", bbox_inches='tight')
    plt.clf()

    # plot adaptive parts
    # plot parts
    x = np.arange(3)
    width = 0.6
    fig, axs = plt.subplots(1, 4, figsize=(12,2), layout='constrained')
    current_count = 0        
    axs[0].set_ylabel("Amort. Query Time Prop")
    for dataset in construction_sizes:
        axs[current_count].set_title(dataset)
        bottom = np.zeros(3)
        plbf_bar_prop = dataset_results[dataset]['plbf']['prop']
        adabf_bar_prop = dataset_results[dataset]['adabf']['prop']
        adaptiveqf_bar_prop = dataset_results[dataset]['adaptiveqf']['prop']
        for (i, part) in enumerate(all_categories):
            values = [plbf_bar_prop[i], adabf_bar_prop[i], adaptiveqf_bar_prop[i]]
            axs[current_count].bar(x, values, bottom=bottom, label=(part if current_count == 0 else ""), color=COMP_COLORS[part], width=width)
            bottom += values
        
        axs[current_count].set_xticks(x, ['plbf', 'ada-bf', 'adaptiveqf'])
        axs[current_count].set_xlim(-0.5, 2.5)
        current_count += 1
    fig.legend(loc='outside center right')
    plt.savefig(f'figures/combined_querytime_prop.pdf')  
    plt.clf()       

def plot_changing_model_exp(learned_filepath):
    learned_df = pd.read_csv(learned_filepath)
    unique_filters = set(learned_df['filter'])
    
    datasets = ["url", "ember", "shalla", "caida"]
    
    fig, axs = plt.subplots(1, 4, figsize=(12,2), layout='constrained')
    current_count = 0        
    axs[0].set_ylabel("False Positive Rate")
    overall_max = None
    overall_min = None
    for dataset in datasets:
        current_dataset_info = learned_df[learned_df['dataset'] == dataset]
        for filter in unique_filters:
            proportions = []
            meds = []
            mins = []
            maxes = []
            current_filter_data = current_dataset_info[current_dataset_info['filter'] == filter]
            unique_model_sizes = sorted(list(set(current_filter_data['model_bytes'])))
            proportion_fpr_mapping = dict()
            for model_size in unique_model_sizes:
                # the issue here is that model sizes end up being very slightly different.
                # what we want to do is collect all the similar model sizes according to the proportion, then process each proportion
                current_data = current_filter_data[(current_filter_data['model_bytes'] == model_size)]
                total_size = current_data['bytes'].iloc[0] # all total sizes for the same filter/dataset should be same, so just grab the first row
                
                proportion = round(model_size / total_size, 2)
                current_fprs = current_data['fpr']
                if proportion not in proportion_fpr_mapping:
                    proportion_fpr_mapping[proportion] = []
                proportion_fpr_mapping[proportion].extend(current_fprs)
            for proportion in proportion_fpr_mapping:
                fprs = proportion_fpr_mapping[proportion]
                proportions.append(proportion)
                meds.append(median(fprs))
                mins.append(min(fprs))
                maxes.append(max(fprs))
                overall_max = max(fprs) if overall_max is None or overall_max < max(fprs) else overall_max
                overall_min = min(fprs) if overall_min is None or overall_min > min(fprs) else overall_min
            axs[current_count].set_title(dataset)
            axs[current_count].plot(proportions, meds, label=(filter if current_count == 0 else ""), color=FILTER_COLORS[filter], marker=FILTER_MARKERS[filter])
            axs[current_count].fill_between(proportions, mins, maxes, alpha=0.2, color=FILTER_COLORS[filter])
            axs[current_count].set_xlabel('Model Space Proportion')
            axs[current_count].set_xticks([0.2, 0.4, 0.6, 0.8])
            axs[current_count].set_xlim([0.1, 0.9])
            axs[current_count].set_yscale('log')
        current_count += 1
    fig.legend(loc="outside center right")
    plt.savefig(f'figures/combined_changing_model.pdf')
    plt.clf()

dynamic_filter_sizes = {"ember": 539890}
def plot_dynamic_exp(learned_filepath, adaptive_filepath, learned_filters=None, unique_datasets=None, num_queries=10000000, output_name="combined_fpr_dynamic.pdf"):
    learned_df = pd.read_csv(learned_filepath)
    adaptive_df = pd.read_csv(adaptive_filepath)
    if learned_filters is None:
        learned_filters = {"plbf", "adabf"}
    if unique_datasets is None:
        unique_datasets = ["url", "ember", "shalla", "caida"]
    
    # each type of query distribution has its own plot
    fig, axs = plt.subplots(1, 3, figsize=(12,2), layout='constrained')
    current_count = 0
    axs[0].set_ylabel("Instantaneous FPR")
    overall_max = None
    overall_min = None
    for dataset in unique_datasets:
        axs[current_count].set_title(dataset)
        data_for_dataset = adaptive_df[adaptive_df['dataset'] == dataset]
        adaptive_data_for_dataset = data_for_dataset[data_for_dataset['num_queries'] == num_queries]
        
        data_for_dataset = learned_df[learned_df['dataset'] == dataset]
        query_counts = sorted(data_for_dataset['curr_query'].unique())

        # now for each filter size, get the median, min, and max
        for filter in learned_filters:
            if filter not in FILTER_COLORS.keys():
                print(f"{filter} not implemented...")
                continue
            data_for_num_queries = data_for_dataset[data_for_dataset['num_queries'] == num_queries]
            data_for_filter = data_for_num_queries[data_for_num_queries['filter'] == filter]
            if len(data_for_filter) == 0:
                print(f"No data found for {filter}-{dataset}...")
                continue
            # now for each filter size, get the median, min, and max 
            fprs = []
            for curr_query in query_counts:
                current_query_data = data_for_filter[data_for_filter['curr_query'] == curr_query]
                if len(current_query_data) == 0:
                    print(f"No data found for {filter}-{dataset}-{curr_query}...")
                    continue
                fpr = current_query_data['fpr'].iloc[0] # technically, should only be one value here
                fprs.append(fpr)
                overall_max = fpr if (overall_max is None or overall_max < fpr) else overall_max
                overall_min = fpr if fpr != 0 and (overall_min is None or overall_min > fpr) else overall_min
            # now plot the data
            axs[current_count].plot(query_counts, fprs, label=(filter if current_count == 0 else ""), color=FILTER_COLORS[filter], marker=FILTER_MARKERS[filter], markersize=3)

        # now process the adaptive filter
        fprs = []
        for curr_query in query_counts:
            current_data = adaptive_data_for_dataset[adaptive_data_for_dataset['curr_query'] == curr_query] 
            if adaptive_data_for_dataset.empty:
                print(f"[EMPTY] {dataset}, {curr_query}")
            fpr = current_data['fpr'].iloc[0]
            fprs.append(fpr)
            overall_max = fpr if (overall_max is None or overall_max < fpr) else overall_max
            overall_min = fpr if fpr != 0 and (overall_min is None or overall_min > fpr) else overall_min
        
        # now plot the data
        axs[current_count].plot(query_counts, fprs, label=('adaptiveqf' if current_count == 0 else ""), color=FILTER_COLORS['aqf'], marker=FILTER_MARKERS['aqf'], markersize=3)
        axs[current_count].set_xlabel('Number of Queries')
        axs[current_count].set_yscale('log')
        # after processing a dataset, move on to the next subplot
        current_count += 1
    fig.legend(loc='outside center right')
    plt.savefig(f'figures/{output_name}')
    plt.clf()


if __name__ == "__main__":
    plot_fpr_space_tradeoff('results/overall_results_with_model_scores.csv', '../adaptiveqf/results/aqf_results.csv')
    plot_adversarial('results/overall_advers_with_model_scores.csv', '../adaptiveqf/results/aqf_advers_results.csv')
    plot_construction_times('results/results_with_model.csv', '../adaptiveqf/results/aqf_results.csv')
    plot_query_times('results/results_with_model.csv', '../adaptiveqf/results/aqf_results.csv')
    plot_model_degradation('results/degrad_results_with_model_scores.csv')
    plot_changing_model_exp('results/changing_model_size.csv')
    plot_dynamic_exp('results/dynamic_results_with_model_scores.csv', '../adaptiveqf/results/aqf_results_dynamic.csv', 
                     learned_filters=["plbf", "adabf"], unique_datasets=["url", "ember", "caida"],
                     output_name="combined_fpr_dynamic.pdf")
    plot_dynamic_exp('results/dynamic_results_with_model_rebuild.csv', '../adaptiveqf/results/aqf_results_dynamic.csv', 
                     learned_filters=["plbf", "adabf"], unique_datasets=["url", "ember", "caida"],
                     output_name="combined_fpr_dynamic_rebuild.pdf")