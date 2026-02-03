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

# unique colors/markers to assign each filter variant, staying constant across all graphs
FILTER_COLORS = {"plbf-forest": "blue", "plbf-tree": "cornflowerblue", "plbf-logistic": "darkturquoise",
                 "adabf-forest": "green", "adabf-tree": "yellowgreen", "adabf-logistic": "limegreen",
                 "stacked": "blueviolet", "aqf": "orange"}
FILTER_MARKERS = {"plbf-forest": "^", "plbf-tree": "<", "plbf-logistic": ">",
                  "adabf-forest": "2", "adabf-tree": "3", "adabf-logistic": "4",
                  "stacked": "o", "aqf": "D"}
# base filter types that are available across all experiments
FILTERS = ["plbf", "adabf", "stacked", "aqf"]

# assign unique colors to each query / construction time operation, staying consistent across all graphs
COLORS = plt.get_cmap('tab20').colors
TIMING_COMP = ['Score Inference', 'Filter Query', 'Reverse Map Adapt', 'Filter Inserts',
                    'Model Training', 'Threshold Finding', 'Reverse Map Updates', 'Opt. & Inserts']
COMP_COLORS = {comp: COLORS[i] for i, comp in enumerate(TIMING_COMP)}

# from the AdaptiveQF experiments, obtain the sizes that all filters follow
dataset_sizes = {"shalla": [4280640, 4807488, 5334336, 5861184, 6388032, 6914880, 7441728], 
                 "ember": [539890, 606338, 672786, 739234, 805682, 872130, 938578], 
                 "url": [69160, 77672, 86184, 94696, 103208, 111720, 120232],
                 "caida": [2144675, 2408635, 2672595, 2936555, 3200515, 3464475, 3728435]}

# the number of queries performed in the one-pass experiments for each dataset
one_pass_sizes = {"url": 162798, "shalla": 3905928, "ember": 800000, "caida": 8493974}

# the number of positive elements in each dataset
dataset_pos = {"ember": 800000, "url": 55681, "shalla": 2926705, "caida": 1196194}

plt.rcParams['text.usetex'] = False

# the models available for each learned filter variant to use, mapping nickname to full name used when reporting experiment data
models = {"forest": "random_forest", "tree": "decision_tree", "logistic": "logistic_regression"}

# which models should be included when reporting experiments describing fpr for different filters
fpr_models = {"tree": "decision_tree"}

# the sizes used by the filters in the adversarial query experiments
ADVERSARIAL_SIZES = {"url": 77672, "ember": 606338, "shalla": 4807488, "caida": 2408635}

# the sizes used by the filters in the query / construction time experiments
construction_sizes = {"url": 69160, "ember": 539890, "shalla": 4280640, "caida": 2144675}


def plot_fpr_space_tradeoff(learned_filename: str, adaptive_filename: str, stacked_filename:str, num_queries=[10000000], learned_filters=None):
    """
    For the learned, stacked, and adaptive filters, saves pdf figures plotting size vs fpr
    on different datasets (url, ember, shalla, caida) and static workloads (one-pass, uniform, Zipfian).

    Parameters
    ----------
    learned_filename : str
        Relative path of csv file storing learned filter results.
    adaptive_filename : str
        Relative path of csv file storing adaptive filter results.
    stacked_filename : str
        Relative path of csv file storing stacked filter results.
    num_queries : List(int)
        List of different numbers of queries used in experiments to plot.
    learned_filters : List(str)
        Optional list of specific learned filters (i.e. plbf or adabf) to use in figures.

    Returns
    -------
    No return value: instead, saves the figure to the '../results/figures/learned' folder.
    """
    # first, grab the data from the learned results, and separate the data based on the filter
    learned_df = pd.read_csv(learned_filename)
    adaptive_df = pd.read_csv(adaptive_filename)
    stacked_df = pd.read_csv(stacked_filename)
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
                    if filter not in FILTERS:
                        print(f"{filter} not implemented...")
                        continue
                    data_for_dataset = learned_df[learned_df['dataset'] == dataset]
                    data_for_num_queries = data_for_dataset[data_for_dataset['num_queries'] == query_num]
                    data_for_query_dist = data_for_num_queries[data_for_num_queries['query_dist'] == query_dist]
                    data_for_filter = data_for_query_dist[data_for_query_dist['filter'] == filter]
                    
                    # now for each filter size, process each type of model...
                    for model in fpr_models:
                        model_data_for_filter = data_for_filter[data_for_filter['model_type'] == models[model]]
                        if model_data_for_filter.empty:
                            print(f"[EMPTY] {dataset}, {filter}-{model}, {query_num}, {query_dist}")
                            continue
                        # get the median, min, and max 
                        sizes = []
                        meds = []
                        mins = []
                        maxes = []
                        avgs = []
                        for size in unique_sizes:
                            current_size_data = model_data_for_filter[model_data_for_filter['bytes'] == size]
                            if len(current_size_data) == 0:
                                print(f"No data found for {filter}-{model}-{dataset}-{size}...")
                                continue
                            fprs = current_size_data['fpr']
                            if len(fprs) == 0:
                                print(f"No data found for {filter}-{model}-{dataset}-{size}...")
                                continue
                            sizes.append(size / dataset_pos[dataset] * 8)
                            meds.append(median(fprs))
                            mins.append(min(fprs))
                            maxes.append(max(fprs))
                            avgs.append(sum(fprs) / len(fprs))
                            overall_max = max(fprs) if (overall_max is None or overall_max < max(fprs)) else overall_max
                            overall_min = max(fprs) if min(fprs) != 0 and (overall_min is None or overall_min > min(fprs)) else overall_min
                        # now plot the data
                        axs[current_count].plot(sizes, meds, lw=1.3, markersize=5, label=(f"{filter}-{model}" if current_count == 0 else ""), color=FILTER_COLORS[f"{filter}-{model}"], marker=FILTER_MARKERS[f"{filter}-{model}"])
                        axs[current_count].fill_between(sizes, mins, maxes, alpha=0.2, color=FILTER_COLORS[f"{filter}-{model}"])
                
                # now process the stacked filter
                sizes = []
                meds = []
                mins = []
                maxes = []
                avgs = []
                for size in unique_sizes:
                    current_size_data = stacked_df[
                        (stacked_df['dataset'] == dataset) &
                        (stacked_df['num_queries'] == query_num) &
                        (stacked_df['query_dist'] == query_dist) &
                        (stacked_df['size'] == size)
                    ]
                    if current_size_data.empty:
                        print(f"[EMPTY] {dataset}, stacked, {query_num}, {size}, {query_dist}")
                    fprs = current_size_data['fpr']
                    sizes.append(size / dataset_pos[dataset] * 8)
                    meds.append(median(fprs))
                    mins.append(min(fprs))
                    maxes.append(max(fprs))                            
                    avgs.append(sum(fprs) / len(fprs))
                    overall_max = max(fprs) if (overall_max is None or overall_max < max(fprs)) else overall_max
                    overall_min = max(fprs) if min(fprs) != 0 and (overall_min is None or overall_min > min(fprs)) else overall_min
                # now plot the data
                axs[current_count].plot(sizes, meds, lw=1.3, markersize=5, label=('stacked' if current_count == 0 else ""), color=FILTER_COLORS['stacked'], marker=FILTER_MARKERS['stacked'])
                axs[current_count].fill_between(sizes, mins, maxes, alpha=0.2, color=FILTER_COLORS['stacked'])

                # now process the adaptive filter
                sizes = []
                meds = []
                mins = []
                maxes = []
                avgs = []
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
                    avgs.append(sum(fprs) / len(fprs))
                    overall_max = max(fprs) if (overall_max is None or overall_max < max(fprs)) else overall_max
                    overall_min = max(fprs) if min(fprs) != 0 and (overall_min is None or overall_min > min(fprs)) else overall_min
                # now plot the data
                axs[current_count].plot(sizes, meds, lw=1.3, markersize=5, label=('adaptiveqf' if current_count == 0 else ""), color=FILTER_COLORS['aqf'], marker=FILTER_MARKERS['aqf'])
                axs[current_count].fill_between(sizes, mins, maxes, alpha=0.2, color=FILTER_COLORS['aqf'])
                # now configure the graph
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
        plt.savefig(f'../results/figures/combined_fpr_{query_dist}_{query_num/1000000 if query_dist != "onepass" else ""}M.pdf')
        plt.clf()

def plot_model_degradation(learned_filename: str):
    """
    For the learned filters, saves pdf figures plotting fpr vs negative set proportion used in the model training set.

    Parameters
    ----------
    learned_filename : str
        Relative path of csv file storing learned filter results.
        
    Returns
    -------
    No return value: instead, saves the figure to the '../results/figures/learned' folder.
    """
    learned_df = pd.read_csv(learned_filename)
    datasets = ["url", "ember", "shalla", "caida"]
    learned_filters = ["plbf", "adabf"]
    set_sizes = [0.2, 0.4, 0.6, 0.8]
    fig, axs = plt.subplots(1, 4, figsize=(12,2), layout='constrained')
    current_count = 0
    axs[0].set_ylabel("Median FPR")
    for dataset in datasets:
        axs[current_count].set_title(dataset)
        size_to_process = min(dataset_sizes[dataset])
        for learned_filter in learned_filters:
            for model in models:
                if learned_filter not in FILTERS:
                    print(f"{learned_filter} not implemented...")
                    continue
                data_for_dataset = learned_df[learned_df['dataset'] == dataset]
                data_for_num_queries = data_for_dataset[data_for_dataset['num_queries'] == 10000000]
                data_for_query_dist = data_for_num_queries[data_for_num_queries['query_dist'] == "unif"]
                data_for_filter = data_for_query_dist[data_for_query_dist['filter'] == learned_filter]
                current_data = data_for_filter[(data_for_filter['bytes'] == size_to_process) & (data_for_filter['model_type'] == models[model])]
                if len(current_data) == 0:
                    print(f"No data found for {learned_filter}-{models[model]}...")
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
                axs[current_count].plot(set_sizes, meds, label=(f"{learned_filter}-{model}" if current_count == 0 else ""), color=FILTER_COLORS[f"{learned_filter}-{model}"], marker=FILTER_MARKERS[f"{learned_filter}-{model}"], markersize=9)
                # axs[current_count].fill_between(set_sizes, mins, maxes, alpha=0.2, color=FILTER_COLORS[f"{learned_filter}-{model}"])
                axs[current_count].set_xlabel('Training Set Proportion')
                axs[current_count].set_yscale('log')
                axs[current_count].set_yticks([0.0001, 0.001, 0.01, 0.1])
        current_count += 1
    fig.legend(loc='outside center right')
    plt.savefig(f'../results/figures/combined_degrad_10M_.pdf')
    plt.clf()

def plot_adversarial(learned_filename: str, adaptive_filename: str, stacked_filename: str, learned_filters=None):
    """
    For the learned, stacked, and adaptive filters, saves a figure plotting fpr vs the proportion of adversarial
    queries included in the experimental workload.

    Parameters
    ----------
    learned_filename : str
        Relative path of csv file storing learned filter results.
    adaptive_filename : str
        Relative path of csv file storing adaptive filter results.
    stacked_filename : str
        Relative path of csv file storing stacked filter results.
    learned_filters : List(str)
        Optional list of learned filters to only represent in the figures (i.e. plbf or adabf).
        
    Returns
    -------
    No return value: instead, saves the figure to the '../results/figures/learned' folder.
    """
    learned_df = pd.read_csv(learned_filename)
    adaptive_df = pd.read_csv(adaptive_filename)
    stacked_df = pd.read_csv(stacked_filename)
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
            unique_freqs = sorted(set(learned_df['freq']))
            for filter in learned_filters:
                for model in fpr_models:
                    filter_model = f"{filter}-{model}"
                    if filter_model not in FILTER_COLORS:
                        print(f"{filter_model} not implemented...")
                        continue
                    current_filter_data = learned_df[
                        (learned_df['dataset'] == dataset) &
                        (learned_df['num_queries'] == num_queries) &
                        (learned_df['filter'] == filter) &
                        (learned_df['model_type'] == models[model])
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
                    axs[current_count].plot(freqs, meds, label=(filter_model if current_count == 0 else ""), color=FILTER_COLORS[filter_model], marker=FILTER_MARKERS[filter_model])
                    axs[current_count].fill_between(freqs, mins, maxes, alpha=0.2, color=FILTER_COLORS[filter_model])
            
            stacked_data = stacked_df[(stacked_df['dataset'] == dataset) &
                                      (stacked_df['num_queries'] == num_queries) &
                                      (stacked_df['size'] == current_size)]
            freqs = []
            meds = []
            mins = []
            maxes = []
            for freq in unique_freqs:
                data_for_freq = stacked_data[stacked_data['freq'] == freq]
                if data_for_freq.empty:
                    print(f"[EMPTY] {dataset}, stacked, {num_queries}, {current_size}")
                    continue
                fprs = data_for_freq['fpr']
                freqs.append(freq)
                meds.append(median(fprs))
                mins.append(min(fprs))
                maxes.append(max(fprs))
                overall_max = max(fprs) if (overall_max is None or overall_max < max(fprs)) else overall_max
                overall_min = max(fprs) if min(fprs) != 0 and (overall_min is None or overall_min > min(fprs)) else overall_min    
            axs[current_count].plot(freqs, meds, label=('stacked' if current_count == 0 else ""), color=FILTER_COLORS['stacked'], marker=FILTER_MARKERS['stacked'])
            axs[current_count].fill_between(freqs, mins, maxes, alpha=0.2, color=FILTER_COLORS['stacked'])
            
            adaptive_data = adaptive_df[
                (adaptive_df['dataset'] == dataset) &
                (adaptive_df['num_queries'] == num_queries) &
                (adaptive_df['size'] == current_size)
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
            axs[current_count].plot(freqs, meds, label=('adaptiveqf' if current_count == 0 else ""), color=FILTER_COLORS['aqf'], marker=FILTER_MARKERS['aqf'])
            axs[current_count].fill_between(freqs, mins, maxes, alpha=0.2, color=FILTER_COLORS['aqf'])

            axs[current_count].set_xlabel('Adversarial Frequency')
            axs[current_count].set_yscale('log')

        current_count += 1
    # label and save the plot
    for ax in axs.flat:
        ax.set_ylim(math.pow(10, math.floor(math.log10(overall_min))), 
                    math.pow(10, math.ceil(math.log10(overall_max))))
    # plt.title(f'Adversarial Query FPR-Space Tradeoff on {dataset} ({num_queries / 1000000}M Queries)')
    fig.legend(loc="outside center right")
    plt.savefig(f'../results/figures/combined_fpr_adversarial_{num_queries/1000000}M.pdf')
    plt.clf()
    
def plot_construction_times(learned_filename: str, adaptive_filename: str, stacked_filename: str):
    """
    For the learned, stacked, and adaptive filters, saves a figure plotting a broken-axis stacked bar chart
    representing the filter construction times, with each bar component corresponding to a different
    construction operation. Learned filter variants using each model type (RandomForest, DecisionTree, LogisticRegression)
    are all included.

    Parameters
    ----------
    learned_filename : str
        Relative path of csv file storing learned filter results.
    adaptive_filename : str
        Relative path of csv file storing adaptive filter results.
    stacked_filename : str
        Relative path of csv file storing stacked filter results.
        
    Returns
    -------
    No return value: instead, saves the figure to the '../results/figures/learned' folder.
    """
    # first establish the categories for the construction
    learned_df = pd.read_csv(learned_filename)
    adaptive_df = pd.read_csv(adaptive_filename)
    stacked_df = pd.read_csv(stacked_filename)
    learned_categories = ['Model Training', 'Threshold Finding', 'Filter Inserts']
    adaptive_categories = ['Filter Inserts', 'Reverse Map Updates']
    stacked_categories = ['Opt. & Inserts']
    all_categories = sorted(list(set(learned_categories + adaptive_categories + stacked_categories)))

    # now, for each dataset, track the different times...
    # these will map dataset => filter dict, which in turn stores filter => height/prop
    dataset_results = dict()

    # now go through each dataset and obtain the results for each kind of filter
    for dataset in construction_sizes:
        print("construction time on ", dataset)
        current_results = dict()
        for model in models:
            plbf_data = learned_df[
                (learned_df['dataset'] == dataset) &
                (learned_df['bytes'] == construction_sizes[dataset]) &
                (learned_df['filter'] == 'plbf') &
                (learned_df['model_type'] == models[model])
            ]
            if len(plbf_data) != 0:
                bar = {'Model Training': np.median(plbf_data['construct_time']) + np.median(plbf_data['train_time']) + np.median(plbf_data['initial_scores']),
                    'Threshold Finding': np.median(plbf_data['segment_division']) + np.median(plbf_data['t_f_finding']),
                    'Filter Inserts': np.median(plbf_data['bloom_init']) + np.median(plbf_data['region_finding']) + np.median(plbf_data['filter_inserts'])}
                plbf_bar_heights = [bar.get(part, 0) for part in all_categories]
                plbf_total_height = np.sum(plbf_bar_heights)
                plbf_bar_prop = [part / plbf_total_height for part in plbf_bar_heights]

            adabf_data = learned_df[
                    (learned_df['dataset'] == dataset) &
                    (learned_df['bytes'] == construction_sizes[dataset]) &
                    (learned_df['filter'] == 'adabf') & 
                    (learned_df['model_type'] == models[model])
                ]
            if len(adabf_data) != 0:
                bar = {'Model Training': median(adabf_data['construct_time']) + median(adabf_data['train_time']) + median(adabf_data['initial_scores']),
                    'Threshold Finding': median(adabf_data['region_finding']),
                    'Filter Inserts': median(adabf_data['bloom_init']) + median(adabf_data['filter_inserts'])}
                adabf_bar_heights = [bar.get(part, 0) for part in all_categories]
                adabf_total_height = sum(adabf_bar_heights)
                adabf_bar_prop = [part / adabf_total_height for part in adabf_bar_heights]
            current_results[f"plbf-{model}"] = {'height': plbf_total_height, 'prop': plbf_bar_prop, 'bar': plbf_bar_heights}
            current_results[f"adabf-{model}"] = {'height': adabf_total_height, 'prop': adabf_bar_prop, 'bar': adabf_bar_heights}
        
        # now handle the stacked filter
        stacked_data = stacked_df[
                (stacked_df['dataset'] == dataset) &
                (stacked_df['size'] == construction_sizes[dataset])
        ]
        if len(stacked_data) != 0:
            bar = {'Opt. & Inserts': median(stacked_data['insert_time'])}
            stacked_bar_heights = [bar.get(part, 0) for part in all_categories]
            stacked_total_height = sum(stacked_bar_heights)
            stacked_bar_prop = [part / stacked_total_height for part in stacked_bar_heights]
        else:
            print("couldn't find data for stacked filter on ", dataset)
            exit(1)
        current_results["stacked"] = {'height': stacked_total_height, 'prop': stacked_bar_prop, 'bar': stacked_bar_heights}
        
        # now handle the adaptiveqf
        adaptive_data = adaptive_df[
                (adaptive_df['dataset'] == dataset) &
                (adaptive_df['size'] == construction_sizes[dataset])
        ]
        if len(adaptive_data) != 0:
            bar = {'Filter Inserts': median(adaptive_data['insert_time']), 'Reverse Map Updates': median(adaptive_data['amortized_adapt'])}
            adaptive_bar_heights = [bar.get(part, 0) for part in all_categories]  
            adaptive_total_height = sum(adaptive_bar_heights)
            adaptive_bar_prop = [part / adaptive_total_height for part in adaptive_bar_heights]
        else:
            print("couldn't find data for aqf on ", dataset)
            exit(1)
        current_results["adaptiveqf"] = {'height': adaptive_total_height, 'prop': adaptive_bar_prop, 'bar': adaptive_bar_heights}
        dataset_results[dataset] = current_results

    fig, axs = plt.subplots(2, 4, figsize=(15, 2.5), layout='constrained', sharex=True)
    fig.set_constrained_layout_pads(w_pad=0.001, h_pad=0.05, wspace=0.05, hspace=0.1)
    fig.text(-0.02, 0.5, 'Construct Time (ks)', va='center', rotation='vertical')
    current_count = 0
    width = 0.6
    for dataset in construction_sizes:
        x = np.arange(8)
        axs[0][current_count].set_title(dataset, fontsize='large')
        axs[0][current_count].spines.bottom.set_visible(False)
        axs[1][current_count].spines.top.set_visible(False)
        axs[0][current_count].tick_params(bottom=False, labelbottom=False)
        axs[1][current_count].tick_params(top=False)
        axs[0][current_count].set_ylim(0.15, 17)
        axs[1][current_count].set_ylim(0, 0.15)
        axs[1][current_count].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        bottom = np.zeros(8)
        plbf_forest_bar_prop = dataset_results[dataset]['plbf-forest']['bar']
        plbf_tree_bar_prop = dataset_results[dataset]['plbf-tree']['bar']
        plbf_logistic_bar_prop = dataset_results[dataset]['plbf-logistic']['bar']
        adabf_forest_bar_prop = dataset_results[dataset]['adabf-forest']['bar']
        adabf_tree_bar_prop = dataset_results[dataset]['adabf-tree']['bar']
        adabf_logistic_bar_prop = dataset_results[dataset]['adabf-logistic']['bar']
        adaptiveqf_bar_prop = dataset_results[dataset]['adaptiveqf']['bar']
        stacked_bar_prop = dataset_results[dataset]['stacked']['bar']
        for (i, part) in enumerate(all_categories):
            # learned filters are in seconds, adaptiveqf is in microseconds, want to convert to kiloseconds
            values = [plbf_forest_bar_prop[i] / 1000, plbf_tree_bar_prop[i] / 1000, plbf_logistic_bar_prop[i] / 1000, 
                      adabf_forest_bar_prop[i] / 1000, adabf_tree_bar_prop[i] / 1000, adabf_logistic_bar_prop[i] / 1000, 
                      stacked_bar_prop[i] / 1000000000, adaptiveqf_bar_prop[i] / 1000000000, ]
            axs[0][current_count].bar(x, values, edgecolor='black', bottom=bottom, color=COMP_COLORS[part], width=width)
            axs[1][current_count].bar(x, values, edgecolor='black', bottom=bottom, label=(textwrap.fill(part, 12) if current_count == 0 else ""), color=COMP_COLORS[part], width=width)
            bottom += values
        for i, height in enumerate(bottom):
            if height < 0.15:
                axs[1][current_count].text(
                    x[i],
                    height,
                    f"{height:.1e}" if height < 1 else f'{height:.1f}',
                    ha='center',
                    va='bottom',
                    fontsize='x-small',
                )
            elif height > 0.15:
                axs[0][current_count].text(
                    x[i],
                    height,
                    f"{height:.1e}" if height < 1 else f'{height:.1f}',
                    ha='center',
                    va='bottom',
                    fontsize='x-small',
                )
        axs[1][current_count].set_xticks(x, ['plbf\nforest', 'plbf\ntree', 'plbf\nlogistic', 'adabf\nforest', 'adabf\ntree', 'adabf\nlogistic', 'stacked', 'aqf'], fontsize='small')
        for label in axs[1][current_count].get_yticklabels():
            label.set_fontsize('x-small')
        for label in axs[0][current_count].get_yticklabels():
            label.set_fontsize('x-small')
        for container in axs[0][current_count].containers:
            for rect in container:
                if rect.get_height() < 0.15:
                    rect.set_visible(False)
        current_count += 1
    fig.legend(loc="upper center", ncol=len(all_categories), bbox_to_anchor=(0.5, 1.25), fontsize='medium', handlelength=1)
    plt.savefig(f'../results/figures/combined_const_with_prop.pdf', bbox_inches='tight')
    plt.clf()

def plot_query_times(learned_filename: str, adaptive_filename: str, stacked_filename:str):
    """
    For the learned, stacked, and adaptive filters, saves a figure plotting a broken-axis stacked bar chart
    representing the filter query times, with each bar component corresponding to a different
    query operation. Learned filter variants using each model type (RandomForest, DecisionTree, LogisticRegression)
    are all included.

    Parameters
    ----------
    learned_filename : str
        Relative path of csv file storing learned filter results.
    adaptive_filename : str
        Relative path of csv file storing adaptive filter results.
    stacked_filename : str
        Relative path of csv file storing stacked filter results.
        
    Returns
    -------
    No return value: instead, saves the figure to the '../results/figures/learned' folder.
    """
    # first establish the categories for the queries
    plbf_df = pd.read_csv(learned_filename)
    adaptive_df = pd.read_csv(adaptive_filename)
    stacked_df = pd.read_csv(stacked_filename)
    learned_categories = ['Score Inference', 'Filter Query']
    adaptiveqf_categories = ['Filter Query', 'Reverse Map Updates']
    stacked_categories = ['Filter Query']
    all_categories = sorted(list(set(learned_categories + adaptiveqf_categories + stacked_categories)))

    # now, for each dataset, track the different times...
    # these will map dataset => filter dict, which in turn stores filter => height/prop
    dataset_results = dict()
    # now go through each dataset and obtain the results for each kind of filter
    for dataset in construction_sizes:
        print("query times on ", dataset)
        current_results = dict()
        for model in models:
            plbf_data = plbf_df[
                (plbf_df['dataset'] == dataset) &
                (plbf_df['bytes'] == construction_sizes[dataset]) & 
                (plbf_df['filter'] == 'plbf') & 
                (plbf_df['model_type'] == models[model])
            ]
            if len(plbf_data) != 0:
                bar = {'Score Inference': sum(plbf_data['amort_score_time']) / len(plbf_data['amort_score_time']),
                        'Filter Query': sum(plbf_data['amort_back_filter_time']) / len(plbf_data['amort_back_filter_time']) + sum(plbf_data['amort_region_time']) / len(plbf_data['amort_region_time'])}
                plbf_bar_heights = [bar.get(part, 0) for part in all_categories]
                plbf_total_height = sum(plbf_bar_heights)
                plbf_bar_prop = [part / plbf_total_height for part in plbf_bar_heights]
                print(f"plbf-{model} timing: {bar} (seconds)")
            adabf_data = plbf_df[
                (plbf_df['dataset'] == dataset) &
                (plbf_df['bytes'] == construction_sizes[dataset]) & 
                (plbf_df['filter'] == 'adabf') & 
                (plbf_df['model_type'] == models[model])
            ]
            if len(adabf_data) != 0:
                bar = {'Score Inference': sum(adabf_data['amort_score_time']) / len(adabf_data['amort_score_time']), 
                    'Filter Query': sum(adabf_data['amort_back_filter_time']) / len(adabf_data['amort_back_filter_time']) + sum(adabf_data['amort_region_time']) / len(adabf_data['amort_region_time'])}
                adabf_bar_heights = [bar.get(part, 0) for part in all_categories]
                adabf_total_height = sum(adabf_bar_heights)
                adabf_bar_prop = [part / adabf_total_height for part in adabf_bar_heights]
                print(f"adabf-{model} timing: {bar} (seconds)")
            current_results[f"plbf-{model}"] = {'height': plbf_total_height, 'prop': plbf_bar_prop, 'bar': plbf_bar_heights}
            current_results[f"adabf-{model}"] = {'height': adabf_total_height, 'prop': adabf_bar_prop, 'bar': adabf_bar_heights}

        #now handle adaptiveqf
        adaptive_data = adaptive_df[
                        (adaptive_df['dataset'] == dataset) &
                        (adaptive_df['size'] == construction_sizes[dataset])
                ]
        if len(adaptive_data) != 0:
            bar = {'Filter Query': sum(adaptive_data['amortized_query']) / len(adaptive_data['amortized_query']), 
                'Reverse Map Updates': sum(adaptive_data['amortized_adapt']) / len(adaptive_data['amortized_adapt'])}
            adaptiveqf_bar_heights = [bar.get(part, 0) for part in all_categories]
            adaptiveqf_total_height = sum(adaptiveqf_bar_heights)
            adaptiveqf_bar_prop = [part / adaptiveqf_total_height for part in adaptiveqf_bar_heights]
            print(f"adaptiveqf timing: {bar} (microseconds)")
        else:
            print("couldn't find data for aqf on ", dataset)
            exit(1)
        current_results[f"adaptiveqf"] = {'height': adaptiveqf_total_height, 'prop': adaptiveqf_bar_prop, 'bar': adaptiveqf_bar_heights}

        # now handle stacked filter
        stacked_data = stacked_df[
                        (stacked_df['dataset'] == dataset) &
                        (stacked_df['size'] == construction_sizes[dataset])
                ]
        if len(stacked_data) != 0:
            bar = {'Filter Query': sum(stacked_data['amortized_query']) / len(stacked_data['amortized_query'])}
            # though we don't actually need to plot this, just need to make sure the categories are aligned
            stacked_bar_heights = [bar.get(part, 0) for part in all_categories]
            stacked_total_height = sum(stacked_bar_heights)
            stacked_bar_prop = [part / stacked_total_height for part in stacked_bar_heights]
            print(f"stacked timing: {bar} (microseconds)")
        current_results[f"stacked"] = {'height': stacked_total_height, 'prop': stacked_bar_prop, 'bar': stacked_bar_heights}
        dataset_results[dataset] = current_results

    # now for the overall query times, need to keep the order of the datasets
    # and keep track of the bar heights across datasets
    fig, axs = plt.subplots(2, 4, figsize=(15, 2.5), layout='constrained', sharex=True)
    fig.set_constrained_layout_pads(w_pad=0.001, h_pad=0.05, wspace=0.05, hspace=0.1)
    fig.text(-0.02, 0.5, 'Amort. Query Time (ms)', va='center', rotation='vertical')

    current_count = 0
    width = 0.6
    for dataset in construction_sizes:
        x = np.arange(8)
        axs[0][current_count].set_title(dataset, fontsize='large')
        axs[0][current_count].spines.bottom.set_visible(False)
        axs[1][current_count].spines.top.set_visible(False)
        axs[0][current_count].tick_params(bottom=False, labelbottom=False)
        axs[1][current_count].tick_params(top=False)
        axs[0][current_count].set_ylim(0.1, 5)
        axs[1][current_count].set_ylim(0, 0.1)
        axs[1][current_count].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        bottom = np.zeros(8)
        plbf_forest_bar_prop = dataset_results[dataset]['plbf-forest']['bar']
        plbf_tree_bar_prop = dataset_results[dataset]['plbf-tree']['bar']
        plbf_logistic_bar_prop = dataset_results[dataset]['plbf-logistic']['bar']
        adabf_forest_bar_prop = dataset_results[dataset]['adabf-forest']['bar']
        adabf_tree_bar_prop = dataset_results[dataset]['adabf-tree']['bar']
        adabf_logistic_bar_prop = dataset_results[dataset]['adabf-logistic']['bar']
        adaptiveqf_bar_prop = dataset_results[dataset]['adaptiveqf']['bar']
        stacked_bar_prop = dataset_results[dataset]['stacked']['bar']
        for (i, part) in enumerate(all_categories):
            # learned filters are recorded in seconds, adaptive in microseconds, want to convert to milliseconds
            values = [plbf_forest_bar_prop[i] * 1000, plbf_tree_bar_prop[i] * 1000, plbf_logistic_bar_prop[i] * 1000,
                      adabf_forest_bar_prop[i] * 1000, adabf_tree_bar_prop[i] * 1000, adabf_logistic_bar_prop[i] * 1000,
                      stacked_bar_prop[i] / 1000, adaptiveqf_bar_prop[i] / 1000,]
            axs[0][current_count].bar(x, values, edgecolor='black', bottom=bottom, color=COMP_COLORS[part], width=width)
            axs[1][current_count].bar(x, values, edgecolor='black', bottom=bottom, label=(textwrap.fill(part, 12) if current_count == 0 else ""), color=COMP_COLORS[part], width=width)
            bottom += values
        for i, height in enumerate(bottom):
            if height < 0.05:
                axs[1][current_count].text(
                    x[i],
                    height,
                    f"{height:.1e}" if height < 1 else f'{height:.1f}',
                    ha='center',
                    va='bottom',
                    fontsize='x-small',
                )
            elif height > 0.1:
                axs[0][current_count].text(
                    x[i],
                    height,
                    f"{height:.1e}" if height < 1 else f'{height:.1f}',
                    ha='center',
                    va='bottom',
                    fontsize='x-small',
                )
        axs[1][current_count].set_xticks(x, ['plbf\nforest', 'plbf\ntree', 'plbf\nlogistic', 'adabf\nforest', 'adabf\ntree', 'adabf\nlogistic', 'stacked', 'aqf'], fontsize='small')
        for label in axs[1][current_count].get_yticklabels():
            label.set_fontsize('x-small')
        for label in axs[0][current_count].get_yticklabels():
            label.set_fontsize('x-small')
        for container in axs[0][current_count].containers:
            for rect in container:
                if rect.get_height() < 0.1:
                    rect.set_visible(False)
        current_count += 1
    fig.legend(loc="upper center", ncol=len(all_categories), bbox_to_anchor=(0.5, 1.25), fontsize='medium', handlelength=1)
    plt.savefig(f'../results/figures/combined_query_with_prop.pdf', bbox_inches='tight')
    plt.clf()

def plot_changing_model_exp(learned_filepath: str):
    """
    For the learned filters, saves a figure plotting the fpr vs the proportion of space used
    by the filter for the internal ML model.

    Parameters
    ----------
    learned_filename : str
        Relative path of csv file storing learned filter results.
    adaptive_filename : str
        Relative path of csv file storing adaptive filter results.
    stacked_filename : str
        Relative path of csv file storing stacked filter results.
        
    Returns
    -------
    No return value: instead, saves the figure to the '../results/figures/learned' folder.
    """
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
            axs[current_count].plot(proportions, meds, label=(f"{filter}-forest" if current_count == 0 else ""), color=FILTER_COLORS[f"{filter}-forest"], marker=FILTER_MARKERS[f"{filter}-forest"])
            axs[current_count].fill_between(proportions, mins, maxes, alpha=0.2, color=FILTER_COLORS[f"{filter}-forest"])
            axs[current_count].set_xlabel('Model Space Proportion')
            axs[current_count].set_xticks([0.2, 0.4, 0.6, 0.8])
            axs[current_count].set_xlim([0.1, 0.9])
            axs[current_count].set_yscale('log')
        current_count += 1
    fig.legend(loc="outside center right")
    plt.savefig(f'../results/figures/combined_changing_model.pdf')
    plt.clf()

def plot_dynamic_exp(learned_filepath : str, adaptive_filepath : str, stacked_filepath : str, learned_filters=None, unique_datasets=None, num_queries=10000000, output_name="combined_fpr_dynamic.pdf"):
    """
    For the learned, stacked, and adaptive filters, saves a figure plotting fpr vs query number,
    where periodic shuffling of the dataset is performed as the filters progress through the query set.
    construction operation. Learned filter variants using each model type (RandomForest, DecisionTree, LogisticRegression)
    are all included.

    Parameters
    ----------
    learned_filename : str
        Relative path of csv file storing learned filter results.
    adaptive_filename : str
        Relative path of csv file storing adaptive filter results.
    stacked_filename : str
        Relative path of csv file storing stacked filter results.
    learned filters : List(str)
        Optional list of learned filters to only include in figure (i.e. plbf or adabf)
    unique_datasets : List(str)
        Optional list of datasets to constrain the figure to (i.e. url, ember, or caida)
    num_queries : int
        Number of queries used in the dynamic experiment
    output_name : str
        Name of output file, useful for distinguishing between dynamic experiments where model retraining is (not) included.
        
    Returns
    -------
    No return value: instead, saves the figure to the '../results/figures/learned' folder.
    """
    learned_df = pd.read_csv(learned_filepath)
    adaptive_df = pd.read_csv(adaptive_filepath)
    stacked_df = pd.read_csv(stacked_filepath)
    if learned_filters is None:
        learned_filters = {"plbf", "adabf"}
    if unique_datasets is None:
        unique_datasets = ["url", "ember", "caida"]
    
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
            if filter not in FILTERS:
                print(f"{filter} not implemented...")
                continue
            for model in fpr_models:
                data_for_num_queries = data_for_dataset[data_for_dataset['num_queries'] == num_queries]
                data_for_filter = data_for_num_queries[(data_for_num_queries['filter'] == filter) & (data_for_num_queries['model_type'] == models[model])]
                if len(data_for_filter) == 0:
                    print(f"No data found for {filter}-{dataset}-{models[model]}...")
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
                axs[current_count].plot(query_counts, fprs, label=(f"{filter}-{model}" if current_count == 0 else ""), color=FILTER_COLORS[f"{filter}-{model}"], marker=FILTER_MARKERS[f"{filter}-{model}"], markersize=2)

        # now process the stacked filter
        fprs = []
        for curr_query in query_counts:
            current_data = stacked_df[(stacked_df['dataset'] == dataset) & (stacked_df['num_queries'] == num_queries) & (stacked_df['curr_query'] == curr_query)]
            if current_data.empty:
                print(f"[EMPTY] {dataset}, stacked, {curr_query}")
            fpr = current_data['fpr'].iloc[0]
            fprs.append(fpr)
            overall_max = fpr if (overall_max is None or overall_max < fpr) else overall_max
            overall_min = fpr if fpr != 0 and (overall_min is None or overall_min > fpr) else overall_min
        # now plot the data
        axs[current_count].plot(query_counts, fprs, label=('stacked' if current_count == 0 else ""), color=FILTER_COLORS['stacked'], marker=FILTER_MARKERS['stacked'], markersize=2)
        axs[current_count].set_xlabel('Number of Queries')

        # now process the adaptive filter
        fprs = []
        for curr_query in query_counts:
            current_data = adaptive_data_for_dataset[adaptive_data_for_dataset['curr_query'] == curr_query] 
            if current_data.empty:
                print(f"[EMPTY] {dataset}, {curr_query}")
            fpr = current_data['fpr'].iloc[0]
            fprs.append(fpr)
            overall_max = fpr if (overall_max is None or overall_max < fpr) else overall_max
            overall_min = fpr if fpr != 0 and (overall_min is None or overall_min > fpr) else overall_min
        # now plot the data
        axs[current_count].plot(query_counts, fprs, label=('adaptiveqf' if current_count == 0 else ""), color=FILTER_COLORS['aqf'], marker=FILTER_MARKERS['aqf'], markersize=2)
        axs[current_count].set_xlabel('Number of Queries')

        axs[current_count].set_yscale('log')
        # after processing a dataset, move on to the next subplot
        current_count += 1
    fig.legend(loc='outside center right')
    plt.savefig(f'../results/figures/{output_name}')
    plt.clf()


if __name__ == "__main__":
    plot_fpr_space_tradeoff('../results/learned/overall_results_with_model_scores.csv', '../results/aqf/aqf_results.csv', '../results/stacked/stacked_results.csv')
    plot_adversarial('../results/learned/overall_advers_with_model_scores.csv', '../results/aqf/aqf_advers_results.csv', '../results/stacked/stacked_advers_results.csv')
    print("construction: ")
    plot_construction_times('../results/learned/results_with_model_builtin.csv', '../results/aqf/aqf_results.csv', '../results/stacked/stacked_results.csv')
    print("query: ")
    plot_query_times('../results/learned/results_with_model_builtin.csv', '../results/aqf/aqf_results.csv', '../results/stacked/stacked_results.csv')
    plot_model_degradation('../results/learned/degrad_results_with_model_scores.csv')
    plot_changing_model_exp('../results/learned/changing_model_size.csv')
    plot_dynamic_exp('../results/learned/dynamic_results_with_model_scores.csv', '../results/aqf/aqf_results_dynamic.csv', '../results/stacked/stacked_results_dynamic.csv',
                     learned_filters=["plbf", "adabf"], unique_datasets=["url", "ember", "caida"],
                     output_name="combined_fpr_dynamic.pdf")
    plot_dynamic_exp('../results/learned/dynamic_results_with_model_rebuild.csv', '../results/aqf/aqf_results_dynamic.csv', '../results/stacked/stacked_results_dynamic.csv',
                     learned_filters=["plbf", "adabf"], unique_datasets=["url", "ember", "caida"],
                     output_name="combined_fpr_dynamic_rebuild.pdf")