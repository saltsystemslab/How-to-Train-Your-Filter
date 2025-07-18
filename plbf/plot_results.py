import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statistics import median

FILTER_COLORS = {"plbf": "blue", "adabf": "green", "aqf": "orange"}
COLORS = plt.get_cmap('tab10').colors

def plot_fpr_space_tradeoff(learned_filename: str, adaptive_filename: str, num_queries=[10000000], learned_filters=None):
    # first, grab the data from the learned results, and separate the data based on the filter
    learned_df = pd.read_csv(learned_filename)
    print(f'length of learned_df: {len(learned_df)}')
    adaptive_df = pd.read_csv(adaptive_filename)
    print(f'length of adaptive_df: {len(adaptive_df)}')
    if learned_filters is None:
        learned_filters = {"plbf", "adabf"}
    unique_datasets = set(learned_df['dataset'])
    unique_query_dists = set(learned_df['query_dist'])
    for dataset in unique_datasets:
        for query_num in num_queries:
            for query_dist in unique_query_dists:
                print("overall rows available: ", len(adaptive_df))
                data_for_dataset = adaptive_df[adaptive_df['dataset'] == dataset]
                print("number of rows that match dataset: ", len(data_for_dataset))
                data_for_num_queries = data_for_dataset[data_for_dataset['num_queries'] == query_num]
                print("query num: ", query_num)
                print("number of rows that match num_queries: ", len(data_for_num_queries))
                adaptive_data_for_query_dist = data_for_num_queries[data_for_num_queries['query_dist'] == query_dist]
                print("number of rows that match dist: ", len(adaptive_data_for_query_dist))
                # now for each filter size, get the median, min, and max
                unique_sizes = sorted(set(adaptive_data_for_query_dist['size']))
                for filter in learned_filters:
                    if filter not in FILTER_COLORS.keys():
                        print(f"{filter} not implemented...")
                        continue
                    data_for_dataset = learned_df[learned_df['dataset'] == dataset]
                    data_for_num_queries = data_for_dataset[data_for_dataset['num_queries'] == query_num]
                    data_for_query_dist = data_for_num_queries[data_for_num_queries['query_dist'] == query_dist]
                    data_for_filter = data_for_query_dist[data_for_query_dist['filter'] == filter]
                    if len(data_for_filter) == 0:
                        print(f"No data found for {filter}...")
                        continue
                    # now for each filter size, get the median, min, and max 
                    sizes = []
                    meds = []
                    mins = []
                    maxes = []
                    for size in unique_sizes:
                        current_size_data = data_for_filter[data_for_filter['bytes'] == size]
                        fprs = current_size_data['fpr']
                        sizes.append(size)
                        meds.append(median(fprs))
                        mins.append(min(fprs))
                        maxes.append(max(fprs))
                    # now plot the data
                    # print(f'{filter} data: {meds}')
                    plt.plot(sizes, meds, label=filter, color=FILTER_COLORS[filter])
                    plt.fill_between(sizes, mins, maxes, alpha=0.2, color=FILTER_COLORS[filter])

                    # TODO - for now, hold off on plotting error bars...
                # now process the adaptive filter
                
                print(f"sizes for adaptivqf: {unique_sizes}")
                sizes = []
                meds = []
                mins = []
                maxes = []
                for size in unique_sizes:
                    current_size_data = adaptive_data_for_query_dist[adaptive_data_for_query_dist['size'] == size]
                    if current_size_data.empty:
                        print(f"[EMPTY] {dataset}, {filter}, {num_queries}, {size}")
                    fprs = current_size_data['fpr']
                    sizes.append(size)
                    meds.append(median(fprs))
                    mins.append(min(fprs))
                    maxes.append(max(fprs))
                # print('adaptiveqf data: ', meds)
                # now plot the data
                print(f"adaptive data: ", meds)
                plt.plot(sizes, meds, label='adaptiveqf', color=FILTER_COLORS['aqf'])
                
                # label and save the data
                plt.xlabel('Filter Total Size (Bytes)')
                plt.ylabel('False-positive Rate')
                # plt.yscale('log')
                plt.title(f'FPR-Space Tradeoff on {dataset} ({query_num / 1000000}M {query_dist} Queries)')
                plt.legend()
                plt.savefig(f'fpr_{dataset}_{query_dist}_{query_num/1000000}M.pdf', bbox_inches='tight')
                plt.clf()

def plot_model_degradation(learned_filename: str, include_throughput=True):
    print("todo")
    # first, grab the data from the csv
    # next, set up row-aligned arrays of train size and fpr



# for this one, we want to try the best size for each filter that still results in a false-positive...
def plot_adversarial(learned_filename: str, adaptive_filename: str, learned_filters=None, max_valid_size=None):
    learned_df = pd.read_csv(learned_filename)
    adaptive_df = pd.read_csv(adaptive_filename)
    if learned_filters is None:
        learned_filters = {"plbf", "adabf"}
    unique_datasets = set(learned_df['dataset'])
    unique_num_queries = set(learned_df['num_queries'])
    for dataset in unique_datasets:
        for num_queries in unique_num_queries:
            current_size = 304992 if dataset == "url" else 1869472
            # find the maximum size such that there is no row where 'fpr' is 0
            # TODO - add this back in when the data is finished collecting...
            # current_size = max_valid_size
            # if max_valid_size is None:
            #     df_for_filter = learned_df[learned_df['dataset'] == dataset]
            #     nonzero_fpr_rows = df_for_filter.groupby('bytes').filter(lambda df: not (df['fpr'] == 0).any())
            #     current_size = nonzero_fpr_rows['bytes'].max()
            #     print(f"{dataset} filter size: {current_size}")
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
                for freq in unique_freqs:
                    data_for_freq = data_for_size[data_for_size['freq'] == freq]
                    fprs = data_for_freq['fpr']
                    freqs.append(freq)
                    meds.append(median(fprs))
                plt.plot(freqs, meds, label=f'{filter} ({current_size} bytes)', color=FILTER_COLORS[filter])
            # TODO - also do the adaptiveqf data...
            adaptive_data = adaptive_df[
                (adaptive_df['dataset'] == dataset) &
                (adaptive_df['num_queries'] == num_queries) &
                (adaptive_df['size'] == (304848 if dataset == "url" else current_size))
            ]
            freqs = []
            meds = []
            for freq in unique_freqs:
                data_for_freq = adaptive_data[adaptive_data['freq'] == freq]
                if data_for_freq.empty:
                    print(f"[EMPTY] {dataset}, adaptiveqf, {num_queries}, {current_size}")
                    continue
                fprs = data_for_freq['fpr']
                freqs.append(freq)
                meds.append(median(fprs))
            plt.plot(freqs, meds, label=f'Adaptiveqf ({current_size} bytes)', color=FILTER_COLORS['aqf'])

            # label and save the plot
            plt.xlabel('Overall Adversarial Query Frequency')
            plt.ylabel('False-Positive Rate')
            plt.yscale('log')
            plt.title(f'Adversarial Query FPR-Space Tradeoff on {dataset} ({num_queries / 1000000}M Queries)')
            plt.legend()
            plt.savefig(f'fpr_adversarial_{dataset}_{num_queries/1000000}M.pdf', bbox_inches='tight')
            plt.clf()


def plot_construction_times(plbf_filename: str, adabf_filename: str, adaptive_filename: str, learned_filters=None):
    plbf_df = pd.read_csv(plbf_filename)
    plbf_categories = ['Model Training', 'Score Inference', 'Segment Division', 'T/F Finding', 'Bloom Filter Init', 'Insertions']
    adabf_categories = ['Model Training', 'Score Inference', 'Threshold-Finding', 'Backup Filter Insertion']
    adaptiveqf_categories = ['CQF Insertion', 'Updating Reverse Map']
    all_categories = list(set(plbf_categories + adabf_categories + adaptiveqf_categories))
    if learned_filters is None:
        learned_filters = {"plbf", "adabf"}
    # construction times only differ across different datasets, since we always use the same model and insert the same keys
    unique_datasets = set(plbf_df['dataset'])
    unique_sizes = set(plbf_df['bytes'])
    for dataset in unique_datasets:
        for size in unique_sizes:
            plbf_data = plbf_df[
                    (plbf_df['dataset'] == dataset) &
                    (plbf_df['bytes'] == size)
                ]
            if len(plbf_data) != 0:
                bar = {'Model Training': median(plbf_data['train_time']), 'Score Inference': median(plbf_data['initial_scores']),
                       'Segment Division': median(plbf_data['segment_division']), 'T/F Finding': median(plbf_data['t_f_finding']),
                       'Bloom Filter Init': median(plbf_data['bloom_init']),
                       'Insertions': median(plbf_data['region_finding']) + median(plbf_data['filter_inserts'])}
                plbf_bar_heights = [bar.get(part, 0) for part in all_categories]
            adabf_data = []
            if len(adabf_data) != 0:
                print("todo")
            adaptive_data = []
            if len(adaptive_data) != 0:
                print("todo")
                
            x = np.arange(1)
            bar_width = 0.5

            bottom = np.zeros(1)
            colors = plt.cm.tab20.colors

            for (i, part) in enumerate(all_categories):
                values = [plbf_bar_heights[i]]
                plt.bar(x, values, bottom=bottom, label=part, color=colors[i], width=bar_width)
                bottom += values
            
            plt.xticks(x, ['plbf'])
            plt.xlim(-0.5, 0.5) # may need to get rid of this later?
            plt.ylabel("Runtime (seconds)")
            plt.legend()
            plt.title("Construction Time Breakdown")
            plt.tight_layout()
            plt.savefig(f'{dataset}_construction.pdf', bbox_inches='tight')  
            plt.clf()          

    if adabf_filename is not None:
        df = pd.read_csv(adabf_filename)
    # for the adaptive, collect...
    if adaptive_filename is not None:
        df = pd.read_csv(adaptive_filename)

def plot_query_times(plbf_filename: str, adabf_filename: str, adaptive_filename: str, learned_filters=None):
    plbf_df = pd.read_csv(plbf_filename)
    plbf_categories = ['Score Inference', 'Region-Finding', 'Backup Filter Query']
    adabf_categories = ['Score Inference', 'Threshold-Finding', 'Backup Filter Query']
    adaptiveqf_categories = ['Querying Reverse Map', 'Adapting']
    all_categories = list(set(plbf_categories + adabf_categories + adaptiveqf_categories))
    if learned_filters is None:
        learned_filters = {"plbf", "adabf"}
    # construction times only differ across different datasets, since we always use the same model and insert the same keys
    unique_datasets = set(plbf_df['dataset'])
    unique_sizes = set(plbf_df['bytes'])
    for dataset in unique_datasets:
        for size in unique_sizes:
            plbf_data = plbf_df[
                    (plbf_df['dataset'] == dataset) &
                    (plbf_df['bytes'] == size)
                ]
            if len(plbf_data) != 0:
                bar = {'Score Inference': median(plbf_data['med_score_time']),
                       'Region-Finding': median(plbf_data['med_region_time']),
                       'Backup Filter Query': median(plbf_data['med_back_filter_time'])}
                plbf_bar_heights = [bar.get(part, 0) for part in all_categories]
            adabf_data = []
            if len(adabf_data) != 0:
                print("todo")
            adaptive_data = []
            if len(adaptive_data) != 0:
                print("todo")
                
            x = np.arange(1)
            bar_width = 0.5

            bottom = np.zeros(1)
            colors = plt.cm.tab20.colors
            print("all categories: ", all_categories)
            for (i, part) in enumerate(all_categories):
                values = [plbf_bar_heights[i]]
                plt.bar(x, values, bottom=bottom, label=part, color=colors[i], width=bar_width)
                bottom += values
            
            plt.xticks(x, ['plbf'])
            plt.xlim(-0.5, 0.5) # may need to get rid of this later?
            plt.ylabel("Runtime (seconds)")
            plt.legend()
            plt.title("Median Query Time Breakdown")
            plt.tight_layout()
            plt.savefig(f'{dataset}_query.pdf', bbox_inches='tight')    
            plt.clf()        

    if adabf_filename is not None:
        df = pd.read_csv(adabf_filename)
    # for the adaptive, collect...
    if adaptive_filename is not None:
        df = pd.read_csv(adaptive_filename)


if __name__ == "__main__":
    plot_fpr_space_tradeoff('results/overall_results_with_prescores.csv', '../adaptiveqf/results/aqf_results.csv')
    plot_adversarial('results/overall_advers_with_prescores.csv', '../adaptiveqf/results/aqf_advers_results.csv')
    plot_construction_times('results/plbf_results_with_model.csv', None, None)
    plot_query_times('results/plbf_results_with_model.csv', None, None)