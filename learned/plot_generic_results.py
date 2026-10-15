"""
Plots the results of the filters run through the adaptiveqf experiment harnesses (test_*_generic) in the same
style as plot_results.py: one subplot per dataset, the median over trials as a line with a band from the
minimum to the maximum, and a log-scaled false positive rate.

The results come from the combined csv files the harnesses write (distribution_results.csv, advers_results.csv,
dynamic_results.csv for the swapping workload and shifting_results.csv). Datasets and sizes without results are
skipped. A false positive rate of 0, which a log axis cannot show, is drawn at the bottom of the axis with a hollow
marker. In the uniform workload, where every negative key is queried about equally often, one negative key that is a
false positive gives an FPR of about 1 / (number of negative keys), so a rate of 0 is drawn on a dashed line at that
resolution instead (on the datasets with a 0).
"""
import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# colors, markers and line styles for each filter, staying constant across all graphs (plot_results.py uses the
# same colors for the filters it plots): the two CascadedLBF filters are a dark and a light blue, told apart by
# their markers and line styles as well
FILTERS = {
    "aqf": {"label": "adaptiveqf", "color": "#FF8C1A", "marker": "D", "linestyle": "-"},
    "stacked": {"label": "stacked", "color": "#1DB954", "marker": "o", "linestyle": "-"},
    "bloom": {"label": "bloom", "color": "#808080", "marker": "s", "linestyle": "-"},
    "plbf": {"label": "plbf", "color": "#1A5FD6", "marker": "^", "linestyle": "-"},
    "clbf": {"label": "clbf", "color": "#3EC1F3", "marker": "v", "linestyle": "--"},
}
DATASETS = ["url", "ember", "shalla", "caida"]
# the number of positive elements in each dataset, to convert sizes to bits per key
DATASET_POS = {"ember": 400000, "url": 55679, "shalla": 2926705, "caida": 1196194}
# the number of negative keys in each dataset, for the uniform workload's resolution (see above)
DATASET_NEG = {"ember": 400000, "url": 107117, "shalla": 979223, "caida": 7297780}
# the AQF sizes (r = 5 to 11) the other filters are matched to
DATASET_SIZES = {"shalla": [4280640, 4807488, 5334336, 5861184, 6388032, 6914880, 7441728],
                 "ember": [539890, 606338, 672786, 739234, 805682, 872130, 938578],
                 "url": [69160, 77672, 86184, 94696, 103208, 111720, 120232],
                 "caida": [2144675, 2408635, 2672595, 2936555, 3200515, 3464475, 3728435]}
# the size used by the filters in the adversarial experiments (the AQF's r = 6 size)
ADVERSARIAL_SIZES = {d: sizes[1] for d, sizes in DATASET_SIZES.items()}
# the dynamic workloads run in windows of this fraction of the queries: a warm-up window (not recorded), a
# window without a churn, then a churn at the start of each later window
DYNAMIC_WINDOW = 0.1
# the size used by the filters in the construction / query time plots (the AQF's r = 5 size)
TIMING_SIZES = {d: sizes[0] for d, sizes in DATASET_SIZES.items()}
# colors for each construction / query time component, staying constant across both timing graphs
TIMING_COLORS = dict(zip(["Configuration", "Model Scoring", "Filter Inserts", "Reverse Map Updates",
                          "Model Inference", "Filter Query", "Reverse Map Adapt", "Model Training"],
                         plt.get_cmap("tab20").colors))


def budget_of(row):
    """
    The AQF size a row was matched to: learned filters report their budget, the others their size. Sizes a little
    under an AQF size (the bloom filter rounds down to whole 64-bit words) are matched to that AQF size.
    """
    params = dict(kv.split("=") for kv in str(row["params"]).split(";") if "=" in kv)
    size = int(params["budget"]) if "budget" in params else int(row["size"])
    return next((s for s in DATASET_SIZES.get(row["dataset"], []) if 0 <= s - size < 8), size)


def read(path):
    df = pd.read_csv(path)
    df = df[df["filter"].isin(FILTERS)].copy()
    df["budget"] = df.apply(budget_of, axis=1)
    return df


def set_log_limits(axs, values):
    nonzero = [v for v in values if v > 0]
    if not nonzero:
        return None
    floor = math.pow(10, math.floor(math.log10(min(nonzero))))
    for ax in axs:
        ax.set_yscale("log")
        ax.set_ylim(floor / 2, math.pow(10, math.ceil(math.log10(max(nonzero)))))
    return floor / 2


def plot_series(ax, f, xs, meds, mins, maxes, zero_y, label, markersize=5):
    style = FILTERS[f]
    clip = lambda vs: [v if v > 0 else zero_y for v in vs]
    ax.plot(xs, clip(meds), lw=1.3, markersize=markersize, label=label, color=style["color"], marker=style["marker"],
            linestyle=style["linestyle"])
    ax.fill_between(xs, clip(mins), clip(maxes), alpha=0.2, color=style["color"])
    zeros = [(x, zero_y) for x, m in zip(xs, meds) if m == 0]
    if zeros:
        ax.plot(*zip(*zeros), linestyle="none", marker=style["marker"], markersize=markersize + 1,
                markerfacecolor="white", markeredgecolor=style["color"])


def figure(datasets, width_per=3):
    fig, axs = plt.subplots(1, len(datasets), figsize=(width_per * len(datasets) + 1.6, 2.2), layout="constrained", squeeze=False)
    return fig, list(axs[0])


def plot_fpr_vs_size(df, out_dir, datasets):
    """FPR against bits per key at each AQF size, for the uniform, Zipfian and one-pass workloads."""
    for dist, name in [("unif", "uniform"), ("zipf", "zipf"), ("onepass", "onepass")]:
        data = df[df["query_dist"] == dist]
        present = [d for d in datasets if len(data[data["dataset"] == d])]
        if not present:
            continue
        fig, axs = figure(present)
        all_values = data["fpr"].tolist()
        resolutions = {}
        if dist == "unif":
            # datasets with an FPR of 0 show it at the one-key resolution, which the axis must then include
            resolutions = {d: 1 / DATASET_NEG[d] for d in present if (data[data["dataset"] == d]["fpr"] == 0).any()}
            all_values += list(resolutions.values())
        zero_y = set_log_limits(axs, all_values)
        for i, d in enumerate(present):
            ax = axs[i]
            ax.set_title(d)
            if d in resolutions:
                ax.axhline(resolutions[d], color="gray", lw=0.8, ls="--", zorder=0)
                ax.annotate("1 negative key", (1, resolutions[d]), xycoords=("axes fraction", "data"),
                            xytext=(-2, -2), textcoords="offset points", ha="right", va="top", fontsize=7, color="gray")
            for f in FILTERS:
                rows = data[(data["dataset"] == d) & (data["filter"] == f)]
                stats = rows.groupby("budget")["fpr"].agg(["median", "min", "max"]).reindex(
                    [s for s in DATASET_SIZES[d] if s in set(rows["budget"])])
                if stats.empty:
                    continue
                bits = [b * 8 / DATASET_POS[d] for b in stats.index]
                plot_series(ax, f, bits, stats["median"], stats["min"], stats["max"], resolutions.get(d, zero_y),
                            FILTERS[f]["label"] if i == 0 else "")
            ax.set_xlabel("Bits per Key")
        axs[0].set_ylabel("False Positive Rate")
        fig.legend(loc="outside center right")
        save(fig, out_dir, f"generic_fpr_{name}")


def plot_adversarial(df, out_dir, datasets):
    """FPR against the proportion of adversarial queries, at the AQF's r = 6 size."""
    present = [d for d in datasets if len(df[(df["dataset"] == d) & (df["budget"] == ADVERSARIAL_SIZES[d])])]
    if not present:
        return
    fig, axs = figure(present)
    zero_y = set_log_limits(axs, df[df["budget"].isin(ADVERSARIAL_SIZES.values())]["fpr"].tolist())
    for i, d in enumerate(present):
        ax = axs[i]
        ax.set_title(d)
        for f in FILTERS:
            rows = df[(df["dataset"] == d) & (df["filter"] == f) & (df["budget"] == ADVERSARIAL_SIZES[d])]
            stats = rows.groupby("freq")["fpr"].agg(["median", "min", "max"])
            if stats.empty:
                continue
            plot_series(ax, f, list(stats.index), stats["median"], stats["min"], stats["max"], zero_y,
                        FILTERS[f]["label"] if i == 0 else "")
        ax.set_xlabel("Adversarial Frequency")
    axs[0].set_ylabel("False Positive Rate")
    fig.legend(loc="outside center right")
    save(fig, out_dir, "generic_fpr_adversarial")


def plot_dynamic(df, out_dir, datasets, name):
    """
    Instantaneous FPR over the query stream of a dynamic workload (swapping or shifting), with the churns marked.
    Each dataset's filters are all built to one size (the AQF's r = 5 size for that workload's positives).
    """
    present = [d for d in datasets if len(df[df["dataset"] == d])]
    if not present:
        return
    fig, axs = figure(present)
    zero_y = set_log_limits(axs, df["fpr"].tolist())
    for i, d in enumerate(present):
        ax = axs[i]
        ax.set_title(d)
        rows = df[df["dataset"] == d]
        # the filters are built to the AQF's size (or, without AQF results, the smallest size present)
        aqf = rows[rows["filter"] == "aqf"]
        budget = aqf["budget"].iloc[0] if len(aqf) else rows["budget"].min()
        num_queries = rows["num_queries"].iloc[0]
        for churn in np.arange(2, round(1 / DYNAMIC_WINDOW)) * DYNAMIC_WINDOW * num_queries:
            ax.axvline(churn, color="lightgray", lw=0.8, zorder=0)
        # (the shifting workload's AQF sizes aren't in DATASET_SIZES, so the bloom filter's size, rounded down to
        # whole 64-bit words, is matched to the AQF's here)
        matched = rows[(budget - rows["budget"] >= 0) & (budget - rows["budget"] < 8)]
        for f in FILTERS:
            stats = matched[matched["filter"] == f].groupby("curr_query")["fpr"].agg(["median", "min", "max"])
            if stats.empty:
                continue
            plot_series(ax, f, list(stats.index), stats["median"], stats["min"], stats["max"], zero_y,
                        FILTERS[f]["label"] if i == 0 else "", markersize=2)
        ax.set_xlabel("Number of Queries")
    axs[0].set_ylabel("Instantaneous FPR")
    fig.legend(loc="outside center right")
    save(fig, out_dir, name)


def timing_rows(df, d, f):
    """The timed uniform-workload rows for a filter at the timing size (rows from untimed builds have no times)."""
    return df[(df["dataset"] == d) & (df["filter"] == f) & (df["query_dist"] == "unif") &
              (df["budget"] == TIMING_SIZES[d]) & df["build_time_ns"].notna()]


def plot_stacked_times(df, out_dir, datasets, components, ylabel, name, fmt):
    """
    One subplot per dataset with a stacked bar per filter, each segment the median of one timing component.
    components maps a component name to a function from a filter's rows to that component's values.
    """
    present = [d for d in datasets if any(len(timing_rows(df, d, f)) for f in FILTERS)]
    if not present:
        return
    fig, axs = figure(present, width_per=3.4)
    used = set()
    for i, d in enumerate(present):
        ax = axs[i]
        ax.set_title(d)
        names = [f for f in FILTERS if len(timing_rows(df, d, f))]
        bottom = np.zeros(len(names))
        for part, values in components.items():
            heights = np.array([np.median(values(timing_rows(df, d, f))) for f in names])
            if not heights.any():
                continue
            ax.bar(range(len(names)), heights, bottom=bottom, width=0.6, color=TIMING_COLORS[part],
                   edgecolor="white", linewidth=1, label=part if part not in used else "")
            used.add(part)
            bottom += heights
        for x, total in enumerate(bottom):
            ax.text(x, total, fmt(total), ha="center", va="bottom", fontsize="x-small")
        ax.set_ylim(0, bottom.max() * 1.15)
        ax.set_xticks(range(len(names)), names, fontsize="small")
    axs[0].set_ylabel(ylabel)
    fig.legend(loc="outside center right")
    save(fig, out_dir, name)


def plot_construction_times(df, out_dir, datasets, train_times=None):
    """
    Construction time (s) at the AQF's r = 5 size on the uniform workload. Configuration is the time spent
    configuring the filter from its parameters (the stacked filter's optimizer; loading and calibrating a learned
    filter's model and its optimization at the target FPR) and is excluded from build_time_ns, so it is shown as its
    own segment. The time spent only matching the filters to the AQF's size (the learned filters' model choice and
    search for the target FPR) is in neither. Model Training is the median time to train each learned filter's
    model (train_times, from learned/utils/time_model_training.sh), measured separately.
    """
    s = lambda col: (lambda rows: rows[col] / 1e9)
    def training(rows):
        if train_times is None:
            return np.zeros(len(rows))
        medians = train_times.groupby(["dataset", "model"])["train_time_ns"].median() / 1e9
        models = rows["params"].str.extract(r"model=([a-z0-9_]+)")[0]
        return np.array([medians.get((d, m), 0.0) for d, m in zip(rows["dataset"], models)])
    plot_stacked_times(df, out_dir, datasets, {
        "Model Training": training,
        "Configuration": s("config_time_ns"),
        "Model Scoring": s("build_model_ns"),
        # the filter's own build time, without model scoring or auxiliary structures
        "Filter Inserts": lambda rows: (rows["build_time_ns"] - rows["aux_build_time_ns"] - rows["build_model_ns"]) / 1e9,
        "Reverse Map Updates": s("aux_build_time_ns"),
    }, "Construction Time (s)", "generic_construction_time", lambda v: f"{v:.2g}")


def plot_query_times(df, out_dir, datasets):
    """
    Amortized query time (ns) at the AQF's r = 5 size on the uniform workload. Learned filters time their model
    inference and backup filters separately on a sample of queries; the extra clock reads make those sampled
    queries slower than the rest, so the measured amortized query time is split in the sampled proportions.
    """
    def model_share(rows):
        sampled = rows["query_model_ns"] + rows["query_filter_ns"]
        return (rows["query_model_ns"] / sampled).where(sampled > 0, 0)
    plot_stacked_times(df, out_dir, datasets, {
        "Model Inference": lambda rows: rows["amortized_query_ns"] * model_share(rows),
        "Filter Query": lambda rows: rows["amortized_query_ns"] * (1 - model_share(rows)),
        "Reverse Map Adapt": lambda rows: rows["amortized_adapt_ns"],
    }, "Amort. Query Time (ns)", "generic_query_time", lambda v: f"{v:.0f}")


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{name}.pdf"))
    fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=160)
    plt.close(fig)
    print(f"wrote {os.path.join(out_dir, name)}.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="../results", help="folder holding the harnesses' combined csv files")
    parser.add_argument("--out", default="../results/figures", help="folder to save the figures to")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, help="datasets to plot (those without results are skipped)")
    args = parser.parse_args()
    plt.rcParams["text.usetex"] = False
    path = lambda name: os.path.join(args.results, name)
    if os.path.exists(path("distribution_results.csv")):
        distribution = read(path("distribution_results.csv"))
        plot_fpr_vs_size(distribution, args.out, args.datasets)
        train_times = pd.read_csv(path("model_train_times.csv")) if os.path.exists(path("model_train_times.csv")) else None
        plot_construction_times(distribution, args.out, args.datasets, train_times)
        plot_query_times(distribution, args.out, args.datasets)
    if os.path.exists(path("advers_results.csv")):
        plot_adversarial(read(path("advers_results.csv")), args.out, args.datasets)
    if os.path.exists(path("dynamic_results.csv")):
        plot_dynamic(read(path("dynamic_results.csv")), args.out, args.datasets, "generic_fpr_dynamic_swapping")
    if os.path.exists(path("shifting_results.csv")):
        plot_dynamic(read(path("shifting_results.csv")), args.out, args.datasets, "generic_fpr_dynamic_shifting")
