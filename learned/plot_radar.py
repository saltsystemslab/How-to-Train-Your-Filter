"""
Summarizes every experiment in one single-column radar chart: one polygon per filter over six axes, further out
is better. The Bloom filter is left out (RADAR_FILTERS), so every axis compares the AQF, the stacked filter and the
two learned filters only.

Each measured axis is scored the same way. Within one comparison group (a dataset, and where it applies a workload
and size), the filters' raw values (lower is better) are put on a log scale, since they span orders of magnitude,
and min-max normalized so that the best filter in the group scores 1 and the worst 0:

    score = log(worst / value) / log(worst / best)

A filter's score on the axis is the mean of its scores over the groups, so every dataset counts equally. The scores
are relative to the filters in RADAR_FILTERS: adding or removing a filter rescales every axis. They are drawn at
radius FLOOR + (1 - FLOOR) * score, so a filter that is worst everywhere is still a visible polygon.

A false positive rate of 0 is counted as 1 / (number of queries), the smallest rate a run can measure.
The scores are also written to radar_scores.csv next to the figure.

How the filters map onto each axis (scores from the current results, see radar_scores.csv):

    Construction Time   build + configuration time, plus the learned filters' model training time (from
                        model_train_times.csv), at the timing size on the uniform workload; per dataset.
                        AQF ~1 and stacked ~0.8: both only insert the keys, though the stacked filter also runs its
                        optimizer over the sampled negatives. PLBF ~0.2 and CLBF ~0: model training (seconds to over
                        a minute on ember) and calibration dominate, and the CLBF's cascade optimization is slowest.
    Query Time          amortized query time plus the AQF's amortized adapt time, same rows; per dataset.
                        AQF 1 (70-95 ns), stacked ~0.8 (120-180 ns, it queries several stacked layers), PLBF and
                        CLBF ~0.1 and ~0.05: every query runs XGBoost inference (0.2-4 us).
    Static Workload     every filter is built to the same sizes, so this is the FPR each filter reaches in a given
    FPR                 space on the static workloads: the median FPR over trials; per dataset, query
                        distribution (uniform, zipf, one-pass) and size.
                        PLBF and CLBF ~0.9: the model removes most negatives before the backup filters. Stacked
                        ~0.4: it spends space on the negatives it saw while sampling. AQF ~0: its remainder bits
                        give it the highest starting FPR per byte, and adapting only helps the repeated queries.
    Best-Case Dynamic   the lowest instantaneous FPR a filter reaches between two churns of the dynamic workloads
    FPR                 (the rate it settles to before the next churn): the median over the churn windows and
                        trials of each window's lowest FPR; per dataset and workload (swapping, shifting).
                        AQF ~0.75: it adapts to each churn's new false positives and settles near 0. The CLBF,
                        PLBF and stacked filter (~0.45, ~0.35, ~0.25) are fixed after construction, so they keep
                        their post-churn FPR until the next churn.
    Worst-Case Adv.     the highest median FPR over the adversarial frequencies; per dataset.
    FPR                 AQF 1: it fixes each false positive the adversary finds (peak FPR 0.0002-0.005). The others
                        ~0: the adversary repeats the false positives it found, so their FPR follows the
                        adversarial frequency (peak FPR 0.12-0.40).
    Workload            not measured, but from what each filter needs to be built (WORKLOAD_KNOWLEDGE).
    Independence        AQF 1 (positive keys only), PLBF and CLBF 0.5 (labeled negatives to train a model on),
                        stacked 0 (a sample of the query stream, the negatives with their frequencies).
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_generic_results import ADVERSARIAL_SIZES, DATASETS, DYNAMIC_WINDOW, FILTERS, TIMING_SIZES, read

# the filters compared (and the order they are drawn in); every axis is normalized among these only
RADAR_FILTERS = ["aqf", "stacked", "plbf", "clbf"]
# the radius a score of 0 is drawn at
FLOOR = 0.15
# how much each filter depends on knowing the workload before it is built: 1 needs nothing but the positive keys
# (the AQF learns from false positives as it goes), 0.5 needs a labeled sample of negative keys to train and
# calibrate a model on, 0 needs a sample of the query stream itself (the negative keys and their frequencies)
WORKLOAD_KNOWLEDGE = {"aqf": 1.0, "bloom": 1.0, "plbf": 0.5, "clbf": 0.5, "stacked": 0.0}
AXES = ["Construction\nTime", "Query\nTime", "Static\nWorkload FPR", "Best-Case\nDynamic FPR", "Worst-Case\nAdv. FPR",
        "Workload\nIndependence"]
# one column of a two-column paper (ACM / VLDB / IEEE are 3.3-3.5 in)
COLUMN_WIDTH = 3.33


def mean_scores(raw, groups):
    """
    raw has a filter column, the group columns and a value column (lower is better); returns each filter's mean
    score over the groups (see the module docstring).
    """
    logs = np.log10(raw["value"].astype(float))
    worst, best = logs.groupby([raw[g] for g in groups]).transform("max"), logs.groupby([raw[g] for g in groups]).transform("min")
    scores = ((worst - logs) / (worst - best)).where(worst > best, 1.0)
    return scores.groupby(raw["filter"]).mean()


def clip_fpr(df):
    return df["fpr"].clip(lower=1 / df["num_queries"])


def construction_and_query(dist, train_times, datasets):
    train = (train_times.groupby(["dataset", "model"])["train_time_ns"].median()
             if train_times is not None else pd.Series(dtype=float))
    rows = dist[dist["dataset"].isin(datasets) & (dist["query_dist"] == "unif") & dist["build_time_ns"].notna() &
                (dist["budget"] == dist["dataset"].map(TIMING_SIZES))].copy()
    models = rows["params"].str.extract(r"model=([a-z0-9_]+)")[0]
    rows["training"] = [train.get((d, m), 0.0) for d, m in zip(rows["dataset"], models)]
    rows["build"] = rows["build_time_ns"] + rows["config_time_ns"].fillna(0) + rows["training"]
    rows["query"] = rows["amortized_query_ns"] + rows["amortized_adapt_ns"].fillna(0)
    medians = rows.groupby(["dataset", "filter"])[["build", "query"]].median().reset_index()
    return (mean_scores(medians.rename(columns={"build": "value"}), ["dataset"]),
            mean_scores(medians.rename(columns={"query": "value"}), ["dataset"]))


def space_efficiency(dist, datasets):
    rows = dist[dist["dataset"].isin(datasets)].assign(value=clip_fpr)
    groups = ["dataset", "query_dist", "budget"]
    medians = rows.groupby(groups + ["filter"])["value"].median().reset_index()
    # only sizes every filter was run at, so each group compares the same filters
    medians = medians.groupby(groups).filter(lambda g: g["filter"].nunique() == medians["filter"].nunique())
    return mean_scores(medians, groups)


def best_dynamic(dynamic, datasets):
    rows = dynamic[dynamic["dataset"].isin(datasets)].assign(value=clip_fpr)
    rows["window"] = (rows["curr_query"] // (DYNAMIC_WINDOW * rows["num_queries"])).astype(int)
    lows = rows.groupby(["dataset", "workload", "filter", "trial", "window"])["value"].min()
    return mean_scores(lows.groupby(["dataset", "workload", "filter"]).median().reset_index(), ["dataset", "workload"])


def peak_adversarial(advers, datasets):
    rows = advers[advers["dataset"].isin(datasets) &
                  (advers["budget"] == advers["dataset"].map(ADVERSARIAL_SIZES))].assign(value=clip_fpr)
    peaks = rows.groupby(["dataset", "filter", "freq"])["value"].median().groupby(["dataset", "filter"]).max()
    return mean_scores(peaks.reset_index(), ["dataset"])


# shorter axis labels for the small radars of the split detail figure, which have less room around them
SHORT_AXES = ["Constr.\nTime", "Query\nTime", "Static\nFPR", "Best-Case\nDyn. FPR", "Worst-Case\nAdv. FPR",
              "Workload\nIndep."]
ANGLES = np.linspace(0, 2 * np.pi, len(AXES), endpoint=False)


def style(ax, labels=None, pad=4):
    """
    A radar axis with the first axis at the top, going clockwise, and rings at scores 0, 0.5 and 1. labels are the
    axis labels (None for none).
    """
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.set_xticks(ANGLES, labels if labels else [])
    ax.set_yticks([FLOOR, FLOOR + (1 - FLOOR) * 0.5, 1], [])
    ax.grid(color="#d0d0d0", lw=0.5)
    ax.spines["polar"].set_color("#b0b0b0")
    if labels:
        ax.tick_params(axis="x", pad=pad)
        # labels on the sides lean away from the circle so they don't run into the polygons
        for label, angle in zip(ax.get_xticklabels(), ANGLES):
            label.set_horizontalalignment("center" if np.isclose(np.sin(angle), 0) else
                                          "left" if np.sin(angle) > 0 else "right")


def polygon(row):
    """A filter's scores as a closed polygon of radii."""
    radii = FLOOR + (1 - FLOOR) * row.values
    return np.append(ANGLES, ANGLES[0]), np.append(radii, radii[0])


def draw_overview(ax, scores, legend):
    """Every filter's polygon overlaid on one labeled radar (with a legend entry each if legend)."""
    style(ax, AXES)
    # the outer ring is the best filter on an axis, the inner ring (radius FLOOR) the worst
    ax.text(np.radians(9), 1, "best", fontsize=5.5, color="#808080", ha="left", va="bottom")
    for f, row in scores.iterrows():
        spec = FILTERS[f]
        ax.plot(*polygon(row), color=spec["color"], lw=1.4, ls=spec["linestyle"], marker=spec["marker"],
                markersize=3.5, zorder=3, label=spec["label"] if legend else None)
        ax.fill(*polygon(row), color=spec["color"], alpha=0.07, zorder=2)


def draw_single(ax, scores, f, labels=None, markersize=2.2):
    """
    One filter's polygon in its color over the other filters in gray, titled with the filter's name. With labels,
    each axis is labeled with its name and the filter's score on it.
    """
    row = scores.loc[f]
    style(ax, [f"{label}\n{score:.2f}" for label, score in zip(labels, row.values)] if labels else None, pad=2)
    for other, orow in scores.iterrows():
        if other != f:
            ax.plot(*polygon(orow), color="#c8c8c8", lw=0.6, zorder=2)
    spec = FILTERS[f]
    ax.plot(*polygon(row), color=spec["color"], lw=1.2, ls=spec["linestyle"], marker=spec["marker"],
            markersize=markersize, zorder=3)
    ax.fill(*polygon(row), color=spec["color"], alpha=0.2, zorder=2)
    # with labels, the title goes above the top axis's three-line label
    ax.set_title(spec["label"], fontsize=7, fontweight="bold" if labels else "normal", pad=24 if labels else 3)


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"wrote {os.path.join(out_dir, name)}.pdf")


def plot(scores, out_dir):
    """
    The overlaid radar of every filter, with a row of small radars under it that replaces the legend: each names
    one filter and draws its polygon in its color, with the other filters in gray behind it.
    """
    # sized to be placed at 100% width in one column, so the 7 pt text matches a paper's caption size
    with plt.rc_context({"font.size": 7, "axes.linewidth": 0.6}):
        # positions in inches: the big radar is kept narrow enough for its side labels to fit in the column,
        # and the row of small radars sits under its bottom label
        height, main_size, small_size, small_bottom, main_bottom = 3.3, 1.62, 0.62, 0.06, 1.25
        fig = plt.figure(figsize=(COLUMN_WIDTH, height))
        inches = lambda left, bottom, w, h: [left / COLUMN_WIDTH, bottom / height, w / COLUMN_WIDTH, h / height]
        draw_overview(fig.add_axes(inches((COLUMN_WIDTH - main_size) / 2, main_bottom, main_size, main_size),
                                   projection="polar"), scores, legend=False)
        slot = COLUMN_WIDTH / len(scores)
        for i, f in enumerate(scores.index):
            draw_single(fig.add_axes(inches(slot * i + (slot - small_size) / 2, small_bottom, small_size, small_size),
                                     projection="polar"), scores, f)
        save(fig, out_dir, "generic_radar")


def plot_split(scores, out_dir):
    """
    The same chart as two single-column figures: generic_radar_overview, the overlaid radar with a legend, and
    generic_radar_detail, a 2x2 grid of one radar per filter with its axes labeled and its scores printed.
    """
    with plt.rc_context({"font.size": 7, "axes.linewidth": 0.6}):
        height, main_size, main_bottom = 2.6, 1.62, 0.6
        fig = plt.figure(figsize=(COLUMN_WIDTH, height))
        inches = lambda left, bottom, w, h: [left / COLUMN_WIDTH, bottom / height, w / COLUMN_WIDTH, h / height]
        ax = fig.add_axes(inches((COLUMN_WIDTH - main_size) / 2, main_bottom, main_size, main_size), projection="polar")
        draw_overview(ax, scores, legend=True)
        fig.legend(loc="lower center", ncol=len(scores), frameon=False, handlelength=2.2, columnspacing=1.2,
                   bbox_to_anchor=(0.5, 0))
        save(fig, out_dir, "generic_radar_overview")

        # each cell of the grid holds one radar, centered, with room for its axis labels (name and score) around it
        cols = 2
        rows = int(np.ceil(len(scores) / cols))
        cell_w, cell_h, size = COLUMN_WIDTH / cols, 1.55, 0.68
        height = cell_h * rows
        fig = plt.figure(figsize=(COLUMN_WIDTH, height))
        inches = lambda left, bottom, w, h: [left / COLUMN_WIDTH, bottom / height, w / COLUMN_WIDTH, h / height]
        with plt.rc_context({"font.size": 5.5}):
            for i, f in enumerate(scores.index):
                # a little right of center: the left-side labels ("Worst-Case") are the widest
                left = cell_w * (i % cols) + (cell_w - size) / 2 + 0.07
                bottom = cell_h * (rows - 1 - i // cols) + (cell_h - size) / 2 - 0.12
                draw_single(fig.add_axes(inches(left, bottom, size, size), projection="polar"), scores, f,
                            labels=SHORT_AXES, markersize=2.5)
        save(fig, out_dir, "generic_radar_detail")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="../results", help="folder holding the harnesses' combined csv files")
    parser.add_argument("--out", default="../results/figures/generic", help="folder to save the figure to")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, help="datasets to score")
    parser.add_argument("--split", action="store_true",
                        help="save the overview radar (with a legend) and the per-filter radars as two figures, "
                             "generic_radar_overview and generic_radar_detail, instead of one generic_radar")
    parser.add_argument("--no-training", action="store_true", help="leave model training out of construction time")
    args = parser.parse_args()
    plt.rcParams["text.usetex"] = False
    path = lambda name: os.path.join(args.results, name)
    # only the compared filters, so each axis is normalized among them
    load = lambda name: (lambda df: df[df["filter"].isin(RADAR_FILTERS)])(read(path(name)))

    dist = load("distribution_results.csv")
    train_times = None if args.no_training or not os.path.exists(path("model_train_times.csv")) else \
        pd.read_csv(path("model_train_times.csv"))
    build, query = construction_and_query(dist, train_times, args.datasets)
    dynamic = pd.concat([load("dynamic_results.csv"), load("shifting_results.csv")])
    scores = pd.DataFrame({
        AXES[0]: build, AXES[1]: query, AXES[2]: space_efficiency(dist, args.datasets),
        AXES[3]: best_dynamic(dynamic, args.datasets),
        AXES[4]: peak_adversarial(load("advers_results.csv"), args.datasets),
        AXES[5]: pd.Series(WORKLOAD_KNOWLEDGE),
    }).reindex([f for f in RADAR_FILTERS if f in build.index])

    print(scores.rename(columns=lambda c: c.replace("\n", " ")).round(3).to_string())
    os.makedirs(args.out, exist_ok=True)
    scores.rename(columns=lambda c: c.replace("\n", " ")).to_csv(os.path.join(args.out, "radar_scores.csv"))
    (plot_split if args.split else plot)(scores, args.out)
