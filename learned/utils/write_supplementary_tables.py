"""
Writes the supplementary tables that explain the false positive rate results of the adaptiveqf experiment harnesses
(test_*_generic), as csv files and LaTeX tables:

- S1 (fp_sources): where the learned filters' false positives come from in each workload, summed over trials and
  sizes (or one size, with --r): unchecked model decisions (the model's score let the key through without a backup
  filter check) or false positives of a backup filter. Only the untimed (TRACK_FP_SOURCE) runs record the sources.
- S2 (repetition): how often the negative keys of each static workload repeat, against the false positives and
  adaptations of the adaptiveqf and the false positives of the other filters, at one size (--r, default 5). A filter
  that cannot adapt repeats a false positive every time its key is queried again; the adaptiveqf adapts to each one.

Run from the learned folder (paths are relative to it by default), e.g.
    python3 utils/write_supplementary_tables.py
    python3 utils/write_supplementary_tables.py --r 8 --out ../results/tables_r8
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_generic_results import DATASETS, DATASET_SIZES, FILTERS, read  # noqa: E402

LEARNED = ["plbf", "clbf"]
# the static workloads, their names in the results and their query index files
WORKLOADS = {"onepass": ("One-pass", "onepass_{}.csv"), "unif": ("Uniform", "hashed_unif_10M_{}.csv"),
             "zipf": ("Zipfian", "hashed_zipf_10M_{}.csv")}
DATASET_NAMES = {"url": "URL", "ember": "Ember", "shalla": "Shalla", "caida": "CAIDA"}
FILTER_NAMES = {"aqf": r"\textsc{AdaptiveQF}", "stacked": "Stacked", "bloom": "Bloom", "plbf": r"\textsc{PLBF}",
                "clbf": r"\textsc{CLBF}"}

parser = argparse.ArgumentParser()
parser.add_argument("--results", default="../results", help="folder with the harnesses' result csv files")
parser.add_argument("--data", default="../data", help="data folder (query index files and cascaded/<dataset> labels)")
parser.add_argument("--out", default="../results/tables", help="folder to write the tables to")
parser.add_argument("--r", type=int, default=None,
                    help="the AQF r-value whose size to use (S1: default all sizes; S2: default 5)")
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)


def at_r(df, r):
    """The rows of the filters matched to the AQF size with the given r (all rows if r is None)."""
    if r is None:
        return df
    sizes = {d: s[r - 5] for d, s in DATASET_SIZES.items()}
    return df[df["budget"] == df["dataset"].map(sizes)]


def count(n):
    return f"{n:,.0f}"


def write(name, table, latex):
    table.to_csv(os.path.join(args.out, f"{name}.csv"), index=False)
    with open(os.path.join(args.out, f"{name}.tex"), "w") as f:
        f.write(latex)
    print(f"wrote {name}.csv and {name}.tex to {args.out}")


# S1: false positive sources of the learned filters
dist = read(os.path.join(args.results, "distribution_results.csv"))
advers = read(os.path.join(args.results, "advers_results.csv"))
advers["query_dist"] = "advers"
sources = pd.concat([at_r(dist, args.r), advers if args.r in (None, 6) else advers.iloc[0:0]])
sources = sources[sources["filter"].isin(LEARNED) & sources["fp_from_model"].notna()]
s1 = (sources.groupby(["dataset", "filter", "query_dist"])[["fp_from_model", "fp_from_backup"]].sum().reset_index())
s1["total"] = s1["fp_from_model"] + s1["fp_from_backup"]
s1["model_share"] = (s1["fp_from_model"] / s1["total"]).where(s1["total"] > 0)

workload_names = {**{k: v[0] for k, v in WORKLOADS.items()}, "advers": "Adversarial"}
datasets = [d for d in DATASETS if d in set(s1["dataset"])]
size_note = "all sizes" if args.r is None else f"the AQF size with $r={args.r}$"
lines = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\begin{tabular}{ll" + "r" * len(datasets) + "}", r"\toprule",
         "Filter & Workload & " + " & ".join(DATASET_NAMES[d] for d in datasets) + r" \\", r"\midrule"]
for f in LEARNED:
    for w in [w for w in workload_names if w in set(s1["query_dist"])]:
        cells = []
        for d in datasets:
            row = s1[(s1["dataset"] == d) & (s1["filter"] == f) & (s1["query_dist"] == w)]
            if row.empty:
                cells.append("--")
            elif row["total"].iloc[0] == 0:
                cells.append("no FPs")
            else:
                cells.append(f"{100 * row['model_share'].iloc[0]:.0f}\\% ({count(row['total'].iloc[0])})")
        lines.append(f"{FILTER_NAMES[f]} & {workload_names[w]} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule" if f != LEARNED[-1] else r"\bottomrule")
lines += [r"\end{tabular}",
          r"\caption{Share of the learned filters' false positives caused by unchecked model decisions (the rest are "
          r"false positives of a backup filter), with the total number of false positives in parentheses, summed over "
          rf"three trials and {size_note} (the adversarial test uses the AQF size with $r=6$ and sums over all "
          r"adversarial proportions).}",
          r"\label{tab:fp-sources}", r"\end{table}", ""]
write("s1_fp_sources", s1, "\n".join(lines))

# S2: repetition of negative queries against false positives and adaptations
r = args.r if args.r is not None else 5
labels = {}
rows = []
for d in DATASETS:
    labels_path = os.path.join(args.data, "cascaded", d, "labels.bin")
    if not os.path.exists(labels_path):
        continue
    labels = np.fromfile(labels_path, dtype=np.float32, offset=16)
    for w, (_, file_pattern) in WORKLOADS.items():
        query_path = os.path.join(args.data, "updated_query_indices", file_pattern.format(d))
        results = at_r(dist[(dist["dataset"] == d) & (dist["query_dist"] == w)], r)
        if not os.path.exists(query_path) or results.empty:
            continue
        queries = pd.read_csv(query_path)["index"].to_numpy()
        negatives = queries[labels[queries] == 0]
        distinct = len(np.unique(negatives))
        row = {"dataset": d, "workload": w, "negative_queries": len(negatives), "distinct_negatives": distinct,
               "repeat_share": 1 - distinct / len(negatives), "queries_per_negative": len(negatives) / distinct}
        # median false positives over the trials (fpr is the false positives over the negative queries)
        for f in FILTERS:
            fr = results[results["filter"] == f]
            row[f"{f}_fps"] = (fr["fpr"] * len(negatives)).round().median() if not fr.empty else np.nan
        aqf = results[results["filter"] == "aqf"]
        row["aqf_adaptations"] = aqf["num_adaptations"].median() if not aqf.empty else np.nan
        rows.append(row)
s2 = pd.DataFrame(rows)

other = [f for f in FILTERS if f != "aqf"]
lines = [r"\begin{table*}[t]", r"\centering", r"\small",
         r"\begin{tabular}{llrrrrr" + "r" * len(other) + "}", r"\toprule",
         r" & & & & & \multicolumn{2}{c}{\textsc{AdaptiveQF}} & \multicolumn{" + str(len(other)) +
         r"}{c}{False positives} \\", r"\cmidrule(lr){6-7} \cmidrule(lr){8-" + str(7 + len(other)) + "}",
         r"Dataset & Workload & Neg.\ queries & Distinct neg. & Queries/neg. & FPs & Adapt. & " +
         " & ".join(FILTER_NAMES[f] for f in other) + r" \\", r"\midrule"]
for d in [d for d in DATASETS if d in set(s2.get("dataset", []))]:
    part = s2[s2["dataset"] == d]
    for i, (_, row) in enumerate(part.iterrows()):
        cells = [DATASET_NAMES[d] if i == 0 else "", WORKLOADS[row["workload"]][0], count(row["negative_queries"]),
                 count(row["distinct_negatives"]), f"{row['queries_per_negative']:.1f}", count(row["aqf_fps"]),
                 count(row["aqf_adaptations"])] + [count(row[f"{f}_fps"]) if pd.notna(row[f"{f}_fps"]) else "--"
                                                   for f in other]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"
lines += [r"\end{tabular}",
          r"\caption{Repetition of the negative queries in each static workload (the negative queries, the distinct "
          r"negative keys among them, and the average number of times each is queried), against the median number of "
          r"false positives of each filter over three trials and the median number of adaptations of the "
          rf"\textsc{{AdaptiveQF}}, at the AQF size with $r={r}$.}}",
          r"\label{tab:repetition}", r"\end{table*}", ""]
write("s2_repetition", s2, "\n".join(lines))
