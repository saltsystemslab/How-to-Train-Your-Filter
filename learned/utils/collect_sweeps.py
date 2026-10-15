"""
Converts the results of the learned filter sweeps (run_sweeps.sh) to the formats of the Python experiments they
replace, so plot_results.py plots them the same way: ../results/learned/changing_model_size_xgboost.csv (as
run_exp_with_changing_model.py writes) and ../results/learned/degrad_results_xgboost.csv (as run_exp_model_degrad.py
writes), with model_type "xgboost". Run from the learned folder.
"""
import os
import re

import pandas as pd

SWEEPS = os.path.join("..", "sweeps")
OUT = os.path.join("..", "results", "learned")


def read(sweep):
    df = pd.read_csv(os.path.join(SWEEPS, sweep, "results", "distribution_results.csv"))
    df = df[df["filter"] == "plbf"].copy()
    df["bytes"] = df["params"].str.extract(r"budget=(\d+)")[0].astype(int)
    df["model"] = df["params"].str.extract(r"model=([a-z0-9_]+)")[0]
    df["model_type"] = "xgboost"
    return df


os.makedirs(OUT, exist_ok=True)
model = read("model_proportion")
model["model_bytes"] = model["model_size"]
model["model_accuracy"] = float("nan")
model[["dataset", "filter", "bytes", "model_bytes", "num_queries", "model_accuracy", "fpr", "fp_from_model",
       "fp_from_backup", "model_type", "model"]].to_csv(os.path.join(OUT, "changing_model_size_xgboost.csv"), index=False)

train = read("train_proportion")
train["train_set_size"] = train["model"].map(lambda m: int(re.match(r"trainneg_(\d+)_", m).group(1)) / 100)
train[["dataset", "filter", "model_type", "bytes", "query_dist", "num_queries", "train_set_size", "fpr",
       "model"]].to_csv(os.path.join(OUT, "degrad_results_xgboost.csv"), index=False)
print(f"wrote {len(model)} model proportion and {len(train)} training proportion rows to {OUT}")
