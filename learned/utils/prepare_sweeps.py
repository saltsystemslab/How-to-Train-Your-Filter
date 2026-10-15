"""
Prepares the two learned filter sweeps that rerun the old Python experiments with CascadedLBF's PLBF and XGBoost
models (see run_sweeps.sh), each in its own copy of the experiment layout under sweeps/<sweep> (adaptiveqf, data and
logs folders as the harnesses expect them, and its own results folder), so the main data and results are untouched:

- model_proportion: the PLBF at the AQF's r = 5 size with models taking about 10%, 20%, ..., 80% of the space, all of
  the depth of the model the PLBF uses at that size (so only the number of trees changes). The dataset's models
  within 3% of a target proportion are reused; otherwise one is trained (same data and parameters as
  train_cascaded_models.sh) into the sweep's own copy of the dataset's folder.
- train_proportion: the PLBF at the AQF's r = 5 size, with the model it uses there (the "auto" choice) retrained on
  20%, 40%, 60% and 80% of the negative keys (and the positives of the dataset's usual training set). Every model is
  calibrated on the same 10% of the negative keys (and the usual calibration positives), disjoint from all training
  sets.

Writes sweeps/<sweep>/jobs.txt, the harness commands to run (from sweeps/<sweep>/adaptiveqf). Models are trained
single-threaded, one at a time. Run from the learned folder.
"""
import json
import os
import subprocess

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
TRAINER = os.path.join(ROOT, "CascadedLBF", "experiment", "build", "train_xgboost")
NUM_QUERIES = 10000000
# the AQF's r = 5 sizes, the PLBF's model there and the dataset files
BUDGETS = {"url": 69160, "ember": 539890, "shalla": 4280640, "caida": 2144675}
AUTO_MODELS = {"url": "xgboost_d4_r7_e03", "ember": "xgboost_d4_r60_e03", "shalla": "xgboost_d4_r100_e03",
               "caida": "xgboost_d8_r30_e03"}
PATHS = {"url": "malicious_url_scores.csv", "ember": "combined_ember_metadata.csv", "shalla": "shalla_combined.csv",
         "caida": "caida.csv"}
MODEL_PROPORTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
PROPORTION_TOLERANCE = 0.03
TRAIN_PROPORTIONS = [0.2, 0.4, 0.6, 0.8]
CALIBRATION_NEG_FRACTION = 0.1


def load_bin(path):
    with open(path, "rb") as f:
        rows, cols = np.fromfile(f, dtype=np.uint64, count=2)
    return np.memmap(path, dtype=np.float32, mode="r", offset=16, shape=(int(rows), int(cols)))


def save_bin(path, data):
    data = np.ascontiguousarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    with open(path, "wb") as f:
        np.array([data.shape[0], data.shape[1]], dtype=np.uint64).tofile(f)
        data.tofile(f)


def train(out_dir, X_train, y_train, depth, rounds):
    """Trains a model into out_dir (skipped if it is already there), single-threaded."""
    if os.path.exists(os.path.join(out_dir, "xgboost_model.bin")):
        return
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run([TRAINER, "-x", X_train, "-y", y_train, "-o", out_dir, "-d", str(depth), "-r", str(rounds), "-e", "0.3"],
                   check=True, stdout=subprocess.DEVNULL, env={**os.environ, "OMP_NUM_THREADS": "1"})
    print(f"trained {out_dir} ({os.path.getsize(os.path.join(out_dir, 'xgboost_model.bin'))} bytes)", flush=True)


def link(src, dst):
    if not os.path.lexists(dst):
        os.symlink(src, dst)


def make_layout(sweep):
    """sweeps/<sweep> with the harnesses, the data (with an empty cascaded folder to fill), logs and results."""
    base = os.path.join(ROOT, "sweeps", sweep)
    for d in ["adaptiveqf", "data/cascaded", "logs", "results"]:
        os.makedirs(os.path.join(base, d), exist_ok=True)
    for entry in os.listdir(os.path.join(ROOT, "adaptiveqf")):
        link(os.path.join(ROOT, "adaptiveqf", entry), os.path.join(base, "adaptiveqf", entry))
    for entry in os.listdir(os.path.join(ROOT, "data")):
        if entry != "cascaded":
            link(os.path.join(ROOT, "data", entry), os.path.join(base, "data", entry))
    return base


def job(dataset, model, sweep_label):
    return (f"./test_distribution_generic_fpsource plbf ../data/{PATHS[dataset]} "
            f"../data/updated_query_indices/hashed_unif_10M_{dataset}.csv {NUM_QUERIES} {BUDGETS[dataset]} {model} "
            f"> ../logs/{sweep_label}_plbf_{dataset}_{model}.txt 2>&1")


def model_proportion():
    base = make_layout("model_proportion")
    jobs = []
    for dataset, budget in BUDGETS.items():
        real = os.path.join(ROOT, "data", "cascaded", dataset)
        mine = os.path.join(base, "data", "cascaded", dataset)
        os.makedirs(mine, exist_ok=True)
        for entry in os.listdir(real):
            link(os.path.join(real, entry), os.path.join(mine, entry))
        sizes = {m: os.path.getsize(os.path.join(real, m, "xgboost_model.bin")) for m in os.listdir(real)
                 if m.startswith("xgboost_") and os.path.exists(os.path.join(real, m, "xgboost_model.bin"))}
        # one family of models, as the old experiment varied one model type: the depth of the model the PLBF uses at
        # this size (AUTO_MODELS), with more or fewer trees
        depth = int(AUTO_MODELS[dataset].split("_d")[1].split("_")[0])
        sizes = {m: s for m, s in sizes.items() if m.startswith(f"xgboost_d{depth}_")}
        largest = max(sizes, key=sizes.get)
        per_round = sizes[largest] / int(largest.split("_r")[1].split("_")[0])
        for target in MODEL_PROPORTIONS:
            close = [m for m, s in sizes.items() if abs(s / budget - target) <= PROPORTION_TOLERANCE]
            if close:
                model = min(close, key=lambda m: abs(sizes[m] / budget - target))
            else:
                rounds = max(1, int(round(target * budget / per_round)))
                model = f"xgboost_d{depth}_r{rounds}_e03"
                if model not in sizes:
                    train(os.path.join(mine, model), os.path.join(real, "X_train.bin"), os.path.join(real, "y_train.bin"),
                          depth, rounds)
                    sizes[model] = os.path.getsize(os.path.join(mine, model, "xgboost_model.bin"))
            print(f"model_proportion {dataset}: target {target:.1f} -> {model} ({sizes[model] / budget:.3f})", flush=True)
            jobs.append(job(dataset, model, "model_proportion"))
    with open(os.path.join(base, "jobs.txt"), "w") as f:
        f.writelines(j + "\n" for j in jobs)


def train_proportion():
    base = make_layout("train_proportion")
    jobs = []
    for dataset in BUDGETS:
        real = os.path.join(ROOT, "data", "cascaded", dataset)
        mine = os.path.join(base, "data", "cascaded", dataset)
        os.makedirs(mine, exist_ok=True)
        for entry in ["features.bin", "labels.bin", "split.json"]:
            link(os.path.join(real, entry), os.path.join(mine, entry))
        split = json.load(open(os.path.join(real, "split.json")))
        X = load_bin(os.path.join(real, "features.bin"))
        labels = np.asarray(load_bin(os.path.join(real, "labels.bin"))[:, 0])
        # the same permutations as export_cascaded_inputs.py; the calibration negatives are the last 10% of the
        # negatives, so they are disjoint from every training set (the first 20%, ..., 80%)
        rng = np.random.default_rng(0)
        pos_rows = rng.permutation(np.flatnonzero(labels == 1))
        neg_rows = rng.permutation(np.flatnonzero(labels == 0))
        count = lambda rows, fraction: int(round(fraction * len(rows)))
        train_pos = pos_rows[:count(pos_rows, split["train_pos_fraction"])]
        val_rows = np.sort(np.concatenate([pos_rows[:count(pos_rows, split["val_pos_fraction"])],
                                           neg_rows[len(neg_rows) - count(neg_rows, CALIBRATION_NEG_FRACTION):]]))
        if not os.path.exists(os.path.join(mine, "X_val.bin")):
            save_bin(os.path.join(mine, "X_val.bin"), X[val_rows])
            save_bin(os.path.join(mine, "y_val.bin"), labels[val_rows])
        auto = AUTO_MODELS[dataset]
        depth, rounds = int(auto.split("_d")[1].split("_")[0]), int(auto.split("_r")[1].split("_")[0])
        for fraction in TRAIN_PROPORTIONS:
            model = f"trainneg_{int(fraction * 100)}_{auto}"
            if not os.path.exists(os.path.join(mine, model, "xgboost_model.bin")):
                train_rows = np.sort(np.concatenate([train_pos, neg_rows[:count(neg_rows, fraction)]]))
                X_path, y_path = os.path.join(mine, "X_train_tmp.bin"), os.path.join(mine, "y_train_tmp.bin")
                save_bin(X_path, X[train_rows])
                save_bin(y_path, labels[train_rows])
                train(os.path.join(mine, model), X_path, y_path, depth, rounds)
                os.remove(X_path)
                os.remove(y_path)
            print(f"train_proportion {dataset}: {fraction:.1f} of the negatives -> {model}", flush=True)
            jobs.append(job(dataset, model, "train_proportion"))
    with open(os.path.join(base, "jobs.txt"), "w") as f:
        f.writelines(j + "\n" for j in jobs)


if __name__ == "__main__":
    model_proportion()
    train_proportion()
