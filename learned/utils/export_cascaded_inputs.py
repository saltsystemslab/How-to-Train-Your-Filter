"""
Exports the inputs the CascadedLBF learned filters (PLBF, CLBF) need to run in the adaptiveqf experiment harnesses:

- features.bin / labels.bin: the feature vector and label of every row of the dataset, in file order, so the
  harnesses can look up a key's features by its row index. Features are the same ones the learned filters in this
  folder use (see updated_classifiers.obtain_raw_and_vectorized_keys).
- X_train.bin / y_train.bin: the XGBoost training set.
- X_val.bin / y_val.bin: the calibration set.
- split.json: the fractions of the positives and negatives in each set, which the learned filters also use when
  they retrain on new data (in the dynamic experiments).

The sets follow the splits of the CLBF paper (Sato and Matsui, "Cascaded Learned Bloom Filter for Optimal
Model-Filter Size Balance and Fast Rejection"): for url, 80% of the negatives for training and another 10% for
calibration, with all positives in both; for ember, 10% of the negatives for training and another 10% for
calibration, with 10% of the positives for training and all of them for calibration. The other datasets use
ember's split. Training and calibration negatives never overlap.

All files use CascadedLBF's binary format (size_t rows, size_t cols, then row-major float32), so the XGBoost model
can be trained with CascadedLBF's own train_xgboost, which uses the same XGBoost library the filters load models with.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from updated_classifiers import obtain_raw_and_vectorized_keys, CONFIG, PATH, OBJ_KEY, DATA_PATH  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', action="store", dest="dataset", type=str, required=True,
                    help="the dataset to export (i.e. url, ember, shalla, caida)")
parser.add_argument('--out_dir', action="store", dest="out_dir", type=str, default=None,
                    help="output folder (default: ../data/cascaded/<dataset>)")
parser.add_argument('--split', action="store", dest="split", type=str, default=None, choices=["url", "ember"],
                    help="which of the CLBF paper's splits to use (default: url's for url, ember's otherwise)")
parser.add_argument('--splits_only', action="store_true", dest="splits_only",
                    help="only rewrite the training and calibration sets, from previously exported features.bin/labels.bin")
parser.add_argument('--seed', action="store", dest="seed", type=int, default=0)
args = parser.parse_args()


def save_data_bin(path, data):
    data = np.ascontiguousarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    with open(path, "wb") as f:
        np.array([data.shape[0], data.shape[1]], dtype=np.uint64).tofile(f)
        data.tofile(f)


out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "cascaded", args.dataset)
os.makedirs(out_dir, exist_ok=True)


def load_data_bin(path):
    with open(path, "rb") as f:
        rows, cols = np.fromfile(f, dtype=np.uint64, count=2)
    return np.memmap(path, dtype=np.float32, mode="r", offset=16, shape=(int(rows), int(cols)))


if args.splits_only:
    X = load_data_bin(os.path.join(out_dir, "features.bin"))
    labels = np.asarray(load_data_bin(os.path.join(out_dir, "labels.bin"))[:, 0])
else:
    keys, X, labels = obtain_raw_and_vectorized_keys(args.dataset)
    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)

    # The harnesses look features up by the row index of the dataset csv they read, so the rows must be in
    # that csv's order (ember's features, for example, come from EMBER's own files rather than the csv).
    csv_keys = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", DATA_PATH, CONFIG[args.dataset][PATH]),
                           usecols=[CONFIG[args.dataset][OBJ_KEY]])[CONFIG[args.dataset][OBJ_KEY]].to_numpy()
    keys = np.asarray(keys)
    if len(keys) != len(csv_keys):
        raise SystemExit(f"{len(keys)} feature rows but {len(csv_keys)} csv rows")
    if not (keys.astype(str) == csv_keys.astype(str)).all():
        position = {k: i for i, k in enumerate(keys.astype(str))}
        if len(position) != len(keys) or set(position) != set(csv_keys.astype(str)):
            raise SystemExit("the feature rows can't be matched to the csv rows by key")
        order = np.array([position[k] for k in csv_keys.astype(str)])
        X, labels = X[order], labels[order]
        print("reordered feature rows to match the csv")
    print(f"{args.dataset}: {X.shape[0]} rows, {X.shape[1]} features, {int(labels.sum())} positives")
    save_data_bin(os.path.join(out_dir, "features.bin"), X)
    save_data_bin(os.path.join(out_dir, "labels.bin"), labels)

# the fractions of each label in the training and calibration sets (the CLBF paper's splits)
SPLITS = {
    "url": {"train_pos_fraction": 1.0, "val_pos_fraction": 1.0, "train_neg_fraction": 0.8, "val_neg_fraction": 0.1},
    "ember": {"train_pos_fraction": 0.1, "val_pos_fraction": 1.0, "train_neg_fraction": 0.1, "val_neg_fraction": 0.1},
}
split = SPLITS[args.split or ("url" if args.dataset == "url" else "ember")]

rng = np.random.default_rng(args.seed)
pos_rows = rng.permutation(np.flatnonzero(labels == 1))
neg_rows = rng.permutation(np.flatnonzero(labels == 0))
# positives may be in both sets; negatives are split into disjoint training and calibration sets
count = lambda rows, fraction: int(round(fraction * len(rows)))
num_train_neg, num_val_neg = count(neg_rows, split["train_neg_fraction"]), count(neg_rows, split["val_neg_fraction"])
train_rows = np.concatenate([pos_rows[:count(pos_rows, split["train_pos_fraction"])], neg_rows[:num_train_neg]])
val_rows = np.concatenate([pos_rows[:count(pos_rows, split["val_pos_fraction"])],
                           neg_rows[num_train_neg:num_train_neg + num_val_neg]])
for name, rows in [("train", train_rows), ("val", val_rows)]:
    rows = rng.permutation(rows)
    save_data_bin(os.path.join(out_dir, f"X_{name}.bin"), X[rows])
    save_data_bin(os.path.join(out_dir, f"y_{name}.bin"), labels[rows])
with open(os.path.join(out_dir, "split.json"), "w") as f:
    json.dump(split, f, indent=2)
print(f"wrote {'' if args.splits_only else 'features, labels, '}training ({len(train_rows)} rows) and calibration "
      f"({len(val_rows)} rows) sets to {out_dir}")
