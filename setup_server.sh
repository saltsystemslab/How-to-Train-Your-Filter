# Script to set up a machine to run the experiments: checks the tools and libraries, builds XGBoost (from
# CascadedLBF) and the experiment harnesses, and checks that the datasets, query files and learned filter models are
# present and consistent. Run from the root directory; it prints a summary and exits with an error if anything is
# missing.

#!/bin/bash

ADAPTIVE_DIR="adaptiveqf"
CASCADED_DIR="CascadedLBF"
datasets=("url" "ember" "shalla" "caida")
declare -A dataset_paths=(
    ["url"]="data/malicious_url_scores.csv"
    ["ember"]="data/combined_ember_metadata.csv"
    ["shalla"]="data/shalla_combined.csv"
    ["caida"]="data/caida.csv"
)
problems=0
problem() {
    echo "PROBLEM: $1"
    problems=$((problems + 1))
}

echo "== machine"
echo "$(hostname): $(nproc) cores, $(free -g | awk '/Mem:/ {print $2 " GB memory, " $7 " GB available"}'), load average $(cut -d' ' -f1-3 /proc/loadavg)"
lscpu | grep -E "Model name|Socket"

echo "== tools"
for tool in gcc g++ make cmake python3; do
    command -v "$tool" > /dev/null || problem "$tool is not installed"
done
echo '#include <openssl/rand.h>
int main(void) { return 0; }' | gcc -x c - -o /dev/null -lssl -lcrypto 2> /dev/null || problem "the OpenSSL development headers are not installed"
python3 -c "import numpy, pandas, matplotlib" 2> /dev/null ||
    problem "python3 needs numpy, pandas and matplotlib (e.g. pip3 install --user numpy pandas matplotlib)"

echo "== XGBoost (CascadedLBF/thirdparty/xgboost)"
xgboost_lib="$CASCADED_DIR/thirdparty/xgboost/lib/libxgboost.so"
if [ ! -f "$xgboost_lib" ] || ldd "$xgboost_lib" | grep -q "not found"; then
    echo "building XGBoost"
    (cd "$CASCADED_DIR/thirdparty/xgboost" && mkdir -p build && cd build && cmake .. > /dev/null && make -j "$(nproc)" > /dev/null) ||
        problem "XGBoost failed to build"
fi
[ -f "$xgboost_lib" ] && echo "found $xgboost_lib"

echo "== harnesses"
if (cd "$ADAPTIVE_DIR" && make clean > /dev/null && make all > /dev/null 2> ../logs_setup_build.txt); then
    echo "built the harnesses"
    rm -f logs_setup_build.txt
    # the harnesses run from the adaptiveqf folder, which their library paths are relative to
    (cd "$ADAPTIVE_DIR" && ldd test_distribution_generic | grep -q "not found") &&
        problem "the harnesses can't find a shared library (see ldd test_distribution_generic in $ADAPTIVE_DIR)"
else
    problem "the harnesses failed to build (see logs_setup_build.txt)"
fi

echo "== data"
# checks that each query index file only refers to rows of its dataset
python3 - "${datasets[@]}" << 'EOF' || problems=$((problems + 1))
import json, os, sys
import numpy as np
import pandas as pd

paths = {"url": "data/malicious_url_scores.csv", "ember": "data/combined_ember_metadata.csv",
         "shalla": "data/shalla_combined.csv", "caida": "data/caida.csv"}
ok = True
def problem(message):
    global ok
    ok = False
    print(f"PROBLEM: {message}")

for dataset in sys.argv[1:]:
    if not os.path.exists(paths[dataset]):
        problem(f"missing {paths[dataset]}")
        continue
    cascaded = f"data/cascaded/{dataset}"
    labels_path = f"{cascaded}/labels.bin"
    if not os.path.exists(labels_path):
        problem(f"missing {labels_path} (run prepare_learned_models.sh)")
        continue
    with open(labels_path, "rb") as f:
        rows, cols = np.fromfile(f, dtype=np.uint64, count=2)
        labels = np.fromfile(f, dtype=np.float32)
    for name in ["features.bin", "X_train.bin", "y_train.bin", "X_val.bin", "y_val.bin", "split.json"]:
        if not os.path.exists(f"{cascaded}/{name}"):
            problem(f"missing {cascaded}/{name} (run prepare_learned_models.sh)")
    models = [m for m in os.listdir(cascaded) if m.startswith("xgboost_") and os.path.exists(f"{cascaded}/{m}/xgboost_model.bin")]
    if "xgboost_d4_r100_e03" not in models:
        problem(f"{cascaded} is missing the clbf model xgboost_d4_r100_e03 (run prepare_learned_models.sh)")
    for kind in ["hashed_unif_10M", "hashed_zipf_10M", "onepass"]:
        path = f"data/updated_query_indices/{kind}_{dataset}.csv"
        if not os.path.exists(path):
            problem(f"missing {path} (run generate_queries.sh)")
            continue
        queries = pd.read_csv(path, usecols=[0]).iloc[:, 0]
        if queries.max() >= rows:
            problem(f"{path} refers to rows past the end of the dataset ({(queries >= rows).sum()} queries; regenerate it with generate_queries.sh)")
    print(f"{dataset}: {int(rows)} rows, {int((labels == 1).sum())} positives, {len(models)} models")
sys.exit(0 if ok else 1)
EOF

echo "== summary"
if [ "$problems" -eq 0 ]; then
    echo "ready: run test_small.sh to check that everything runs, then run_queries.sh"
else
    echo "$problems problem(s) found"
    exit 1
fi
