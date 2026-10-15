# Reruns the Python learned filter experiments on the model's share of the space (run_exp_with_changing_model.py)
# and on the proportion of negative keys in the training set (run_exp_model_degrad.py) with CascadedLBF's PLBF and
# XGBoost models, through the C++ harness (untimed, uniform workload, 3 trials, at the AQF's r = 5 size). Each runs
# in its own copy of the experiment layout under sweeps/ (see learned/utils/prepare_sweeps.py), so the main results
# are untouched. Then converts the results to the Python experiments' formats (learned/utils/collect_sweeps.py) and
# plots them as before, to results/figures/combined_changing_model_xgboost.pdf and combined_degrad_10M_xgboost.pdf.

#!/bin/bash
set -e
NUM_PARALLEL=16
MIN_FREE_GB=40
export OMP_NUM_THREADS=1
echo 1000 > /proc/self/oom_score_adj
LOW_PRIORITY="nice -n 19 ionice -c 3"
ROOT_DIR=$(pwd)

run_jobs() {
    while IFS= read -r cmd; do
        while [ "$(jobs -rp | wc -l)" -ge "$NUM_PARALLEL" ] || \
              [ "$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)" -lt "$MIN_FREE_GB" ]; do
            sleep 10
        done
        ( $LOW_PRIORITY bash -c "$cmd" || echo "failed (exit $?): $cmd" >> "$ROOT_DIR/sweeps/failed_jobs.txt" ) &
        sleep 10
    done < "$1"
    wait
}

echo "$(date +%H:%M) choosing and training the models"
(cd learned && $LOW_PRIORITY python3 utils/prepare_sweeps.py)

for sweep in model_proportion train_proportion; do
    echo "$(date +%H:%M) running $(wc -l < sweeps/$sweep/jobs.txt) $sweep experiments, $NUM_PARALLEL at a time"
done
cat sweeps/model_proportion/jobs.txt | sed 's|^|cd '"$ROOT_DIR"'/sweeps/model_proportion/adaptiveqf \&\& |' > sweeps/all_jobs.txt
cat sweeps/train_proportion/jobs.txt | sed 's|^|cd '"$ROOT_DIR"'/sweeps/train_proportion/adaptiveqf \&\& |' >> sweeps/all_jobs.txt
run_jobs sweeps/all_jobs.txt

echo "$(date +%H:%M) collecting and plotting"
cd learned
python3 utils/collect_sweeps.py
python3 -c "
import plot_results as pr
pr.plot_changing_model_exp('../results/learned/changing_model_size_xgboost.csv', 'combined_changing_model_xgboost.pdf')
pr.plot_model_degradation('../results/learned/degrad_results_xgboost.csv', 'combined_degrad_10M_xgboost.pdf')
"
cd ..
[ -s sweeps/failed_jobs.txt ] && { echo "FAILED:"; cat sweeps/failed_jobs.txt; }
echo "$(date +%H:%M) done"
