# Times training the XGBoost models the learned filters (PLBF, CLBF) use in the timed uniform experiments, for the
# construction time figure: each model is retrained 3 times, single-threaded, with CascadedLBF's trainer and the
# same data and parameters as train_cascaded_models.sh, into a temporary folder (the saved models are left as they
# are). The trainer reports the time spent boosting, as in CascadedLBF's own experiments. Writes
# ../results/model_train_times.csv. Run from the learned folder, after the timed experiments.
set -e

export OMP_NUM_THREADS=1
trainer="$(realpath ../CascadedLBF/experiment/build/train_xgboost)"
results="../results/distribution_results.csv"
out="../results/model_train_times.csv"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# the models of the timed (uniform, r = 5) learned filter experiments
models=$(python3 -c "
import pandas as pd
d = pd.read_csv('$results')
t = d[d['build_time_ns'].notna() & d['filter'].isin(['plbf', 'clbf'])]
for dataset, model in sorted(set(zip(t['dataset'], t['params'].str.extract(r'model=([a-z0-9_]+)')[0]))):
    print(dataset, model)
")

echo "dataset,model,trial,train_time_ns,same_as_saved_model" > "$out"
while read -r dataset model; do
    data_dir="../data/cascaded/$dataset"
    depth=$(echo "$model" | sed -E 's/xgboost_d([0-9]+)_r.*/\1/')
    rounds=$(echo "$model" | sed -E 's/.*_r([0-9]+)_e.*/\1/')
    for trial in 0 1 2; do
        # a new folder for each run, since the trainer skips training when its folder already has a model
        run_dir="$tmp/${dataset}_${model}_$trial"
        mkdir "$run_dir"
        "$trainer" -x "$data_dir/X_train.bin" -y "$data_dir/y_train.bin" -o "$run_dir" -d "$depth" -r "$rounds" -e 0.3 > /dev/null
        ms=$(grep -o '"model_train_time_ms": [0-9.]*' "$run_dir/info.json" | awk '{print $2}')
        same=$(cmp -s "$run_dir/xgboost_model.bin" "$data_dir/$model/xgboost_model.bin" && echo 1 || echo 0)
        echo "$dataset,$model,$trial,$(python3 -c "print(int(float('$ms') * 1e6))"),$same" >> "$out"
        echo "$dataset $model trial $trial: $ms ms (same as the saved model: $same)"
    done
done <<< "$models"
