#!/bin/bash
# Trains XGBoost models for the CascadedLBF learned filters (PLBF, CLBF) on a dataset exported by
# export_cascaded_inputs.py, using CascadedLBF's own trainer so the models load with the same XGBoost
# library the filters use. Models are written to ../data/cascaded/<dataset>/xgboost_d<depth>_r<rounds>_e03.
#
# The PLBF adapter can pick among these by size ("auto"), so a range of boosting rounds is trained;
# depth 4 with 100 rounds is the model CascadedLBF uses for its CLBF experiments; it also uses depth 8.
# Run from the learned folder: ./utils/train_cascaded_models.sh url
set -e

dataset="$1"
if [ -z "$dataset" ]; then
    echo "usage: $0 <dataset>"
    exit 1
fi

cascaded_dir="../CascadedLBF"
data_dir="$(realpath ../data/cascaded/$dataset)"
trainer="$cascaded_dir/experiment/build/train_xgboost"

if [ ! -f "$data_dir/X_train.bin" ]; then
    echo "no training data in $data_dir; run utils/export_cascaded_inputs.py --dataset $dataset first"
    exit 1
fi
if [ ! -x "$trainer" ]; then
    echo "building $trainer"
    mkdir -p "$cascaded_dir/experiment/build"
    xgboost="$(realpath $cascaded_dir/thirdparty/xgboost)"
    g++ -O3 -DNDEBUG -std=c++17 -fno-strict-aliasing -I"$xgboost/include" -I"$cascaded_dir/thirdparty" \
        "$cascaded_dir/experiment/xgboost/train_xgboost.cpp" -o "$trainer" \
        "$xgboost/lib/libxgboost.so" -Wl,-rpath="$xgboost/lib"
fi

eta=0.3
train() {
    local depth=$1 rounds=$2
    local out_dir="$data_dir/xgboost_d${depth}_r${rounds}_e03"
    if [ -f "$out_dir/xgboost_model.bin" ]; then
        return
    fi
    mkdir -p "$out_dir"
    "$trainer" -x "$data_dir/X_train.bin" -y "$data_dir/y_train.bin" -o "$out_dir" -d $depth -r $rounds -e $eta > /dev/null
    echo "trained $out_dir ($(stat -c %s "$out_dir/xgboost_model.bin") bytes)"
}
# depth 4: small models (about 1.6 KB per round) for small budgets
for rounds in $(seq 1 20) 30 40 50 60 70 80 90 100; do
    train 4 $rounds
done
# depth 8 (also used in CascadedLBF's experiments): larger models, for the PLBF on larger datasets
for rounds in 10 20 30 40 50 60 70 80 90 100; do
    train 8 $rounds
done
