# Script to prepare the learned filters run from the adaptiveqf experiment harnesses (plbf, clbf) ahead of time.
# For each dataset, it exports the feature vectors the python learned filters use (plus XGBoost training and
# calibration sets) and trains the XGBoost models, saving everything under data/cascaded/<dataset>.
# Experiments then only load the saved models. Datasets and models that already exist are skipped.

#!/bin/bash
set -e # exit immediately when a command fails

LEARNED_DIR="learned"
datasets=("url" "ember" "shalla" "caida")

cd "$LEARNED_DIR"
for dataset in "${datasets[@]}"; do
    if [ -f "../data/cascaded/$dataset/features.bin" ]; then
        echo "features for $dataset already exported"
    else
        echo "exporting features for $dataset"
        python3 utils/export_cascaded_inputs.py --dataset "$dataset"
    fi
    echo "training models for $dataset"
    ./utils/train_cascaded_models.sh "$dataset"
done
cd ..
