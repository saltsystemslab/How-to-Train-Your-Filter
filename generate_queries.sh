# Script to generate query files for experiments.

#!/bin/bash
set -e # exit immediately when a command fails

ADAPTIVE_DIR="adaptiveqf"
LEARNED_DIR="learned"
NUM_QUERIES=10000000
declare -A dataset_paths=(
    ["url"]="$LEARNED_DIR/data/malicious_url_scores.csv"
    ["ember"]="$LEARNED_DIR/data/combined_ember_metadata.csv"
    ["shalla"]="$LEARNED_DIR/data/shalla_combined.csv"
    ["caida"]="$LEARNED_DIR/data/caida.csv"
)

# taken from counting properly-formatted non-header rows in datasets
declare -A dataset_rows=(
    ["url"]=162798
    ["ember"]=800000
    ["shalla"]=3905928
    ["caida"]=8493974
)

BINARY_TARGET="write_queries"

cd "$ADAPTIVE_DIR" || { echo "Failed to cd into $ADAPTIVE_DIR"; exit 1; }
if [ -f "$BINARY_TARGET" ]; then
    echo "make already done, $BINARY_TARGET exists"
else
    echo "need to run make"
    make all
fi
for dataset in "${!dataset_paths[@]}"; do
    # first, set up binaries containing list of query indexes
    path="${dataset_paths[$dataset]}"
    count="${dataset_rows[$dataset]}"
    ./"$BINARY_TARGET" "$NUM_QUERIES" "$count" "$dataset"
    ./"$BINARY_TARGET" "$NUM_QUERIES" "$count" "$dataset" "1.5"
    echo "finished making binaries for $dataset"
done
# now, set up a clean csv of query indexes to take advantage
# of vectorized operations in Python
cd "../$LEARNED_DIR"
python3 utils/convert_query_binaries.py
echo "finished converting binaries"
cd ..