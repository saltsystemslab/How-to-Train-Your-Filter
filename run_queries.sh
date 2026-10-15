# Script to run all experiments for the filters run from the adaptiveqf experiment harnesses (aqf, stacked, bloom,
# plbf, clbf). Results are appended to the csv files in results/ and logs are written to logs/.
#
# The experiments run in two phases:
# 1. Timed: the uniform workload at the size the construction and query times are reported for (TIMING_R), one
#    experiment at a time with nothing else running, so its build and query times are not affected by other
#    experiments (including waiting for CascadedLBF's temp-file lock). The adaptiveqf runs first on each dataset,
#    since its sizes are the sizes every other filter is built to.
# 2. Untimed: the uniform workload at the other sizes and the Zipfian, one-pass, adversarial and dynamic (swapping
#    and shifting) workloads, run in parallel with the harnesses built with TRACK_FP_SOURCE, which record whether
#    the learned filters' false positives came from unchecked model decisions or backup filter errors instead of
#    timing operations.
# The learned filters need prepare_learned_models.sh to have been run first.

#!/bin/bash
set -e # exit immediately when a command fails

ADAPTIVE_DIR="adaptiveqf"
LEARNED_DIR="learned"
NUM_QUERIES=${NUM_QUERIES:-10000000}
# the number of untimed experiments run at once, and the free memory (in GB) needed to start another one
NUM_PARALLEL=16
MIN_FREE_GB=40

# To share the machine without getting in the way of other users' work: every experiment runs at the lowest CPU and
# I/O priority, single-threaded (XGBoost otherwise uses every core; single-threaded is also faster for these models
# and matches the other filters), and is the first process the kernel kills if memory runs out.
export OMP_NUM_THREADS=1
echo 1000 > /proc/self/oom_score_adj || echo "couldn't lower this script's priority for the OOM killer"
LOW_PRIORITY="nice -n 19 ionice -c 3"

# Runs the commands in the given file, NUM_PARALLEL at a time, starting each only once MIN_FREE_GB of memory is free.
run_jobs() {
    while IFS= read -r cmd; do
        while [ "$(jobs -rp | wc -l)" -ge "$NUM_PARALLEL" ] || \
              [ "$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)" -lt "$MIN_FREE_GB" ]; do
            sleep 10
        done
        ( $LOW_PRIORITY bash -c "$cmd" || echo "failed (exit $?): $cmd" >> "$ROOT_DIR/logs/failed_jobs.txt" ) &
        # let the experiment allocate its memory before checking whether another one fits
        sleep 10
    done < "$1"
    wait
}
# the proportion of the queries the stacked filter learns its negatives from
STACKED_PROP=0.1
# the models the learned filters use: plbf picks the largest saved model up to 0.2x its budget
PLBF_MODEL="auto"
CLBF_MODEL="xgboost_d4_r100_e03"
# Set to 1 to also run the python learned filter experiments (plbf and adabf from learned/), which plot_results.py
# plots. They run in background processes, so they are run after both phases.
RUN_PYTHON_LEARNED=0

declare -A dataset_paths=(
    ["url"]="data/malicious_url_scores.csv"
    ["ember"]="data/combined_ember_metadata.csv"
    ["shalla"]="data/shalla_combined.csv"
    ["caida"]="data/caida.csv"
)
# DATASETS, R_VALUES and NUM_QUERIES can be set to run a smaller version of the experiments (e.g. to check that
# everything runs): DATASETS="url" R_VALUES="5 6" NUM_QUERIES=1000000 ./run_queries.sh
read -r -a datasets <<< "${DATASETS:-url ember shalla caida}"
filters=("aqf" "stacked" "bloom" "plbf" "clbf")
read -r -a r_values <<< "${R_VALUES:-5 6 7 8 9 10 11}"
# the adaptiveqf r-value whose size is used in the adversarial and in the dynamic experiments, and the one whose
# uniform experiments are timed (the size the construction and query times are reported for)
ADVERSARIAL_R=6
DYNAMIC_R=5
TIMING_R=5

ROOT_DIR=$(pwd)
mkdir -p logs results
cd "$ADAPTIVE_DIR" || { echo "Failed to cd into $ADAPTIVE_DIR"; exit 1; }
make clean
make all
cd ..

# the arguments of a filter built to the given size (the adaptiveqf takes q and r instead)
filter_args() {
    local filter=$1 size=$2 q=$3 r=$4
    case $filter in
        aqf) echo "$q $r";;
        stacked) echo "$size $STACKED_PROP";;
        bloom) echo "$size";;
        plbf) echo "$size $PLBF_MODEL";;
        clbf) echo "$size $CLBF_MODEL";;
    esac
}

# the size of the adaptiveqf with the given q and r, from the given results file
aqf_size() {
    (cd "$ROOT_DIR/$LEARNED_DIR" && python3 utils/obtain_filter_size.py --dataset "$1" --path "$ROOT_DIR/results/$2" --q "$3" --r "$4")
}

declare -A q_values shift_q_values
for dataset in "${datasets[@]}"; do
    q_values[$dataset]=$(cd "$ROOT_DIR/$LEARNED_DIR" && python3 read_pos_counts.py --dataset "$dataset")
    # the shifting workload's filters hold half of the positives
    shift_q_values[$dataset]=$(cd "$ROOT_DIR/$LEARNED_DIR" && python3 read_pos_counts.py --dataset "$dataset" --fraction 0.5)
done

# phase 1: the timed uniform experiments, one at a time
cd "$ADAPTIVE_DIR"
for dataset in "${datasets[@]}"; do
    path="../${dataset_paths[$dataset]}"
    q=${q_values[$dataset]}
    queries="../data/updated_query_indices/hashed_unif_10M_$dataset.csv"
    r=$TIMING_R
    for filter in "${filters[@]}"; do
        size=""
        [ "$filter" != "aqf" ] && size=$(aqf_size "$dataset" distribution_results.csv "$q" "$r")
        echo "running timed uniform: $filter $dataset q=$q r=$r ${size:+(size $size)}"
        $LOW_PRIORITY ./test_distribution_generic "$filter" "$path" "$queries" "$NUM_QUERIES" $(filter_args "$filter" "$size" "$q" "$r") \
            > "../logs/uniform_${filter}_${dataset}_${q}_${r}.txt" 2>&1
    done
done
# time training the timed learned filters' models (for the construction time figure), also with nothing else running
echo "timing the learned filters' model training"
(cd "$ROOT_DIR/$LEARNED_DIR" && $LOW_PRIORITY bash utils/time_model_training.sh)

# phase 2: the untimed experiments, in parallel. The adaptiveqf's untimed uniform experiments and the shifting
# workload's adaptiveqf (for half of the positives) run first, since their sizes are the sizes the other filters
# are built to.
sizing_jobs=$(mktemp)
for dataset in "${datasets[@]}"; do
    path="../${dataset_paths[$dataset]}"
    q=${q_values[$dataset]}
    for r in "${r_values[@]}"; do
        if [ "$r" -ne "$TIMING_R" ]; then
            echo "./test_distribution_generic_fpsource aqf $path ../data/updated_query_indices/hashed_unif_10M_$dataset.csv $NUM_QUERIES $q $r > ../logs/uniform_aqf_${dataset}_${q}_${r}.txt 2>&1" >> "$sizing_jobs"
        fi
    done
    shift_q=${shift_q_values[$dataset]}
    echo "./test_shifting_generic_fpsource aqf $path $NUM_QUERIES $shift_q $DYNAMIC_R > ../logs/shifting_aqf_${dataset}_${shift_q}_${DYNAMIC_R}.txt 2>&1" >> "$sizing_jobs"
done
echo "running the adaptiveqf experiments the other filters' sizes come from"
run_jobs "$sizing_jobs"
rm "$sizing_jobs"

jobs_file=$(mktemp)
for dataset in "${datasets[@]}"; do
    path="../${dataset_paths[$dataset]}"
    q=${q_values[$dataset]}
    queries="../data/updated_query_indices"
    for filter in "${filters[@]}"; do
        for r in "${r_values[@]}"; do
            size=$(aqf_size "$dataset" distribution_results.csv "$q" "$r")
            args=$(filter_args "$filter" "$size" "$q" "$r")
            if [ "$r" -ne "$TIMING_R" ] && [ "$filter" != "aqf" ]; then
                echo "./test_distribution_generic_fpsource $filter $path $queries/hashed_unif_10M_$dataset.csv $NUM_QUERIES $args > ../logs/uniform_${filter}_${dataset}_${q}_${r}.txt 2>&1" >> "$jobs_file"
            fi
            echo "./test_distribution_generic_fpsource $filter $path $queries/hashed_zipf_10M_$dataset.csv $NUM_QUERIES $args > ../logs/zipfian_${filter}_${dataset}_${q}_${r}.txt 2>&1" >> "$jobs_file"
            echo "./test_one_pass_generic_fpsource $filter $path $queries/onepass_$dataset.csv $args > ../logs/onepass_${filter}_${dataset}_${q}_${r}.txt 2>&1" >> "$jobs_file"
            if [ "$r" -eq "$ADVERSARIAL_R" ]; then
                echo "./test_advers_dist_generic_fpsource $filter $path $queries/hashed_unif_10M_$dataset.csv $NUM_QUERIES $args > ../logs/adversarial_${filter}_${dataset}_${q}_${r}.txt 2>&1" >> "$jobs_file"
            fi
            if [ "$r" -eq "$DYNAMIC_R" ]; then
                echo "./test_dynamic_generic_fpsource $filter $path $queries/hashed_unif_10M_$dataset.csv $NUM_QUERIES $args > ../logs/swapping_${filter}_${dataset}_${q}_${r}.txt 2>&1" >> "$jobs_file"
            fi
        done
        if [ "$filter" != "aqf" ]; then
            shift_q=${shift_q_values[$dataset]}
            size=$(aqf_size "$dataset" shifting_results.csv "$shift_q" "$DYNAMIC_R")
            args=$(filter_args "$filter" "$size" "$shift_q" "$DYNAMIC_R")
            echo "./test_shifting_generic_fpsource $filter $path $NUM_QUERIES $args > ../logs/shifting_${filter}_${dataset}_${shift_q}_${DYNAMIC_R}.txt 2>&1" >> "$jobs_file"
        fi
    done
done
echo "running $(wc -l < "$jobs_file") untimed experiments, $NUM_PARALLEL at a time"
run_jobs "$jobs_file"
rm "$jobs_file"
cd ..

if [ "$RUN_PYTHON_LEARNED" -eq 1 ]; then
    for dataset in "${datasets[@]}"; do
        q=${q_values[$dataset]}
        cd "$LEARNED_DIR"
        for r in "${r_values[@]}"; do
            filter_size=$(python3 utils/obtain_filter_size.py --dataset "$dataset" --path "../results/distribution_results.csv" --q "$q" --r "$r")
            echo "running the python learned filter experiments in the background: $dataset $q $r (size $filter_size)"
            # one-pass test
            python3 run_exp_with_model_prescores.py --datasets "$dataset" --N "1000" --k "5" --M "$filter_size" --filters "plbf" "adabf" --trials "3" --new_model > "../logs/one_pass_learned_${dataset}_${q}_${r}.txt" 2>&1 &
            # uniform (+ adversarial) test
            python3 run_exp_with_model_prescores.py --datasets "$dataset" --query_path "../data/updated_query_indices/hashed_unif_10M_$dataset.csv" --N "1000" --k "5" --M "$filter_size" --filters "plbf" "adabf" --trials "3" --new_model --adv > "../logs/uniform_learned_${dataset}_${q}_${r}.txt" 2>&1 &
            # Zipfian test
            python3 run_exp_with_model_prescores.py --datasets "$dataset" --query_path "../data/updated_query_indices/hashed_zipf_10M_$dataset.csv" --N "1000" --k "5" --M "$filter_size" --filters "plbf" "adabf" --trials "3" --new_model > "../logs/zipfian_learned_${dataset}_${q}_${r}.txt" 2>&1 &
            if [ "$r" -eq 5 ] && [ "$dataset" != "shalla" ]; then
                # timing test
                for model_type in random_forest decision_tree logistic_regression; do
                    python3 run_exp_with_model.py --datasets "$dataset" --query_path "../data/updated_query_indices/hashed_unif_10M_$dataset.csv" --N "1000" --k "5" --bytes "$filter_size" --filters "plbf" "adabf" --trials "3" --new_model --model_type "$model_type" > "../logs/timing_learned_${dataset}_${model_type}_${q}_${r}.txt" 2>&1 &
                done
                # dynamic tests
                python3 run_exp_dynamic_with_model_rebuild.py --datasets "$dataset" --query_path "../data/updated_query_indices/hashed_unif_10M_$dataset.csv" --N "1000" --k "5" --M "$filter_size" --filters "plbf" "adabf" --trials "1" > "../logs/dynamic_learned_rebuild_${dataset}_${q}_${r}.txt" 2>&1 &
                python3 run_exp_dynamic_with_prescores.py --datasets "$dataset" --query_path "../data/updated_query_indices/hashed_unif_10M_$dataset.csv" --N "1000" --k "5" --M "$filter_size" --filters "plbf" "adabf" --trials "1" --new_model > "../logs/dynamic_learned_${dataset}_${q}_${r}.txt" 2>&1 &
                # training set proportion test
                python3 run_exp_model_degrad.py --datasets "$dataset" --query_path "../data/updated_query_indices/hashed_unif_10M_$dataset.csv" --N "1000" --k "5" --M "$filter_size" --filters "plbf" "adabf" --trials "3" --new_model > "../logs/degrad_learned_${dataset}_${q}_${r}.txt" 2>&1 &
            fi
        done
        # the model proportion test is independent of adaptive filter size but dependent on the dataset
        python3 run_exp_with_changing_model.py --dataset "$dataset" --query_path "../data/updated_query_indices/hashed_unif_10M_$dataset.csv" --filters "plbf" "adabf" --trials "3" > "../logs/model_prop_learned_$dataset.txt" 2>&1 &
        cd ..
    done
fi
