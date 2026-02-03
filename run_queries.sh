# Script to run all experiments. Learned filter experiments are run in background processes.

#!/bin/bash
set -e # exit immediately when a command fails

ADAPTIVE_DIR="adaptiveqf"
LEARNED_DIR="learned"
NUM_QUERIES=10000000

declare -A dataset_paths=(
    ["url"]="data/malicious_url_scores.csv"
    ["ember"]="data/combined_ember_metadata.csv"
    ["shalla"]="data/shalla_combined.csv"
    ["caida"]="data/caida.csv"
)

cd "$ADAPTIVE_DIR" || { echo "Failed to cd into $ADAPTIVE_DIR"; exit 1; }
make clean
make all
cd ..

r_values=(5 6 7 8 9 10 11)

for dataset in "${!dataset_paths[@]}"; do
    path="${dataset_paths[$dataset]}"
    cd "$LEARNED_DIR"
    q=$(python3 -m read_pos_counts.py --dataset "$dataset")
    cd ../"$ADAPTIVE_DIR"
    for r in "${r_values[@]}"; do
        # first, perform experiments with the adaptiveqf to set the size that the other filter types will use
        echo "running adaptive one-pass: $dataset $q $r"
        ./test_one_pass "../$path" "$q" "$r" > "../logs/one_pass_adaptive_$q_$r.txt" 2>&1
        sync
        sleep 0.5
        echo "running adaptive uniform: $dataset $q $r"
        ./test_distribution "../$path" "../data/updated_query_indices/hashed_unif_10M_$dataset.csv" "$NUM_QUERIES" "$q" "$r" > "../logs/uniform_adaptive_$dataset_$q_$r.txt" 2>&1
        sync
        sleep 0.5
        echo "running adaptive zipfian: $dataset $q $r"
        ./test_distribution "../$path" "../data/updated_query_indices/hashed_zipf_10M_$dataset.csv" "$NUM_QUERIES" "$q" "$r" > "../logs/zipfian_adaptive_$dataset_$q_$r.txt" 2>&1
        sync
        sleep 0.5
        if [ "$r" -eq 5 ] && [ "$dataset" != "shalla" ]; then
            echo "running adaptive dynamic: $dataset $q $r"
            ./test_dynamic "../$path" "../data/updated_query_indices/hashed_unif_10M_$dataset.csv" "$NUM_QUERIES" "$q" "$r" > "../logs/dynamic_adaptive_$dataset_$q_$r.txt" 2>&1
        fi
        sync
        sleep 0.5
        if [ "$r" -eq 6 ]; then
            echo "running adaptive adversarial: $dataset $q $r"
            (./test_advers_dist "../$path" "../data/updated_query_indices/hashed_unif_10M_$dataset.csv" "$NUM_QUERIES" "$q" "$r" > "../logs/advers_adaptive_$dataset_$q_$r.txt" 2>&1) || echo "Adversarial test failed, continuing..."
        fi
        sync
        sleep 0.5

        # now, go through each learned filter and run the same tests :)
        use the python script to get the correct size for the q and r values
        cd ../"$LEARNED_DIR"
        filter_size=$(python3 utils/obtain_filter_size.py --dataset "$dataset" --path "../results/aqf/aqf_results.csv" --q "$q" --r "$r")
        echo "size for $dataset $q $r: $filter_size"
        echo "rerunning all learned filter tests in the background..."
        # one-pass test
        python3 run_exp_with_model_prescores.py --datasets "$dataset" --N "1000" --k "5" --M "$filter_size" --filters "plbf" "adabf" --trials "3" --new_model > "../logs/one_pass_learned_$dataset_$q_$r.txt" 2>&1 &
        # uniform (+ adversarial) test
        python3 run_exp_with_model_prescores.py --datasets "$dataset" --query_path "../data/updated_query_indices/hashed_unif_10M_$dataset.csv" --N "1000" --k "5" --M "$filter_size" --filters "plbf" "adabf" --trials "3" --new_model --adv > "../logs/uniform_adaptive_$dataset_$q_$r.txt" 2>&1 &
        # Zipfian test
        python3 run_exp_with_model_prescores.py --datasets "$dataset" --query_path "../data/updated_query_indices/hashed_zipf_10M_$dataset.csv" --N "1000" --k "5" --M "$filter_size" --filters "plbf" "adabf" --trials "3" --new_model > "../logs/zipfian_learned_$dataset_$q_$r.txt" 2>&1 &
        if [ "$r" -eq 5 ] && [ "$dataset" != "shalla" ]; then
            # timing test
            python3 run_exp_with_model.py --datasets "$dataset" --query_path "../data/updated_query_indices/hashed_unif_10M_$dataset.csv" --N "1000" --k "5" --M "$filter_size" --filters "plbf" "adabf" --trials "3" --new_model > "../logs/timing_learned_$dataset_$q_$r.txt" 2>&1 &
            # dynamic tests
            python3 run_exp_dynamic_with_model_rebuild.py --datasets "$dataset" --query_path "../data/updated_query_indices/hashed_unif_10M_$dataset.csv" --N "1000" --k "5" --M "$filter_size" --filters "plbf" "adabf" --trials "1" --new_model > "../logs/dynamic_learned_rebuild_$dataset_$q_$r.txt" 2>&1 &
            python3 run_exp_dynamic_with_prescores.py --datasets "$dataset" --query_path "../data/updated_query_indices/hashed_unif_10M_$dataset.csv" --N "1000" --k "5" --M "$filter_size" --filters "plbf" "adabf" --trials "1" --new_model > "../logs/dynamic_learned_$dataset_$q_$r.txt" 2>&1 &
            # training set proportion test
            python3 run_exp_model_degrad.py --datasets "$dataset" --query_path "../data/updated_query_indices/hashed_unif_10M_$dataset.csv" --N "1000" --k "5" --M "$filter_size" --filters "plbf" "adabf" --trials "3" --new_model > "../logs/degrad_learned_$dataset_$q_$r.txt" 2>&1 &
        fi
        cd ../"$ADAPTIVE_DIR"

        # now, process the stacked filters >:)
        echo "running stacked filter experiments for $dataset $q $r with filter size $filter_size"
        echo "running stacked filter: $dataset $q $r one-pass"
        ./test_one_pass_stacked "../$path" "$filter_size" "0.25" "../data/updated_query_indices/onepass_$dataset.csv" 2>&1 > "../logs/onepass_stacked_$dataset_$q_$r.txt" 2>&1
        sync
        sleep 0.5
        echo "running stacked filter: $dataset $q $r uniform"
        ./test_distribution_stacked "../$path" "../data/updated_query_indices/hashed_unif_10M_$dataset.csv" "$NUM_QUERIES" "$filter_size" "0.25" > "../logs/uniform_stacked_$dataset_$q_$r.txt" 2>&1
        sync
        sleep 0.5
        echo "running stacked filter: $dataset $q $r zipfian"
        ./test_distribution_stacked "../$path" "../data/updated_query_indices/hashed_zipf_10M_$dataset.csv" "$NUM_QUERIES" "$filter_size" "0.25" > "../logs/zipfian_stacked_$dataset_$q_$r.txt" 2>&1
        sync
        sleep 0.5
        if [ "$r" -eq 5 ] && [ "$dataset" != "shalla" ]; then
            echo "running stacked filter: $dataset $q $r dynamic"
            ./test_dynamic_stacked "../$path" "../data/updated_query_indices/hashed_unif_10M_$dataset.csv" "$NUM_QUERIES" "$filter_size" "0.25" > "../logs/dynamic_stacked_$dataset_$q_$r.txt" 2>&1
            sync
            sleep 0.5
        fi
        if [ "$r" -eq 6 ]; then
            echo "running stacked adversarial: $dataset $q $r"
            ./test_advers_dist_stacked "../$path" "../data/updated_query_indices/hashed_unif_10M_$dataset.csv" "$NUM_QUERIES" "$filter_size" "0.25" > "../logs/adversarial_stacked_$dataset_$q_$r.txt" 2>&1
            sync
            sleep 0.5
        fi
    done
    the model proportion test is independent of adaptive filter size but dependent on the dataset
    cd ../"$LEARNED_DIR"
    python3 run_exp_with_changing_model.py --dataset "$dataset" --query_path "../data/updated_query_indices/hashed_unif_10M_$dataset.csv" --filters "plbf" "adabf" --trials "3" > "../logs/model_prop_learned_$dataset.txt" 2>&1 &
    cd ..
done