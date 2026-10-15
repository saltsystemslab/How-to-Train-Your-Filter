# Script to plot results from experiments. All tests must be finished running.

#!/bin/bash
set -e # exit immediately when a command fails

LEARNED_DIR="learned"

cd "$LEARNED_DIR"
# plot the comparison of the filters run from the adaptiveqf experiment harnesses (FPR for each workload, and
# construction and query times) to ../results/figures/generic
python3 plot_generic_results.py --results ../results --out ../results/figures/generic
# summarize every experiment in one radar chart, also saved to ../results/figures/generic
python3 plot_radar.py --results ../results --out ../results/figures/generic
# write the supplementary tables S1 and S2 (where false positives come from, and query repetition) to ../results/tables
python3 utils/write_supplementary_tables.py
# plot figures 7-14 from the python learned filter experiments, if they were run (RUN_PYTHON_LEARNED in run_queries.sh)
if [ -f ../results/learned/overall_results_with_model_scores.csv ]; then
    python3 plot_results.py
fi
# plot figure 6
python3 utils/graph_score_distributions.py
cd ..
