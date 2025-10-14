# Script to plot results from experiments. All tests must be finished running.

#!/bin/bash
set -e # exit immediately when a command fails

LEARNED_DIR="learned"

cd "$LEARNED_DIR"
# plot figures 7-14
python3 plot_results.py
# plot figure 6
python3 utils/graph_score_distributions.py
cd ..