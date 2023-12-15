#/bin/bash

input_folder="results/64nnz_per_thread"
input_v100="${input_folder}/v100/results-v100.csv"
input_a100="${input_folder}/a100/results-a100.csv"
output_a100="${input_folder}/a100/bars_percentile/"
output_v100="${input_folder}/v100/bars_percentile/"

# 0.5 percentile
python3 plot_bars_percentile.py --input $input_v100 --output_folder $output_v100 --percentile 0.5 --gpu v100
python3 plot_bars_percentile.py --input $input_a100 --output_folder $output_a100 --percentile 0.5 --gpu a100

# 0.95 percentile
python3 plot_bars_percentile.py --input $input_v100 --output_folder $output_v100 --percentile 0.95 --gpu v100
python3 plot_bars_percentile.py --input $input_a100 --output_folder $output_a100 --percentile 0.95 --gpu a100