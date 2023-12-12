#/bin/bash

# 0.5 percentile
python3 plot_bars_percentile.py --input results/results-v100.csv --output_folder results/ --percentile 0.5 --gpu v100
python3 plot_bars_percentile.py --input results/results-a100.csv --output_folder results/ --percentile 0.5 --gpu a100

# 0.95 percentile
python3 plot_bars_percentile.py --input results/results-v100.csv --output_folder results/ --percentile 0.95 --gpu v100
python3 plot_bars_percentile.py --input results/results-a100.csv --output_folder results/ --percentile 0.95 --gpu a100