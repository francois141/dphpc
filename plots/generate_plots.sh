#/bin/bash

### ================ Percentile bar plots ================ ###

### V100 ###
python3 plot_bars_percentile.py --input results/results-v100.csv --output_folder results/ --gpu v100 --K all --percentile 0.5
python3 plot_bars_percentile.py --input results/results-v100.csv --output_folder results/ --gpu v100 --K all --percentile 0.95

# K = 32
python3 plot_bars_percentile.py --input results/results-v100.csv --output_folder results/ --gpu v100 --K 32 --percentile 0.5
python3 plot_bars_percentile.py --input results/results-v100.csv --output_folder results/ --gpu v100 --K 32 --percentile 0.95

# K = 64
python3 plot_bars_percentile.py --input results/results-v100.csv --output_folder results/ --gpu v100 --K 64 --percentile 0.5
python3 plot_bars_percentile.py --input results/results-v100.csv --output_folder results/ --gpu v100 --K 64 --percentile 0.95

# K = 128
python3 plot_bars_percentile.py --input results/results-v100.csv --output_folder results/ --gpu v100 --K 128 --percentile 0.5
python3 plot_bars_percentile.py --input results/results-v100.csv --output_folder results/ --gpu v100 --K 128 --percentile 0.95

### A100 ###
python3 plot_bars_percentile.py --input results/results-a100.csv --output_folder results/ --gpu a100 --K all --percentile 0.5
python3 plot_bars_percentile.py --input results/results-a100.csv --output_folder results/ --gpu a100 --K all --percentile 0.95

# K = 32
python3 plot_bars_percentile.py --input results/results-a100.csv --output_folder results/ --gpu a100 --K 32 --percentile 0.5
python3 plot_bars_percentile.py --input results/results-a100.csv --output_folder results/ --gpu a100 --K 32 --percentile 0.95

# K = 64
python3 plot_bars_percentile.py --input results/results-a100.csv --output_folder results/ --gpu a100 --K 64 --percentile 0.5
python3 plot_bars_percentile.py --input results/results-a100.csv --output_folder results/ --gpu a100 --K 64 --percentile 0.95

# K = 128
python3 plot_bars_percentile.py --input results/results-a100.csv --output_folder results/ --gpu a100 --K 128 --percentile 0.5
python3 plot_bars_percentile.py --input results/results-a100.csv --output_folder results/ --gpu a100 --K 128 --percentile 0.95

### ================ Speedup plots ================ ###

### V100 ###
python3 plot_speedup.py --input results/results-v100.csv --output_folder results/ --gpu v100 --ci 0.95

### A100 ###
python3 plot_speedup.py --input results/results-a100.csv --output_folder results/ --gpu a100 --ci 0.95

### ================ Performance plots ================ ###

### V100 ###
python3 plot_performance.py --input results/results-v100.csv --output_folder results/ --gpu v100 --ci 0.95

### A100 ###
python3 plot_performance.py --input results/results-a100.csv --output_folder results/ --gpu a100 --ci 0.95

### ================ Roofline plots ================ ###

### V100 ###
python3 plot_roofline.py --input results/results-v100.csv --output_folder results/ --gpu v100 --percentile 0.95

### A100 ###
python3 plot_roofline.py --input results/results-a100.csv --output_folder results/ --gpu a100 --percentile 0.95