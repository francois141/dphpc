#/bin/bash

folder="results/pytorch-invest/"
input_v100="${folder}v100/results-v100.csv"
input_a100="${folder}a100/results-a100.csv"

### ================ Percentile bar plots ================ ###

### V100 ###
# python3 plot_bars_percentile.py --input $input_v100 --output_folder $folder --gpu v100 --K all --percentile 0.95
python3 plot_bars_percentile.py --input $input_v100 --output_folder $folder --gpu v100 --K 32 --percentile 0.95
# python3 plot_bars_percentile.py --input $input_v100 --output_folder $folder --gpu v100 --K 64 --percentile 0.95
# python3 plot_bars_percentile.py --input $input_v100 --output_folder $folder --gpu v100 --K 128 --percentile 0.95
# python3 plot_bars_percentile.py --input $input_v100 --output_folder $folder --gpu v100 --K 256 --percentile 0.95

### A100 ###
# python3 plot_bars_percentile.py --input $input_a100 --output_folder $folder --gpu a100 --K all --percentile 0.95
python3 plot_bars_percentile.py --input $input_a100 --output_folder $folder --gpu a100 --K 32 --percentile 0.95
# python3 plot_bars_percentile.py --input $input_a100 --output_folder $folder --gpu a100 --K 64 --percentile 0.95
# python3 plot_bars_percentile.py --input $input_a100 --output_folder $folder --gpu a100 --K 128 --percentile 0.95
# python3 plot_bars_percentile.py --input $input_a100 --output_folder $folder --gpu a100 --K 256 --percentile 0.95

## ================ Violin plots ================ ###

### V100 ###
python3 plot_violin.py --input $input_v100 --output_folder $folder --gpu v100 --K 32
# python3 plot_violin.py --input $input_v100 --output_folder $folder --gpu v100 --K 64
# python3 plot_violin.py --input $input_v100 --output_folder $folder --gpu v100 --K 128

### A100 ###
python3 plot_violin.py --input $input_a100 --output_folder $folder --gpu a100 --K 32
# python3 plot_violin.py --input $input_a100 --output_folder $folder --gpu a100 --K 64
# python3 plot_violin.py --input $input_a100 --output_folder $folder --gpu a100 --K 128

### ================ Speedup plots ================ ###

### V100 ###
python3 plot_speedup.py --input $input_v100 --output_folder $folder --gpu v100 --ci 0.95

### A100 ###
python3 plot_speedup.py --input $input_a100 --output_folder $folder --gpu a100 --ci 0.95

### ================ Performance plots ================ ###

### V100 ###
python3 plot_performance.py --input $input_v100 --output_folder $folder --gpu v100 --ci 0.95

### A100 ###
python3 plot_performance.py --input $input_a100 --output_folder $folder --gpu a100 --ci 0.95

### ================ Roofline plots ================ ###

### V100 ###
python3 plot_roofline.py --input $input_v100 --output_folder $folder --gpu v100 --percentile 0.95

### A100 ###
python3 plot_roofline.py --input $input_a100 --output_folder $folder --gpu a100 --percentile 0.95