import argparse
import math
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import pandas as pd
import seaborn as sns

pd.set_option('mode.chained_assignment', None) # should remove and fix the warning ...

CPU_SPEC = "Intel(R) Core(TM) iX-XXXXXX CPU @ X.00GHz"
GPU_SPEC = "NVIDIA XXX"

def read_df(path):
    df = pd.read_csv(path)
    return df

def plot_runtime(args: argparse.Namespace, df: pd.DataFrame, dataset_name):
    
    # Change figure size before plotting
    plt.figure(figsize=(12, 10))

    ### Titles ###
    plt.xlabel("K", loc="center", fontdict={ "size": "medium" })
    plt.ylabel("Runtime [ms]")
    plt.title(f"SDDMM runtime on the {dataset_name} dataset\nRunning on {CPU_SPEC} & {GPU_SPEC}\n", loc="center", y=1.05, fontdict={ "weight": "bold", "size": "large" })

    ### Scale ###
    plt.xscale("log", base=2)
    plt.yscale("log", base=2)
    
    ### Plot Computations ###
    df['exec_time_ms'] = df['exec_time'] / 1_000_000
    df['comp_repr'] = df[['competitor', 'mat_repr']].agg(' - '.join, axis=1)

    sns.lineplot(df, x="K", y="exec_time_ms", hue="comp_repr", legend=True, zorder=1)
    sns.scatterplot(df, x="K", y="exec_time_ms", hue="comp_repr", style="mat_repr", s=100, legend=False, zorder=5)

    sns.despine(left=True, bottom=False) # do not show axis line on the left but show it on the bottom (needs axes.linewidth & axes.edgecolor set)

    ### Legend ###
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.175), ncol=6, fancybox=True, fontsize="small")

    plt.tight_layout(rect=[ 0.05, 0.1, 0.95, 0.9 ])
    
    plt.savefig(args.output_folder + dataset_name + ".png", format="png") # plt.show()
    plt.close()

def main(args: argparse.Namespace):
    sns.set_theme(context="notebook", font_scale=1, style="darkgrid", rc={ "lines.linewidth": 2, "axes.linewidth": 1, "axes.edgecolor":"black" })

    df = read_df(args.input)

    datasets = pd.unique(df['dataset'])
    for dataset in datasets:
        plot_runtime(args, df[df['dataset'] == dataset], dataset)

if __name__ == "__main__":    
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--input", default="results/results.csv", type=str, help="CSV input path")
    argParser.add_argument("--output_folder", default="results/", type=str, help="Output folder")
    args = argParser.parse_args()
    main(args)
    
# csv = pd.read_csv("dummy_measures.csv")
# arr = np.array(csv).transpose()

# fig, ax = plt.subplots()

# keys = np.unique(arr[0])

# for key in keys:
#        ax.plot(arr[1][arr[0] == key], arr[2][arr[0] == key], label=key)

# ax.set(xlabel='Size', ylabel='Speed',
#        title='Benchmark SDDMM - dphpc')
# ax.grid()

# plt.legend(loc="upper left")
# fig.savefig("benchmark.png")
# #plt.show()