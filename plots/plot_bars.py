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

def plot_bars(args: argparse.Namespace, df: pd.DataFrame, dataset_name):
    fig, ax = plt.subplots()

    # Change figure size before plotting
    fig.set_size_inches((12, 10))

    ### Scale & Ticks ###
    ax.set_yscale("linear")
    # ax.get_yaxis().set_major_locator(plticker.LogLocator(base=10)) # ax.get_yaxis().set_major_locator(plticker.AutoLocator()) 
    # ax.get_yaxis().set_major_formatter(plticker.LogFormatterMathtext())

    # ax.get_yaxis().set_minor_locator(plticker.LogLocator(base=10, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))) # ax.get_yaxis().set_minor_locator(plticker.AutoMinorLocator())
    # ax.get_yaxis().set_minor_formatter(plticker.NullFormatter())

    # ax.tick_params(which='major', axis='both', width=1, length=7)
    # ax.tick_params(which='minor', axis='both', width=0.5, length=4, color='black')

    ### Plot Computations ###
    runtime_cols_ns = [ 'total_ns', 'init_ns', 'comp_ns', 'cleanup_ns' ]
    runtime_cols_ms = [ 'total_ms', 'Initialization', 'Computation', 'Cleanup' ]
    df[runtime_cols_ms] = df[runtime_cols_ns] / 1_000_000

    plot_df = df[ [ 'competitor', 'Initialization', 'Computation', 'Cleanup' ] ]
    plot_df = plot_df.set_index("competitor")
    plot_df = plot_df.groupby("competitor").mean()
    plot_df.plot(stacked=True, kind='bar', ax=ax)

    sns.despine(left=True, bottom=False) # do not show axis line on the left but show it on the bottom (needs axes.linewidth & axes.edgecolor set)

    ### Titles ###
    plt.xlabel("Competitor", loc="center", fontdict={ "size": "medium" })
    plt.ylabel("Runtime [ms]")
    plt.title(f"SDDMM runtime on the {dataset_name} dataset ({df.iloc[0]['N']}x{df.iloc[0]['M']})\nRunning on {CPU_SPEC} & {GPU_SPEC}\n", loc="center", y=1.05, fontdict={ "weight": "bold", "size": "large" })

    ### Legend ###
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1), ncol=6, fancybox=True, fontsize="small")

    plt.tight_layout(rect=[ 0.05, 0.1, 0.95, 0.9 ])
    
    os.makedirs(args.output_folder + "bars/", exist_ok=True)
    plt.savefig(args.output_folder + "bars/" + dataset_name + ".png", format="png") # plt.show()
    plt.close()

def main(args: argparse.Namespace):
    sns.set_theme(context="notebook", font_scale=1, style="darkgrid", rc={ "lines.linewidth": 2, "axes.linewidth": 1, "axes.edgecolor":"black", "xtick.bottom": True, "ytick.left": True }) # rc={ "xtick.top": True, "ytick.left": True }

    df = read_df(args.input)

    datasets = pd.unique(df['dataset'])
    for dataset in datasets:
        plot_bars(args, df[df['dataset'] == dataset], dataset)

if __name__ == "__main__":    
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--input", default="results/results.csv", type=str, help="CSV input path")
    argParser.add_argument("--output_folder", default="results/", type=str, help="Output folder")
    args = argParser.parse_args()
    main(args)
    