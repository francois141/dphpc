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

BASELINE = { "competitor": "CPU-Basic", "mat_repr": "CSR" }

RUNTIME_FIELD = "comp_ns" # total_ns, init_ns, comp_ns, cleanup_ns

def read_df(path):
    df = pd.read_csv(path)
    return df

def plot_speedup(args: argparse.Namespace, df: pd.DataFrame, dataset_name):
    fig, ax = plt.subplots()

    # Change figure size before plotting
    fig.set_size_inches((12, 10))

    ### Titles ###
    plt.xlabel("K", loc="center", fontdict={ "size": "medium" })
    plt.ylabel(f"Speedup [Baseline is {BASELINE['competitor']} ({BASELINE['mat_repr']})]")
    plt.title(f"SDDMM speedup on the {dataset_name} dataset ({df.iloc[0]['N']}x{df.iloc[0]['M']})\nRunning on {CPU_SPEC} & {GPU_SPEC}\n", loc="center", y=1.05, fontdict={ "weight": "bold", "size": "large" })

    ### Scale & Ticks ###
    ax.set_xscale("log") # log, linear
    ax.get_xaxis().set_major_locator(plticker.LogLocator(base=2)) # LogLocator, LinearLocator, AutoLocator
    ax.get_xaxis().set_major_formatter(plticker.LogFormatter(base=2)) # LogFormatter, LinearFormatter, NullFormatter
    ax.get_xaxis().set_minor_locator(plticker.NullLocator()) # AutoMinorLocator, NullLocator
    ax.get_xaxis().set_minor_formatter(plticker.NullFormatter())

    ax.set_yscale("log") # linear
    ax.get_yaxis().set_major_locator(plticker.LogLocator(base=10)) # ax.get_yaxis().set_major_locator(plticker.AutoLocator()) 
    ax.get_yaxis().set_major_formatter(plticker.LogFormatterMathtext())

    ax.get_yaxis().set_minor_locator(plticker.LogLocator(base=10, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))) # ax.get_yaxis().set_minor_locator(plticker.AutoMinorLocator())
    ax.get_yaxis().set_minor_formatter(plticker.NullFormatter())

    ax.tick_params(which='major', axis='both', width=1, length=7)
    ax.tick_params(which='minor', axis='both', width=0.5, length=4, color='black')

    ### Plot Computations ###
    baselines = df[np.logical_and.reduce([ df[k] == v for k,v in BASELINE.items() ])]
    baseline_times = baselines[args.runtime_field].reset_index(drop=True)

    df['comp_repr'] = df[['competitor', 'mat_repr']].agg(' - '.join, axis=1)
    df['speedup'] = 0.0
    grouped = df.groupby([ "competitor", "mat_repr" ])
    for _, group in grouped:
        reset_group = group.reset_index()
        reset_group['speedup'] = baseline_times / reset_group[args.runtime_field]
        reset_group = reset_group.set_index("index")
        df.loc[reset_group.index, 'speedup'] = reset_group['speedup']
    # => there must be a smarter way to do this ...

    sns.lineplot(df, x="K", y="speedup", hue="comp_repr", legend=True, zorder=1, ax=ax)
    sns.scatterplot(df, x="K", y="speedup", hue="comp_repr", style="mat_repr", s=100, legend=False, zorder=5, ax=ax)

    sns.despine(left=True, bottom=False) # do not show axis line on the left but show it on the bottom (needs axes.linewidth & axes.edgecolor set)

    ### Legend ###
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.175), ncol=6, fancybox=True, fontsize="small")

    plt.tight_layout(rect=[ 0.05, 0.1, 0.95, 0.9 ])
    
    speedup_str = "speedup_" + args.runtime_field.split("_")[0]
    os.makedirs(args.output_folder + speedup_str + "/", exist_ok=True)
    plt.savefig(args.output_folder + speedup_str + "/" + dataset_name + ".png", format="png") # plt.show()
    plt.close()

def main(args: argparse.Namespace):
    sns.set_theme(context="notebook", font_scale=1, style="darkgrid", rc={ "lines.linewidth": 2, "axes.linewidth": 1, "axes.edgecolor":"black", "xtick.bottom": True, "ytick.left": True }) # rc={ "xtick.top": True, "ytick.left": True }

    df = read_df(args.input)

    datasets = pd.unique(df['dataset'])
    for dataset in datasets:
        plot_speedup(args, df[df['dataset'] == dataset], dataset)

if __name__ == "__main__":    
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--input", default="results/results.csv", type=str, help="CSV input path")
    argParser.add_argument("--output_folder", default="results/", type=str, help="Output folder")
    argParser.add_argument("--runtime_field", default=RUNTIME_FIELD, type=str, help="Runtime field")
    args = argParser.parse_args()
    main(args)
    