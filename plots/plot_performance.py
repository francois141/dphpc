import argparse
import math
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import pandas as pd
import seaborn as sns

pd.set_option('mode.chained_assignment', None) # should remove and fix the warning ...

SPECS = {
    "v100": [ "Intel(R) 6140 @ 2.30 GHz", "NVIDIA V100" ],
    "a100": [ "AMD EPYC 7742 @ 2.25 GHz", "NVIDIA A100" ],
}

RUNTIME_FIELD = "comp_ns" # total_ns, init_ns, comp_ns, cleanup_ns
flops = lambda NZ, K : 2*K * NZ + NZ

def read_df(path):
    df = pd.read_csv(path)
    return df

def plot_runtime(args: argparse.Namespace, df: pd.DataFrame, dataset_name):
    fig, ax = plt.subplots()

    # Change figure size before plotting
    fig.set_size_inches((12, 10))

    # Device & Input Metadata
    cpu, gpu = SPECS[args.gpu]
    first = df.iloc[0]
    runs = df[ (df["competitor"] == first['competitor']) & (df['mat_repr'] == first['mat_repr']) & (df['K'] == first['K']) ].shape[0]
    N = df.iloc[0]['N']
    M = df.iloc[0]['M']
    NZ = df.iloc[0]['NZ']
    density = (NZ / (N * M)) * 100
    density = round(density, 2 if density > 0.01 else 4)
    ci = int(args.ci * 100)
    dataset_name = str(dataset_name[0].upper() + dataset_name[1:])
    print(f"Plotting results for {dataset_name}")

    ### Titles ###
    plt.xlabel("K", loc="center", fontdict={ "size": "medium" })
    plt.ylabel("Performance [GFlops/s]")
    plt.title(f"SDDMM performance with R={runs} runs (showing {ci}% CI)\n{dataset_name} dataset: {N}x{M} with {density}% density\nRunning on {cpu} and {gpu}\n", loc="center", y=1, fontdict={ "weight": "bold", "size": "large" })

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
    # df['exec_time_s'] = df[args.runtime_field] / 1_000_000_000 # from ns to s
    # df['GFlops'] = df.apply(lambda row: flops(row['NZ'], row['K']) / 1_000_000_000, axis=1) # from flops to GFlops
    # df['performance'] = df['GFlops'] / df['exec_time_s']

    df['flops'] = df.apply(lambda row: flops(row['NZ'], row['K']), axis=1)
    df['performance'] = df['flops'] / df[args.runtime_field] # (flops / 10^9) / (ns / 10^9) = flops / ns

    df['comp_repr'] = df[['competitor', 'mat_repr']].agg(' - '.join, axis=1)

    sns.lineplot(df, x="K", y="performance", hue="comp_repr", legend=True, zorder=1, errorbar=("ci", ci), err_style="band", ax=ax)

    sns.despine(left=True, bottom=False) # do not show axis line on the left but show it on the bottom (needs axes.linewidth & axes.edgecolor set)

    ### Legend ###
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.175), ncol=4, fancybox=True, fontsize="small")

    plt.tight_layout(rect=[ 0.05, 0.1, 0.95, 0.9 ])
    
    plot_dir = f"{args.output_folder}{args.gpu}/performance_{args.runtime_field.split('_')[0]}/"
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(plot_dir + dataset_name + ".png", format="png") # plt.show()
    plt.close()

def main(args: argparse.Namespace):
    sns.set_theme(context="notebook", font_scale=1, style="darkgrid", rc={ "lines.linewidth": 2, "axes.linewidth": 1, "axes.edgecolor":"black", "xtick.bottom": True, "ytick.left": True }) # rc={ "xtick.top": True, "ytick.left": True }

    df = read_df(args.input)

    datasets = pd.unique(df['dataset'])
    for dataset in datasets:
        plot_runtime(args, df[df['dataset'] == dataset], dataset)

if __name__ == "__main__":    
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--input", default="results/results-v100.csv", type=str, help="CSV input path")
    argParser.add_argument("--output_folder", default="results/", type=str, help="Output folder")
    argParser.add_argument("--runtime_field", default=RUNTIME_FIELD, type=str, help="Runtime field")
    argParser.add_argument("--ci", default=0.95, type=float, help="Percentage confidence interval to draw")
    argParser.add_argument("--gpu", default="v100", type=str, help="GPU model")
    args = argParser.parse_args()
    main(args)