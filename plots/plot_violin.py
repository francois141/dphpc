import argparse
import math
import os
import string

import matplotlib as mat
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

def read_df(path):
    df = pd.read_csv(path)
    return df

def plot_violin(args: argparse.Namespace, df: pd.DataFrame, dataset_name: str):
    fig, ax = plt.subplots()

    # Change figure size before plotting
    fig.set_size_inches((12, 12))

    # Device & Input Metadata
    cpu, gpu = SPECS[args.gpu]
    first = df.iloc[0]
    runs = df[ (df["competitor"] == first['competitor']) & (df['mat_repr'] == first['mat_repr']) & (df['K'] == first['K']) ].shape[0]
    N = df.iloc[0]['N']
    M = df.iloc[0]['M']
    NZ = df.iloc[0]['NZ']
    density = (NZ / (N * M)) * 100
    density = round(density, 2 if density > 0.01 else 4)
    dataset_name = str(dataset_name[0].upper() + dataset_name[1:])
    print(f"[{dataset_name}] Plotting results with K={args.K}")

    ### Scale & Ticks ###
    plt.xticks(rotation=90)
    ax.set_yscale("linear")

    ### Plot Computations ###
    runtime_cols_ns = [ 'total_ns', 'init_ns', 'comp_ns', 'cleanup_ns' ]
    runtime_cols_us = [ 'total_us', 'Initialization', 'Computation', 'Cleanup' ]
    df[runtime_cols_us] = df[runtime_cols_ns] / 1_000

    plot_df = df[ [ 'competitor', 'mat_repr', 'K', 'Computation' ] ]
    plot_df = plot_df.sort_values(by=[ 'K', 'competitor', 'mat_repr' ])
    plot_df['identifier'] = plot_df.apply(lambda row: f"{row['competitor']} - {row['mat_repr']} ({row['K']})", axis=1)

    violin_plot = sns.violinplot(data=plot_df, x='identifier', y='Computation', scale='width', fill=True, ax=ax)

    medians = plot_df.groupby([ 'competitor', 'mat_repr', 'K' ]).median(numeric_only=True)['Computation']
    for xtick in violin_plot.get_xticks():
        median = medians.iloc[xtick]
        y_loc = median + medians.max() * 0.025
        violin_plot.text(xtick, y_loc, int(median), horizontalalignment='center', size='small', color='black', weight='semibold')
        
    sns.despine(left=True, bottom=False) # do not show axis line on the left but show it on the bottom (needs axes.linewidth & axes.edgecolor set)

    ### Titles ###
    plt.xlabel("Competitor - Matrix representation (K)", loc="center", fontdict={ "size": "medium" })
    plt.ylabel(f"Runtime [μs]")
    plt.title(f"SDDMM runtime with R={runs} runs\n{dataset_name} dataset: {N}x{M} with {density}% density\nRunning on {cpu} and {gpu}\n", loc="center", y=1, fontdict={ "weight": "bold", "size": "large" })

    plt.tight_layout(rect=[ 0.05, 0.1, 0.95, 0.9 ])
    
    plot_dir = f"{args.output_folder}{args.gpu}/violin/K_{args.K}/"
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(plot_dir + dataset_name + ".png", format="png") # plt.show()
    plt.close()

def main(args: argparse.Namespace):
    sns.set_theme(context="notebook", font_scale=1, style="darkgrid", rc={ "lines.linewidth": 2, "axes.linewidth": 1, "axes.edgecolor":"black", "xtick.bottom": True, "ytick.left": True }) # rc={ "xtick.top": True, "ytick.left": True }

    df = read_df(args.input)
    
    drop_mask = df['competitor'] == 'CPU-Basic'
    drop_mask |= df['competitor'] == 'CPU-PyTorch'
    drop_mask |= df['competitor'] == 'GPU-Basic'
    drop_mask |= df['competitor'] == 'GPU-Thread-Dispatcher'
    drop_mask |= df['competitor'] == 'GPU-Tiled'
    df = df.drop(df.index[drop_mask])

    datasets = pd.unique(df['dataset'])
    for dataset_name in datasets:
        K = int(args.K)
        df_mask = (df['dataset'] == dataset_name) & (df['K'] == K)
        df_dataset = df[df_mask]

        if df_dataset.empty:
            print(f"Dataset {dataset_name} has no measurements for K={K}")
            continue
            
        plot_violin(args, df_dataset, dataset_name)

if __name__ == "__main__":    
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--input", default="results/results-v100.csv", type=str, help="CSV input path")
    argParser.add_argument("--output_folder", default="results/", type=str, help="Output folder")
    argParser.add_argument("--gpu", default="v100", type=str, help="GPU model")
    argParser.add_argument("--K", default="32", type=str, help="Plot only for a specified K")
    args = argParser.parse_args()
    main(args)