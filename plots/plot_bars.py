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

K = [ 32, 64, 128 ]
ALPHABET = list(string.ascii_lowercase)
COMPETITORS = [ "GPU-PyTorch", "GPU-DGL", "GPU-Dynamic" ]

BAR_WIDTH = 0.25

def read_df(path):
    df = pd.read_csv(path)
    return df

def plot_bars(args: argparse.Namespace, ax: plt.Axes, df: pd.DataFrame, k: int):
    percentile = int(args.percentile * 100)
    
    ### Scale ###
    ax.set_yscale("log") # linear
    ax.get_yaxis().set_major_locator(plticker.LogLocator(base=10)) # ax.get_yaxis().set_major_locator(plticker.AutoLocator()) 
    ax.get_yaxis().set_major_formatter(plticker.LogFormatterMathtext())

    ax.get_yaxis().set_minor_locator(plticker.NullLocator())
    ax.get_yaxis().set_minor_formatter(plticker.NullFormatter())

    ### Plot Computations ###
    df['computation'] = df['comp_ns'] / 1_000

    plot_df = df[ [ 'dataset', 'competitor', 'mat_repr', 'computation' ] ]
    plot_df = plot_df.groupby([ 'dataset', 'competitor', 'mat_repr' ]).quantile(args.percentile)
    plot_df = plot_df.sort_values(by=[ "dataset", "competitor", "mat_repr" ])
    plot_df = plot_df.reset_index()

    datasets = plot_df['dataset'].unique()
    competitors = COMPETITORS

    multiplier = 0
    x = np.arange(len(datasets))
    for competitor in competitors:
        comp_df = plot_df[plot_df['competitor'] == competitor]
        if (comp_df.shape[0] != datasets.shape[0]): # missing measurements
            print("not done " + str(k) + " " + str(competitor))
            print(comp_df['dataset'].unique())
            continue

        offset = BAR_WIDTH * multiplier
        cont = ax.bar(x + offset, comp_df['computation'], BAR_WIDTH, label=competitor)

        labels = comp_df['computation'].apply(lambda res: "NaN" if res == 1 else "")
        ax.bar_label(cont, labels=labels, padding=1)

        multiplier += 1

    ### Titles ###
    ax.set_ylabel(f"p{percentile} runtime [μs]")

    ### xTicks (last Axes only) ###
    if k == K[-1]:
        ax.set_xticks(x + BAR_WIDTH, datasets)
        plt.xticks(rotation=90)
    else:
        ax.set_xticks(x + BAR_WIDTH, [])

    ### Axes Titles ###
    ax.text(-0.5, 10e6 * 2, f"{ALPHABET[K.index(k)]}.) K={k}")


def plot_all(args: argparse.Namespace, df: pd.DataFrame):
    fig, ax = plt.subplots(3, 1)

    # Change figure size before plotting
    fig.set_size_inches((14, 8))

    # Device & Input Metadata
    cpu, gpu = SPECS[args.gpu]
    runs = 20
    percentile = int(args.percentile * 100)
    print(f"[All] Plotting bars")

    ### Plot Computations ###    
    for i, k in enumerate(K):
        df_K = df[df["K"] == k]
        plot_bars(args, ax[i], df_K, k)

    
    sns.despine(left=True, bottom=False) # do not show axis line on the left but show it on the bottom (needs axes.linewidth & axes.edgecolor set)

    ### Titles ###
    plt.xlabel("Dataset", loc="center", fontdict={ "size": "medium" })
    ax[0].set_title(f"SDDMM {percentile}-percentile runtime with R={runs}\nRunning on {cpu} and {gpu}\n", loc="center", fontdict={ "weight": "bold", "size": "large" })

    ### Legend (first Axes only) ###
    ax[0].legend(loc="best", ncols=3, fancybox=True, fontsize="small")

    plt.tight_layout(rect=[ 0.01, 0.01, 0.99, 0.99 ])
    
    plot_dir = f"{args.output_folder}{args.gpu}/bars/"
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(plot_dir + "all.png", format="png") # plt.show()
    plt.close()

        

def main(args: argparse.Namespace):
    sns.set_theme(context="notebook", font_scale=1, style="darkgrid", rc={ "lines.linewidth": 2, "axes.linewidth": 1, "axes.edgecolor":"black", "xtick.bottom": True, "ytick.left": True }) # rc={ "xtick.top": True, "ytick.left": True }
    sns.set_palette("Set1")

    df = read_df(args.input)
    
    drop_mask = df['competitor'] == 'CPU-Basic'
    drop_mask |= df['competitor'] == 'CPU-PyTorch'
    drop_mask |= df['competitor'] == 'GPU-Basic'
    drop_mask |= df['competitor'] == 'GPU-Thread-Dispatcher'
    drop_mask |= df['competitor'] == 'GPU-Tiled'
    drop_mask |= df['competitor'] == 'GPU-Shared'
    drop_mask |= df['competitor'] == 'GPU-Convert'
    drop_mask |= df['competitor'] == 'GPU-Preprocessing'
    drop_mask |= df['competitor'] == 'GPU-cuSPARSE'
    drop_mask |= df['dataset'] == 'RandomWithDensity'
    drop_mask |= df['dataset'] == 'LatinHypercube'
    drop_mask |= df['dataset'] == 'EMail-Enron'
    df = df.drop(df.index[drop_mask])

    df['dataset'] = df['dataset'].apply(lambda name: str(name[0].upper() + name[1:]))

    plot_all(args, df)
    

if __name__ == "__main__":    
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--input", default="results/results-v100.csv", type=str, help="CSV input path")
    argParser.add_argument("--output_folder", default="results/", type=str, help="Output folder")
    argParser.add_argument("--percentile", default=0.95, type=float, help="Runtime percentile")
    argParser.add_argument("--gpu", default="v100", type=str, help="GPU model")
    args = argParser.parse_args()
    main(args)
