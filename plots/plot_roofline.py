import argparse
import math
import os

import colorcet as cc
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import pandas as pd
import seaborn as sns

pd.set_option('mode.chained_assignment', None) # should remove and fix the warning ...
np.seterr(divide = 'ignore') # same

# peak_perf: GPU peak performance on FP32 (theoretical) in [flops/s]
# mem_bandwidth: GPU memory bandwidth (theoretical) in [bytes/s]
SPECS = {
    "v100": { "cpu": "Intel(R) 6140 @ 2.30 GHz", "gpu": "NVIDIA V100", "peak_perf": 14.13 * 10e12, "mem_bandwidth": 0.897 * 10e12 },
    "a100": { "cpu": "AMD EPYC 7742 @ 2.25 GHz", "gpu": "NVIDIA A100", "peak_perf": 19.49 * 10e12, "mem_bandwidth": 1.555 * 10e12 },
}

RUNTIME_FIELD = "comp_ns" # total_ns, init_ns, comp_ns, cleanup_ns
flops = lambda NZ, K : 2*K * NZ + NZ # each NZ requires K multiplications and K additions to sum up. Finally to scale the result by S need NZ multiplications.
trans_bytes = lambda M, N, K, NZ: (M*K + N*K)*4 + NZ*4  # Assuming infinite cache and floats: need to bring in dense A and B and sparse S (ignoring writes to P)

MAX_OP_INTENSITY = 500

def read_df(path):
    df = pd.read_csv(path)
    return df

def empty_legend_label(idx, labels, handles):
    labels.insert(idx, "") # insert empty label & handle to balance columns
    handles.insert(idx, matplotlib.patches.Rectangle((0,0), 1, 1, fill=False, edgecolor='none', visible=False))


def plot_roofline_all(args: argparse.Namespace, df: pd.DataFrame):
    fig, ax = plt.subplots()

    # Change figure size before plotting
    fig.set_size_inches((12, 10))
    
    # Device & Input Metadata
    specs = SPECS[args.gpu]
    cpu = specs['cpu']
    gpu = specs['gpu']
    gpu_pi = specs['peak_perf'] # GPU peak performance on FP32
    gpu_beta = specs['mem_bandwidth'] # GPU peak memory bandwidth
    gpu_bound = gpu_pi / gpu_beta # memory/compute bound (horizontal line)
    print(f"Plotting results for all")

    ### Titles ###
    plt.xlabel("Operational Intensity [Flops/byte]", loc="center", fontdict={ "size": "medium" })
    plt.ylabel("Performance [Flops/s]")
    plt.title(f"SDDMM roofline plot\nRunning on {cpu} and {gpu}\n", loc="center", y=1, fontdict={ "weight": "bold", "size": "large" })

    ### Scale & Ticks ###
    ax.set_xscale("log") # log, linear
    ax.get_xaxis().set_major_locator(plticker.LogLocator(base=10)) # LogLocator, LinearLocator, AutoLocator
    ax.get_xaxis().set_major_formatter(plticker.LogFormatterMathtext()) # LogFormatter, LinearFormatter, NullFormatter

    ax.get_xaxis().set_minor_locator(plticker.LogLocator(base=10, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))) # ax.get_yaxis().set_minor_locator(plticker.AutoMinorLocator())
    ax.get_xaxis().set_minor_formatter(plticker.NullFormatter())

    ax.set_yscale("log") # linear
    ax.get_yaxis().set_major_locator(plticker.LogLocator(base=10)) # ax.get_yaxis().set_major_locator(plticker.AutoLocator()) 
    ax.get_yaxis().set_major_formatter(plticker.LogFormatterMathtext())

    ax.get_yaxis().set_minor_locator(plticker.LogLocator(base=10, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))) # ax.get_yaxis().set_minor_locator(plticker.AutoMinorLocator())
    ax.get_yaxis().set_minor_formatter(plticker.NullFormatter())

    ax.tick_params(which='major', axis='both', width=1, length=7)
    ax.tick_params(which='minor', axis='both', width=0.5, length=4, color='black')

    ### Rooflines ###
    x = np.arange(0, MAX_OP_INTENSITY, 0.25)
    y = gpu_beta * x
    sns.lineplot(x=x, y=y, color="black")

    plt.hlines(gpu_pi, color="black", ls="solid", xmin=gpu_bound, xmax=MAX_OP_INTENSITY)
    plt.vlines(gpu_bound, color="black", alpha=0.4, ls="--", ymin=0, ymax=gpu_pi)

    ### Plot Computations ###
    df['comp_repr'] = df[['competitor', 'mat_repr']].agg(' - '.join, axis=1)

    df['flops'] = df.apply(lambda row: flops(row['NZ'], row['K']), axis=1)
    df['seconds'] = df[RUNTIME_FIELD] / 10e9
    df['bytes'] = df.apply(lambda row: trans_bytes(row['M'], row['N'], row['K'], row['NZ']), axis=1)

    df['performance'] = df['flops'] / df['seconds']
    df['op_intensity'] = df['flops'] / df['bytes']

    df = df[ [ 'dataset', 'comp_repr', 'K', 'performance', 'op_intensity' ] ]
    df = df.groupby([ 'dataset', 'comp_repr', 'K' ]).quantile(args.percentile)

    sns.lineplot(df, x="op_intensity", y="performance", hue="dataset", style="comp_repr", dashes=False, legend=False, zorder=1, ax=ax)
    sns.scatterplot(df, x="op_intensity", y="performance", hue="dataset", style="comp_repr", s=100, legend=True, zorder=5, ax=ax)

    sns.despine(left=True, bottom=False) # do not show axis line on the left but show it on the bottom (needs axes.linewidth & axes.edgecolor set)

    ### Legend ###
    handles, labels = ax.get_legend_handles_labels()
    empty_legend_label(labels.index("comp_repr"), labels, handles) # insert empty label & handle to balance columns

    del handles[labels.index("dataset")] # Remove legend sub-titles
    labels.remove("dataset")
    del handles[labels.index("comp_repr")]
    labels.remove("comp_repr")

    ax.legend(handles=handles, labels=labels, loc="upper left", ncol=4, fancybox=True, fontsize="small")

    plt.tight_layout(rect=[ 0.05, 0.1, 0.95, 0.9 ])
    
    plot_dir = f"{args.output_folder}{args.gpu}/roofline/"
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(plot_dir + "all.png", format="png") # plt.show()
    plt.close()


def plot_roofline(args: argparse.Namespace, df: pd.DataFrame, dataset_name: str):
    fig, ax = plt.subplots()

    # Change figure size before plotting
    fig.set_size_inches((12, 10))
    
    # Device & Input Metadata
    specs = SPECS[args.gpu]
    cpu = specs['cpu']
    gpu = specs['gpu']
    gpu_pi = specs['peak_perf'] # GPU peak performance on FP32
    gpu_beta = specs['mem_bandwidth'] # GPU peak memory bandwidth
    gpu_bound = gpu_pi / gpu_beta # memory/compute bound (horizontal line)
    first = df.iloc[0]
    runs = df[ (df["competitor"] == first['competitor']) & (df['mat_repr'] == first['mat_repr']) & (df['K'] == first['K']) ].shape[0]
    N = int(df.iloc[0]['N'])
    M = int(df.iloc[0]['M'])
    NZ = int(df.iloc[0]['NZ'])
    density = (NZ / (N * M)) * 100
    density = round(density, 2 if density > 0.01 else 4)
    dataset_name = str(dataset_name[0].upper() + dataset_name[1:])
    print(f"Plotting results for {dataset_name}")

    ### Titles ###
    plt.xlabel("Operational Intensity [Flops/byte]", loc="center", fontdict={ "size": "medium" })
    plt.ylabel("Performance [Flops/s]")
    plt.title(f"SDDMM roofline plot with R={runs}\n{dataset_name} dataset: {N}x{M} with {density}% density\nRunning on {cpu} and {gpu}\n", loc="center", y=1, fontdict={ "weight": "bold", "size": "large" })

    ### Scale & Ticks ###
    ax.set_xscale("log") # log, linear
    ax.get_xaxis().set_major_locator(plticker.LogLocator(base=10)) # LogLocator, LinearLocator, AutoLocator
    ax.get_xaxis().set_major_formatter(plticker.LogFormatterMathtext()) # LogFormatter, LinearFormatter, NullFormatter

    ax.get_xaxis().set_minor_locator(plticker.LogLocator(base=10, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))) # ax.get_yaxis().set_minor_locator(plticker.AutoMinorLocator())
    ax.get_xaxis().set_minor_formatter(plticker.NullFormatter())

    ax.set_yscale("log") # linear
    ax.get_yaxis().set_major_locator(plticker.LogLocator(base=10)) # ax.get_yaxis().set_major_locator(plticker.AutoLocator()) 
    ax.get_yaxis().set_major_formatter(plticker.LogFormatterMathtext())

    ax.get_yaxis().set_minor_locator(plticker.LogLocator(base=10, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))) # ax.get_yaxis().set_minor_locator(plticker.AutoMinorLocator())
    ax.get_yaxis().set_minor_formatter(plticker.NullFormatter())

    ax.tick_params(which='major', axis='both', width=1, length=7)
    ax.tick_params(which='minor', axis='both', width=0.5, length=4, color='black')

    ### Rooflines ###
    x = np.arange(0, MAX_OP_INTENSITY, 1)
    y = gpu_beta * x
    sns.lineplot(x=x, y=y, color="black")

    plt.hlines(gpu_pi, color="black", ls="solid", xmin=gpu_bound, xmax=MAX_OP_INTENSITY)
    plt.vlines(gpu_bound, color="black", alpha=0.4, ls="--", ymin=0, ymax=gpu_pi)

    ### Plot Computations ###
    df['comp_repr'] = df[['competitor', 'mat_repr']].agg(' - '.join, axis=1)

    df['flops'] = df.apply(lambda row: flops(row['NZ'], row['K']), axis=1)
    df['seconds'] = df[RUNTIME_FIELD] / 10e9
    df['bytes'] = df.apply(lambda row: trans_bytes(row['M'], row['N'], row['K'], row['NZ']), axis=1)

    df['performance'] = df['flops'] / df['seconds']
    df['op_intensity'] = df['flops'] / df['bytes']

    df = df[ [ 'dataset', 'comp_repr', 'K', 'performance', 'op_intensity' ] ]
    df = df.groupby([ 'dataset', 'comp_repr', 'K' ]).quantile(args.percentile)

    sns.lineplot(df, x="op_intensity", y="performance", hue="comp_repr", legend=False, zorder=1, ax=ax)
    sns.scatterplot(df, x="op_intensity", y="performance", hue="comp_repr", style="K", s=100, legend=True, zorder=5, ax=ax)

    sns.despine(left=True, bottom=False) # do not show axis line on the left but show it on the bottom (needs axes.linewidth & axes.edgecolor set)

    ### Legend ###
    handles, labels = ax.get_legend_handles_labels()
    del handles[labels.index("comp_repr")]
    labels.remove("comp_repr")
    del handles[labels.index("K")] # Remove legend sub-titles
    labels.remove("K")

    ax.legend(handles=handles, labels=labels, loc="best", ncol=3, fancybox=True, fontsize="small")

    plt.tight_layout(rect=[ 0.05, 0.1, 0.95, 0.9 ])
    
    plot_dir = f"{args.output_folder}{args.gpu}/roofline/"
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(plot_dir + dataset_name + ".png", format="png") # plt.show()
    plt.close()


def main(args: argparse.Namespace):
    sns.set_theme(context="notebook", font_scale=1, rc={ "lines.linewidth": 2, "axes.linewidth": 1, "axes.edgecolor":"black", "xtick.bottom": True, "ytick.left": True }) # rc={ "xtick.top": True, "ytick.left": True }
    sns.set_palette(sns.color_palette(cc.glasbey_hv), 30)

    df = read_df(args.input)
    
    drop_mask = df['competitor'] == 'CPU-Basic'
    drop_mask |= df['competitor'] == 'CPU-PyTorch'
    drop_mask |= df['competitor'] == 'GPU-Basic'
    drop_mask |= df['competitor'] == 'GPU-Thread-Dispatcher'
    drop_mask |= df['competitor'] == 'GPU-Tiled'
    df = df.drop(df.index[drop_mask])

    df = df.apply(lambda row: row if math.log2(row['K']).is_integer() else None, axis=1).dropna()
    df['K'] = df['K'].apply(int)

    plot_roofline_all(args, df)

    sns.reset_defaults()
    sns.set_theme(context="notebook", font_scale=1, rc={ "lines.linewidth": 2, "axes.linewidth": 1, "axes.edgecolor":"black", "xtick.bottom": True, "ytick.left": True }) # rc={ "xtick.top": True, "ytick.left": True }

    datasets = pd.unique(df['dataset'])
    for dataset in datasets:
        plot_roofline(args, df[df['dataset'] == dataset], dataset)

if __name__ == "__main__":    
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--input", default="results/results-v100.csv", type=str, help="CSV input path")
    argParser.add_argument("--output_folder", default="results/", type=str, help="Output folder")
    argParser.add_argument("--percentile", default=0.95, type=float, help="Performance & Operational Intensity percentile")
    argParser.add_argument("--gpu", default="v100", type=str, help="GPU model")
    args = argParser.parse_args()
    main(args)