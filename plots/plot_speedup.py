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

BASELINE = { "competitor": "GPU-PyTorch", "mat_repr": "CSR" }

COMPETITORS = [ "GPU-PyTorch", "GPU-DGL", "GPU-Dynamic", "GPU-cuSPARSE", "GPU-Preprocessing" ]

RUNTIME_FIELD = "comp_ns" # total_ns, init_ns, comp_ns, cleanup_ns

def read_df(path):
    df = pd.read_csv(path)
    return df

def mean_baseline_group(group):
    group[args.runtime_field] = int(group[args.runtime_field].mean())
    return group

def calc_speedup(group, baseline_times):
    global args

    group = group.reset_index()
    group['speedup'] = baseline_times / group[args.runtime_field]
    return group.set_index("index")

def plot_speedup(args: argparse.Namespace, df: pd.DataFrame, dataset_name):
    fig, ax = plt.subplots()

    # Change figure size before plotting
    fig.set_size_inches((12, 8))

    # Device & Input Metadata
    cpu, gpu = SPECS[args.gpu]
    # first = df.iloc[0]
    runs = 20 # df[ (df["competitor"] == first['competitor']) & (df['mat_repr'] == first['mat_repr']) & (df['K'] == first['K']) ].shape[0]
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
    plt.ylabel(f"Speedup [Baseline is {BASELINE['competitor']} ({BASELINE['mat_repr']})]")
    plt.title(f"SDDMM speedup with R={runs} runs (showing {ci}% CI)\n{dataset_name} dataset: {N}x{M} with {density}% density\nRunning on {cpu} and {gpu}\n", loc="center", y=1, fontdict={ "weight": "bold", "size": "large" })

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
    baselines = baselines.groupby("K").apply(mean_baseline_group)
    baseline_times = baselines[args.runtime_field].reset_index(drop=True)

    df['comp_repr'] = df[['competitor', 'mat_repr']].agg(' - '.join, axis=1)
    df['speedup'] = 0.0
    df = df.groupby([ "competitor", "mat_repr" ]).apply(lambda group: calc_speedup(group, baseline_times))
    df = df.reset_index(drop=True)
    df = df.sort_values(by=[ 'competitor' ], key=lambda x: x.map(lambda comp: COMPETITORS.index(comp))) # sort by custom dataset order

    # dyn_df = df[df['competitor'] == BASELINE['competitor']]
    # print(f"{dyn_df['speedup'].min()} ; {dyn_df['speedup'].max()}")

    sns.lineplot(df, x="K", y="speedup", hue="comp_repr", legend=True, zorder=1, errorbar=("ci", ci), err_style="band", ax=ax)

    sns.despine(left=True, bottom=False) # do not show axis line on the left but show it on the bottom (needs axes.linewidth & axes.edgecolor set)

    ### Legend ###
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5, fancybox=True, fontsize="small")

    plt.tight_layout(rect=[ 0.01, 0.01, 0.99, 0.99 ])
    
    plot_dir = f"{args.output_folder}{args.gpu}/speedup_{args.runtime_field.split('_')[0]}/"
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(plot_dir + dataset_name + ".png", format="png") # plt.show()
    plt.close()

def main(args: argparse.Namespace):
    sns.set_theme(context="notebook", font_scale=1, style="darkgrid", rc={ "lines.linewidth": 2, "axes.linewidth": 1, "axes.edgecolor":"black", "xtick.bottom": True, "ytick.left": True }) # rc={ "xtick.top": True, "ytick.left": True }
    sns.set_palette("Set1")

    df = read_df(args.input)

    df = df.apply(lambda row: row if math.log2(row['K']).is_integer() else None, axis=1).dropna()
    df['K'] = df['K'].apply(int)

    drop_mask = df['competitor'] == 'CPU-Basic'
    drop_mask |= df['competitor'] == 'CPU-PyTorch'
    drop_mask |= df['competitor'] == 'GPU-Basic'
    drop_mask |= df['competitor'] == 'GPU-Thread-Dispatcher'
    drop_mask |= df['competitor'] == 'GPU-Tiled'
    drop_mask |= df['competitor'] == 'GPU-Shared'
    drop_mask |= df['competitor'] == 'GPU-Convert'
    drop_mask |= df['dataset'] == 'RandomWithDensity'
    drop_mask |= df['dataset'] == 'LatinHypercube'
    drop_mask |= df['dataset'] == 'EMail-Enron'
    drop_mask |= df['K'] == 256
    df = df.drop(df.index[drop_mask])

    datasets = pd.unique(df['dataset'])
    for dataset in datasets:
        plot_speedup(args, df[df['dataset'] == dataset], dataset)

if __name__ == "__main__":    
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--input", default="results/results-v100.csv", type=str, help="CSV input path")
    argParser.add_argument("--output_folder", default="results/", type=str, help="Output folder")
    argParser.add_argument("--runtime_field", default=RUNTIME_FIELD, type=str, help="Runtime field")
    argParser.add_argument("--ci", default=0.95, type=float, help="Percentage confidence interval to draw")
    argParser.add_argument("--gpu", default="v100", type=str, help="GPU model")
    args = argParser.parse_args()
    main(args)
    