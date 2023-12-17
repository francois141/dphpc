import argparse
import pandas as pd

def get_best_thread_combination_overall_runtime(competitor: str, df: pd.DataFrame):
    # returns the best combination of (tb, t) when summing up all the runtimes of a given combination
    sum_df = df.groupby(["num_thread_blocks", "num_threads_per_block"])["comp_ns"].sum().reset_index()
    sorted_df = sum_df.sort_values(by=["comp_ns"]).reset_index(drop=True)
    #print(f"\tsummed runtime: ({sorted_df.loc[0, 'num_thread_blocks']}, {sorted_df.loc[0, 'num_threads_per_block']})")
    print(sorted_df)

def get_best_thread_combination(competitor: str, df: pd.DataFrame):
    # average runtime for each combination of (tb, t) for each datasets
    avg_df = df.groupby(["mat_repr", "dataset", "num_thread_blocks", "num_threads_per_block"]).agg({"comp_ns": "mean"}).reset_index()
    
    # rank each average within the group
    avg_df["rank"] = avg_df.groupby(["mat_repr", "dataset"])["comp_ns"].rank(method="dense")
    
    # sum up the ranks of each combination (tb, t) across all datasets and matrix_repr
    sum_df = avg_df[["num_thread_blocks", "num_threads_per_block", "comp_ns", "rank"]]
    sum_df = sum_df.groupby(["num_thread_blocks", "num_threads_per_block"]).sum()
    
    # sort the ranks in ascending order, lowest rank is supposed to perform best in general
    sorted_df = sum_df.sort_values(by=["rank"]).reset_index()
    #print(f"\trank: ({sorted_df.loc[0, 'num_thread_blocks']}, {sorted_df.loc[0, 'num_threads_per_block']})")
    print(sorted_df)

def main(args: argparse.Namespace):
    df = pd.read_csv(args.input)

    # keep only values for K=64
    df = df[df["K"] == 64]

    competitors = df["competitor"].unique()
    for competitor in competitors:
        if competitor in ["GPU-Shared", "GPU-Preprocessing", "GPU-PyTorch"]:
            continue
        print(competitor)
        get_best_thread_combination(competitor, df[df["competitor"] == competitor])
        #get_best_thread_combination_overall_runtime(competitor, df[df["competitor"] == competitor])


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--input", default="v100/results-v100.csv", type=str, help="CSV input path")
    args = argParser.parse_args()
    main(args)