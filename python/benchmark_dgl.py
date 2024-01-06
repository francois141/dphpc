import scipy
import torch
import os
import time
import argparse

os.environ['TORCH'] = torch.__version__
os.environ['DGLBACKEND'] = "pytorch"

import dgl.sparse as dglsp


def load_matrix(filename):
    m = scipy.io.mmread(filename)
    m = m.tocsr()
    val = torch.tensor(m.data).to('cuda')
    shape = m.shape
    indptr = torch.tensor(m.indptr).to('cuda')
    indices = torch.tensor(m.indices).to('cuda')
    m = dglsp.from_csr(indptr, indices, val, shape)
    return m

def benchmark(S, K):
    M = S.shape[0]
    N = S.shape[1]

    A = torch.rand((M,K)).to('cuda')
    B = torch.rand((K,N)).to('cuda')

    start = time.time_ns()
    # result is not checked
    result = dglsp.sddmm(S,A,B)
    torch.cuda.synchronize()
    timing = time.time_ns() - start
    return timing

def benchmark_dataset(dataset_name, filename, K, num_runs):
    S = load_matrix(filename)
    M = S.shape[0]
    N = S.shape[1]

    for _ in range(num_runs):
        timing = benchmark(S, K)
        line = "GPU-Dgl,"+dataset_name+",CSR,"+str(M)+","+str(N)+","+str(K)+","+str(S.nnz)+","+str(timing)+","+str(0)+","+str(timing)+","+str(0)+",-1,-1,-1"
        print(line)

# Parsing
parser = argparse.ArgumentParser(
                    prog='Benchmark DGL',
                    description='-',
                    epilog='-')
parser.add_argument('--data_folder', default="../data/")
parser.add_argument('--num_runs', default=1)
parser.add_argument('-k', default=32)
args = parser.parse_args()

print("Running benchmark with K="+str(args.k)+" and num_runs="+str(args.num_runs))

K = int(args.k)
num_runs = int(args.num_runs)

# Benchmark matrix market
benchmark_dataset("fluid", args.data_folder+"/ex21/ex21.mtx", K, num_runs)
benchmark_dataset("oil", args.data_folder+"/bcsstk02/bcsstk02.mtx", K, num_runs)
benchmark_dataset("biochemical", args.data_folder+"/N_biocarta/N_biocarta.mtx", K, num_runs)
benchmark_dataset("circuit", args.data_folder+"/fpga_dcop_06/fpga_dcop_06.mtx", K, num_runs)
benchmark_dataset("heat", args.data_folder+"/epb0/epb0.mtx", K, num_runs)
benchmark_dataset("mass", args.data_folder+"/bcsstk07/bcsstk07.mtx", K, num_runs)
benchmark_dataset("adder", args.data_folder+"/adder_dcop_33/adder_dcop_33.mtx", K, num_runs)
benchmark_dataset("trackball", args.data_folder+"/bcsstm37/bcsstm37.mtx", K, num_runs)
benchmark_dataset("ND12K", args.data_folder+"/nd12k/nd12k.mtx", K, num_runs)
benchmark_dataset("HumanGene2", args.data_folder+"/human_gene2/human_gene2.mtx", K, num_runs)
benchmark_dataset("Boeing", args.data_folder+"/ct20stif/ct20stif.mtx", K, num_runs)
benchmark_dataset("Boeing Diagonal", args.data_folder+"/pwtk/pwtk.mtx", K, num_runs)
benchmark_dataset("Stiffness", args.data_folder+"/inline_1/inline_1.mtx", K, num_runs)
benchmark_dataset("Semi-conductor", args.data_folder+"/vas_stokes_1M/vas_stokes_1M.mtx", K, num_runs)
benchmark_dataset("VLSI", args.data_folder+"/nv2/nv2.mtx", K, num_runs)
benchmark_dataset("stack-overflow", args.data_folder+"/sx-stackoverflow/sx-stackoverflow.mtx", K, num_runs)
benchmark_dataset("chip", args.data_folder+"/FullChip/FullChip.mtx", K, num_runs)
benchmark_dataset("mix", args.data_folder+"/mixtank_new/mixtank_new.mtx", K, num_runs)
benchmark_dataset("mechanics", args.data_folder+"/sme3Db/sme3Db.mtx", K, num_runs)
benchmark_dataset("power", args.data_folder+"/TSC_OPF_1047/TSC_OPF_1047.mtx", K, num_runs)
benchmark_dataset("combinatorics", args.data_folder+"/c8_mat11/c8_mat11.mtx", K, num_runs)
benchmark_dataset("stress", args.data_folder+"/smt/smt.mtx", K, num_runs)
benchmark_dataset("mouse-gene", args.data_folder+"/mouse_gene/mouse_gene.mtx", K, num_runs)
