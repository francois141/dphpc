"""
Prints the number of threads used for each implementation on the V100 and A100
depending on the number of nnz elements that should be handled by each thread
the number of threads per block we would like to have

For each value we make sure that we don't exceed the maximum value specified by
the device
"""

def nnz_elements_per_thread(
    nnz: int,
    nnz_per_thread: int,
    threads_per_block: int,
    num_sm: int,
    max_threads_per_sm: int,
    max_thread_blocks_per_sm: int,
    max_threads_per_block: int,
) -> tuple[int, int]:
    """
    nnz                         : number of non-zero elements in the matrix
    nnz_per_thread              : number of nonzero elements to be handled by each thread
    threads_per_block           : maximum number of threads per block we want to use
    num_sm                      : number of streaming multi_processors on the GPU (devide prop)
    max_threads_per_sm          : max number of threads per streaming multiprocessors (device prop)
    max_thread_blocks_per_sm    : max number of thread blocks per streaming multiprocessor (device prop)
    max_threads_per_block       : max number of threads per thread block (device prop)
    """

    # don't use more threads per block than device allows
    threads_per_block = min(threads_per_block, max_threads_per_block)

    # max number of threads to use at most
    max_num_threads = num_sm * max_threads_per_sm

    # num threads that we will use should not be higher than max_num_threads
    num_threads = min((nnz + nnz_per_thread - 1) // nnz_per_thread, max_num_threads)
    num_thread_blocks = (num_threads + threads_per_block - 1) // threads_per_block

    return num_thread_blocks, threads_per_block

def get_num_threads_for_shared(
    m: int,
    n: int,
    k: int,
    t_i: int,
    t_k: int,
    threads_per_block: int,
) -> tuple[int, int]:
    """
    returns the number of thread blocks as well as the number of threads per block
    for the gpu_shared.cu implementation

    m   : number of rows of the sparse matrix
    n   : number of columns of the sparse matrix
    k   : number of columns of the A matrix in (A x B)@sparse(S)
    """
    nb_tiles_row = (m + t_i - 1) // t_i
    nb_tiles_k = (k + t_k - 1) // t_k
    num_thread_blocks = nb_tiles_row * nb_tiles_k
    return num_thread_blocks, threads_per_block

if __name__ == "__main__":

    gpus = {
        "A100": {
            "num_sm": 108,
            "max_threads_per_sm": 2048,
            "max_threads_per_block": 1024,
            "max_thread_blocks_per_sm": 32,
        },
        "V100": {
            "num_sm": 80,
            "max_threads_per_sm": 2048,
            "max_threads_per_block": 1024,
            "max_thread_blocks_per_sm": 32,
        },
    }
    
    datasets = {
        "fluid": {"m": 656, "n": 656, "nnz": 18964},
        "oil": {"m": 66, "n": 66, "nnz": 4356},
        "biochemical": {"m": 1922, "n": 1922, "nnz": 4335},
        "circuit": {"m": 1220, "n": 1220, "nnz": 5860},
        "heat": {"m": 1794, "n": 1794, "nnz": 7764},
        "mass": {"m": 420, "n": 420, "nnz": 7860},
        "adder": {"m": 1813, "n": 1813, "nnz": 11246},
        "trackball": {"m": 25503, "n": 25503, "nnz": 15525},
        "human_gene_2": {"m": 14340, "n": 14340, "nnz": 18068388},
        "nd12k": {"m": 36000, "n": 36000, "nnz": 14220946},
        "platform": {"m": 28924, "n": 28924, "nnz": 2043492},
        "mecanics": {"m": 29067, "n": 29067, "nnz": 2081063},
        "power": {"m": 8140, "n": 8140, "nnz": 2012833},
        "combinatorics": {"m": 4562, "n": 5761, "nnz": 2462970},
        "stress": {"m": 25710, "n": 25710, "nnz": 3749582},
        "mouse": {"m": 45101, "n": 45101, "nnz": 28967291},
        "email_enron": {"m": 36692, "n": 36692, "nnz": 367662},
        "boeing": {"m": 52329, "n": 52329, "nnz": 2600295},
        "boeing_diagonal": {"m": 217918, "n": 217918, "nnz": 11524432},
        "stiffness": {"m": 503712, "n": 503712, "nnz": 36816170},
        "semi_conductor": {"m": 1090664, "n": 1090664, "nnz": 34767207},
        "vlsi": {"m": 1453908, "n": 1453908, "nnz": 37475646},
        "stack_overflow": {"m": 2601977, "n": 2601977, "nnz": 36233450},
        "chip": {"m": 2987012, "n": 2987012, "nnz": 26621983},
    }

    Ks = [32, 64, 96, 128]

    for gpu, properties in gpus.items():
        print(f"\n#### {gpu} ####\n")
        for ds, dim in datasets.items():
            # all implementations that set constant num elements per thread
            num_thread_blocks, threads_per_block = nnz_elements_per_thread(
                nnz=dim["nnz"],
                nnz_per_thread=16,
                threads_per_block=256,
                num_sm=properties["num_sm"],
                max_threads_per_sm=properties["max_threads_per_sm"],
                max_thread_blocks_per_sm=properties["max_thread_blocks_per_sm"],
                max_threads_per_block=properties["max_threads_per_block"],
            )
            print(f"{ds} ({dim['m']}x{dim['n']}), {dim['nnz']} nnz")
            print(f"\tNum Threads for all implementations that limit num elements per thread")
            print(f"\t\tnum thread blocks:\t\t{num_thread_blocks}")
            print(f"\t\tnum threads per block:\t\t{threads_per_block}")
            print(f"\t\ttotal number of threads:\t{num_thread_blocks * threads_per_block}")

            # num threads for the shared implementation
            print(f"\tNum Threads for the shared implementation")
            threads_per_coef = 4
            t_i = 64
            t_k = 32
            for k in Ks:
                num_thread_blocks, threads_per_block = get_num_threads_for_shared(
                    m=dim["m"],
                    n=dim["n"],
                    k=k,
                    t_i=t_i,
                    t_k=t_k,
                    threads_per_block=t_i * threads_per_coef,
                )
                print(f"\t\tFor K = {k}")
                print(f"\t\t\tnum thread blocks:\t\t{num_thread_blocks}")
                print(f"\t\t\tnum threads per block:\t\t{threads_per_block}")
                print(f"\t\t\ttotal number of threads:\t{num_thread_blocks * threads_per_block}")

