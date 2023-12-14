#include "gpu_thread_dispatcher.hpp"
#include <cuda_runtime.h>

__global__ void gpu_thread_dispatcher_csr_kernel(float* A, float* B, float* S, float* P, int* cols, int* rows, int* start_idx, int M, int K, int N, int sparse_size, int row_size) {
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int row_start = start_idx[thread_idx];
    int row_end = start_idx[thread_idx + 1];

    for (int row_idx = row_start; row_idx < row_end; row_idx++) {
        int idx = rows[row_idx];

        int row = row_idx;
        while(idx < rows[row_idx+1]) {
            int col = cols[idx];

            float result = 0.f;
            for (int i = 0; i < K; i++) {
                result += A[row * K + i] * B[col * K + i];
            }
            result *= S[idx];
            P[idx] = result;
            idx++;
        }
    }
}

template <typename T>
void gpu_thread_dispatcher_csr_wrapper(T* A_gpu, T* B_gpu, T* S_gpu, T* P_gpu, int* cols_gpu, int* rows_gpu, int* start_idx, int M, int K, int N, int sparse_size, int row_size) {
    // Perform SDDMM on the GPU
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);  // Assumes device 0, change if using multiple GPUs

	int num_sm = prop.multiProcessorCount;
	int max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
	// int max_thread_blocks_per_sm = prop.maxBlocksPerMultiProcessor;
	int max_threads_per_block = prop.maxThreadsPerBlock;

	// Use maximum number of threads per streaming multiprocessor
	// int threads_per_block = min(max_threads_per_block, (max_threads_per_sm + max_thread_blocks_per_sm - 1) / max_thread_blocks_per_sm);

	// calculate number of thread blocks by using all available streaming multiprocessors
	// int num_thread_blocks = (max_threads_per_sm * num_sm + threads_per_block - 1) / threads_per_block;

    // number of non-zero elements per thread
	int nnz_per_thread = 64;

	// set the number of threads per block
	int threads_per_block = min(max_threads_per_block, 512);

	int max_num_threads = num_sm * max_threads_per_sm;
	int num_threads = min((sparse_size + nnz_per_thread - 1) / nnz_per_thread, max_num_threads);
	int num_thread_blocks = (num_threads + threads_per_block - 1) / threads_per_block;

    gpu_thread_dispatcher_csr_kernel<<<num_thread_blocks, threads_per_block>>>(A_gpu, B_gpu, S_gpu, P_gpu, cols_gpu, rows_gpu, start_idx, M, K, N, sparse_size, row_size);
}

/* Workaround because the wrappers need to be inside the CUDA file (Would normally write templated functions inside the header file!) */
template void gpu_thread_dispatcher_csr_wrapper<float>(float* A_gpu, float* B_gpu, float* S_gpu, float* P_gpu, int* cols_gpu, int* rows_gpu, int* start_idx, int M, int K, int N, int sparse_size, int row_size);
