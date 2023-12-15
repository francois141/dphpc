#include "gpu_tiled.hpp"

#include <cuda_runtime.h>
#include <algorithm>

const int TILE_SIZE = 8;

// perform SDDMM, compute P = (A*B^T) dot S (where dot is the term by term product)
// A is MxK, B is NxK, S and P are MxN sparse
__global__ void gpu_tiled_csr_kernel(float* A, float* B, float* S, float* P, int* cols, int* rows, int M, int K, int N) {
	int nb_running = gridDim.x * blockDim.x;
	
	// first compute the tile decomposition of the center matrix
	int nb_tile_rows = (M + TILE_SIZE - 1) / TILE_SIZE;
	int nb_tile_cols = (N + TILE_SIZE - 1) / TILE_SIZE;

	// divide the rows of tiles evenly among the processes
	int min_per_instance = nb_tile_rows / nb_running;
	int leftovers = nb_tile_rows % nb_running;

	// Each instance will compute multiple contiguous rows of tiles
	int instance_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int tile_row_start = min_per_instance * instance_idx + min(instance_idx, leftovers);
	int tile_row_end = min_per_instance * (instance_idx + 1) + min(instance_idx + 1, leftovers);

	// perform the SDDMM algorithm on the range [range_start, range_end[
	for (int tile_row = tile_row_start; tile_row < tile_row_end; tile_row++) {
		// now compute all the entries inside this tile
		
		// we need to remember for each line where we were at (in the sparse matrix)
		int start_row = tile_row * TILE_SIZE;
		int row_position[TILE_SIZE];
		for (int row = 0; row < TILE_SIZE; row++) {
			row_position[row] = rows[start_row + row];
		}

		// now compute all the entries
		for (int tile_col = 0; tile_col < nb_tile_cols; tile_col++) {
			// we are computing the entries from columns tile_col * TILE_SIZE to col_end
			int col_end = (tile_col + 1) * TILE_SIZE;

			// we are in the tile at coordinate (tile_row, tile_col), now compute all the entries in this tile
			for (int row = 0; row < TILE_SIZE; row++) {
				int& sparse_index = row_position[row];
				while (sparse_index < rows[start_row + row + 1] && cols[sparse_index] < col_end) {
					int curr_row = start_row + row;
					int curr_col = cols[sparse_index];

					float result = 0.f;
					// matrix multiplication
					for (int i = 0; i < K; i++) {
						// B is transposed
						result += A[curr_row * K + i] * B[curr_col * K + i];
					}
					result *= S[sparse_index];
					P[sparse_index] = result;

					sparse_index++;
				}
			}
		}
	}
}

template <typename T>
void gpu_tiled_csr_wrapper(T* A_gpu, T* B_gpu, T* S_gpu, T* P_gpu, int* cols_gpu, int* rows_gpu, int M, int K, int N, int sparse_size) {
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
	
	// Perform SDDMM on the GPU
	gpu_tiled_csr_kernel<<<num_thread_blocks, threads_per_block>>>(A_gpu, B_gpu, S_gpu, P_gpu, cols_gpu, rows_gpu, M, K, N);
}

/* Workaround because the wrappers need to be inside the CUDA file (Would normally write templated functions inside the header file!) */
template void gpu_tiled_csr_wrapper<float>(float* A_gpu, float* B_gpu, float* S_gpu, float* P_gpu, int* cols_gpu, int* rows_gpu, int M, int K, int N, int sparse_size);


