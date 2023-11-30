#include "gpu_blocked.hpp"

#include <cuda_runtime.h>
#include <algorithm>

const int BLOCK_SIZE = 16;

// perform SDDMM, compute P = (A*B^T) dot S (where dot is the term by term product)
// A is MxK, B is NxK, S and P are MxN sparse
// each thread block is responsible for one submatrix Ssub of S
// each thread in a thread block is responsible for a submatrix SSsub of Ssub
// cf https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory
__global__ void gpu_blocked_csr_kernel(float* A, float* B, float* S, float* P, int* cols, int* rows, int M, int K, int N) {
	// get size of grid and thread block
	int grid_width = gridDim.x;
	int grid_height = gridDim.y;
	int tb_width = blockDim.x;
	int tb_height = blockDim.y;
	
	// compute number of rows and columns of a thread block (submatrix Ssub of S)
	int tb_num_rows = (M + grid_height - 1) / grid_height;
	int tb_num_cols = (N + grid_width - 1) / grid_width;

	// compute number of rows and columns of a tile (submatrix SSsub of Ssub)
	int tile_num_rows = (tb_num_rows + tb_height - 1) / tb_height;
	int tile_num_cols = (tb_num_cols + tb_width - 1) / tb_width;

	// This thread is in charge of the tile [start_row:end_row, start_col:end_col]
	int tile_start_row = tb_num_rows * blockIdx.y + tile_num_rows * threadIdx.y;
	int tile_start_col = tb_num_cols * blockIdx.x + tile_num_cols * threadIdx.x;

	// need to take leftovers into account if M or N are not properly divisible by thread block and grid dimensions
	int tile_end_row = min(tile_start_row + tile_num_rows, tb_num_rows * (blockIdx.y + 1));
	int tile_end_col = min(tile_start_col + tile_num_cols, tb_num_cols * (blockIdx.x + 1));

	// perform the SDDMM algorithm on the tile [tile_start_row:tile_end_row, tile_start_col:tile_end_col]
	for (int row = tile_start_row; row < tile_end_row; row++) {
		int sparse_index = rows[row];

		// iterate over all non-zero elements in this row which are in [tile_start_col:til_end_col]
		while (sparse_index < rows[row + 1] && cols[sparse_index] < tile_end_col) {
			int curr_col = cols[sparse_index];

			// we're not yet in our tile -> this is not ideal, is there no other way to get to tile_start_col faster?
			if (curr_col < tile_start_col){
				sparse_index++;
				continue;
			}
			
			// TODO: load A and B into shared memory here

			// for each non-zero element in S, compute the corresponding scalar product
			float result = 0.f;
			for (int i = 0; i < K; i++){
				// B is transposed
				result += A[row * K + i] * B[curr_col * K + i];
			}
			result *= S[sparse_index];
			P[sparse_index] = result;

			sparse_index++;
		}
	}
}

template <typename T>
void gpu_blocked_csr_wrapper(T* A_gpu, T* B_gpu, T* S_gpu, T* P_gpu, int* cols_gpu, int* rows_gpu, int M, int K, int N) {
	
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	// dim3 dimGrid(N / dimBlock.x, M / dimBlock.y);
	dim3 dimGrid(2, 2);

	// Perform SDDMM on the GPU
	gpu_blocked_csr_kernel<<<dimGrid, dimBlock>>>(A_gpu, B_gpu, S_gpu, P_gpu, cols_gpu, rows_gpu, M, K, N);
}

/* Workaround because the wrappers need to be inside the CUDA file (Would normally write templated functions inside the header file!) */
template void gpu_blocked_csr_wrapper<float>(float* A_gpu, float* B_gpu, float* S_gpu, float* P_gpu, int* cols_gpu, int* rows_gpu, int M, int K, int N);


