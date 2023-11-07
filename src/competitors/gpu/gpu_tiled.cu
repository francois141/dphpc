#include "gpu_basic.hpp"

#include <cuda_runtime.h>
#include <algorithm>

const int TILE_SIZE = 16;

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
void gpu_tiled_csr_wrapper(Dense<T>& A, Dense<T>& B, CSR<T>& S, CSR<T>& P) {
	// A is MxK, B is NxK, S and P are MxN sparse
	int M = A.getRows();
	int K = A.getCols();
	int N = B.getRows();

	assert(K == B.getCols());

	// get the size needed for each matrix
	size_t A_size = M * K * sizeof(T);
	size_t B_size = K * N * sizeof(T);
	size_t SP_size = S.getValues().size() * sizeof(T);
	size_t sparse_col_size = S.getValues().size() * sizeof(int);
	size_t sparse_row_size = (S.getRows() + 1) * sizeof(int);

	static_assert(sizeof(T) == sizeof(float), "the kernel is specialized for single precision floating points");

	// allocate the matrices on the GPU
	float* A_gpu, * B_gpu, * S_gpu, * P_gpu;
	int* cols_gpu, * rows_gpu;
	cudaMalloc(&A_gpu, A_size);
	cudaMalloc(&B_gpu, B_size);
	cudaMalloc(&S_gpu, SP_size);
	cudaMalloc(&P_gpu, SP_size);
	cudaMalloc(&cols_gpu, sparse_col_size);
	cudaMalloc(&rows_gpu, sparse_row_size);

	// copy from RAM to GPU
	cudaMemcpy(A_gpu, &A.getValue(0, 0), A_size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, &B.getValue(0, 0), B_size, cudaMemcpyHostToDevice);
	cudaMemcpy(S_gpu, S.getValues().data(), SP_size, cudaMemcpyHostToDevice);
	cudaMemcpy(cols_gpu, S.getColPositions().data(), sparse_col_size, cudaMemcpyHostToDevice);
	cudaMemcpy(rows_gpu, S.getRowPositions().data(), sparse_row_size, cudaMemcpyHostToDevice);

	// Perform SDDMM on the GPU
	gpu_tiled_csr_kernel<<<32, 32 >>>(A_gpu, B_gpu, S_gpu, P_gpu, cols_gpu, rows_gpu, M, K, N);

	// copy result back to RAM
	cudaMemcpy(P.getValues().data(), P_gpu, SP_size, cudaMemcpyDeviceToHost);
	P.setColPositions(S.getColPositions());
	P.setRowPositions(S.getRowPositions());

	// free all the GPU allocated memory
	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(S_gpu);
	cudaFree(P_gpu);
	cudaFree(cols_gpu);
	cudaFree(rows_gpu);
}

/* Workaround because the wrappers need to be inside the CUDA file (Would normally write templated functions inside the header file!) */

template void gpu_tiled_csr_wrapper<float>(Dense<float>& A, Dense<float>& B, CSR<float>& S, CSR<float>& P);


