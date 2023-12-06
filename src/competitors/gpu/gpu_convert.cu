#include "gpu_convert.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>

constexpr int block_size = 32;

__global__ void gpu_convert_kernel(int* rows, int* rows_coo, int M) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= M) return;
	int start = rows[row];
	int end = rows[row + 1];
	for (int i = start; i < end; i++) {
		rows_coo[i] = row;
	}
}

// perform SDDMM, compute P = (A*B^T) dot S (where dot is the term by term product)
// A is MxK, B is NxK, S and P are MxN sparse
__global__ void gpu_basic_coo_kernel_2(float* A, float* B, float* S, float* P, int* cols, int* rows, int M, int K, int N, int sparse_size) {
	int nb_running = gridDim.x * blockDim.x;
	int min_per_instance = sparse_size / nb_running;
	int leftovers = sparse_size % nb_running;

	// We have to compute sparse_size, each instance will compute a contiguous part of it
	// such that each entrie is computed once and they are evenly distributed
	int instance_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int range_start = min_per_instance * instance_idx + min(instance_idx, leftovers);
	int range_end = min_per_instance * (instance_idx + 1) + min(instance_idx + 1, leftovers);

	// perform the SDDMM algorithm on the range [range_start, range_end[
	for (int entry = range_start; entry < range_end; entry++) {
		int row = rows[entry];
		int col = cols[entry];

		float result = 0.f;
		// matrix multiplication
		for (int i = 0; i < K; i++) {
			// B is transposed
			result += A[row * K + i] * B[col * K + i];
		}
		result *= S[entry];
		P[entry] = result;
	}
}

namespace Competitors {

	void GPUConvert::run_csr(Dense<T>& A, Dense<T>& B, CSR<T>& S, CSR<T>& P) {
		// A is MxK, B is NxK, S and P are MxN sparse
		int M = A.getRows();
		int K = A.getCols();
		int N = B.getRows();

		size_t sparse_size = S.getValues().size();
		int n_blocks = (M + block_size - 1) / block_size;

		// Convert to COO
		gpu_convert_kernel << < n_blocks, block_size >> > (rows_gpu, rows_coo_gpu, M);
		// Perform SDDMM on the GPU
		gpu_basic_coo_kernel_2 << <32, 32 >> > (A_gpu, B_gpu, S_gpu, P_gpu, cols_gpu, rows_coo_gpu, M, K, N, sparse_size);
		// No need to convert back to CSR, just reuse S

		cudaDeviceSynchronize();
	}
}
