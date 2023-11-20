#include "gpu_basic.hpp"

#include <cuda_runtime.h>
#include <algorithm>

// perform SDDMM, compute P = (A*B^T) dot S (where dot is the term by term product)
// A is MxK, B is NxK, S and P are MxN sparse
__global__ void gpu_basic_coo_kernel(float* A, float* B, float* S, float* P, int* cols, int* rows, int M, int K, int N, int sparse_size) {
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

template <typename T>
void gpu_basic_coo_wrapper(T* A_gpu, T* B_gpu, T* S_gpu, T* P_gpu, int* cols_gpu, int* rows_gpu, int M, int K, int N, int sparse_size) {
	
	// Perform SDDMM on the GPU
	gpu_basic_coo_kernel<<<32, 32>>>(A_gpu, B_gpu, S_gpu, P_gpu, cols_gpu, rows_gpu, M, K, N, sparse_size);
}

/* Workaround because the wrappers need to be inside the CUDA file (Would normally write templated functions inside the header file!) */
template void gpu_basic_coo_wrapper<float>(float* A_gpu, float* B_gpu, float* S_gpu, float* P_gpu, int* cols_gpu, int* rows_gpu, int M, int K, int N, int sparse_size);


