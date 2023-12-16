#include "gpu_convert.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>


__global__ void gpu_convert_kernel(int* rows, int* rows_coo, int M) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	for (; row < M; row += (gridDim.x * blockDim.x)){
		int start = rows[row];
		int end = rows[row + 1];
		for (int i = start; i < end; i++) {
			rows_coo[i] = row;
		}
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
        for (int i = 0; i < K; i += 8) {
            float4 a1 = *reinterpret_cast<float4*>(&A[row * K + i]);
            float4 a2 = *reinterpret_cast<float4*>(&A[row * K + i + 4]);
            float4 b1 = *reinterpret_cast<float4*>(&B[col * K + i]);
            float4 b2 = *reinterpret_cast<float4*>(&B[col * K + i + 4]);
            result += a1.x * b1.x + a1.y * b1.y + a1.z * b1.z + a1.w * b1.w;
            result += a2.x * b2.x + a2.y * b2.y + a2.z * b2.z + a2.w * b2.w;
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
		
		int num_thread_blocks = this->get_num_thread_blocks();
        int num_threads_per_block = this->get_num_threads_per_block();

		// Convert to COO
		gpu_convert_kernel <<< num_thread_blocks, num_threads_per_block >>> (rows_gpu, rows_coo_gpu, M);
		// Perform SDDMM on the GPU
		gpu_basic_coo_kernel_2 <<< num_thread_blocks, num_threads_per_block >>> (A_gpu, B_gpu, S_gpu, P_gpu, cols_gpu, rows_coo_gpu, M, K, N, sparse_size);
		// No need to convert back to CSR, just reuse S
	}
}
