#include "gpu_basic.hpp"

#include <cuda_runtime.h>
#include <algorithm>

__global__ void gpu_basic_csr_kernel(void) {

}

// perform SDDMM, compute P = (A*B^T) dot S (where dot is the term by term product)
// A is MxK, B is NxK, S and P are MxN sparse
__global__ void gpu_basic_coo_kernel(double* A, double* B, double* S, double* P, int* cols, int* rows, int M, int K, int N, int sparse_size) {
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

		double result = 0.0;
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
void gpu_basic_csr_wrapper(Dense<T>& A, Dense<T>& B, CSR<T>& S, CSR<T>& P) {
	gpu_basic_csr_kernel << <1, 1 >> > ();
}

template <typename T>
void gpu_basic_coo_wrapper(Dense<T>& A, Dense<T>& B, COO<T>& S, COO<T>& P) {
	// A is MxK, B is NxK, S and P are MxN sparse
	int M = A.getRows();
	int K = A.getCols();
	int N = B.getRows();

	assert(K == B.getCols());

	// get the size needed for each matrix
	size_t A_size = M * K * sizeof(T);
	size_t B_size = K * N * sizeof(T);
	size_t SP_size = S.getValues().size() * sizeof(T);
	size_t sparse_dim_size = S.getValues().size() * sizeof(int);

    static_assert(sizeof(T) == sizeof(double), "the kernel is specialized for double precision floating points");

	// allocate the matrices on the GPU
	double* A_gpu, * B_gpu, * S_gpu, * P_gpu;
	int* cols_gpu, * rows_gpu;
	cudaMalloc(&A_gpu, A_size);
	cudaMalloc(&B_gpu, B_size);
	cudaMalloc(&S_gpu, SP_size);
	cudaMalloc(&P_gpu, SP_size);
	cudaMalloc(&cols_gpu, sparse_dim_size);
	cudaMalloc(&rows_gpu, sparse_dim_size);

	// copy from RAM to GPU
	cudaMemcpy(A_gpu, &A.getValue(0, 0), A_size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, &B.getValue(0, 0), B_size, cudaMemcpyHostToDevice);
	cudaMemcpy(S_gpu, S.getValues().data(), SP_size, cudaMemcpyHostToDevice);
	cudaMemcpy(cols_gpu, S.getColPositions().data(), sparse_dim_size, cudaMemcpyHostToDevice);
	cudaMemcpy(rows_gpu, S.getRowPositions().data(), sparse_dim_size, cudaMemcpyHostToDevice);

	// Perform SDDMM on the GPU
	gpu_basic_coo_kernel << <32, 32 >> > (A_gpu, B_gpu, S_gpu, P_gpu, cols_gpu, rows_gpu, M, K, N, S.getValues().size());

	// copy result back to RAM
	cudaMemcpy(P.getValues().data(), P_gpu, SP_size, cudaMemcpyDeviceToHost);

	// free all the GPU allocated memory
	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(S_gpu);
	cudaFree(P_gpu);
	cudaFree(cols_gpu);
	cudaFree(rows_gpu);
}

/* Workaround because the wrappers need to be inside the CUDA file (Would normally write templated functions inside the header file!) */

template void gpu_basic_csr_wrapper<double>(Dense<double>& A, Dense<double>& B, CSR<double>& S, CSR<double>& P);
template void gpu_basic_coo_wrapper<double>(Dense<double>& A, Dense<double>& B, COO<double>& S, COO<double>& P);


