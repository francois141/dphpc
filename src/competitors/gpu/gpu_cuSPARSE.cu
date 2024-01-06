#include "gpu_cuSPARSE.hpp"


__global__ void gpu_cuSPARSE_scale_kernel(float* S, float* P, int S_nnz) {
   	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (; i < S_nnz; i += stride)
		P[i] = S[i] * P[i];

}

template <typename T>
void gpu_cuSPARSE_scale_wrapper(T* S, T* P, int S_nnz) {

	int threads_per_block = 1024;
	int thread_blocks = (S_nnz + threads_per_block - 1) / threads_per_block;

	// Perform SDDMM on the GPU
	gpu_cuSPARSE_scale_kernel<<<thread_blocks, threads_per_block>>>(S, P, S_nnz);
}

/* Workaround because the wrappers need to be inside the CUDA file (Would normally write templated functions inside the header file!) */
template void gpu_cuSPARSE_scale_wrapper<float>(float* S, float* P, int S_nnz);
