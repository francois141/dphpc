#include "gpu_tensor.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>

using namespace nvcuda;

#include <stdio.h>

#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }

void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
    if (stat != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
    }
}

half *A_tensor, *B_tensor;

__global__ void convertFp32ToFp16(half *out, float *in, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}

__global__ void
gpu_tensor_csr_kernel(half *A, half *B, float *S, float *P, int *cols, int *rows, int *start_idx, int M, int K, int N,
                      int sparse_size, int row_size) {

    int warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warp_offset = (blockIdx.x * blockDim.x + threadIdx.x) % 16;

    int row_start = start_idx[warp_idx];
    int row_end = start_idx[warp_idx + 1];

    // Distpatch at warp level
    // Iterate over stride of 16
    for (int row_idx = row_start; row_idx < row_end; row_idx += 16) {

        for (int k = 0; k < K; k += 16) {

            int row = row_idx + warp_offset;
            int idx = rows[row];

            while (true) {
                // Get next row = minium de tous les threads
                __shared__ int nextColValue;
                nextColValue = INT_MAX;

                // Check if next col est infinie ==> si c'est le cas ==> break la loop et passe à la suite
                __syncthreads();

                if (idx < rows[row + 1]) {
                    nextColValue = min(nextColValue, cols[idx] - cols[idx] % 16);
                }

                __syncthreads();

                // If we are done we can break
                if (nextColValue == INT_MAX) {
                    break;
                }

                // Otherwise we do the local computations
                __shared__ float c[256];

                // Declare the fragments
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

                // Load matrices
                wmma::load_matrix_sync(a_frag, A + K * row_idx + k, K);
                wmma::load_matrix_sync(b_frag, B + K * nextColValue + k, K);

                // Equivalent to memset
                wmma::fill_fragment(acc_frag, 0.0f);

                // Perform the matrix multiplication
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

                // Store matrix in shared memory
                wmma::store_matrix_sync(c, acc_frag, 16, wmma::mem_col_major);
                // Update 16x16 tile
                while (idx < rows[row + 1] && cols[idx] < nextColValue + 16) {
                    // We convert global index into a "local" index
                    // TODO: replace modulos by faster operations
                    int colLocal = cols[idx] % 16;
                    int rowLocal = row % 16;
                    P[idx] += c[colLocal * 16 + rowLocal] * S[idx];
                    idx++;
                }

                __syncthreads();
            }
        }
    }
}

template<typename T>
void gpu_tensor_csr_wrapper(T *S_gpu, T *P_gpu, int *cols_gpu, int *rows_gpu, int *start_idx, int M, int K, int N,
                            int sparse_size, int row_size) {
    // Perform SDDMM on the GPU
    gpu_tensor_csr_kernel<<<32, 32>>>(A_tensor, B_tensor, S_gpu, P_gpu, cols_gpu, rows_gpu, start_idx, M, K, N,
                                      sparse_size, row_size);
    cudaErrCheck(cudaDeviceSynchronize());
}

/* Workaround because the wrappers need to be inside the CUDA file (Would normally write templated functions inside the header file!) */
template void
gpu_tensor_csr_wrapper<float>(float *S_gpu, float *P_gpu, int *cols_gpu, int *rows_gpu, int *start_idx, int M, int K,
                              int N, int sparse_size, int row_size);

void setupTensorData(float *A, float *B, size_t A_size, size_t B_size) { // test

    // Malloc half floats
    cudaErrCheck(cudaMalloc(&A_tensor, A_size * sizeof(half)));
    cudaMalloc(&B_tensor, B_size * sizeof(half));

    convertFp32ToFp16<<<A_size / 32, 32>>>(A_tensor, A, A_size);
    cudaErrCheck(cudaDeviceSynchronize());
    convertFp32ToFp16<<<B_size / 32, 32>>>(B_tensor, B, B_size);
    cudaErrCheck(cudaDeviceSynchronize());
}

void freeTensorData() {
    // free values for the tensors
    cudaFree(A_tensor);
    cudaFree(B_tensor);
}