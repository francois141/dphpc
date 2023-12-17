#include "gpu_preprocessing.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>

// how many wraps are used to compute each coefficient
// this is in turn how many wraps we affect to each row
constexpr int threads_per_coef = 4;
constexpr int Tk = 32;
constexpr int Ti = Competitors::GPUPreprocessing::TileRows;
constexpr int Tj = Ti * threads_per_coef;
// make blocksize = Tj to make the copy code easier (this is not necessary)
constexpr int blocksize = Tj;

// perform SDDMM, compute P = (A*B^T) dot S (where dot is the term by term product)
// A is MxTk, B is NxTk, S and P are MxN sparse
__global__ void gpu_preprocessing_csr_kernel(const float* __restrict__  A, const float* __restrict__  B, const float* __restrict__ S, float* __restrict__ P, const int* cols, const int* rows, const int* prep_pos, const int* prep_ind, const int* prep_cols, int M, int K, int N) {
	// row_delta: which row this thread handles compared to the start row (blockIdx.x * Ti)
	int row_delta = threadIdx.x / threads_per_coef;
	int curr_thread = threadIdx.x % threads_per_coef;

	int nb_tiles_row = (M + Ti - 1) / Ti;
	int my_tile_row = blockIdx.x % nb_tiles_row;
	int my_row = my_tile_row * Ti + row_delta;
	int tile_k = (blockIdx.x / nb_tiles_row) * Tk;

	// local B matrix which contains the value used in one iteration
	__shared__ float local_B[Tj][Tk];
	int sparse_index = rows[my_row];

	// number of coefficients stored locally by this single thread
	constexpr int nb_coefs_stored = Tk / threads_per_coef;
	// we vectorize the operation
	assert(nb_coefs_stored % 4 == 0);
	float4 line_coefs[nb_coefs_stored / 4];
	for (int k = 0; k < nb_coefs_stored / 4; k++) {
		line_coefs[k] = *reinterpret_cast<const float4*>(&A[my_row * K + tile_k + curr_thread * nb_coefs_stored + k * 4]);
	}

	int prep_ind_offset = prep_pos[my_tile_row];
	int cols_for_line = prep_pos[my_tile_row + 1] - prep_ind_offset;
	for (int tile_j = 0; tile_j < cols_for_line; tile_j += Tj) {

		__syncthreads();

		// blocksize being Tj makes this easier
		// copy the part of B we are going to use to local_B
		if (threadIdx.x < cols_for_line - tile_j) {
			int copy_col = prep_ind[prep_ind_offset + tile_j + threadIdx.x];
			// each thread copies one full column
			for (int k = 0; k < Tk; k += 4) {
				*reinterpret_cast<float4*>(&local_B[threadIdx.x][k]) = *reinterpret_cast<const float4*>(&B[copy_col * K + tile_k + k]);
			}
		}
		__syncthreads();

		if (my_row >= M)
			continue;

		int last_col = prep_ind[min(prep_ind_offset + tile_j + Tj - 1, prep_ind_offset + cols_for_line - 1)];

		// while we are still in the current tile
		while (sparse_index < rows[my_row + 1] && cols[sparse_index] <= last_col) {
			// now perform the matrix multiplication

			// using prep_cols, we know directly which column in local_B we need to use
			int local_idx = prep_cols[sparse_index] - (prep_ind_offset + tile_j);

			float result = 0.0f;
			for (int k = 0; k < nb_coefs_stored / 4; k++) {
				float4 B_col = *reinterpret_cast<const float4*>(&local_B[local_idx][curr_thread * nb_coefs_stored + k * 4]);
				// perform the dot product
				result += line_coefs[k].x * B_col.x + line_coefs[k].y * B_col.y + line_coefs[k].z * B_col.z + line_coefs[k].w * B_col.w;
			}
			result *= S[sparse_index];

			// reduction process
			/*
			const unsigned wraps_idx = (row_delta * wraps_per_coef) % 32;
			const unsigned reduction_mask = ((1U << wraps_per_coef) - 1) << wraps_idx;
			for (int idx = wraps_per_coef / 2; idx >= 1; idx /= 2)
				result += __shfl_xor_sync(reduction_mask, result, idx);

			// use atomic operations because multiple invocation can modify this value at the same time
			if (curr_wrap == 0)
				atomicAdd(P + sparse_index, result);
			*/
			// It looks like using one atomicAdd (with a lot of congestion) is better performance-wise
			// than performing reductions in the wrap
			atomicAdd(P + sparse_index, result);

			sparse_index++;
		}
	}
}

namespace Competitors {

	void GPUPreprocessing::run_csr(Dense<T>& A, Dense<T>& B, CSR<T>& S, CSR<T>& P) {
		// A is MxK, B is NxK, S and P are MxN sparse
		int M = A.getRows();
		int K = A.getCols();
		int N = B.getRows();

		if (K % Tk != 0) {
			std::cerr << "Implementation requires that K is a multiple of " << Tk << " (current K = " << K << ")" << std::endl;
			return;
		}

		int nb_tiles_row = (M + Ti - 1) / Ti;
		int nb_tiles_k = (K + Tk - 1) / Tk;
		int block_count = nb_tiles_row * nb_tiles_k;
		this->set_num_thread_blocks(block_count);
		this->set_num_threads_per_block(blocksize);
		gpu_preprocessing_csr_kernel << < block_count, blocksize >> > (A_gpu, B_gpu, S_gpu, P_gpu, cols_gpu, rows_gpu, prep_pos_gpu, prep_ind_gpu, prep_col_gpu, M, K, N);

		cudaDeviceSynchronize();
	}

}
