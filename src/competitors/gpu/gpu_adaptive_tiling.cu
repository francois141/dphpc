#include "gpu_adaptive_tiling.hpp"

#include <cuda_runtime.h>
#include <algorithm>

const int TILE_SIZE = 256;
const int PANEL_SIZE = 3;
const int THRESHOLD = 2;

/*
This function reorders the columns of the sparse matrix S (CSR) such that all columns with a density
above threshold are at the beginning and all columns with a low denisty are at the end of the panel
*/
__global__ void reorder_csr_row_panel(int* rows, int* cols, float* vals, int* reordered_cols, float* reordered_vals, int* panel_ptr, int num_rows, int num_cols){
	int* col_count = (int*)malloc(num_cols * sizeof(int));
	int* col_to_tile_id = (int*)malloc(num_cols * sizeof(int));
	for (int i = 0; i < num_cols; i++){
		col_count[i] = 0;
		col_to_tile_id[i] = 0;
	}

	int start_row = PANEL_SIZE * threadIdx.x;
	int end_row = min(start_row + PANEL_SIZE, num_rows);

	// count the number of non-zero element in current row_panel
	int num_heavy_cols = 0;
	for (int row = start_row; row < end_row; row++){
		int sparse_index = rows[row];
		while (sparse_index < rows[row+1]){
			col_count[cols[sparse_index]]++;

			// store number of heavy columns
			if (col_count[cols[sparse_index]] == THRESHOLD)
				num_heavy_cols++;

			sparse_index++;
		}
	}

	// each heavy tile has TILE_SIZE columns and there is one additional tile for the sparse columns
	int num_tiles = (num_heavy_cols + TILE_SIZE - 1) / TILE_SIZE + 1;
	panel_ptr[threadIdx.x+1] = num_tiles;

	// run over columns and save the tile_id for each col
	int ctr = 0;
	for (int i = 0; i < num_cols; i++){
		if (col_count[i] >= THRESHOLD){
			col_to_tile_id[i] = ctr / TILE_SIZE;
			ctr++;
		} else {
			col_to_tile_id[i] = num_tiles - 1; // sparse column
		}
	}

	// reoder each row, heavy columns at the front and sparse columns at the back
	for (int row = start_row; row < end_row; row++){
		int sparse_index = rows[row];
		int heavy_ptr = rows[row];
		int sparse_ptr = rows[row+1]-1; // we fill sparse columns from the back
		while (sparse_index < rows[row+1]){
			if (col_count[cols[sparse_index]] >= THRESHOLD){
				reordered_cols[heavy_ptr] = cols[sparse_index];
				reordered_vals[heavy_ptr] = vals[sparse_index];
				heavy_ptr++;
			} else {
				reordered_cols[sparse_index] = cols[sparse_index];
				reordered_vals[sparse_index] = vals[sparse_index];
				sparse_ptr--;
			}
			sparse_index++;
		}
	}
}

// perform SDDMM, compute P = (A*B^T) dot S (where dot is the term by term product)
// A is MxK, B is NxK, S and P are MxN sparse
__global__ void gpu_adaptive_tiling_csr_wrapper(float* A, float* B, float* S, float* P, int* cols, int* rows, int M, int K, int N) {
	int nb_running = gridDim.x * blockDim.x;
}

template <typename T>
void gpu_adaptive_tiling_csr_wrapper(T* A_gpu, T* B_gpu, T* S_gpu, T* P_gpu, int* cols_gpu, int* rows_gpu, int M, int K, int N) {
	
	// Perform SDDMM on the GPU
	// gpu_tiled_csr_kernel<<<32, 512>>>(A_gpu, B_gpu, S_gpu, P_gpu, cols_gpu, rows_gpu, M, K, N);
}

template <typename T>
void gpu_reorder_csr_row_panel_wrapper(int* rows, int* cols, T* vals, int* reordered_cols, T* reordered_vals, int* panel_ptr, int num_rows, int num_cols){
	int num_threads = (num_rows + PANEL_SIZE - 1) / PANEL_SIZE;
	reorder_csr_row_panel<<<1, num_threads>>>(rows, cols, vals, reordered_cols, reordered_vals, panel_ptr, num_rows, num_cols);
}

/* Workaround because the wrappers need to be inside the CUDA file (Would normally write templated functions inside the header file!) */
template void gpu_adaptive_tiling_csr_wrapper<float>(float* A_gpu, float* B_gpu, float* S_gpu, float* P_gpu, int* cols_gpu, int* rows_gpu, int M, int K, int N);

template void gpu_reorder_csr_row_panel_wrapper<float>(int* rows, int* cols, float* vals, int* reordered_cols, float* reordered_vals, int* panel_ptr, int num_rows, int num_cols);
