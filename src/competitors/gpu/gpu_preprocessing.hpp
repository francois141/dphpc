#pragma once

#include "benchmark/competitor.hpp"
#include "matrices/matrices.h"

namespace Competitors {

	class GPUPreprocessing : public SDDMM::Competitor<float> {
		using T = float;
	public:

		static constexpr int TileRows = 32;

		GPUPreprocessing()
			: SDDMM::Competitor<T>("GPU-Preprocessing")
		{}

		inline void init_csr(Dense<T>& A, Dense<T>& B, CSR<T>& S, CSR<T>& P) override {
			// A is MxK, B is NxK, S and P are MxN sparse
			unsigned int M = A.getRows();
			unsigned int K = A.getCols();
			unsigned int N = B.getRows();

			assert(K == B.getCols());

			// get the size needed for each matrix
			size_t A_size = M * K * sizeof(T);
			size_t B_size = K * N * sizeof(T);
			size_t SP_size = S.getValues().size() * sizeof(T);
			size_t sparse_col_size = S.getValues().size() * sizeof(int);
			size_t sparse_row_size = (S.getRows() + 1) * sizeof(int);

			static_assert(sizeof(T) == sizeof(float), "the kernel is specialized for single precision floating points");

			// allocate the matrices on the GPU
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
			// the kernel needs to assume that P is sets to 0
			cudaMemset(P_gpu, 0, SP_size);

			// preprocessing part
			int nb_tile_rows = (M + TileRows - 1) / TileRows;
			// first part, get the size of the block to allocate
			int preprocessing_block_size = 0;
			std::vector<int> preprocessing_positions(nb_tile_rows + 1);
			// preprocessing_cols maps a sparse entry to its compressed column
			std::vector<int> preprocessing_cols(S.getValues().size());

			for (int row_tile = 0; row_tile < nb_tile_rows; row_tile++) {
				int row_start = row_tile * TileRows;

				// we need the TileRows coefficients in a row to be 0 to ignore it
				int col_indices[TileRows];
				int curr_rows = std::min<int>(TileRows, M - row_start);
				for (int row = 0; row < curr_rows; row++) {
					col_indices[row] = S.getRowPositions()[row_start + row];
				}

				while (true) {
					int min_col = INT_MAX;
					// get the first column with a nonzero entry
					for (int row = 0; row < curr_rows; row++) {
						int col_index = col_indices[row];
						if (col_index < S.getRowPositions()[row_start + row + 1])
							min_col = std::min(min_col, S.getColPositions()[col_index]);
					}
					if (min_col == INT_MAX)
						break;

					for (int row = 0; row < curr_rows; row++) {
						int& col_index = col_indices[row];
						if (col_index < S.getRowPositions()[row_start + row + 1] && S.getColPositions()[col_index] == min_col) {
							preprocessing_cols[col_index] = preprocessing_block_size;
							col_index++;
						}
					}
					preprocessing_block_size++;
				}

				preprocessing_positions[row_tile + 1] = preprocessing_block_size;
			}

			// now that we have the size, we can compute the indices
			std::vector<int> preprocessing_indices(preprocessing_block_size);
			preprocessing_block_size = 0;
			for (int row_tile = 0; row_tile < nb_tile_rows; row_tile++) {
				int row_start = row_tile * TileRows;

				// we need the TileRows coefficients in a row to be 0 to ignore it
				int col_indices[TileRows];
				int curr_rows = std::min<int>(TileRows, M - row_start);
				for (int row = 0; row < curr_rows; row++) {
					col_indices[row] = S.getRowPositions()[row_start + row];
				}

				while (true) {
					int min_col = INT_MAX;
					for (int row = 0; row < curr_rows; row++) {
						int col_index = col_indices[row];
						if (col_index < S.getRowPositions()[row_start + row + 1])
							min_col = std::min(min_col, S.getColPositions()[col_index]);
					}
					if (min_col == INT_MAX)
						break;

					preprocessing_indices[preprocessing_block_size++] = min_col;
					for (int row = 0; row < curr_rows; row++) {
						int& col_index = col_indices[row];
						if (col_index < S.getRowPositions()[row_start + row + 1] && S.getColPositions()[col_index] == min_col)
							col_index++;
					}
				}
			}

			// create and copy the preprocessing data to the GPU
			size_t prep_pos_size = preprocessing_positions.size() * sizeof(int);
			size_t prep_ind_size = preprocessing_indices.size() * sizeof(int);
			cudaMalloc(&prep_pos_gpu, prep_pos_size);
			cudaMalloc(&prep_ind_gpu, prep_ind_size);
			cudaMalloc(&prep_col_gpu, sparse_col_size);

			cudaMemcpy(prep_pos_gpu, preprocessing_positions.data(), prep_pos_size, cudaMemcpyHostToDevice);
			cudaMemcpy(prep_ind_gpu, preprocessing_indices.data(), prep_ind_size, cudaMemcpyHostToDevice);
			cudaMemcpy(prep_col_gpu, preprocessing_cols.data(), sparse_col_size, cudaMemcpyHostToDevice);

			cudaDeviceSynchronize();
		}

		void run_csr(Dense<T>& A, Dense<T>& B, CSR<T>& S, CSR<T>& P);

		inline void cleanup_csr(Dense<T>& A, Dense<T>& B, CSR<T>& S, CSR<T>& P) override {
			size_t SP_size = S.getValues().size() * sizeof(T);

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

			cudaFree(prep_pos_gpu);
			cudaFree(prep_ind_gpu);
			cudaFree(prep_col_gpu);
		}

		inline void run_coo(Dense<T>& A, Dense<T>& B, COO<T>& S, COO<T>& P) override {}

		bool csr_supported() override { return true; };
		bool coo_supported() override { return false; };

		bool is_gpu() override { return true; };

	private:
		float* A_gpu, * B_gpu, * S_gpu, * P_gpu;
		int* cols_gpu, * rows_gpu;
		int* prep_pos_gpu, * prep_ind_gpu, * prep_col_gpu;
	};

}