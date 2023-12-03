#pragma once

#include "benchmark/competitor.hpp"
#include "matrices/matrices.h"

template <typename T>
void gpu_adaptive_tiling_csr_wrapper(T* A_gpu, T* B_gpu, T* S_gpu, T* P_gpu, int* cols_gpu, int* rows_gpu, int M, int K, int N);

template <typename T>
void gpu_reorder_csr_row_panel_wrapper(int* cols, T* vals, int* reordered_cols, T* reordered_vals, int* panel_ptr, int num_rows, int num_cols);

namespace Competitors {

    template <typename T>
    class GPUAdaptiveTiling : public SDDMM::Competitor<T> {
    public:

        GPUAdaptiveTiling()
            : SDDMM::Competitor<T>("GPU-Adaptive-Tiling")
        {}

        virtual inline void init_csr(Dense<T>& A, Dense<T>& B, CSR<T>& S, CSR<T>& P) override {
            int panel_size = 128; // num rows per panel
            int tile_width = 256; // num columns per tile

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
            size_t num_panels_size = ((S.getRows() + panel_size - 1) / panel_size + 1) * sizeof(int);

            static_assert(sizeof(T) == sizeof(float), "the kernel is specialized for single precision floating points");

            T* reordered_vals = (T*)malloc(SP_size);
            int* reordered_cols = (int*)malloc(sparse_col_size);
            int* panel_ptr = (int*)malloc(num_panels_size);

            // allocate the matrices on the GPU
            cudaMalloc(&A_gpu, A_size);
            cudaMalloc(&B_gpu, B_size);
            cudaMalloc(&S_gpu, SP_size);
            cudaMalloc(&P_gpu, SP_size);
            cudaMalloc(&cols_gpu, sparse_col_size);
            cudaMalloc(&rows_gpu, sparse_row_size);
            cudaMalloc(&reordered_cols_gpu, sparse_col_size);
            cudaMalloc(&reordered_vals_gpu, SP_size);
            cudaMalloc(&panel_ptr_gpu, num_panels_size);

            // copy from RAM to GPU
            cudaMemcpy(A_gpu, &A.getValue(0, 0), A_size, cudaMemcpyHostToDevice);
            cudaMemcpy(B_gpu, &B.getValue(0, 0), B_size, cudaMemcpyHostToDevice);
            cudaMemcpy(S_gpu, S.getValues().data(), SP_size, cudaMemcpyHostToDevice);
            cudaMemcpy(cols_gpu, S.getColPositions().data(), sparse_col_size, cudaMemcpyHostToDevice);
            cudaMemcpy(rows_gpu, S.getRowPositions().data(), sparse_row_size, cudaMemcpyHostToDevice);
            
            gpu_reorder_csr_row_panel_wrapper(cols_gpu, S_gpu, reordered_cols_gpu, reordered_vals_gpu, panel_ptr_gpu, M, N);

            // copy from GPU to RAM
            cudaMemcpy(reordered_cols_gpu, reordered_cols, sparse_col_size, cudaMemcpyDeviceToHost);
            cudaMemcpy(reordered_vals_gpu, reordered_vals, sparse_col_size, cudaMemcpyDeviceToHost);
            cudaMemcpy(panel_ptr_gpu, panel_ptr, num_panels_size, cudaMemcpyDeviceToHost);
        }

        virtual inline void run_csr(Dense<T>& A, Dense<T>& B, CSR<T>& S, CSR<T>& P) override {
            // A is MxK, B is NxK, S and P are MxN sparse
            int M = A.getRows();
            int K = A.getCols();
            int N = B.getRows();

            // gpu_adaptive_tiling_csr_wrapper(A_gpu, B_gpu, S_gpu, P_gpu, cols_gpu, rows_gpu, M, K, N);            
            // cudaDeviceSynchronize();
        }

        virtual inline void cleanup_csr(Dense<T>& A, Dense<T>& B, CSR<T>& S, CSR<T>& P) override {           
            size_t SP_size = S.getValues().size() * sizeof(T);

            // copy result back to RAM
            // cudaMemcpy(P.getValues().data(), P_gpu, SP_size, cudaMemcpyDeviceToHost);
            // P.setColPositions(S.getColPositions());
            // P.setRowPositions(S.getRowPositions());

            // free all the CPU allocated memory
            free(reordered_vals);
            free(reordered_cols);
            free(panel_ptr);

            // free all the GPU allocated memory
            cudaFree(A_gpu);
            cudaFree(B_gpu);
            cudaFree(S_gpu);
            cudaFree(P_gpu);
            cudaFree(cols_gpu);
            cudaFree(rows_gpu);
            cudaFree(reordered_cols_gpu);
            cudaFree(reordered_vals_gpu);
            cudaFree(panel_ptr_gpu);
        }

        virtual inline void run_coo(Dense<T>& A, Dense<T>& B, COO<T>& S, COO<T>& P) override {}

        virtual bool csr_supported() override { return true; };
        virtual bool coo_supported() override { return false; };

        virtual bool is_gpu() override { return true; }

        virtual float* get_reordered_vals() { return reordered_vals; }
        virtual int* get_reordered_cols() { return reordered_cols; }
        virtual int* get_panel_ptr() { return panel_ptr };

    private:
        float* A_gpu, * B_gpu, * S_gpu, * P_gpu, * reordered_vals, * reordered_vals_gpu;
        int* cols_gpu, * rows_gpu, * reordered_cols, * reordered_cols_gpu, * panel_ptr_gpu, * panel_ptr;
    };

}