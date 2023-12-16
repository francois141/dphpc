#pragma once

#include "benchmark/competitor.hpp"
#include "matrices/matrices.h"

template <typename T>
void gpu_tiled_csr_wrapper(T* A_gpu, T* B_gpu, T* S_gpu, T* P_gpu, int* cols_gpu, int* rows_gpu, int M, int K, int N, int sparse_size, int num_thread_blocks, int num_threads_per_block);

namespace Competitors {

    template <typename T>
    class GPUTiled : public SDDMM::Competitor<T> {
    public:

        GPUTiled()
            : SDDMM::Competitor<T>("GPU-Tiled")
        {}

        GPUTiled(int threads_per_block, int thread_blocks)
            : SDDMM::Competitor<T>("GPU-Tiled", threads_per_block, thread_blocks)
        {}

        virtual inline void init_csr(Dense<T>& A, Dense<T>& B, CSR<T>& S, CSR<T>& P) override {
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
        }

        virtual inline void run_csr(Dense<T>& A, Dense<T>& B, CSR<T>& S, CSR<T>& P) override {
            // A is MxK, B is NxK, S and P are MxN sparse
            int M = A.getRows();
            int K = A.getCols();
            int N = B.getRows();

            size_t sparse_size = S.getValues().size();
            int num_thread_blocks = this->get_num_thread_blocks();
            int num_threads_per_block = this->get_num_threads_per_block();

            gpu_tiled_csr_wrapper(A_gpu, B_gpu, S_gpu, P_gpu, cols_gpu, rows_gpu, M, K, N, sparse_size, num_thread_blocks, num_threads_per_block);            
            cudaDeviceSynchronize();
        }

        virtual inline void cleanup_csr(Dense<T>& A, Dense<T>& B, CSR<T>& S, CSR<T>& P) override {           
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
        }

        virtual inline void run_coo(Dense<T>& A, Dense<T>& B, COO<T>& S, COO<T>& P) override {}

        virtual bool csr_supported() override { return true; };
        virtual bool coo_supported() override { return false; };

        virtual bool is_gpu() override { return true; }

    private:
        float* A_gpu, * B_gpu, * S_gpu, * P_gpu;
        int* cols_gpu, * rows_gpu;
    };

}