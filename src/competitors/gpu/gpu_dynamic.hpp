#pragma once

#include <cuda_runtime.h>

#include "benchmark/competitor.hpp"
#include "matrices/matrices.h"
#include "gpu_shared.hpp"
#include "gpu_convert.hpp"

template <typename T>
void gpu_basic_dynamic_wrapper(T* A_gpu, T* B_gpu, T* S_gpu, T* P_gpu, int* cols_gpu, int* rows_gpu, int M, int K, int N, int sparse_size, int row_size);

namespace Competitors {

    template <typename T>
    class GPUDynamic : public SDDMM::Competitor<T> {
    public:

        GPUDynamic()
                : SDDMM::Competitor<T>("GPU-Dynamic")
        {}

        GPUDynamic(int threads_per_block, int thread_blocks)
            : SDDMM::Competitor<T>("GPU-Dynamic", threads_per_block, thread_blocks)
        {}

        virtual inline void init_csr(Dense<T>& A, Dense<T>& B, CSR<T>& S, CSR<T>& P) override {
            if(S.getDensity() >= 0.0001) {
                this->denseCompetitor.init_csr(A,B,S,P);
            } else {
                this->sparseCompetitor.init_csr(A,B,S,P);
            }
        }

        virtual inline void run_csr(Dense<T>& A, Dense<T>& B, CSR<T>& S, CSR<T>& P) override {
            if(S.getDensity() >= 0.0001) {
                this->denseCompetitor.run_csr(A,B,S,P);
            } else {
                this->sparseCompetitor.run_csr(A,B,S,P);
            }
        }

        virtual inline void cleanup_csr(Dense<T>& A, Dense<T>& B, CSR<T>& S, CSR<T>& P) override {
            if(S.getDensity() >= 0.0001 ) {
                this->denseCompetitor.cleanup_csr(A,B,S,P);
            } else {
                this->sparseCompetitor.cleanup_csr(A,B,S,P);
            }
        }

        virtual inline void init_coo(Dense<T>& A, Dense<T>& B, COO<T>& S, COO<T>& P) override {}

        virtual inline void run_coo(Dense<T>& A, Dense<T>& B, COO<T>& S, COO<T>& P) override {}

        virtual inline void cleanup_coo(Dense<T>& A, Dense<T>& B, COO<T>& S, COO<T>& P) override {}

        virtual bool csr_supported() override { return true; };
        virtual bool coo_supported() override { return false; };

        virtual bool is_gpu() override { return true; };

    private:
        float* A_gpu, * B_gpu, * S_gpu, * P_gpu;
        int* cols_gpu, * rows_gpu;

        Competitors::GPUPreprocessing denseCompetitor = Competitors::GPUPreprocessing(this->get_num_thread_blocks(), this->get_num_threads_per_block());
        Competitors::GPUConvert sparseCompetitor = Competitors::GPUConvert(this->get_num_thread_blocks(), this->get_num_threads_per_block());
    };

}