#pragma once

#include "benchmark/competitor.hpp"
#include "matrices/matrices.h"

template <typename T>
void gpu_tiled_csr_wrapper(Dense<T>& A, Dense<T>& B, CSR<T>& S, CSR<T>& P);


namespace Competitors {

    template <typename T>
    class GPUTiled : public SDDMM::Competitor<T> {
    public:

        GPUTiled()
            : SDDMM::Competitor<T>("GPU-Tiled")
        {}

        inline void run_csr(Dense<T>& A, Dense<T>& B, CSR<T>& S, CSR<T>& P) override {
            gpu_tiled_csr_wrapper(A, B, S, P);
        }

        inline void run_coo(Dense<T>& A, Dense<T>& B, COO<T>& S, COO<T>& P) override {}

        bool csr_supported() override { return true; };
        bool coo_supported() override { return false; };

    };

}