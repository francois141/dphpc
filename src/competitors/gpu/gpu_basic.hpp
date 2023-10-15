#pragma once

#include "benchmark/competitor.hpp"
#include "matrices/matrices.h"

template <typename T>
void gpu_basic_csr_wrapper(Dense<T> &A, Dense<T> &B, CSR<T> &S, CSR<T> &P);

template <typename T>
void gpu_basic_coo_wrapper(Dense<T> &A, Dense<T> &B, COO<T> &S, COO<T> &P);

namespace Competitors {

    template <typename T>
    class GPUBasic : public SDDMM::Competitor<T> {
        public:

            GPUBasic()
            : SDDMM::Competitor<T>("GPU-Basic")
            {}

            virtual inline void run_csr(Dense<T> &A, Dense<T> &B, CSR<T> &S, CSR<T> &P) {
                gpu_basic_csr_wrapper(A, B, S, P);
            }

            virtual inline void run_coo(Dense<T> &A, Dense<T> &B, COO<T> &S, COO<T> &P) {
                gpu_basic_coo_wrapper(A, B, S, P);
            }

            virtual bool csr_supported() { return false; };
            virtual bool coo_supported() { return true; };

    };

}