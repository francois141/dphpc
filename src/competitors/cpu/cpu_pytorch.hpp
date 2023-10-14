#include <iostream>

#include "benchmark/competitor.hpp"
#include "matrices/matrices.h"

namespace Competitors {

    template <typename T>
    class CPUPyTorch : public SDDMM::Competitor<T> {
        public:

            CPUPyTorch()
            : SDDMM::Competitor<T>("CPU-PyTorch")
            {}

            virtual inline void run_csr(Dense<T> &A, Dense<T> &B, CSR<T> &S, CSR<T> &P) {
                std::cout << "(Not supported)" << std::endl;
            }

            virtual inline void run_coo(Dense<T> &A, Dense<T> &B, COO<T> &S, COO<T> &P) {
                std::cout << "(Not supported)" << std::endl;
            }

            // TODO - implement one of these by calling library func
            // TODO - timing measurements might be negatively affected if we need to convert to other data types

    };

}