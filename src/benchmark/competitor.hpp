#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include "matrices/matrices.h"

namespace SDDMM {

    template<typename T>
    class Competitor {
        public:
            const std::string name;
        
            Competitor(const std::string& name)
            : name(name)
            {}

            ~Competitor() {}

            virtual inline void run_csr(Dense<T> &A, Dense<T> &B, CSR<T> &S, CSR<T> &P) = 0;
            virtual inline void run_coo(Dense<T> &A, Dense<T> &B, COO<T> &S, COO<T> &P) = 0;

            virtual bool csr_supported() = 0;
            virtual bool coo_supported() = 0;
    };
}
