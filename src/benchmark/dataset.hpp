#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>

#include "matrices/matrices.h"

namespace SDDMM {

    template<typename T>
    class Dataset {
        private:
            Dense<T> A;
            Dense<T> B;

            CSR<T> S_csr;
            COO<T> S_coo;
            
            CSR<T> P_csr;
            COO<T> P_coo;
        public:
        
            Dataset(const std::string& dataset_path) {
                // load & allocate dataset from FS (A, B, S)
            }

            Dataset(Dense<T> &A, Dense<T> &B, CSR<T> &S_csr)
            : A(A), B(B),
            S_csr(S_csr), S_coo(S_csr),
            P_csr(), P_coo()
            {}

            Dataset(Dense<T> &A, Dense<T> &B, COO<T> &S_coo)
            : A(A), B(B),
            S_csr(S_coo), S_coo(S_coo),
            P_csr(), P_coo()
            {}
            
            Dataset(Dense<T> &A, Dense<T> &B, CSR<T> &S_csr, CSR<T> &P_csr)
            : A(A), B(B),
            S_csr(S_csr), S_coo(S_csr),
            P_csr(P_csr), P_coo(P_csr)
            {}

            Dataset(Dense<T> &A, Dense<T> &B, COO<T> &S_coo, COO<T> &P_coo)
            : A(A), B(B),
            S_csr(S_coo), S_coo(S_coo),
            P_csr(P_coo), P_coo(P_coo)
            {}


            ~Dataset()
            {}

            Dense<T> &getA() {
                return this->A;
            }

            Dense<T> &getB() {
                return this->B;
            }

            CSR<T> &getS_CSR() {
                return this->S_csr;
            }

            COO<T> &getS_COO() {
                return this->S_coo;
            }

            bool hasExpected_CSR() {
                return this->P_csr.getRows() > 0;
            }

            CSR<T> &getExpected_CSR() {
                return this->P_csr;
            }

            const bool hasExpected_COO() {
                return this->P_coo.getRows() > 0;
            }

            COO<T> &getExpected_COO() {
                return this->P_coo;
            }
    };
}
