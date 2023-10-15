#include <iostream>
#include <vector>
#include <assert.h>

#include "benchmark/competitor.hpp"
#include "matrices/matrices.h"

namespace Competitors {

    template <typename T>
    class CPUBasic : public SDDMM::Competitor<T> {
        public:

            CPUBasic()
            : SDDMM::Competitor<T>("CPU-Basic")
            {}

            virtual inline void run_csr(Dense<T> &A, Dense<T> &B, CSR<T> &S, CSR<T> &P) {
                const unsigned int K = B.getCols();
                const unsigned int M = S.getRows();

                for (int i = 0; i < M; i++) {
                    for (int j = S.getRowPositions()[i]; j < S.getRowPositions()[i+1];j++) {
                        for (int k = 0; k < K; k++) {
                            (*P.getValue(j)) += A.getValue(i,k) * B.getValue(S.getColPositions()[j],k);
                        }
                    }
                }

                for (int i = 0; i < S.getRowPositions().size() - 1; i++) {
                    for (int j = S.getColPositions()[i]; j < S.getRowPositions()[i+1]; j++) {
                        (*P.getValue(j)) *= S.getValues()[j];
                    }
                }
            }

            virtual inline void run_coo(Dense<T> &A, Dense<T> &B, COO<T> &S, COO<T> &P) {
                const unsigned int K = B.getCols();
                const unsigned int M = S.getRows();

                for (int i = 0; i < S.getValues().size(); i++) {
                    for (int k = 0; k < K; k++) {
                        (*P.getValue(i)) += A.getValue(k, S.getRowPositions()[i]) * B.getValue(S.getColPositions()[i], k);
                    }
                }

                for (int i = 0; i < S.getValues().size(); i++) {
                    (*P.getValue(i)) *= S.getValues()[i];
                }
            }

            virtual bool csr_supported() { return true; };
            virtual bool coo_supported() { return true; };

    };

}