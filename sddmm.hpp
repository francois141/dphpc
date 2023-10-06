#include <iostream>
#include <vector>
#include <assert.h>
#include "Dense.h"
#include "CSR.h"

// TODO: Optimize naive version
template<typename T>
CSR<T> SequentialSDDMM(CSR<T> &S, Dense<T> &A, Dense<T> &B) {
    CSR<T> P(S);
    P.clearValues();

    // TODO: Do we want assertions?
    assert(A.getCols() == B.getCols());
    assert(A.getRows() == P.getRows());
    assert(B.getCols() == P.getCols());

    const unsigned int K = B.getCols();
    const unsigned int M = S.getRows();

    for(int i = 0; i < M;i++) {
        for(int j = S.getRowPositions()[i]; j < S.getRowPositions()[i+1];j++) {
            for(int k = 0; k < K; k++) {
                // TODO: Can we do better in c++?
                (*P.getValue(j)) += A.getValue(i,k) * B.getValue(S.getColPositions()[j],k);
            }
        }
    }

    for(int i = 0; i < S.getRowPositions().size() - 1;i++) {
        for(int j = S.getColPositions()[i]; j < S.getRowPositions()[i+1]; j++){
            // TODO: Can we do better in c++?
            (*P.getValue(j)) *= S.getValues()[j];
        }
    }

    return P;
}