#include <iostream>
#include <vector>
#include <assert.h>
#include "Dense.h"
#include "Dense.cpp"
#include "CSR.h"
#include "CSR.cpp"

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
                (*P.getValue(j)) = A.getValue(i,k) * B.getValue(S.getColPositions()[j],k);
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

// TODO: Add benchmark infrastructure in the code
// TODO: Write unit tests - use google framework for example
int main() {
    std::vector<std::vector<int>> matrix(2, std::vector<int>(2,1));
    Dense<int> d(matrix);

    std::vector<int> values = {1,1,1,1};
    std::vector<std::pair<int,int>> positions = {{0,0},{0,1},{1,0},{1,1}};
    CSR<int> sparseMatrix(2,2,positions, values);

    std::cout << "SDDMM Operation" << std::endl;
    std::cout << SequentialSDDMM(sparseMatrix, d, d) << std::endl;
}