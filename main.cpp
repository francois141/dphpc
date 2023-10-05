#include <iostream>
#include <vector>
#include "Dense.h"
#include "Dense.cpp"
#include "CSR.h"
#include "CSR.cpp"

// TODO: Check bounds
// TODO: Optimize naive version
template<typename T>
CSR<T> SequentialSDDMM(CSR<T> &S, Dense<T> &A, Dense<T> &B) {
    CSR<T> P(S);
    P.clearValues();

    const unsigned int K = B.getCols();
    // TODO: Create a more clean code
    const unsigned int M = S.getRowPositions().size()-1;

    for(int i = 0; i < M;i++) {
        for(int j = S.getRowPositions()[i]; j < S.getRowPositions()[i+1];j++) {
            for(int k = 0; k < K; k++) {
                // TODO: Rewrite this hugly and slow code
                T res = A.getValue(i,k) * B.getValue(S.getColPositions()[j],k);
                std::vector<T> values = P.getValues();
                values[j] += res;
                P.setValues(values);
            }
        }
    }

    for(int i = 0; i < S.getRowPositions().size() - 1;i++) {
        for(int j = S.getColPositions()[i]; j < S.getRowPositions()[i+1]; j++){
            // TODO: Rewrite this slow and hugly code
            std::vector<T> values = P.getValues();
            values[j] *= S.getValues()[j];
            P.setValues(values);
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
    CSR<int> sparseMatrix(positions, values);

    std::cout << "SDDMM Operation" << std::endl;
    std::cout << SequentialSDDMM(sparseMatrix, d, d) << std::endl;
}