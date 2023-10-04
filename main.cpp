#include <iostream>
#include <vector>
#include "Dense.h"
#include "Dense.cpp"
#include "CSR.h"
#include "CSR.cpp"

template<typename T>
Dense<T> SequentialSDDMM(const CSR<T> &S, const Dense<T> &A, const Dense<T> &B) {
    return Dense<T>();
}

int main() {
    std::cout << "Dense Matrix" << std::endl;

    std::vector<std::vector<int>> matrix(4, std::vector<int>(4,4));

    Dense<int> d(matrix);
    std::cout << d;

    std::cout << "Sparse Matrix" << std::endl;

    std::vector<int> values = {1,2,3,4};
    std::vector<std::pair<int,int>> positions = {{1,1},{1,2},{1,3},{4,4}};

    CSR<int> sparseMatrix(positions, values);
    std::cout << sparseMatrix;
}