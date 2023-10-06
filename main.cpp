#include <iostream>
#include <vector>
#include "Dense.h"
#include "Dense.cpp"
#include "CSR.h"
#include "CSR.cpp"
#include "sddmm.hpp"

// TODO: Add benchmark infrastructure in the code
int main() {
    std::vector<std::vector<int>> matrix(2, std::vector<int>(2,1));
    Dense<int> d(matrix);

    std::vector<int> values = {1,1,1,1};
    std::vector<std::pair<int,int>> positions = {{0,0},{0,1},{1,0},{1,1}};
    CSR<int> sparseMatrix(2,2,positions, values);

    std::cout << "SDDMM Operation" << std::endl;
    std::cout << SequentialSDDMM(sparseMatrix, d, d) << std::endl;
}