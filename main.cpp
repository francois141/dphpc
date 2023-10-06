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

    std::vector<Triplet<int>> triplets{{0,0,1},{0,1,1},{1,0,1},{1,1,1}};
    CSR<int> sparseMatrix(2,2, triplets);

    std::cout << "SDDMM Operation" << std::endl;
    std::cout << SDDMM::Basic(sparseMatrix, d, d) << std::endl;
}