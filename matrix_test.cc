#include <gtest/gtest.h>
#include "Dense.h"
#include "Dense.cpp"
#include "CSR.h"
#include "CSR.cpp"
#include "sddmm.hpp"

// Smoke test
TEST(BasicTest, SmokeTest) {
    std::vector<std::vector<int>> matrix(2, std::vector<int>(2,1));
    Dense<int> d(matrix);

    std::vector<Triplet<int>> triplets{{0,0,1},{0,1,1},{1,0,1},{1,1,1}};
    CSR<int> sparseMatrix(2,2,triplets);

    const CSR<int> sddmmResult = SDDMM::Basic(sparseMatrix, d, d);

    std::vector<Triplet<int>> triplets2{{0,0,2},{0,1,2},{1,0,2},{1,1,2}};
    const CSR<int> expected(2,2,triplets2);

    EXPECT_EQ(sddmmResult, expected);
}

