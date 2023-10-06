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

    CSR<int> sparseMatrix(2,2, {{0,0},{0,1},{1,0},{1,1}}, {1,1,1,1});

    const CSR<int> sddmmResult = SequentialSDDMM(sparseMatrix, d, d);
    const CSR<int> expected(2,2,{{0,0},{1,0},{0,1},{1,1}}, {2,2,2,2});

    EXPECT_EQ(sddmmResult, expected);
}

