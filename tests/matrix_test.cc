#include <gtest/gtest.h>
#include "matrices/matrices.h"
#include "competitors/cpu/cpu_basic.hpp"

// Smoke test
TEST(BasicTest, SmokeTest) {
    // std::vector<std::vector<int>> matrix(2, std::vector<int>(2,1));
    // Dense<int> d(matrix);

    // std::vector<Triplet<int>> triplets{{0,0,1},{0,1,1},{1,0,1},{1,1,1}};
    // CSR<int> sparseMatrix(2,2,triplets);

    // const CSR<int> sddmmResult = SDDMM::Basic(sparseMatrix, d, d);

    // std::vector<Triplet<int>> triplets2{{0,0,2},{0,1,2},{1,0,2},{1,1,2}};
    // const CSR<int> expected(2,2,triplets2);

    // EXPECT_EQ(sddmmResult, expected);
}

TEST(BasicTest, COO) {
    std::vector<Triplet<int>> triplets{{0,0,1},{0,1,1},{1,0,1},{1,1,1}};
    COO<int> coo(2, 2, triplets);

    EXPECT_EQ(4, coo.getValues().size());
    EXPECT_EQ(4, coo.getRowPositions().size());
    EXPECT_EQ(4, coo.getColPositions().size());
}

TEST(BasicTest, RepresentationConversion) {
    
    std::vector<Triplet<int>> triplets_csr{{0,0,1},{0,1,1},{1,0,1},{1,1,1}};
    CSR<int> csr_mat(2,2,triplets_csr);

    std::vector<Triplet<int>> triplets{{0,0,1}, {1,0,1}, {0,1,1}, {1,1,1}};
    COO<int> coo_mat(2, 2, triplets);

    CSR<int> csr_converted(coo_mat);
    COO<int> coo_converted(csr_mat);
    
    // std::cout << csr_mat << std::endl;
    // std::cout << coo_mat << std::endl;
    // std::cout << csr_converted << std::endl;

    EXPECT_EQ(csr_mat, csr_converted);
    EXPECT_EQ(coo_mat, coo_converted);
}
