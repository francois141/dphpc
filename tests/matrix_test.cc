#include <gtest/gtest.h>
#include "matrices/matrices.h"
#include "competitors/cpu/cpu_basic.hpp"
#include "competitors/gpu/gpu_basic.hpp"
#include "benchmark/dataset.hpp"
#include "utils/random_generator.hpp"
#include "utils/util.hpp"

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
    std::vector<Triplet<double>> triplets {{0,0,1.0},{0,1,1.0},{1,0,1.0},{1,1,1.0}};
    COO<double> coo(2, 2, triplets);

    EXPECT_EQ(4, coo.getValues().size());
    EXPECT_EQ(4, coo.getRowPositions().size());
    EXPECT_EQ(4, coo.getColPositions().size());
}

TEST(BasicTest, RepresentationConversion) {
    
    std::vector<Triplet<double>> triplets_csr{{0,0,1.0},{1,0,1.0},{0,1,1.0},{1,1,1.0}};
    CSR<double> csr_mat(2,2,triplets_csr);

    std::vector<Triplet<double>> triplets{{0,0,1.0},{0,1,1.0},{1,0,1.0},{1,1,1.0}};
    COO<double> coo_mat(2, 2, triplets);

    CSR<double> csr_converted(coo_mat);
    COO<double> coo_converted(csr_mat);

    EXPECT_EQ(csr_mat, csr_converted);
    EXPECT_EQ(coo_mat, coo_converted);
}

TEST(BasicTest, BiggerConversion) {
    for(int i = 100; i <= 1000; i += 25) {
        int cols = i;
        int rows = 2*i;
        int nbSamples = i;

        std::vector<Triplet<double>> triplets = sampleTriplets<double>(rows, cols, nbSamples);

        auto S_csr = CSR<double>(rows, cols, triplets);
        auto S_coo = COO<double>(rows, cols, triplets);

        EXPECT_EQ(S_csr,CSR<double>(S_coo));
        EXPECT_EQ(S_coo,COO<double>(S_csr));
    }
}

TEST(BasicTest, GPU_basic)
{
    auto gpu_basic =
        std::shared_ptr<Competitors::GPUBasic<double>>(new Competitors::GPUBasic<double>);

    auto cpu_basic =
        std::shared_ptr<Competitors::CPUBasic<double>>(new Competitors::CPUBasic<double>);

    std::vector<std::vector<double>> A_vals { { 1.0, 2.0 }, { 3.0, 4.0 } };
    Dense<double> A(A_vals);

    std::vector<std::vector<double>> B_vals { { -1.0, 2.0 }, { -3.0, 4.0 } };
    Dense<double> B(B_vals);

    std::vector<Triplet<double>> triplets {
        { 0, 0, 1.0 }, { 0, 1, 1.0 }, { 1, 0, 1.0 }, { 1, 1, 1.0 }
    };
    COO<double> S(2, 2, triplets);
    COO<double> P1(S);
    COO<double> P2(S);
    gpu_basic->run_coo(A, B, S, P1);
    cpu_basic->run_coo(A, B, S, P2);

    EXPECT_EQ(P1, P2);
}

TEST(BasicTest, GPU_basic2)
{
    auto gpu_basic =
        std::unique_ptr<Competitors::GPUBasic<double>>(new Competitors::GPUBasic<double>);

    auto cpu_basic =
        std::unique_ptr<Competitors::CPUBasic<double>>(new Competitors::CPUBasic<double>);

    std::vector<std::vector<double>> A_vals { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } };
    Dense<double> A(A_vals);

    std::vector<std::vector<double>> B_vals { { -1.0, -3.0, 5.0 }, { 2.0, 4.0, -6.0 } };
    Dense<double> B(B_vals);

    std::vector<Triplet<double>> triplets { { 0, 0, 1.0 }, { 1, 0, 1.0 }, { 1, 1, 1.0 } };
    COO<double> S(2, 2, triplets);
    COO<double> P1(S);
    COO<double> P2(S);
    gpu_basic->run_coo(A, B, S, P1);
    cpu_basic->run_coo(A, B, S, P2);

    EXPECT_EQ(P1, P2);
}

TEST(BasicTest, GPU_basic3)
{
    auto gpu_basic =
        std::unique_ptr<Competitors::GPUBasic<double>>(new Competitors::GPUBasic<double>);

    auto cpu_basic =
        std::unique_ptr<Competitors::CPUBasic<double>>(new Competitors::CPUBasic<double>);

    std::vector<std::vector<double>> A_vals { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } };
    Dense<double> A(A_vals);

    std::vector<std::vector<double>> B_vals { { -1.0, -3.0, 5.0 } };
    Dense<double> B(B_vals);

    std::vector<Triplet<double>> triplets { { 0, 0, 1.0 }, { 1, 0, -1.0 } };
    COO<double> S(2, 2, triplets);
    COO<double> P1(S);
    COO<double> P2(S);
    gpu_basic->run_coo(A, B, S, P1);
    cpu_basic->run_coo(A, B, S, P2);

    EXPECT_EQ(P1, P2);
}

TEST(BasicTest, GPU_basic4)
{
    auto gpu_basic =
        std::unique_ptr<Competitors::GPUBasic<double>>(new Competitors::GPUBasic<double>);

    auto cpu_basic =
        std::unique_ptr<Competitors::CPUBasic<double>>(new Competitors::CPUBasic<double>);

    std::vector<std::vector<double>> A_vals { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 } };
    Dense<double> A(A_vals);

    std::vector<std::vector<double>> B_vals { { -1.0, -3.0 }, { 2.0, 2.0 }, { 3.0, 1.0 } };
    Dense<double> B(B_vals);

    std::vector<Triplet<double>> triplets {
        { 0, 0, 1.0 }, { 1, 0, -1.0 }, { 2, 0, 1.0 }, { 2, 2, 1.0 }
    };
    COO<double> S(3, 3, triplets);
    COO<double> P1(S);
    COO<double> P2(S);
    gpu_basic->run_coo(A, B, S, P1);
    cpu_basic->run_coo(A, B, S, P2);

    EXPECT_EQ(P1, P2);
}

TEST(BasicTest, GPU_advanced)
{
    const int rows = 500;
    const int cols = 500;

    auto gpu_basic =
            std::unique_ptr<Competitors::GPUBasic<double>>(new Competitors::GPUBasic<double>);

    auto cpu_basic =
            std::unique_ptr<Competitors::CPUBasic<double>>(new Competitors::CPUBasic<double>);

    auto dataset =
            std::unique_ptr<SDDMM::RandomWithDensityDataset<double>>(new SDDMM::RandomWithDensityDataset<double>(rows, cols, 32, 0.3));

    auto S = dataset->getS_COO();
    auto A = dataset->getA();
    auto B = dataset->getB();

    COO<double> P1(S);
    COO<double> P2(S);

    gpu_basic->run_coo(A, B, S, P1);
    cpu_basic->run_coo(A, B, S, P2);

    EXPECT_EQ(P1, P2);
}

TEST(BasicTest, GPU_advanced_dense)
{
    const int rows = 250;
    const int cols = 250;

    auto gpu_basic =
            std::unique_ptr<Competitors::GPUBasic<double>>(new Competitors::GPUBasic<double>);

    auto cpu_basic =
            std::unique_ptr<Competitors::CPUBasic<double>>(new Competitors::CPUBasic<double>);

    auto dataset =
            std::unique_ptr<SDDMM::RandomWithDensityDataset<double>>(new SDDMM::RandomWithDensityDataset<double>(rows, cols, 32, 0.7));

    auto S = dataset->getS_COO();
    auto A = dataset->getA();
    auto B = dataset->getB();

    COO<double> P1(S);
    COO<double> P2(S);

    gpu_basic->run_coo(A, B, S, P1);
    cpu_basic->run_coo(A, B, S, P2);

    EXPECT_EQ(P1, P2);
}

TEST(BasicTest, GPU_advanced_sparse)
{
    const int rows = 10000;
    const int cols = 10000;

    auto gpu_basic =
            std::unique_ptr<Competitors::GPUBasic<double>>(new Competitors::GPUBasic<double>);

    auto cpu_basic =
            std::unique_ptr<Competitors::CPUBasic<double>>(new Competitors::CPUBasic<double>);

    auto dataset =
            std::unique_ptr<SDDMM::RandomWithDensityDataset<double>>(new SDDMM::RandomWithDensityDataset<double>(rows, cols, 32, 0.001));

    auto S = dataset->getS_COO();
    auto A = dataset->getA();
    auto B = dataset->getB();

    COO<double> P1(S);
    COO<double> P2(S);

    gpu_basic->run_coo(A, B, S, P1);
    cpu_basic->run_coo(A, B, S, P2);

    EXPECT_EQ(P1, P2);
}

