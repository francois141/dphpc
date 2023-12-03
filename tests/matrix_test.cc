#include <gtest/gtest.h>
#include "matrices/matrices.h"
#include "competitors/cpu/cpu_basic.hpp"
#include "competitors/cpu/cpu_pytorch.hpp"
#include "competitors/gpu/gpu_basic.hpp"
#include "competitors/gpu/gpu_tiled.hpp"
#include "competitors/gpu/gpu_thread_dispatcher.hpp"
#include "competitors/gpu/gpu_pytorch.hpp"
#include "competitors/gpu/gpu_adaptive_tiling.hpp"
#include "benchmark/dataset.hpp"
#include "utils/random_generator.hpp"
#include "utils/util.hpp"

TEST(BasicTest, COO) {
    std::vector<Triplet<float>> triplets {{0,0,1.0},{0,1,1.0},{1,0,1.0},{1,1,1.0}};
    COO<float> coo(2, 2, triplets);

    EXPECT_EQ(4, coo.getValues().size());
    EXPECT_EQ(4, coo.getRowPositions().size());
    EXPECT_EQ(4, coo.getColPositions().size());
}

TEST(BasicTest, RepresentationConversion) {
    
    std::vector<Triplet<float>> triplets_csr{{0,0,1.0},{1,0,1.0},{0,1,1.0},{1,1,1.0}};
    CSR<float> csr_mat(2,2,triplets_csr);

    std::vector<Triplet<float>> triplets{{0,0,1.0},{0,1,1.0},{1,0,1.0},{1,1,1.0}};
    COO<float> coo_mat(2, 2, triplets);

    CSR<float> csr_converted(coo_mat);
    COO<float> coo_converted(csr_mat);

    EXPECT_EQ(csr_mat, csr_converted);
    EXPECT_EQ(coo_mat, coo_converted);
}

TEST(BasicTest, BiggerConversion) {
    for(int i = 100; i <= 1500; i += 25) {
        int cols = i;
        int rows = 2*i;
        int nbSamples = i;

        std::vector<Triplet<float>> triplets = sampleTriplets<float>(rows, cols, nbSamples);

        auto S_csr = CSR<float>(rows, cols, triplets);
        auto S_coo = COO<float>(rows, cols, triplets);

        EXPECT_EQ(S_csr,CSR<float>(S_coo));
        EXPECT_EQ(S_coo,COO<float>(S_csr));
    }
}

TEST(BasicTest, GPU_basic)
{
    auto gpu_basic =
        std::shared_ptr<Competitors::GPUBasic<float>>(new Competitors::GPUBasic<float>);

    auto cpu_basic =
        std::shared_ptr<Competitors::CPUBasic<float>>(new Competitors::CPUBasic<float>);

    std::vector<std::vector<float>> A_vals { { 1.0, 2.0 }, { 3.0, 4.0 } };
    Dense<float> A(A_vals);

    std::vector<std::vector<float>> B_vals { { -1.0, 2.0 }, { -3.0, 4.0 } };
    Dense<float> B(B_vals);

    std::vector<Triplet<float>> triplets {
        { 0, 0, 1.0 }, { 0, 1, 1.0 }, { 1, 0, 1.0 }, { 1, 1, 1.0 }
    };
    COO<float> S(2, 2, triplets);
    COO<float> P1(S);
    COO<float> P2(S);

    gpu_basic->init_coo(A, B, S, P1);
    gpu_basic->run_coo(A, B, S, P1);
    gpu_basic->cleanup_coo(A, B, S, P1);

    cpu_basic->run_coo(A, B, S, P2);

    EXPECT_EQ(P1, P2);
}

TEST(BasicTest, GPU_basic2)
{
    auto gpu_basic =
        std::unique_ptr<Competitors::GPUBasic<float>>(new Competitors::GPUBasic<float>);

    auto cpu_basic =
        std::unique_ptr<Competitors::CPUBasic<float>>(new Competitors::CPUBasic<float>);

    std::vector<std::vector<float>> A_vals { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } };
    Dense<float> A(A_vals);

    std::vector<std::vector<float>> B_vals { { -1.0, -3.0, 5.0 }, { 2.0, 4.0, -6.0 } };
    Dense<float> B(B_vals);

    std::vector<Triplet<float>> triplets { { 0, 0, 1.0 }, { 1, 0, 1.0 }, { 1, 1, 1.0 } };
    COO<float> S(2, 2, triplets);
    COO<float> P1(S);
    COO<float> P2(S);

    gpu_basic->init_coo(A, B, S, P1);
    gpu_basic->run_coo(A, B, S, P1);
    gpu_basic->cleanup_coo(A, B, S, P1);

    cpu_basic->run_coo(A, B, S, P2);

    EXPECT_EQ(P1, P2);
}

TEST(BasicTest, GPU_basic3)
{
    auto gpu_basic =
        std::unique_ptr<Competitors::GPUBasic<float>>(new Competitors::GPUBasic<float>);

    auto cpu_basic =
        std::unique_ptr<Competitors::CPUBasic<float>>(new Competitors::CPUBasic<float>);

    std::vector<std::vector<float>> A_vals { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } };
    Dense<float> A(A_vals);

    std::vector<std::vector<float>> B_vals { { -1.0, -3.0, 5.0 } };
    Dense<float> B(B_vals);

    std::vector<Triplet<float>> triplets { { 0, 0, 1.0 }, { 1, 0, -1.0 } };
    COO<float> S(2, 2, triplets);
    COO<float> P1(S);
    COO<float> P2(S);

    gpu_basic->init_coo(A, B, S, P1);
    gpu_basic->run_coo(A, B, S, P1);
    gpu_basic->cleanup_coo(A, B, S, P1);

    cpu_basic->run_coo(A, B, S, P2);

    EXPECT_EQ(P1, P2);
}

TEST(BasicTest, GPU_basic4)
{
    auto gpu_basic =
        std::unique_ptr<Competitors::GPUBasic<float>>(new Competitors::GPUBasic<float>);

    auto cpu_basic =
        std::unique_ptr<Competitors::CPUBasic<float>>(new Competitors::CPUBasic<float>);

    std::vector<std::vector<float>> A_vals { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 } };
    Dense<float> A(A_vals);

    std::vector<std::vector<float>> B_vals { { -1.0, -3.0 }, { 2.0, 2.0 }, { 3.0, 1.0 } };
    Dense<float> B(B_vals);

    std::vector<Triplet<float>> triplets {
        { 0, 0, 1.0 }, { 1, 0, -1.0 }, { 2, 0, 1.0 }, { 2, 2, 1.0 }
    };
    COO<float> S(3, 3, triplets);
    COO<float> P1(S);
    COO<float> P2(S);

    gpu_basic->init_coo(A, B, S, P1);
    gpu_basic->run_coo(A, B, S, P1);
    gpu_basic->cleanup_coo(A, B, S, P1);

    cpu_basic->run_coo(A, B, S, P2);

    EXPECT_EQ(P1, P2);
}

TEST(BasicTest, GPU_advanced)
{
    const int rows = 500;
    const int cols = 500;

    auto gpu_basic =
            std::unique_ptr<Competitors::GPUBasic<float>>(new Competitors::GPUBasic<float>);

    auto cpu_basic =
            std::unique_ptr<Competitors::CPUBasic<float>>(new Competitors::CPUBasic<float>);

    auto dataset =
            std::unique_ptr<SDDMM::RandomWithDensityDataset<float>>(new SDDMM::RandomWithDensityDataset<float>(rows, cols, 32, 0.3));

    auto S = dataset->getS_COO();
    auto A = dataset->getA();
    auto B = dataset->getB();

    COO<float> P1(S);
    COO<float> P2(S);

    gpu_basic->init_coo(A, B, S, P1);
    gpu_basic->run_coo(A, B, S, P1);
    gpu_basic->cleanup_coo(A, B, S, P1);

    cpu_basic->run_coo(A, B, S, P2);

    EXPECT_EQ(P1, P2);
}

TEST(BasicTest, GPU_advanced_dense)
{
    const int rows = 250;
    const int cols = 250;

    auto gpu_basic =
            std::unique_ptr<Competitors::GPUBasic<float>>(new Competitors::GPUBasic<float>);

    auto cpu_basic =
            std::unique_ptr<Competitors::CPUBasic<float>>(new Competitors::CPUBasic<float>);

    auto dataset =
            std::unique_ptr<SDDMM::RandomWithDensityDataset<float>>(new SDDMM::RandomWithDensityDataset<float>(rows, cols, 32, 0.7));

    auto S = dataset->getS_COO();
    auto A = dataset->getA();
    auto B = dataset->getB();

    COO<float> P1(S);
    COO<float> P2(S);

    gpu_basic->init_coo(A, B, S, P1);
    gpu_basic->run_coo(A, B, S, P1);
    gpu_basic->cleanup_coo(A, B, S, P1);

    cpu_basic->run_coo(A, B, S, P2);

    EXPECT_EQ(P1, P2);
}

TEST(BasicTest, GPU_advanced_sparse)
{
    const int rows = 10000;
    const int cols = 10000;

    auto gpu_basic =
            std::unique_ptr<Competitors::GPUBasic<float>>(new Competitors::GPUBasic<float>);

    auto cpu_basic =
            std::unique_ptr<Competitors::CPUBasic<float>>(new Competitors::CPUBasic<float>);

    auto dataset =
            std::unique_ptr<SDDMM::RandomWithDensityDataset<float>>(new SDDMM::RandomWithDensityDataset<float>(rows, cols, 32, 0.001));

    auto S = dataset->getS_COO();
    auto A = dataset->getA();
    auto B = dataset->getB();

    COO<float> P1(S);
    COO<float> P2(S);

    gpu_basic->init_coo(A, B, S, P1);
    gpu_basic->run_coo(A, B, S, P1);
    gpu_basic->cleanup_coo(A, B, S, P1);

    cpu_basic->run_coo(A, B, S, P2);

    EXPECT_EQ(P1, P2);
}

TEST(BasicTest, GPU_test_tiled)
{
    const int rows = 500;
    const int cols = 500;

    auto gpu_basic =
            std::unique_ptr<Competitors::GPUBasic<float>>(new Competitors::GPUBasic<float>);

    auto gpu_tiled =
            std::unique_ptr<Competitors::GPUTiled<float>>(new Competitors::GPUTiled<float>);

    auto dataset =
            std::unique_ptr<SDDMM::RandomWithDensityDataset<float>>(new SDDMM::RandomWithDensityDataset<float>(rows, cols, 32, 0.3));

    auto S_coo = dataset->getS_COO();
    auto S_csr = dataset->getS_CSR();
    auto A = dataset->getA();
    auto B = dataset->getB();

    COO<float> P1(S_coo);
    CSR<float> P2(S_csr);

    gpu_basic->init_coo(A, B, S_coo, P1);
    gpu_basic->run_coo(A, B, S_coo, P1);
    gpu_basic->cleanup_coo(A, B, S_coo, P1);

    gpu_tiled->init_csr(A, B, S_csr, P2);
    gpu_tiled->run_csr(A, B, S_csr, P2);
    gpu_tiled->cleanup_csr(A, B, S_csr, P2);

    EXPECT_EQ(P1, COO<float>(P2));
}

/*TEST(BasicTest, CPU_PyTorch)
{
    auto cpu_basic =
        std::unique_ptr<Competitors::CPUBasic<float>>(new Competitors::CPUBasic<float>);

    auto cpu_pytorch =
        std::unique_ptr<Competitors::CPUPyTorch<float>>(new Competitors::CPUPyTorch<float>);

    std::vector<std::vector<float>> A_vals { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 } };
    Dense<float> A(A_vals);

    std::vector<std::vector<float>> B_vals { { -1.0, -3.0 }, { 2.0, 2.0 }, { 3.0, 1.0 } };
    Dense<float> B(B_vals);

    std::vector<Triplet<float>> triplets {
        { 0, 0, 1.0 }, { 1, 0, -1.0 }, { 2, 0, 2.0 }, { 2, 2, 1.0 }
    };
    CSR<float> S(3, 3, triplets);
    CSR<float> P1(S);
    CSR<float> P2(S);
    cpu_basic->run_csr(A, B, S, P1);
    cpu_pytorch->run_csr(A, B, S, P2);

    EXPECT_EQ(P1, P2);
}

TEST(BasicTest, CPU_PyTorch_advanced)
{
    const int rows = 10000;
    const int cols = 10000;

    auto cpu_pytorch =
            std::unique_ptr<Competitors::CPUPyTorch<float>>(new Competitors::CPUPyTorch<float>);

    auto cpu_basic =
            std::unique_ptr<Competitors::CPUBasic<float>>(new Competitors::CPUBasic<float>);

    auto dataset =
            std::unique_ptr<SDDMM::RandomWithDensityDataset<float>>(new SDDMM::RandomWithDensityDataset<float>(rows, cols, 32, 0.001));

    auto S = dataset->getS_CSR();
    auto A = dataset->getA();
    auto B = dataset->getB();

    CSR<float> P1(S);
    CSR<float> P2(S);

    cpu_pytorch->run_csr(A, B, S, P1);
    cpu_basic->run_csr(A, B, S, P2);

    EXPECT_EQ(P1, P2);
}

TEST(BasicTest, GPU_PyTorch)
{
    auto cpu_basic =
        std::unique_ptr<Competitors::CPUBasic<float>>(new Competitors::CPUBasic<float>);

    auto gpu_pytorch =
        std::unique_ptr<Competitors::GPUPyTorch<float>>(new Competitors::GPUPyTorch<float>);

    std::vector<std::vector<float>> A_vals { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 } };
    Dense<float> A(A_vals);

    std::vector<std::vector<float>> B_vals { { -1.0, -3.0 }, { 2.0, 2.0 }, { 3.0, 1.0 } };
    Dense<float> B(B_vals);

    std::vector<Triplet<float>> triplets {
        { 0, 0, 1.0 }, { 1, 0, -1.0 }, { 2, 0, 2.0 }, { 2, 2, 1.0 }
    };
    CSR<float> S(3, 3, triplets);
    CSR<float> P1(S);
    CSR<float> P2(S);
    cpu_basic->run_csr(A, B, S, P1);
    gpu_pytorch->run_csr(A, B, S, P2);

    EXPECT_EQ(P1, P2);
}

TEST(BasicTest, GPU_PyTorch_advanced)
{
    const int rows = 10000;
    const int cols = 10000;

    auto gpu_pytorch =
            std::unique_ptr<Competitors::GPUPyTorch<float>>(new Competitors::GPUPyTorch<float>);

    auto cpu_basic =
            std::unique_ptr<Competitors::CPUBasic<float>>(new Competitors::CPUBasic<float>);

    auto dataset =
            std::unique_ptr<SDDMM::RandomWithDensityDataset<float>>(new SDDMM::RandomWithDensityDataset<float>(rows, cols, 32, 0.001));

    auto S = dataset->getS_CSR();
    auto A = dataset->getA();
    auto B = dataset->getB();

    CSR<float> P1(S);
    CSR<float> P2(S);

    gpu_pytorch->run_csr(A, B, S, P1);
    cpu_basic->run_csr(A, B, S, P2);

    EXPECT_EQ(P1, P2);
}*/

TEST(BasicTest, dispatcher_more_threads_than_rows){
    std::vector<Triplet<float>> triplets_csr{{0,0,1.0},{1,0,1.0},{0,1,1.0},{1,1,1.0}};
    CSR<float> csr_mat(2,2,triplets_csr);

    int num_threads = 4;
    csr_mat.recomputeDispatcher(num_threads);

    std::vector<int> start_idx = csr_mat.getStartIdx();
    std::vector<int> expected{0,1,2,2,2};

    EXPECT_EQ(start_idx, expected);
}

TEST(BasicTest, dispatcher_equal_num_threads_than_rows){
    std::vector<Triplet<float>> triplets_csr{{0,0,1.0},{1,0,1.0},{0,1,1.0},{1,1,1.0}};
    CSR<float> csr_mat(2,2,triplets_csr);

    int num_threads = 2;
    csr_mat.recomputeDispatcher(num_threads);

    std::vector<int> start_idx = csr_mat.getStartIdx();
    std::vector<int> expected{0,1,2};

    EXPECT_EQ(start_idx, expected);
}

TEST(BasicTest, dispatcher_less_threads_than_rows){
    std::vector<Triplet<float>> triplets_csr{{0,0,1.0},{1,0,1.0},{2,1,1.0},{3,1,1.0}};
    CSR<float> csr_mat(4,2,triplets_csr);

    int num_threads = 2;
    csr_mat.recomputeDispatcher(num_threads);

    std::vector<int> start_idx = csr_mat.getStartIdx();
    std::vector<int> expected{0,2,4};

    EXPECT_EQ(start_idx, expected);
}

TEST(BasicTest, dispatcher_empty_rows){
    std::vector<Triplet<float>> triplets_csr{{1,0,1.0}};
    CSR<float> csr_mat(4,2,triplets_csr);

    int num_threads = 2;
    csr_mat.recomputeDispatcher(num_threads);

    std::vector<int> start_idx = csr_mat.getStartIdx();
    std::vector<int> expected{0,4,4};

    EXPECT_EQ(start_idx, expected);
}

TEST(BasicTest, dispatcher_large_regular_input){
    std::vector<Triplet<float>> triplets_csr;

    int n = 150000;
    for(int i = 0; i < n;i++) {
        triplets_csr.push_back({i,i,4});
    }
    CSR<float> csr_mat(n,n,triplets_csr);

    int num_threads = n/2;
    csr_mat.recomputeDispatcher(num_threads);

    std::vector<int> start_idx = csr_mat.getStartIdx();
    std::vector<int> expected;
    for(int i = 0; i <= num_threads;i++) {
        expected.push_back(2*i);
    }

    EXPECT_EQ(start_idx, expected);
}

TEST(BasicTest, GPU_test_dispatcher){
    const int rows = 500;
    const int cols = 500;

    auto gpu_basic =
            std::unique_ptr<Competitors::GPUBasic<float>>(new Competitors::GPUBasic<float>);

    auto gpu_dispatcher =
            std::unique_ptr<Competitors::GPUThreadDispatcher<float>>(new Competitors::GPUThreadDispatcher<float>);

    auto dataset =
            std::unique_ptr<SDDMM::RandomWithDensityDataset<float>>(new SDDMM::RandomWithDensityDataset<float>(rows, cols, 32, 0.3));

    auto S_coo = dataset->getS_COO();
    auto S_csr = dataset->getS_CSR();
    auto A = dataset->getA();
    auto B = dataset->getB();

    COO<float> P1(S_coo);
    CSR<float> P2(S_csr);

    gpu_basic->init_coo(A, B, S_coo, P1);
    gpu_basic->run_coo(A, B, S_coo, P1);
    gpu_basic->cleanup_coo(A, B, S_coo, P1);

    gpu_dispatcher->init_csr(A, B, S_csr, P2);
    gpu_dispatcher->run_csr(A, B, S_csr, P2);
    gpu_dispatcher->cleanup_csr(A, B, S_csr, P2);

    EXPECT_EQ(P1, COO<float>(P2));
}

TEST(BasicTest, Adaptie_tiling_reorder_s){

    auto gpu_adaptive_tiler =
            std::unique_ptr<Competitors::GPUAdaptiveTiling<float>>(new Competitors::GPUAdaptiveTiling<float>);

    auto dataset =
            std::unique_ptr<SDDMM::RandomWithDensityDataset<float>>(new SDDMM::RandomWithDensityDataset<float>(6, 8, 32, 0.3));

    auto A = dataset->getA();
    auto B = dataset->getB();

    std::vector<Triplet<float>> triplets_csr{
        {0,0,1.0},{0,4,2.0},{0,6,3.0},
        {1,1,4.0},{1,2,5.0},{1,4,5.0},{1,7,6.0},
        {2,1,8.0},{2,4,9.0},{2,6,10.0},{2,7,11.0},
        {3,1,12.0},{3,3,13.0},
        {4,5,14.0},
        {5,1,15.0},{5,5,16.0}
    };
    CSR<float> S_csr(6,8,triplets_csr);
    CSR<float> P1(S_csr);

    int expected_cols[] = {4,6,0,1,4,7,2,1,4,6,7,1,3,5,1,5};
    float expected_vals[] = {2,3,1,4,6,7,5,8,9,10,11,12,13,14,15,16};

    gpu_adaptive_tiler->init_csr(A, B, S_csr, P1);

    EXPECT_EQ(expected_cols, gpu_adaptive_tiler->get_reordered_cols());
    EXPECT_EQ(expected_vals, gpu_adaptive_tiler->get_reordered_vals());
}