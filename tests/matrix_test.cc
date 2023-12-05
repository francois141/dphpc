#include <gtest/gtest.h>
#include "matrices/matrices.h"
#include "competitors/cpu/cpu_basic.hpp"
#include "competitors/cpu/cpu_pytorch.hpp"
#include "competitors/gpu/gpu_basic.hpp"
#include "competitors/gpu/gpu_tiled.hpp"
#include "competitors/gpu/gpu_shared.hpp"
#include "competitors/gpu/gpu_thread_dispatcher.hpp"
#include "competitors/gpu/gpu_tensor.hpp"
#include "competitors/gpu/gpu_pytorch.hpp"
#include "benchmark/dataset.hpp"
#include "utils/random_generator.hpp"
#include "utils/util.hpp"


void init_float_competitors(std::vector<std::shared_ptr<SDDMM::Competitor<float>>>& float_competitors) {
    /* CPU Competitors */
    auto cpu_basic = std::make_shared<Competitors::CPUBasic<float>>();
    float_competitors.push_back(cpu_basic);

    auto cpu_pytorch = std::make_shared<Competitors::CPUPyTorch<float>>();
    float_competitors.push_back(cpu_pytorch);

    /* GPU Competitors */
    auto gpu_basic = std::make_shared<Competitors::GPUBasic<float>>();
    float_competitors.push_back(gpu_basic);

    auto gpu_pytorch = std::make_shared<Competitors::GPUPyTorch<float>>();
    float_competitors.push_back(gpu_pytorch);

    auto gpu_tiled = std::make_shared<Competitors::GPUTiled<float>>();
    float_competitors.push_back(gpu_tiled);

    auto gpu_thread_dispatcher = std::make_shared<Competitors::GPUThreadDispatcher<float>>();
    float_competitors.push_back(gpu_thread_dispatcher);

    auto gpu_shared = std::make_shared<Competitors::GPUShared>();
    float_competitors.push_back(gpu_shared);
}

COO<float> test_all_competitors_coo(Dense<float>& A, Dense<float>& B, COO<float>& S) {
    static std::vector<std::shared_ptr<SDDMM::Competitor<float>>> float_competitors;
    if (float_competitors.size() == 0) init_float_competitors(float_competitors);

    COO<float> P1(S);
    COO<float> P2(S);

    float_competitors[0]->init_coo(A, B, S, P1);
    float_competitors[0]->run_coo(A, B, S, P1);
    float_competitors[0]->cleanup_coo(A, B, S, P1);

    for (const auto& competitor : float_competitors) {
        if (!competitor->coo_supported()) continue;
        competitor->init_coo(A, B, S, P2);
        competitor->run_coo(A, B, S, P2);
        competitor->cleanup_coo(A, B, S, P2);
        EXPECT_EQ(P1, P2);
    }

    return P1;
}

CSR<float> test_all_competitors_csr(Dense<float>& A, Dense<float>& B, CSR<float>& S) {
    static std::vector<std::shared_ptr<SDDMM::Competitor<float>>> float_competitors;
    if (float_competitors.size() == 0) init_float_competitors(float_competitors);

    CSR<float> P1(S);
    CSR<float> P2(S);

    float_competitors[0]->init_csr(A, B, S, P1);
    float_competitors[0]->run_csr(A, B, S, P1);
    float_competitors[0]->cleanup_csr(A, B, S, P1);

    for (const auto& competitor : float_competitors) {
        if (!competitor->csr_supported()) continue;
        competitor->init_csr(A, B, S, P2);
        competitor->run_csr(A, B, S, P2);
        competitor->cleanup_csr(A, B, S, P2);
        EXPECT_EQ(P1, P2);
    }

    return P1;
}

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

TEST(BasicTest, sddmm_coo1)
{
    std::vector<std::vector<float>> A_vals { { 1.0, 2.0 }, { 3.0, 4.0 } };
    Dense<float> A(A_vals);

    std::vector<std::vector<float>> B_vals { { -1.0, 2.0 }, { -3.0, 4.0 } };
    Dense<float> B(B_vals);

    std::vector<Triplet<float>> triplets {
        { 0, 0, 1.0 }, { 0, 1, 1.0 }, { 1, 0, 1.0 }, { 1, 1, 1.0 }
    };
    COO<float> S(2, 2, triplets);
    
    test_all_competitors_coo(A, B, S);
}

TEST(BasicTest, sddmm_coo2)
{
    std::vector<std::vector<float>> A_vals { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } };
    Dense<float> A(A_vals);

    std::vector<std::vector<float>> B_vals { { -1.0, -3.0, 5.0 }, { 2.0, 4.0, -6.0 } };
    Dense<float> B(B_vals);

    std::vector<Triplet<float>> triplets { { 0, 0, 1.0 }, { 1, 0, 1.0 }, { 1, 1, 1.0 } };
    COO<float> S(2, 2, triplets);

    test_all_competitors_coo(A, B, S);
}

TEST(BasicTest, sddmm_coo3)
{
    std::vector<std::vector<float>> A_vals { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } };
    Dense<float> A(A_vals);

    std::vector<std::vector<float>> B_vals { { -1.0, -3.0, 5.0 } };
    Dense<float> B(B_vals);

    std::vector<Triplet<float>> triplets { { 0, 0, 1.0 }, { 1, 0, -1.0 } };
    COO<float> S(2, 2, triplets);

    test_all_competitors_coo(A, B, S);
}

TEST(BasicTest, sddmm_coo4)
{
    std::vector<std::vector<float>> A_vals { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 } };
    Dense<float> A(A_vals);

    std::vector<std::vector<float>> B_vals { { -1.0, -3.0 }, { 2.0, 2.0 }, { 3.0, 1.0 } };
    Dense<float> B(B_vals);

    std::vector<Triplet<float>> triplets {
        { 0, 0, 1.0 }, { 1, 0, -1.0 }, { 2, 0, 1.0 }, { 2, 2, 1.0 }
    };
    COO<float> S(3, 3, triplets);

    test_all_competitors_coo(A, B, S);
}

TEST(BasicTest, sddmm_random_coo)
{
    const int rows = 500;
    const int cols = 500;

    auto dataset =
            std::unique_ptr<SDDMM::RandomWithDensityDataset<float>>(new SDDMM::RandomWithDensityDataset<float>(rows, cols, 32, 0.3));

    auto& S = dataset->getS_COO();
    auto& A = dataset->getA();
    auto& B = dataset->getB();
    
    test_all_competitors_coo(A, B, S);
}

TEST(BasicTest, sddmm_random_dense_coo)
{
    const int rows = 250;
    const int cols = 250;

    auto dataset =
            std::unique_ptr<SDDMM::RandomWithDensityDataset<float>>(new SDDMM::RandomWithDensityDataset<float>(rows, cols, 32, 0.7));

    auto& S = dataset->getS_COO();
    auto& A = dataset->getA();
    auto& B = dataset->getB();

    test_all_competitors_coo(A, B, S);
}

TEST(BasicTest, sddmm_random_sparse_coo)
{
    const int rows = 10000;
    const int cols = 10000;

    auto dataset =
            std::unique_ptr<SDDMM::RandomWithDensityDataset<float>>(new SDDMM::RandomWithDensityDataset<float>(rows, cols, 32, 0.001));

    auto& S = dataset->getS_COO();
    auto& A = dataset->getA();
    auto& B = dataset->getB();

    test_all_competitors_coo(A, B, S);
}

TEST(BasicTest, sddmm_csr_coo)
{
    const int rows = 500;
    const int cols = 500;

    auto dataset =
            std::unique_ptr<SDDMM::RandomWithDensityDataset<float>>(new SDDMM::RandomWithDensityDataset<float>(rows, cols, 32, 0.3));

    auto& S_coo = dataset->getS_COO();
    auto& S_csr = dataset->getS_CSR();
    auto& A = dataset->getA();
    auto& B = dataset->getB();

    auto P1_csr = test_all_competitors_csr(A, B, S_csr);
    auto P1_coo = test_all_competitors_coo(A, B, S_coo);

    EXPECT_EQ(P1_coo, COO<float>(P1_csr));
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
    csr_mat.computeDispatcher(num_threads);

    std::vector<int> start_idx = csr_mat.getStartIdx();
    std::vector<int> expected{0,1,2,2,2};

    EXPECT_EQ(start_idx, expected);
}

TEST(BasicTest, dispatcher_equal_num_threads_than_rows){
    std::vector<Triplet<float>> triplets_csr{{0,0,1.0},{1,0,1.0},{0,1,1.0},{1,1,1.0}};
    CSR<float> csr_mat(2,2,triplets_csr);

    int num_threads = 2;
    csr_mat.computeDispatcher(num_threads);

    std::vector<int> start_idx = csr_mat.getStartIdx();
    std::vector<int> expected{0,1,2};

    EXPECT_EQ(start_idx, expected);
}

TEST(BasicTest, dispatcher_less_threads_than_rows){
    std::vector<Triplet<float>> triplets_csr{{0,0,1.0},{1,0,1.0},{2,1,1.0},{3,1,1.0}};
    CSR<float> csr_mat(4,2,triplets_csr);

    int num_threads = 2;
    csr_mat.computeDispatcher(num_threads);

    std::vector<int> start_idx = csr_mat.getStartIdx();
    std::vector<int> expected{0,2,4};

    EXPECT_EQ(start_idx, expected);
}

TEST(BasicTest, dispatcher_empty_rows){
    std::vector<Triplet<float>> triplets_csr{{1,0,1.0}};
    CSR<float> csr_mat(4,2,triplets_csr);

    int num_threads = 2;
    csr_mat.computeDispatcher(num_threads);

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
    csr_mat.computeDispatcher(num_threads);

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


TEST(BasicTest, test_dispatch_tensor){
    std::vector<Triplet<float>> triplets_csr;

    int n = 64;
    for(int i = 0; i < n;i++) {
        triplets_csr.push_back({i,i,1});
    }

    CSR<float> csr_mat(n,n,triplets_csr);

    int num_threads = n/2;
    csr_mat.computeDispatcherTensorCores(4);

    std::vector<int> start_idx = csr_mat.getStartIdx();
    std::vector<int> expected = {0,16,32,48,64};

    EXPECT_EQ(start_idx, expected);
}

TEST(BasicTest, GPU_test_tensor){
    const int rows = 512;
    const int cols = 512;

    auto gpu_basic =
            std::unique_ptr<Competitors::GPUBasic<float>>(new Competitors::GPUBasic<float>);

    auto gpu_dispatcher =
            std::unique_ptr<Competitors::GPUTensor<float>>(new Competitors::GPUTensor<float>);

    auto dataset =
            std::unique_ptr<SDDMM::RandomWithDensityDataset<float>>(new SDDMM::RandomWithDensityDataset<float>(rows, cols, 64, 0.4));

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