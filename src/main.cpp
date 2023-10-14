#include <iostream>
#include <vector>

#include "benchmark/benchmark.hpp"
#include "competitors/competitors.h"

#include "matrices/matrices.h"

#define DATASET_PATH "data/dataset_1/"

int main(int argc, char* argv[]) {

    std::vector<std::vector<double>> matrixA(2, std::vector<double>(2,1));
    Dense<double> A(matrixA); 

    std::vector<std::vector<double>> matrixB(2, std::vector<double>(2,1));
    Dense<double> B(matrixB); 

    std::vector<Triplet<double>> S_triplets{{0,0,1}, {0,1,1}, {1,0,1}, {1,1,1}};
    COO<double> S(2, 2, S_triplets);

    std::vector<Triplet<double>> P_triplets{{0,0,2}, {0,1,2}, {1,0,2}, {1,1,2}};
    COO<double> P(2, 2, P_triplets); // expected result

    SDDMM::Dataset<double> dataset(A, B, S, P);
    SDDMM::Benchmark<double> benchmark(dataset);

    /* CPU Competitors */
    auto cpu_basic = std::shared_ptr<Competitors::CPUBasic<double>>(new Competitors::CPUBasic<double>);
    benchmark.add_competitor(cpu_basic);

    auto cpu_pytorch = std::shared_ptr<Competitors::CPUPyTorch<double>>(new Competitors::CPUPyTorch<double>);
    benchmark.add_competitor(cpu_pytorch);

    /* GPU Competitors */
    auto gpu_basic = std::shared_ptr<Competitors::GPUBasic<double>>(new Competitors::GPUBasic<double>);
    benchmark.add_competitor(gpu_basic);

    /* Run the benchmark */
    benchmark.benchmark();

    return 0;
}