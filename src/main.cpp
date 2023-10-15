#include <iostream>
#include <vector>

#include "benchmark/benchmark.hpp"
#include "benchmark/dataset.hpp"
#include "competitors/competitors.h"

#include "matrices/matrices.h"

/* =========================== */
/* Benchmark the dummy dataset */
/* =========================== */
void benchmark_dummy() {
    SDDMM::DummyDataset dataset;
    SDDMM::Benchmark<int> benchmark(dataset);

    /* CPU Competitors */
    auto cpu_basic = std::shared_ptr<Competitors::CPUBasic<int>>(new Competitors::CPUBasic<int>);
    benchmark.add_competitor(cpu_basic);

    auto cpu_pytorch = std::shared_ptr<Competitors::CPUPyTorch<int>>(new Competitors::CPUPyTorch<int>);
    benchmark.add_competitor(cpu_pytorch);

    /* GPU Competitors */
    auto gpu_basic = std::shared_ptr<Competitors::GPUBasic<int>>(new Competitors::GPUBasic<int>);
    benchmark.add_competitor(gpu_basic);

    /* Run the benchmark */
    benchmark.benchmark();
}

/* ========================== */
/* Benchmark the NIPS dataset */
/* ========================== */
void benchmark_NIPS(const std::string& data_folder, const int K) {
    SDDMM::NIPSDataset<double> nips_dataset(data_folder, K);
    SDDMM::Benchmark<double> benchmark(nips_dataset);

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
}

/* ================================= */
/* Benchmark the EMail-Enron dataset */
/* ==================================*/
void benchmark_email_enron(const std::string& data_folder, const int K) {
    SDDMM::EMailEnronDataset<double> email_enron_dataset(data_folder, K);
    SDDMM::Benchmark<double> benchmark(email_enron_dataset);

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
}

int main(int argc, char* argv[]) {

    const int K = 32; // 32 // 128 // 512 // TODO: read from args

    const std::string data_folder = "../data/"; // TODO: read from args

    // DEBUG_OUT("\n=====================================================\n" << std::endl);
    // benchmark_dummy();

    DEBUG_OUT("\n=====================================================\n" << std::endl);

    benchmark_NIPS(data_folder, K);

    DEBUG_OUT("\n=====================================================\n" << std::endl);

    benchmark_email_enron(data_folder, K);    

    return 0;
}