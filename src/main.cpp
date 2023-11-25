#include <iostream>
#include <vector>

#ifndef _WIN32
#include <getopt.h>
#endif

#include "benchmark/benchmark.hpp"
#include "benchmark/dataset.hpp"
#include "benchmark/competitor.hpp"

#include "competitors/competitors.h"

#include "matrices/matrices.h"

static std::vector<std::shared_ptr<SDDMM::Competitor<float>>> float_competitors;

void init_float_competitors() {
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
    
    auto gpu_shared = std::make_shared< Competitors::GPUShared>();
    float_competitors.push_back(gpu_shared);
}

/* =========================== */
/* Benchmark the dummy dataset */
/* =========================== */
void benchmark_dummy() {
    SDDMM::DummyDataset dataset;
    SDDMM::Benchmark<float> benchmark(dataset, float_competitors, "dummy_measures.csv");

    /* Run the benchmark */
    benchmark.benchmark();
}

/* ========================== */
/* Benchmark the NIPS dataset */
/* ========================== */
void benchmark_NIPS(const std::string& data_folder, const int K) {
    SDDMM::NIPSDataset<float> nips_dataset(data_folder, K);
    SDDMM::Benchmark<float> benchmark(nips_dataset, float_competitors, "nips_measures.csv");

    /* Run the benchmark */
    benchmark.benchmark();
}

/* ================================= */
/* Benchmark the EMail-Enron dataset */
/* ==================================*/
void benchmark_email_enron(const std::string& data_folder, const int K) {
    SDDMM::EMailEnronDataset<float> email_enron_dataset(data_folder, K);
    SDDMM::Benchmark<float> benchmark(email_enron_dataset, float_competitors, "enron-measures.csv");

    /* Run the benchmark */
    benchmark.benchmark();
}

/* ================================= */
/* Benchmark the ND12K dataset */
/* ==================================*/
void benchmark_ND12K(const std::string& data_folder, const int K) {
    SDDMM::ND12KDataset<float> nd12k_dataset(data_folder, K);
    SDDMM::Benchmark<float> benchmark(nd12k_dataset, float_competitors, "nd12k-measures.csv");

    /* Run the benchmark */
    benchmark.benchmark();
}

/* ================================= */
/* Benchmark the Human Gene 2 dataset */
/* ==================================*/
void benchmark_human_gene2(const std::string& data_folder, const int K) {
    SDDMM::HumanGene2Dataset<float> human_gene2(data_folder, K);
    SDDMM::Benchmark<float> benchmark(human_gene2, float_competitors, "human_gene2-measures.csv");

    /* Run the benchmark */
    benchmark.benchmark();
}

/* ================================= */
/* Benchmark the Random dataset */
/* ==================================*/
void benchmark_random(const int K) {
    SDDMM::RandomWithDensityDataset<float> random_matrix_dataset(4000, 4000, K, 0.1); // 40k x 40k with 0.01/0.05
    SDDMM::Benchmark<float> benchmark(random_matrix_dataset, float_competitors, "random-matrix-measures.csv");

    /* Run the benchmark */
    benchmark.benchmark();
}

/* ================================= */
/* Benchmark the Cage14 dataset */
/* ==================================*/
void benchmark_cage14(const std::string& data_folder, const int K) {
    SDDMM::Cage14Dataset<float> cage14_dataset(data_folder, K);
    SDDMM::Benchmark<float> benchmark(cage14_dataset, float_competitors, "cage14-measures.csv");

    /* Run the benchmark */
    benchmark.benchmark();
}

/* ====================================================================================================================== */

struct Config {
    std::string data_folder;
	int K; // 32 // 128 // 512
    bool no_csv_header;
};

static void print_config(Config config) {
    DEBUG_OUT("----------------------------------------" << std::endl);
    DEBUG_OUT("Program configuration" << std::endl);
    DEBUG_OUT("  Data Folder: " << config.data_folder << std::endl);
    DEBUG_OUT("  K: " << config.K << std::endl);
    DEBUG_OUT("  No CSV header: " << config.no_csv_header << std::endl);
    DEBUG_OUT("----------------------------------------" << std::endl);
}

static void usage() {
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -d, --data_folder <folder> (folder to read datasets from)" << std::endl;
    std::cout << "  -k, --K <k> (dimension to generate dense matrices A & B from)" << std::endl;
    std::cout << "  -h, --no_csv_header (If you pass this option, no CSV header is printed)" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
}

/* ====================================================================================================================== */

// Runs by default with: ./src/dphpc -data_folder ../data/ -K 32

int main(int argc, char* argv[]) {
    // default values config
    Config config = {
        std::string("../data/"),
        32,
        false
    };

    #ifndef _WIN32
    static struct option long_options[] = {
        { .name = "data_folder", .has_arg = 1, .val = 'd' },
		{ .name = "K", .has_arg = 1, .val = 'k' },
        { .name = "no_csv_header", .has_arg = 0, .val = 'h' },
		{ .name = NULL, .has_arg = 0, .val = '\0' }
	};

    int c;
    while ((c = getopt_long(argc, argv, "d:k:", long_options, NULL)) != -1) {
        if (c == 'd') {
            config.data_folder = std::string(optarg);
        } else if (c == 'k') {
            config.K = std::stoi(optarg);
        } else if (c == 'h') {
            config.no_csv_header = true;
        } else {
            usage();
            return 1;
        }
	}
    #endif

    print_config(config);

    init_float_competitors();

    // CSV Header Format: Competitor_Name,Dataset_Name,Matrix_Representation,M,N,K,Total_Execution_Time,Initialization_Time,Computation_Time,Cleanup_Time,Correctness
    if (!config.no_csv_header) {
        FILE_DUMP("competitor,dataset,mat_repr,M,N,K,total_ns,init_ns,comp_ns,cleanup_ns,correctness" << std::endl);
    }

    DEBUG_OUT("\n=====================================================\n" << std::endl);
    benchmark_NIPS(config.data_folder, config.K);

    DEBUG_OUT("\n=====================================================\n" << std::endl);
    benchmark_email_enron(config.data_folder, config.K);

    DEBUG_OUT("\n=====================================================\n" << std::endl);
    benchmark_ND12K(config.data_folder, config.K);

    DEBUG_OUT("\n=====================================================\n" << std::endl);
    benchmark_human_gene2(config.data_folder, config.K);

    DEBUG_OUT("\n=====================================================\n" << std::endl);
    benchmark_random(config.K);

    // DEBUG_OUT("\n=====================================================\n" << std::endl);
    // benchmark_cage14(config.data_folder, config.K);

    return 0;
}