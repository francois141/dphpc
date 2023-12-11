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

    // auto gpu_tiled = std::make_shared<Competitors::GPUTiled<float>>();
    // float_competitors.push_back(gpu_tiled);

    // auto gpu_thread_dispatcher = std::make_shared<Competitors::GPUThreadDispatcher<float>>();
    // float_competitors.push_back(gpu_thread_dispatcher);

    // auto gpu_tensor = std::make_shared<Competitors::GPUTensor<float>>();
    // float_competitors.push_back(gpu_tensor);

    auto gpu_shared = std::make_shared<Competitors::GPUShared>();
    float_competitors.push_back(gpu_shared);

    auto gpu_convert = std::make_shared<Competitors::GPUConvert>();
    float_competitors.push_back(gpu_convert);
}

void benchmark_dummy() {
    SDDMM::DummyDataset dataset;
    SDDMM::Benchmark<float> benchmark(dataset, float_competitors, "dummy_measures.csv");

    /* Run the benchmark */
    benchmark.benchmark();
}

void benchmark_email_enron(const std::string& data_folder, const int K) {
    SDDMM::EMailEnronDataset<float> dataset(data_folder, K);
    SDDMM::Benchmark<float> benchmark(dataset, float_competitors, "enron-measures.csv");

    /* Run the benchmark */
    benchmark.benchmark();
}

void benchmark_ND12K(const std::string& data_folder, const int K) {
    SDDMM::ND12KDataset<float> dataset(data_folder, K);
    SDDMM::Benchmark<float> benchmark(dataset, float_competitors, "nd12k-measures.csv");

    /* Run the benchmark */
    benchmark.benchmark();
}

void benchmark_human_gene2(const std::string& data_folder, const int K) {
    SDDMM::HumanGene2Dataset<float> dataset(data_folder, K);
    SDDMM::Benchmark<float> benchmark(dataset, float_competitors, "human_gene2-measures.csv");

    /* Run the benchmark */
    benchmark.benchmark();
}

void benchmark_warmup(const int K) {
    SDDMM::RandomWithDensityDataset<float> dataset(4000, 4000, K, 0.1); // 40k x 40k with 0.01/0.05
    SDDMM::Benchmark<float> benchmark(dataset, float_competitors, "random-matrix-measures.csv");

    /* Run the benchmark */
    benchmark.benchmark();
}

void benchmark_random(const int K) {
    SDDMM::RandomWithDensityDataset<float> dataset(4000, 4000, K, 0.1); // 40k x 40k with 0.01/0.05
    SDDMM::Benchmark<float> benchmark(dataset, float_competitors, "random-matrix-measures.csv");

    /* Run the benchmark */
    benchmark.benchmark();
}

void benchmark_latin(const int K) {
    SDDMM::LatinHypercubeDataset<float> dataset(4000, 4000, K); // 40k x 40k with 0.01/0.05
    SDDMM::Benchmark<float> benchmark(dataset, float_competitors, "latin-matrix-measures.csv");

    /* Run the benchmark */
    benchmark.benchmark();
}

void benchmark_cage14(const std::string& data_folder, const int K) {
    SDDMM::Cage14Dataset<float> dataset(data_folder, K);
    SDDMM::Benchmark<float> benchmark(dataset, float_competitors, "cage14-measures.csv");

    /* Run the benchmark */
    benchmark.benchmark();
}

void benchmark_boeing(const std::string& data_folder, const int K) {
    SDDMM::BoeingDataset<float> dataset(data_folder, K);
    SDDMM::Benchmark<float> benchmark(dataset, float_competitors, "boeing-measures.csv");

    /* Run the benchmark */
    benchmark.benchmark();
}

void benchmark_boeing_diagonal(const std::string& data_folder, const int K) {
    SDDMM::BoeingDiagonalDataset<float> dataset(data_folder, K);
    SDDMM::Benchmark<float> benchmark(dataset, float_competitors, "boeing-diagonal-measures.csv");

    /* Run the benchmark */
    benchmark.benchmark();
}


void benchmark_stiffness(const std::string& data_folder, const int K) {
    SDDMM::StiffnessDataset<float> dataset(data_folder, K);
    SDDMM::Benchmark<float> benchmark(dataset, float_competitors, "stiffness-measures.csv");

    /* Run the benchmark */
    benchmark.benchmark();
}

void benchmark_semi_conductor(const std::string& data_folder, const int K) {
    SDDMM::SemiConductorDataset<float> dataset(data_folder, K);
    SDDMM::Benchmark<float> benchmark(dataset, float_competitors, "semi-conductor-measures.csv");

    /* Run the benchmark */
    benchmark.benchmark();
}

void benchmark_vlsi(const std::string& data_folder, const int K) {
    SDDMM::VLSIDataset<float> dataset(data_folder, K);
    SDDMM::Benchmark<float> benchmark(dataset, float_competitors, "vlsi-measures.csv");

    /* Run the benchmark */
    benchmark.benchmark();
}

void benchmark_stack_overflow(const std::string& data_folder, const int K) {
    SDDMM::StackOverflowDataset<float> dataset(data_folder, K);
    SDDMM::Benchmark<float> benchmark(dataset, float_competitors, "stack-overflow-measures.csv");

    /* Run the benchmark */
    benchmark.benchmark();
}

void benchmark_chip(const std::string& data_folder, const int K) {
    SDDMM::ChipDataset<float> dataset(data_folder, K);
    SDDMM::Benchmark<float> benchmark(dataset, float_competitors, "chip-measures.csv");

    /* Run the benchmark */
    benchmark.benchmark();
}

void benchmark_combinatorics(const std::string& data_folder, const int K) {
    SDDMM::CombinatoricsDataset<float> dataset(data_folder, K);
    SDDMM::Benchmark<float> benchmark(dataset, float_competitors, "combinatorics-measures.csv");

    /* Run the benchmark */
    benchmark.benchmark();
}

void benchmark_mechanics(const std::string& data_folder, const int K) {
    SDDMM::MechanicsDataset<float> dataset(data_folder, K);
    SDDMM::Benchmark<float> benchmark(dataset, float_competitors, "mechanics-measures.csv");

    /* Run the benchmark */
    benchmark.benchmark();
}

void benchmark_mouse(const std::string& data_folder, const int K) {
    SDDMM::MouseGeneDataset<float> dataset(data_folder, K);
    SDDMM::Benchmark<float> benchmark(dataset, float_competitors, "mouse-gene-measures.csv");

    /* Run the benchmark */
    benchmark.benchmark();
}

void benchmark_platform(const std::string& data_folder, const int K) {
    SDDMM::PlatformDataset<float> dataset(data_folder, K);
    SDDMM::Benchmark<float> benchmark(dataset, float_competitors, "platform-measures.csv");

    /* Run the benchmark */
    benchmark.benchmark();
}

void benchmark_power(const std::string& data_folder, const int K) {
    SDDMM::PowerDataset<float> dataset(data_folder, K);
    SDDMM::Benchmark<float> benchmark(dataset, float_competitors, "power-measures.csv");

    /* Run the benchmark */
    benchmark.benchmark();
}

void benchmark_stress(const std::string& data_folder, const int K) {
    SDDMM::StressDataset<float> dataset(data_folder, K);
    SDDMM::Benchmark<float> benchmark(dataset, float_competitors, "stress-measures.csv");

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

    // CSV Header Format: Competitor_Name,Dataset_Name,Matrix_Representation,M,N,K,Non_Zeros,Total_Execution_Time,Initialization_Time,Computation_Time,Cleanup_Time,Correctness
    if (!config.no_csv_header) {
        FILE_DUMP("competitor,dataset,mat_repr,M,N,K,NZ,total_ns,init_ns,comp_ns,cleanup_ns,correctness" << std::endl);
    }

    // Warmup dataset

    DEBUG_OUT("\n=====================================================\n" << std::endl);
    benchmark_warmup(config.K);

    // Artificial datasets

    DEBUG_OUT("\n=====================================================\n" << std::endl);
    benchmark_random(config.K);

    DEBUG_OUT("\n=====================================================\n" << std::endl);
    benchmark_latin(config.K);

    // Dense datasets

    DEBUG_OUT("\n=====================================================\n" << std::endl);
    benchmark_human_gene2(config.data_folder, config.K);

    DEBUG_OUT("\n=====================================================\n" << std::endl);
    benchmark_ND12K(config.data_folder, config.K);

    DEBUG_OUT("\n=====================================================\n" << std::endl);
    benchmark_platform(config.data_folder, config.K);

    DEBUG_OUT("\n=====================================================\n" << std::endl);
    benchmark_mechanics(config.data_folder, config.K);

    DEBUG_OUT("\n=====================================================\n" << std::endl);
    benchmark_power(config.data_folder, config.K);

    DEBUG_OUT("\n=====================================================\n" << std::endl);
    benchmark_combinatorics(config.data_folder, config.K);

    DEBUG_OUT("\n=====================================================\n" << std::endl);
    benchmark_stress(config.data_folder, config.K);

    DEBUG_OUT("\n=====================================================\n" << std::endl);
    benchmark_mouse(config.data_folder, config.K);

    // Sparse datasets

    DEBUG_OUT("\n=====================================================\n" << std::endl);
    benchmark_email_enron(config.data_folder, config.K);

    DEBUG_OUT("\n=====================================================\n" << std::endl);
    benchmark_boeing(config.data_folder,config.K);

    DEBUG_OUT("\n=====================================================\n" << std::endl);
    benchmark_boeing_diagonal(config.data_folder,config.K);

    DEBUG_OUT("\n=====================================================\n" << std::endl);
    benchmark_stiffness(config.data_folder,config.K);

    DEBUG_OUT("\n=====================================================\n" << std::endl);
    benchmark_semi_conductor(config.data_folder,config.K);

    DEBUG_OUT("\n=====================================================\n" << std::endl);
    benchmark_vlsi(config.data_folder,config.K);

    DEBUG_OUT("\n=====================================================\n" << std::endl);
    benchmark_stack_overflow(config.data_folder,config.K);

    DEBUG_OUT("\n=====================================================\n" << std::endl);
    benchmark_chip(config.data_folder, config.K);

    return 0;
}