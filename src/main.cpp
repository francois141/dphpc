#include <iostream>
#include <vector>
#include <getopt.h>

#include "benchmark/benchmark.hpp"
#include "benchmark/dataset.hpp"
#include "benchmark/competitor.hpp"

#include "competitors/competitors.h"

#include "matrices/matrices.h"
#include "utils/file_writer.hpp"

static std::vector<std::shared_ptr<SDDMM::Competitor<double>>> double_competitors;

void init_double_competitors() {
    /* CPU Competitors */
    auto cpu_basic = std::make_shared<Competitors::CPUBasic<double>>();
    double_competitors.push_back(cpu_basic);

    auto cpu_pytorch = std::make_shared<Competitors::CPUPyTorch<double>>();
    double_competitors.push_back(cpu_pytorch);

    /* GPU Competitors */
    auto gpu_basic = std::make_shared<Competitors::GPUBasic<double>>();
    double_competitors.push_back(gpu_basic);
}

/* =========================== */
/* Benchmark the dummy dataset */
/* =========================== */
void benchmark_dummy() {
    SDDMM::DummyDataset dataset;
    SDDMM::Benchmark<double> benchmark(dataset, double_competitors, "dummy_measures.csv");

    /* Run the benchmark */
    benchmark.benchmark();
}

/* ========================== */
/* Benchmark the NIPS dataset */
/* ========================== */
void benchmark_NIPS(const std::string& data_folder, const int K) {
    SDDMM::NIPSDataset<double> nips_dataset(data_folder, K);
    SDDMM::Benchmark<double> benchmark(nips_dataset, double_competitors, "nips_measures.csv");

    /* Run the benchmark */
    benchmark.benchmark();
}

/* ================================= */
/* Benchmark the EMail-Enron dataset */
/* ==================================*/
void benchmark_email_enron(const std::string& data_folder, const int K) {
    SDDMM::EMailEnronDataset<double> email_enron_dataset(data_folder, K);
    SDDMM::Benchmark<double> benchmark(email_enron_dataset, double_competitors, "enron-measures.csv");

    /* Run the benchmark */
    benchmark.benchmark();
}

/* ====================================================================================================================== */

struct config_t {
    std::string data_folder;
	int K; // 32 // 128 // 512
};

struct config_t config = { // default values
    std::string("../data/"),
    32
};

static void print_config(void) {
    DEBUG_OUT("----------------------------------------" << std::endl);
    DEBUG_OUT("Program configuration" << std::endl);
    DEBUG_OUT("  Data Folder: " << config.data_folder << std::endl);
    DEBUG_OUT("  K: " << config.K << std::endl);
    DEBUG_OUT("----------------------------------------" << std::endl);
}

static void usage(const char *argv0) {
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -d, --data_folder <folder> (folder to read datasets from)" << std::endl;
    std::cout << "  -k, --K <k> (dimension to generate dense matrices A & B from)" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
}

/* ====================================================================================================================== */

// Runs by default with: ./src/dphpc -data_folder ../data/ -K 32

int main(int argc, char* argv[]) {
    int K = 32;
    static struct option long_options[] = {
        { .name = "data_folder", .has_arg = 1, .val = 'd' },
		{ .name = "K", .has_arg = 1, .val = 'k' },
		{ .name = NULL, .has_arg = 0, .val = '\0' }
	};

    int c;
    while ((c = getopt_long(argc, argv, "d:k:", long_options, NULL)) != -1) {
        if (c == 'd') {
            config.data_folder = std::string(optarg);
        } else if (c == 'k') {
            config.K = std::stoi(optarg);
        } else {
            usage(argv[0]);
            return 1;
        }
	}

    print_config();

    init_double_competitors();

    // Format: Competitor_Name,Dataset_Name,Matrix_Representation,Supported,M,N,K,Execution_Time,Correctness
    FILE_DUMP("competitor,dataset,mat_repr,supported,N,M,K,exec_time,correctness" << std::endl);

    DEBUG_OUT("\n=====================================================\n" << std::endl);
    benchmark_dummy();

    DEBUG_OUT("\n=====================================================\n" << std::endl);

    benchmark_NIPS(config.data_folder, config.K);

    DEBUG_OUT("\n=====================================================\n" << std::endl);

    benchmark_email_enron(config.data_folder, config.K);

    return 0;
}