#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>
#include <functional>
#include <chrono>
#include <cmath>

#include "competitor.hpp"
#include "dataset.hpp"
#include "utils/file_writer.hpp"
#include "utils/helpers.hpp"

#define header "algorithm; size; time"

namespace SDDMM {

    template<typename T>
    class Benchmark {
        public:
                                
            Benchmark(Dataset<T> &dataset, std::string path)
            : dataset(dataset), path(path)
            {
                output = std::make_unique<SDDMM::CSVWriter>(path, header);
            }

            ~Benchmark() = default;

            Dataset<T> &getDataset() {
                return this->dataset;
            }
            
            void add_competitor(std::shared_ptr<Competitor<T>> competitor) {
                competitors.push_back(competitor);
            }

            void clear_competitors() {
                competitors.clear();
            }
            

            void benchmark() {

                /* Select and run correctness baseline */
                if (!dataset.hasExpected()) { // if the dataset has no inherent correct result, take the first competitor as baseline
                    auto baseline_competitor = competitors[0].get();
                    
                    CSR<T> P_csr(this->getDataset().getS_CSR());
                    P_csr.clearValues();
                    baseline_competitor->run_csr(this->getDataset().getA(), this->getDataset().getB(), this->getDataset().getS_CSR(), P_csr);

                    dataset.setExpected_CSR(P_csr);
                }
                
                /* Run the benchmark with all competitors */
                std::for_each(competitors.begin(), competitors.end(), [this](std::shared_ptr<Competitor<T>> competitor_ptr) {
                    auto competitor = competitor_ptr.get();
                    uint64_t ns;

                    /* ============================= */
                    /* Sparse matrices in CSR format */
                    /* ============================= */
                    DEBUG_OUT("Running competitor " << competitor->name << " (Sparse matrices represented as CSR)" << std::endl);

                    CSR<T> P_csr(this->getDataset().getS_CSR());
                    P_csr.clearValues();

                    // Running competitor
                    ns = timing([&] { // TODO: Cold/Warm Cache? CPU Calibration?
                        competitor->run_csr(this->getDataset().getA(), this->getDataset().getB(), this->getDataset().getS_CSR(), P_csr);
                    });

                    // Checking correctness if available
                    if (competitor->csr_supported() && this->getDataset().hasExpected()) {
                        assert(P_csr == this->getDataset().getExpected_CSR());
                        DEBUG_OUT(" - Correction of results asserted." << std::endl);
                    }
                    DEBUG_OUT(" - Execution took " << SECOND(ns) << " seconds (" << ns << "ns)" << std::endl << std::endl);

                    // Format: name,dataset,CSR,M,N,K,time
                    FILE_DUMP(competitor->name << "," << this->getDataset().getName() << ",CSR," << this->getDataset().getS_COO().getRows() << "," << this->getDataset().getS_COO().getCols() << "," << this->getDataset().getA().getCols() << "," << ns << std::endl);

                    /* ============================= */
                    /* Sparse matrices in COO format */
                    /* ============================= */
                    DEBUG_OUT("Running competitor " << competitor->name << " (Sparse matrices represented as COO)" << std::endl);
                    
                    COO<T> P_coo(this->getDataset().getS_COO());
                    P_coo.clearValues();

                    ns = timing([&] { // TODO: Cold/Warm Cache? CPU Calibration?
                        competitor->run_coo(this->getDataset().getA(), this->getDataset().getB(), this->getDataset().getS_COO(), P_coo);
                    });

                    // Checking correctness if available
                     if (competitor->coo_supported() && this->getDataset().hasExpected()) {
                        assert(P_coo == this->getDataset().getExpected_COO());
                        DEBUG_OUT(" - Correction of results asserted." << std::endl);
                    }
                    DEBUG_OUT(" - Execution took " << SECOND(ns) << " seconds (" << ns << "ns)" << std::endl << std::endl);

                    // TODO: Make size dynamic
                    this->output->writeLine(competitor->name, "10", std::to_string(MICROSECOND(ns)));

                    // Format: name,COO,M,N,K,time
                    FILE_DUMP(competitor->name << "," << this->getDataset().getName() << ",COO," << this->getDataset().getS_COO().getRows() << "," << this->getDataset().getS_COO().getCols() << "," << this->getDataset().getA().getCols() << "," << ns << std::endl);
                   
                });
            }
            
        private:
            std::vector<std::shared_ptr<Competitor<T>>> competitors;
            SDDMM::Dataset<T> &dataset;
            std::string path;
            std::unique_ptr<SDDMM::Output> output;

            uint64_t timing(std::function<void()> fn) {
                const auto start = std::chrono::high_resolution_clock::now();
                fn();
                const auto end = std::chrono::high_resolution_clock::now();
                return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            }
    };
}
