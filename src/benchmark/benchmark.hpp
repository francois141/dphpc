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
#include "utils/util.hpp"
#include "utils/helpers.hpp"

#define header "algorithm; size; time"

namespace SDDMM {

    template<typename T>
    class Benchmark {
        public:
                                
            Benchmark(Dataset<T> &dataset, std::vector<std::shared_ptr<Competitor<T>>> &competitors, std::string path)
            : dataset(dataset), competitors(competitors), path(path) {}

            ~Benchmark() = default;

            Dataset<T> &getDataset() {
                return this->dataset;
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
                    uint64_t ns = 0;
                    bool csr_correctness = false, coo_correcntess = false;

                    /* ============================= */
                    /* Sparse matrices in CSR format */
                    /* ============================= */
                    if (competitor->csr_supported()) {
                        DEBUG_OUT("Running competitor " << competitor->name << " (Sparse matrices represented as CSR)" << std::endl);

                        CSR<T> P_csr(this->getDataset().getS_CSR());
                        P_csr.clearValues();

                        // Running competitor
                        ns = timing([&] { // TODO: Cold/Warm Cache? CPU Calibration?
                            competitor->run_csr(this->getDataset().getA(), this->getDataset().getB(), this->getDataset().getS_CSR(), P_csr);
                        });

                        // Checking correctness if available
                        if (this->getDataset().hasExpected()) {
                            csr_correctness = (P_csr == this->getDataset().getExpected_CSR());
                            if (csr_correctness) { DEBUG_OUT(" - Calculated correct results." << std::endl); }
                            else { DEBUG_OUT(" - !!! Wrong results calculated compared to CPU-Basic (CSR) !!!" << std::endl); }
                        }
                        DEBUG_OUT(" - Execution took " << SECOND(ns) << " seconds (" << ns << "ns)" << std::endl << std::endl);
                        FILE_DUMP(competitor->name << "," << this->getDataset().getName() << ",CSR," << this->getDataset().getS_COO().getRows() << "," << this->getDataset().getS_COO().getCols() << "," << this->getDataset().getA().getCols() << "," << ns << "," << csr_correctness << std::endl);
                    } else {
                        DEBUG_OUT("Skipping competitor " << competitor->name << " (does not support CSR)" << std::endl << std::endl);
                    }

                    /* ============================= */
                    /* Sparse matrices in COO format */
                    /* ============================= */
                    if (competitor->coo_supported()) {
                        DEBUG_OUT("Running competitor " << competitor->name << " (Sparse matrices represented as COO)" << std::endl);
                        
                        COO<T> P_coo(this->getDataset().getS_COO());
                        P_coo.clearValues();

                        ns = timing([&] { // TODO: Cold/Warm Cache? CPU Calibration?
                            competitor->run_coo(this->getDataset().getA(), this->getDataset().getB(), this->getDataset().getS_COO(), P_coo);
                        });

                        // Checking correctness if available
                        if (competitor->coo_supported() && this->getDataset().hasExpected()) {
                            coo_correcntess = (P_coo == this->getDataset().getExpected_COO());
                            if (coo_correcntess) { DEBUG_OUT(" - Calculated correct results." << std::endl); }
                            else { DEBUG_OUT(" - !!! Wrong results calculated compared to CPU-Basic (CSR) !!!" << std::endl); }
                        }
                        DEBUG_OUT(" - Execution took " << SECOND(ns) << " seconds (" << ns << "ns)" << std::endl << std::endl);
                        FILE_DUMP(competitor->name << "," << this->getDataset().getName() << ",COO," << this->getDataset().getS_COO().getRows() << "," << this->getDataset().getS_COO().getCols() << "," << this->getDataset().getA().getCols() << "," << ns << "," << coo_correcntess << std::endl);
                    } else {
                        DEBUG_OUT("Skipping competitor " << competitor->name << " (does not support COO)" << std::endl << std::endl);
                    }
                });
            }
            
        private:
            SDDMM::Dataset<T> dataset;
            std::vector<std::shared_ptr<Competitor<T>>> competitors;

            std::string path;

            uint64_t timing(std::function<void()> fn) {
                const auto start = std::chrono::high_resolution_clock::now();
                fn();
                const auto end = std::chrono::high_resolution_clock::now();
                return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            }
    };
}
