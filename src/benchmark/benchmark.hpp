#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>
#include <functional>
#include <chrono>
#include <cmath>

#include "competitor.hpp"
#include "dataset.hpp"

namespace SDDMM {

    template<typename T>
    class Benchmark {
        public:
                                
            Benchmark(Dataset<T> &dataset)
            : dataset(dataset)
            {}

            ~Benchmark() 
            {}

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
                std::for_each(competitors.begin(), competitors.end(), [this](std::shared_ptr<Competitor<T>> competitor_ptr) {
                    auto competitor = competitor_ptr.get();
                    uint64_t ns;

                    /* Sparse matrices in CSR format */
                    std::cout << "Running competitor " << competitor->name << " (Sparse matrices represented as CSR)" << std::endl;

                    CSR<T> P_csr(this->getDataset().getS_CSR());
                    P_csr.clearValues();

                    ns = timing([&] { // TODO: Cold/Warm Cache? CPU Calibration?
                        competitor->run_csr(this->getDataset().getA(), this->getDataset().getB(), this->getDataset().getS_CSR(), P_csr);
                    });
                    std::cout << "Execution took " << ((double) ns / 1e9) << " seconds (" << ns << "ns)" << std::endl << std::endl;

                    if (competitor->csr_supported() && this->getDataset().hasExpected_CSR()) {
                        assert(P_csr == this->getDataset().getExpected_CSR());
                    }

                    /* Sparse matrices in COO format */
                    std::cout << "Running competitor " << competitor->name << " (Sparse matrices represented as COO)" << std::endl;
                    
                    COO<T> P_coo(this->getDataset().getS_COO());
                    P_coo.clearValues();

                    ns = timing([&] { // TODO: Cold/Warm Cache? CPU Calibration?
                        competitor->run_coo(this->getDataset().getA(), this->getDataset().getB(), this->getDataset().getS_COO(), P_coo);
                    });
                    std::cout << "Execution took " << ((double) ns / 1e9) << " seconds (" << ns << "ns)" << std::endl << std::endl;

                    if (competitor->coo_supported() && this->getDataset().hasExpected_COO()) {
                        assert(P_coo == this->getDataset().getExpected_COO());
                    }
                });
            }
            
        private:
            std::vector<std::shared_ptr<Competitor<T>>> competitors;
            SDDMM::Dataset<T> &dataset;

            
            uint64_t timing(std::function<void()> fn) {
                const auto start = std::chrono::high_resolution_clock::now();
                fn();
                const auto end = std::chrono::high_resolution_clock::now();
                return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            }
    };
}
