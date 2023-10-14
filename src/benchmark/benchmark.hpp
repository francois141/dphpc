#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>

#include "competitor.hpp"
#include "dataset.hpp"

namespace SDDMM {

    template<typename T>
    class Benchmark {
        private:
            std::vector<std::shared_ptr<Competitor<T>>> competitors;
            SDDMM::Dataset<T> &dataset;

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

                    /* Sparse matrices in CSR format */
                    std::cout << "Running competitor " << competitor->name << " (Sparse matrices represented as CSR)" << std::endl;

                    CSR<T> P_csr(this->dataset.getS_CSR());
                    P_csr.clearValues();

                    // TODO: Start timing (Cold/Warm Cache? CPU Calibration?)
                    competitor->run_csr(this->dataset.getA(), this->dataset.getB(), this->dataset.getS_CSR(), P_csr);
                    // TODO: End timing

                    // TODO: Check correctness

                    /* Sparse matrices in COO format */
                    std::cout << "Running competitor " << competitor->name << " (Sparse matrices represented as COO)" << std::endl;
                    
                    COO<T> P_coo(this->dataset.getS_COO());
                    P_coo.clearValues();

                    // TODO: Start timing (Cold/Warm Cache? CPU Calibration?)
                    competitor->run_coo(this->dataset.getA(), this->dataset.getB(), this->dataset.getS_COO(), P_coo);
                    // TODO: End timing

                    // TODO: Check correctness
                });
            }
            
    };
}
