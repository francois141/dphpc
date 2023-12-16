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

#include <cuda_runtime.h>

#include "competitor.hpp"
#include "dataset.hpp"
#include "utils/util.hpp"
#include "utils/helpers.hpp"

#define header "algorithm; size; time"

namespace SDDMM {

    typedef struct {
        uint64_t init_ns;
        uint64_t comp_ns;
        uint64_t cleanup_ns;

        uint64_t total_ns;
    } timing_result;

    template<typename T>
    class Benchmark {
        public:
                                
            Benchmark(Dataset<T> &dataset, std::vector<std::shared_ptr<Competitor<T>>> &competitors, std::string path, int num_runs)
            : dataset(dataset), competitors(competitors), path(path), num_runs(num_runs) {}

            ~Benchmark() = default;

            Dataset<T> &getDataset() {
                return this->dataset;
            }           

            void benchmark() {

                cudaFree(0); // lazy context establishment to prevent first call to cudaMalloc to perform device initialization (and take substantially longer than other cudaMalloc calls)

                /* Select and run correctness baseline */
                if (!dataset.hasExpected()) { // if the dataset has no inherent correct result, take the first competitor as baseline
                    auto baseline_competitor = competitors[0].get();
                    
                    CSR<T> P_csr(this->getDataset().getS_CSR());
                    P_csr.clearValues();

                    baseline_competitor->init_csr(this->getDataset().getA(), this->getDataset().getB(), this->getDataset().getS_CSR(), P_csr);
                    baseline_competitor->run_csr(this->getDataset().getA(), this->getDataset().getB(), this->getDataset().getS_CSR(), P_csr);
                    baseline_competitor->cleanup_csr(this->getDataset().getA(), this->getDataset().getB(), this->getDataset().getS_CSR(), P_csr);

                    dataset.setExpected_CSR(P_csr);
                }
                
                /* Run the benchmark with all competitors */
                std::for_each(competitors.begin(), competitors.end(), [this](std::shared_ptr<Competitor<T>> competitor_ptr) {
                    auto competitor = competitor_ptr.get();
                    bool csr_correctness = false, coo_correcntess = false;

                    // possible values for thread blocks and num threads per block
                    std::vector<int> num_thread_blocks_v{ 128, 256, 512, 1024, 2048 };
                    std::vector<int> num_threads_per_block_v{ 64, 128, 256, 512, 1024};

                    /* ============================= */
                    /* Sparse matrices in CSR format */
                    /* ============================= */
                    if (competitor->csr_supported()) {
                        DEBUG_OUT("Running competitor " << competitor->name << " (Sparse matrices represented as CSR) " << num_runs << " times" << std::endl);

                        for (int j = 0; j < num_thread_blocks_v.size(); j++){
                            for (int l = 0; l < num_threads_per_block_v.size(); l++){
                                // check best combination
                                int num_threads_per_block = num_threads_per_block_v[l];
                                int num_thread_blocks = num_thread_blocks_v[j];
                                competitor->set_num_threads_per_block(num_threads_per_block);
                                competitor->set_num_thread_blocks(num_thread_blocks);
                            
                                for (int i = 0; i < num_runs; i++){
                                    CSR<T> P_csr(this->getDataset().getS_CSR());
                                    P_csr.clearValues();

                                    // Running competitor
                                    SDDMM::timing_result res = timing(
                                        competitor->is_gpu(),
                                        [&] { competitor->init_csr(this->getDataset().getA(), this->getDataset().getB(), this->getDataset().getS_CSR(), P_csr); },
                                        [&] { competitor->run_csr(this->getDataset().getA(), this->getDataset().getB(), this->getDataset().getS_CSR(), P_csr); },
                                        [&] { competitor->cleanup_csr(this->getDataset().getA(), this->getDataset().getB(), this->getDataset().getS_CSR(), P_csr); }
                                    );
                                    
                                    // Checking correctness if available
                                    if (this->getDataset().hasExpected()) {
                                        csr_correctness = (P_csr == this->getDataset().getExpected_CSR());
                                        if (!csr_correctness) {
                                            //DEBUG_OUT(" - !!! Wrong results calculated compared to CPU-Basic (CSR) !!!" << std::endl);
                                            //FILE_DUMP("[ " << competitor->name << "] !!! Wrong results calculated compared to CPU-Basic (CSR) !!!" << std::endl);
                                        }
                                    }
                                    DEBUG_OUT(" - Execution took " << MILLISECOND(res.comp_ns) << " milliseconds (" << res.comp_ns << "ns)" << std::endl << std::endl);
                                    FILE_DUMP(competitor->name << "," << this->getDataset().getName() << ",CSR,"
                                        << this->getDataset().getS_COO().getRows() << "," << this->getDataset().getS_COO().getCols() << "," << this->getDataset().getA().getCols() << "," << this->getDataset().getS_COO().getValues().size() << ","
                                        << res.total_ns << "," << res.init_ns << "," << res.comp_ns << "," << res.cleanup_ns << "," << csr_correctness << "," << competitor->get_num_thread_blocks() << "," << competitor->get_num_threads_per_block() <<  std::endl
                                    );
                                }
                            }
                        }

                    }

                    /* ============================= */
                    /* Sparse matrices in COO format */
                    /* ============================= */
                    if (competitor->coo_supported()) {
                        DEBUG_OUT("Running competitor " << competitor->name << " (Sparse matrices represented as COO) " << num_runs << " times" << std::endl);
                        
                        for (int j = 0; j < num_thread_blocks_v.size(); j++){
                            for (int l = 0; l < num_threads_per_block_v.size(); l++){
                                // check best combination
                                int num_threads_per_block = num_threads_per_block_v[l];
                                int num_thread_blocks = num_thread_blocks_v[j];
                                competitor->set_num_threads_per_block(num_threads_per_block);
                                competitor->set_num_thread_blocks(num_thread_blocks);
                            
                                for (int i = 0; i < num_runs; i++){
                                    COO<T> P_coo(this->getDataset().getS_COO());
                                    P_coo.clearValues();

                                    // Running competitor
                                    SDDMM::timing_result res = timing(
                                        competitor->is_gpu(),
                                        [&] { competitor->init_coo(this->getDataset().getA(), this->getDataset().getB(), this->getDataset().getS_COO(), P_coo); },
                                        [&] { competitor->run_coo(this->getDataset().getA(), this->getDataset().getB(), this->getDataset().getS_COO(), P_coo); },
                                        [&] { competitor->cleanup_coo(this->getDataset().getA(), this->getDataset().getB(), this->getDataset().getS_COO(), P_coo); }
                                    );

                                    // Checking correctness if available
                                    if (competitor->coo_supported() && this->getDataset().hasExpected()) {
                                        coo_correcntess = (P_coo == this->getDataset().getExpected_COO());
                                        if (!coo_correcntess) {
                                            //DEBUG_OUT(" - !!! Wrong results calculated compared to CPU-Basic (CSR) !!!" << std::endl);
                                            //FILE_DUMP("[ " << competitor->name << "] !!! Wrong results calculated compared to CPU-Basic (CSR) !!!" << std::endl);
                                        }
                                    }
                                    DEBUG_OUT(" - Execution took " << MILLISECOND(res.comp_ns) << " milliseconds - (" << res.comp_ns << "ns)" << std::endl << std::endl);
                                    FILE_DUMP(competitor->name << "," << this->getDataset().getName() << ",COO,"
                                        << this->getDataset().getS_COO().getRows() << "," << this->getDataset().getS_COO().getCols() << "," << this->getDataset().getA().getCols() << "," << this->getDataset().getS_COO().getValues().size() << ","
                                        << res.total_ns << "," << res.init_ns << "," << res.comp_ns << "," << res.cleanup_ns << "," << coo_correcntess << "," << competitor->get_num_thread_blocks() << "," << competitor->get_num_threads_per_block() << std::endl
                                    );
                                }
                            }
                        }
                    }
                });
            }
            
        private:
            SDDMM::Dataset<T> dataset;
            std::vector<std::shared_ptr<Competitor<T>>> competitors;

            std::string path;

            int num_runs;

            SDDMM::timing_result timing(bool is_gpu, std::function<void()> init_fnc, std::function<void()> comp_fnc, std::function<void()> cleanup_fnc) {
                std::chrono::time_point<std::chrono::high_resolution_clock> start, init_checkpoint, comp_checkpoint, end;
                cudaEvent_t gpu_start, gpu_end;
                float gpu_ms = 0;

                cudaEventCreate(&gpu_start);
                cudaEventCreate(&gpu_end);

                start = std::chrono::high_resolution_clock::now();
                init_fnc();

                if (is_gpu) {
                    init_checkpoint = std::chrono::high_resolution_clock::now();
                    cudaEventRecord(gpu_start);
                    comp_fnc();
                    cudaEventRecord(gpu_end);
                    comp_checkpoint = std::chrono::high_resolution_clock::now();
                } else {
                    init_checkpoint = std::chrono::high_resolution_clock::now();
                    comp_fnc();
                    comp_checkpoint = std::chrono::high_resolution_clock::now();
                }
                
                cleanup_fnc();
                end = std::chrono::high_resolution_clock::now();
                
                SDDMM::timing_result result;
                result.init_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(init_checkpoint - start).count();
                result.cleanup_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - comp_checkpoint).count();
                result.total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

                if (is_gpu) {
                    cudaEventElapsedTime(&gpu_ms, gpu_start, gpu_end);
                    result.comp_ns = gpu_ms * 1000000;
                } else {
                    result.comp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(comp_checkpoint - init_checkpoint).count();
                }

                cudaEventDestroy(gpu_start);
                cudaEventDestroy(gpu_end);

                return result;
            }
    };
}
