#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include "matrices/matrices.h"

namespace SDDMM {

    template<typename T>
    class Competitor {
        public:
            const std::string name;
        
            Competitor(const std::string& name)
            : name(name), num_threads_per_block(1), num_thread_blocks(1)
            {}

            Competitor(const std::string& name, int num_threads_per_block, int num_thread_blocks)
            : name(name), num_threads_per_block(num_threads_per_block), num_thread_blocks(num_thread_blocks)
            {}

            ~Competitor() {}

            virtual inline void init_csr(Dense<T> &A, Dense<T> &B, CSR<T> &S, CSR<T> &P) {};
            virtual inline void init_coo(Dense<T> &A, Dense<T> &B, COO<T> &S, COO<T> &P) {};

            virtual inline void run_csr(Dense<T> &A, Dense<T> &B, CSR<T> &S, CSR<T> &P) = 0;
            virtual inline void run_coo(Dense<T> &A, Dense<T> &B, COO<T> &S, COO<T> &P) = 0;

            virtual inline void cleanup_csr(Dense<T> &A, Dense<T> &B, CSR<T> &S, CSR<T> &P) {};
            virtual inline void cleanup_coo(Dense<T> &A, Dense<T> &B, COO<T> &S, COO<T> &P) {};

            virtual bool csr_supported() = 0;
            virtual bool coo_supported() = 0;

            virtual bool is_gpu() { return false; }

            void set_num_threads_per_block(int num_threads_per_block) {
                this->num_threads_per_block = num_threads_per_block;
            }

            int get_num_threads_per_block() {
                return this->num_threads_per_block;
            }

            void set_num_thread_blocks(int num_thread_blocks) {
                this->num_thread_blocks = num_thread_blocks;
            }

            int get_num_thread_blocks() {
                return this->num_thread_blocks;
            }
        
        private:
            int num_threads_per_block;
            int num_thread_blocks;
    };
}
