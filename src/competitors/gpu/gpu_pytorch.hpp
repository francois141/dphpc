#pragma once

#include <torch/torch.h>

#include "benchmark/competitor.hpp"
#include "matrices/matrices.h"

namespace Competitors {

    template <typename T>
    class GPUPyTorch : public SDDMM::Competitor<T> {
        public:

            GPUPyTorch()
            : SDDMM::Competitor<T>("GPU-PyTorch")
            {}

            virtual inline void init_csr(Dense<T> &A, Dense<T> &B, CSR<T> &S, CSR<T> &P) override {
                if (std::is_same<T, double>::value) {
                    scalar_type = torch::kDouble;
                } else if (std::is_same<T, float>::value) {
                    scalar_type = torch::kFloat;
                } else {
                    DEBUG_OUT("PyTorch SDDMM CSR kernel only supports floating point types. Skipping calculation.");
                    return;
                }

                const unsigned int A_col = A.getCols();
                const unsigned int A_row = A.getRows();
                const unsigned int B_col = B.getCols();
                const unsigned int B_row = B.getRows();
                
                const auto gpu = torch::device(torch::kCUDA);

                // create tensors out of flat vector, by setting (row_stride, col_stride) to (num_cols, 1) on CPU, then copy them to the GPU
                A_tensor = torch::from_blob(A.get_pointer(), { A_row, A_col }, { A_col, 1 }, scalar_type).to(gpu);
                B_tensor = at::transpose(torch::from_blob(B.get_pointer(), {B_row, B_col}, {B_col, 1}, scalar_type), 0, 1).to(gpu);

                // create sparse CSR tensor for CPU
                torch::ScalarType int_type = torch::kInt;
                torch::Tensor crow_indices = torch::tensor(S.getRowPositions(), int_type);
                torch::Tensor col_indices = torch::tensor(S.getColPositions(), int_type);
                torch::Tensor values = torch::tensor(S.getValues(), scalar_type);
                std::vector<int64_t> size = {static_cast<int64_t>(S.getRows()), static_cast<int64_t>(S.getCols())};
                sparse_tensor = torch::sparse_csr_tensor(crow_indices, col_indices, values, size, scalar_type).to(gpu);
            }

            virtual inline void run_csr(Dense<T> &A, Dense<T> &B, CSR<T> &S, CSR<T> &P) override {
                torch::Tensor non_scaled_result = at::native::sparse_sampled_addmm_sparse_csr_cuda(sparse_tensor, A_tensor, B_tensor, 0, 1);
                result = at::native::mul_out_sparse_csr(non_scaled_result, sparse_tensor, non_scaled_result);
                torch::cuda::synchronize();
            }

            virtual inline void cleanup_csr(Dense<T> &A, Dense<T> &B, CSR<T> &S, CSR<T> &P) override {
                const auto cpu = torch::device(torch::kCPU);
                result = result.to(cpu);

                P.setRowPositions(copy_pytorch_mat<int>(result.crow_indices().const_data_ptr<int64_t>(), result.crow_indices().numel()));
                P.setColPositions(copy_pytorch_mat<int>(result.col_indices().const_data_ptr<int64_t>(), result.col_indices().numel()));
                P.setValues(std::vector<T>(result.values().const_data_ptr<T>(), result.values().const_data_ptr<T>() + result.values().numel()));
            }
            
            // https://pytorch.org/docs/stable/generated/torch.sparse.sampled_addmm.html
            virtual inline void run_coo(Dense<T> &A, Dense<T> &B, COO<T> &S, COO<T> &P) {}

            virtual bool csr_supported() override { return true; }; 
            virtual bool coo_supported() override { return false; };

            virtual bool is_gpu() override { return true; };
        
        private:
            torch::ScalarType scalar_type;

            torch::Tensor A_tensor;
            torch::Tensor B_tensor;
            torch::Tensor sparse_tensor;

            torch::Tensor result;
    };

}
