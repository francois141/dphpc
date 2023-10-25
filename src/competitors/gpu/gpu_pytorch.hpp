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

            virtual inline void run_csr(Dense<T> &A, Dense<T> &B, CSR<T> &S, CSR<T> &P) {
                torch::ScalarType scalar_type;
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
                
                const auto cpu = torch::device(torch::kCPU);
                const auto gpu = torch::device(torch::kCUDA);

                // create tensors out of flat vector, by setting (row_stride, col_stride) to (num_cols, 1) on CPU, then copy them to the GPU
                torch::Tensor A_tensor = torch::from_blob(A.get_pointer(), { A_row, A_col }, { A_col, 1 }, scalar_type).to(gpu);
                torch::Tensor B_tensor = at::transpose(torch::from_blob(B.get_pointer(), {B_row, B_col}, {B_col, 1}, scalar_type), 0, 1).to(gpu);

                
                // create sparse CSR tensor for CPU
                torch::ScalarType int_type = torch::kInt;
                torch::Tensor crow_indices = torch::tensor(S.getRowPositions(), int_type);
                torch::Tensor col_indices = torch::tensor(S.getColPositions(), int_type);
                torch::Tensor values = torch::tensor(S.getValues(), scalar_type);
                torch::IntArrayRef size = {S.getRows(), S.getCols()};
                torch::Tensor sparse_tensor = torch::sparse_csr_tensor(crow_indices, col_indices, values, size, scalar_type).to(gpu);

                torch::Tensor result = at::native::sparse_sampled_addmm_sparse_csr_cuda(sparse_tensor, A_tensor, B_tensor, 0, 1).to(cpu);

                P.setRowPositions(std::vector<int>(result.crow_indices().const_data_ptr<int64_t>(), result.crow_indices().const_data_ptr<int64_t>() + result.crow_indices().numel()));
                P.setColPositions(std::vector<int>(result.col_indices().const_data_ptr<int64_t>(), result.col_indices().const_data_ptr<int64_t>() + result.col_indices().numel()));
                P.setValues(std::vector<T>(result.values().const_data_ptr<T>(), result.values().const_data_ptr<T>() + result.values().numel()));
            }

            virtual inline void run_coo(Dense<T> &A, Dense<T> &B, COO<T> &S, COO<T> &P) {
                // https://pytorch.org/docs/stable/generated/torch.sparse.sampled_addmm.html
                DEBUG_OUT(" - PyTorch SDDMM kernel only supports CSR format." << std::endl);
            }

            virtual bool csr_supported() { return true; }; 
            virtual bool coo_supported() { return false; };

    };

}
