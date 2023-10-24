#pragma once

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
                
                // create tensors out of flat vector, by setting (row_stride, col_stride) to (num_cols, 1) on GPU
                at::TensorOptions options_scalar = torch::device(torch::kCUDA).dtype(scalar_type);
                torch::Tensor A_tensor = torch::from_blob(A.get_pointer(), {A_row, A_col}, {A_col, 1}, options_scalar);
                torch::Tensor B_tensor = at::transpose(torch::from_blob(B.get_pointer(), {B_row, B_col}, {B_col, 1}, options_scalar), 0, 1);

                
                // create sparse CSR tensor for CPU
                at::TensorOptions options_int = torch::device(torch::kCUDA).dtype(torch::kInt);
                torch::Tensor crow_indices = torch::tensor(S.getRowPositions(), options_int);
                torch::Tensor col_indices = torch::tensor(S.getColPositions(), options_int);
                torch::Tensor values = torch::tensor(S.getValues(), options_scalar);
                torch::IntArrayRef size = {S.getRows(), S.getCols()};
                torch::Tensor sparse_tensor = torch::sparse_csr_tensor(crow_indices, col_indices, values, size, options_scalar);

                torch::Tensor result = at::native::sparse_sampled_addmm_sparse_csr_cuda(sparse_tensor, A_tensor, B_tensor, 0, 1);

                P.setRowPositions(std::vector<int>(result.crow_indices().data_ptr<int64_t>(), result.crow_indices().data_ptr<int64_t>() + result.crow_indices().numel()));
                P.setColPositions(std::vector<int>(result.col_indices().data_ptr<int64_t>(), result.col_indices().data_ptr<int64_t>() + result.col_indices().numel()));
                P.setValues(std::vector<T>(result.values().data_ptr<T>(), result.values().data_ptr<T>() + result.values().numel()));
            }

            virtual inline void run_coo(Dense<T> &A, Dense<T> &B, COO<T> &S, COO<T> &P) {
                // https://pytorch.org/docs/stable/generated/torch.sparse.sampled_addmm.html
                DEBUG_OUT(" - PyTorch SDDMM kernel only supports CSR format." << std::endl);
            }

            virtual bool csr_supported() { return true; }; 
            virtual bool coo_supported() { return false; };

    };

}