#include <iostream>
#include <torch/torch.h>

#include "utils/util.hpp"
#include "benchmark/competitor.hpp"
#include "matrices/matrices.h"

namespace Competitors {

    template <typename T>
    class CPUPyTorch : public SDDMM::Competitor<T> {
        public:

            CPUPyTorch()
            : SDDMM::Competitor<T>("CPU-PyTorch")
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

                // torch::set_num_threads(X);
                
                // create tensors out of flat vector, by setting (row_stride, col_stride) to (num_cols, 1)
                torch::Tensor A_tensor = torch::from_blob(A.get_pointer(), {A_row, A_col}, {A_col, 1}, scalar_type);
                torch::Tensor B_tensor = at::transpose(torch::from_blob(B.get_pointer(), {B_row, B_col}, {B_col, 1}, scalar_type), 0, 1);

                
                // create sparse CSR tensor for CPU
                torch::Tensor crow_indices = torch::tensor(S.getRowPositions(), torch::kInt);
                torch::Tensor col_indices = torch::tensor(S.getColPositions(), torch::kInt);
                torch::Tensor values = torch::tensor(S.getValues(), scalar_type);
                torch::IntArrayRef size = {S.getRows(), S.getCols()};
                at::TensorOptions options = torch::device(torch::kCPU).dtype(scalar_type);
                torch::Tensor sparse_tensor = torch::sparse_csr_tensor(crow_indices, col_indices, values, size, options);

                torch::Tensor result = at::native::sparse_sampled_addmm_sparse_csr_cpu(sparse_tensor, A_tensor, B_tensor, 0, 1);

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