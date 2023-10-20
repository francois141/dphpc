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
                const unsigned int A_col = A.getCols();
                const unsigned int A_row = A.getRows();
                const unsigned int B_col = B.getCols();
                const unsigned int B_row = B.getRows();

                at::Tensor A_tensor = torch::from_blob(A.get_pointer(), {A_row, A_col}, torch::kCPU);
                at::Tensor B_tensor = torch::from_blob(B.get_pointer(), {B_row, B_col}, torch::kCPU);
                
                /*
                torch::Tensor crow_indices = torch::tensor(P.getRowPositions(), torch::kInt);
                torch::Tensor col_indices = torch::tensor(P.getColPositions(), torch::kInt);
                torch::Tensor values = torch::tensor(P.getValues(), torch::kCPU);
                torch::IntArrayRef size = {P.getRows(), P.getCols()};
                at::TensorOptions options = at::device(at::kCPU).dtype(at::kLong);

                // Create the sparse CSR tensor
                torch::Tensor sparse_tensor = at::sparse_csr_tensor(crow_indices, col_indices, values, size, options);
                */
                // Print the sparse tensor
                torch::Tensor A_t = torch::tensor({{1, 1, 1},
                                                {1, 1, 1},
                                                {1, 1, 1}}, torch::kFloat);
                torch::Tensor B_t = torch::tensor({{1, 1, 1},
                                                {1, 1, 1},
                                                {1, 1, 1}}, torch::kFloat);

                //torch::Tensor res = at::native::dense_to_sparse_csr(A_t);
                /*
                torch::Tensor crow = torch::tensor({0,1,1,1}, torch::kInt);
                torch::Tensor cols = torch::tensor({0}, torch::kInt);
                torch::Tensor val = torch::tensor({1}, torch::kFloat);
                torch::Tensor self = at::native::sparse_compressed_tensor(crow, cols, val, {3,3}, torch::kFloat, torch::Layout::SparseCsr);
                */
                //torch::Tensor res = at::native::sparse_sampled_addmm_sparse_csr_cpu(self, A_t, B_t, 0, 1);
                std::cout << A_t.to_sparse_csr() << std::endl;
                DEBUG_OUT(" - (Not supported)" << std::endl);
            }

            virtual inline void run_coo(Dense<T> &A, Dense<T> &B, COO<T> &S, COO<T> &P) {
                DEBUG_OUT(" - (Not supported)" << std::endl);
            }

            // TODO - implement at least one of these by calling library func
            // timing measurements might be negatively affected if we need to convert to other data types

            virtual bool csr_supported() { return false; };
            virtual bool coo_supported() { return false; };
    };

}