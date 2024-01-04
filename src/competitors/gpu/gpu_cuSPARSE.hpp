#pragma once

#include "benchmark/competitor.hpp"
#include "matrices/matrices.h"

#include <cuda_runtime_api.h>
#include <cusparse.h>         // cusparseSDDMM

namespace Competitors {

    class GPUcuSPARSE : public SDDMM::Competitor<float> {
            using T = float;
        public:

            GPUcuSPARSE()
            : SDDMM::Competitor<T>("GPU-cuSPARSE")
            {}

            // https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSPARSE/sddmm_csr/sddmm_csr_example.c
            virtual inline void init_csr(Dense<T> &A, Dense<T> &B, CSR<T> &S, CSR<T> &P) override {                
                // A is MxK, B is NxK, S and P are MxN sparse
                unsigned int M = A.getRows();
                unsigned int K = A.getCols();
                unsigned int N = B.getRows();

                int S_nnz = S.getValues().size();

                int   A_size = M * K;
                int   B_size = K * N;

                cudaMalloc(&dA, A_size * sizeof(float));
                cudaMalloc(&dB, B_size * sizeof(float));
                cudaMalloc(&dS_offsets, (M + 1) * sizeof(int));
                cudaMalloc(&dS_columns, S_nnz * sizeof(int));
                cudaMalloc(&dS_values,  S_nnz * sizeof(float));

                cudaMemcpy(dA, &A.getValue(0, 0), A_size * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(dB, &B.getValue(0, 0), B_size * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(dS_offsets, S.getRowPositions().data(), (M + 1) * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(dS_columns, S.getColPositions().data(), S_nnz * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(dS_values, S.getValues().data(), S_nnz * sizeof(float), cudaMemcpyHostToDevice);

                // Create handle
                cusparseCreate(&handle);

                // Create dense matrices A and B and sparse matrix S in CSR format
                cusparseCreateDnMat(&matA, M, K, K, dA, CUDA_R_32F, CUSPARSE_ORDER_ROW);
                cusparseCreateDnMat(&matB, K, N, K, dB, CUDA_R_32F, CUSPARSE_ORDER_COL);
                
                cusparseCreateCsr(&matS, M, N, S_nnz, dS_offsets, dS_columns, dS_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

                // allocate an external buffer if needed
                cusparseSDDMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matS, CUDA_R_32F, CUSPARSE_SDDMM_ALG_DEFAULT, &bufferSize);
                cudaMalloc(&dBuffer, bufferSize);
            }

            // https://docs.nvidia.com/cuda/cusparse/#cusparsesddmm
            virtual inline void run_csr(Dense<T> &A, Dense<T> &B, CSR<T> &S, CSR<T> &P) override {                
                // execute preprocess (optional)
                // cusparseSDDMM_preprocess(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matS, CUDA_R_32F, CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer);

                // execute SDDMM
                cusparseSDDMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matS, CUDA_R_32F, CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer);
            }

            virtual inline void cleanup_csr(Dense<T> &A, Dense<T> &B, CSR<T> &S, CSR<T> &P) override {
                // destroy matrix/vector descriptors
                cusparseDestroyDnMat(matA);
                cusparseDestroyDnMat(matB);
                cusparseDestroySpMat(matS);
                cusparseDestroy(handle);

                // copy result back to RAM
                int S_nnz = S.getValues().size();
                cudaMemcpy(P.getValues().data(), dS_values, S_nnz * sizeof(float), cudaMemcpyDeviceToHost);
                
                P.setColPositions(S.getColPositions());
                P.setRowPositions(S.getRowPositions());

                // device memory deallocation
                cudaFree(dBuffer);
                cudaFree(dA);
                cudaFree(dB);
                cudaFree(dS_offsets);
                cudaFree(dS_columns);
                cudaFree(dS_values);
            }
            
            virtual inline void run_coo(Dense<T> &A, Dense<T> &B, COO<T> &S, COO<T> &P) {}

            virtual bool csr_supported() override { return true; }; 
            virtual bool coo_supported() override { return false; };

            virtual bool is_gpu() override { return true; };
        
        private:
            float alpha        = 1.0f;
            float beta         = 0.0f;

            // Device memory management
            int   *dS_offsets, *dS_columns;
            float *dS_values, *dB, *dA;

            // cuSPARSE APIs
            cusparseHandle_t     handle = NULL;
            cusparseDnMatDescr_t matA, matB;
            cusparseSpMatDescr_t matS;
            void*                dBuffer    = NULL;
            size_t               bufferSize = 0;
    };

}
