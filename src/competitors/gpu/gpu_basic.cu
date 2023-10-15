#include "gpu_basic.hpp"

__global__ void gpu_basic_csr_kernel(void) {
}

__global__ void gpu_basic_coo_kernel(void) {
}

template <typename T>
void gpu_basic_csr_wrapper(Dense<T> &A, Dense<T> &B, CSR<T> &S, CSR<T> &P) {
    gpu_basic_csr_kernel<<<1, 1>>>();
}

template <typename T>
void gpu_basic_coo_wrapper(Dense<T> &A, Dense<T> &B, COO<T> &S, COO<T> &P) {
    gpu_basic_csr_kernel<<<1, 1>>>();
}

/* Workaround because the wrappers need to be inside the CUDA file (Would normally write templated functions inside the header file!) */
template void gpu_basic_csr_wrapper<int>(Dense<int> &A, Dense<int> &B, CSR<int> &S, CSR<int> &P);
template void gpu_basic_coo_wrapper<int>(Dense<int> &A, Dense<int> &B, COO<int> &S, COO<int> &P);

template void gpu_basic_csr_wrapper<float>(Dense<float> &A, Dense<float> &B, CSR<float> &S, CSR<float> &P);
template void gpu_basic_coo_wrapper<float>(Dense<float> &A, Dense<float> &B, COO<float> &S, COO<float> &P);

// template void gpu_basic_csr_wrapper<double>(Dense<double> &A, Dense<double> &B, CSR<double> &S, CSR<double> &P);
// template void gpu_basic_coo_wrapper<double>(Dense<double> &A, Dense<double> &B, COO<double> &S, COO<double> &P);

