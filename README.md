# SDDMM Project

### List of optimizations ideas

#### Libraries 

> https://github.com/NVIDIA/cutlass

> https://developer.nvidia.com/cublas

> https://developer.nvidia.com/cuda-math-library

> https://developer.nvidia.com/cusparse

> https://developer.nvidia.com/cutensor

#### Paper 

* Modify their most optimal versions for CSR & adapt for our paper

#### Parameter tuning

* Benchmark over multiple tiling sizes 
* Benchmark with #threads and #blocks
* Analysis between different streaming dimensions

#### Optimizations

* Unroll outermost loop with 2d grid for example
* Maybe unroll while loop after outermost loop
* Try compiling with hardcoded K

#### Cuda specific

Software pipelining: https://towardsdatascience.com/matrix-multiplication-on-the-gpu-e920e50207a8 and https://abhishekudupa.github.io/files/audupa-cgo-2009.pdf

Analyse and remove bank conflicts : https://stackoverflow.com/questions/3841877/what-is-a-bank-conflict-doing-cuda-opencl-programming

Vectorize loads and stores: https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/

#### Caching & memory

Is it possible to improve locality while accessing cols?

```c
int curr_col = cols[sparse_index];
```
