#!/bin/bash

mkdir -p build/

cd build/
cmake -DCMAKE_PREFIX_PATH=$HOME/libtorch ../
# For François
#cmake -DCMAKE_PREFIX_PATH=$HOME/libtorch -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10 ../