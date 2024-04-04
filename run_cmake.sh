#!/bin/bash

mkdir -p build/
cd build/

cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$HOME/libtorch ../
#cmake -DCMAKE_PREFIX_PATH=$HOME/libtorch -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10 -DCMAKE_BUILD_TYPE=Release ../