#!/bin/bash

DPHPC_BUILD_TYPE=DEBUG # set to DEBUG if asserts are needed

mkdir -p build/

cd build/
cmake -DCMAKE_BUILD_TYPE=$DPHPC_BUILD_TYPE -DCMAKE_PREFIX_PATH=$HOME/libtorch ../
# For François
#cmake -DCMAKE_PREFIX_PATH=$HOME/libtorch -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10 ../
#cmake -DCMAKE_PREFIX_PATH=$HOME/libtorch -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10 -DCMAKE_BUILD_TYPE=Release ../
