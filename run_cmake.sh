#!/bin/bash

# SETTINGS

# set to DEBUG if asserts are needed
DPHPC_BUILD_TYPE=Release

# set this to the location of libtorch
TORCH_LOCATION="$HOME/perso/ethz/dphpc/libtorch"

# set this to the location of the DGL library, it should contain the include folders
DGL_LOCATION="$HOME/perso/ethz/dphpc/dgl"

# END OF SETTINGS

mkdir -p build/
cd build/
cmake -DCMAKE_BUILD_TYPE=$DPHPC_BUILD_TYPE -DCMAKE_PREFIX_PATH=$TORCH_LOCATION -DDGL_LOCATION=$DGL_LOCATION ../

# For François
#cmake -DCMAKE_PREFIX_PATH=$HOME/libtorch -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10 ../
#cmake -DCMAKE_PREFIX_PATH=$HOME/libtorch -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10 -DCMAKE_BUILD_TYPE=Release ../