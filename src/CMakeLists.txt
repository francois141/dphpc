set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# FILE(GLOB_RECURSE cppSources *.cpp *.hpp *.h)
# FILE(GLOB_RECURSE cudaFiles *.cu)
# add_executable(dphpc ${cppSources} ${cudaFiles})

add_executable(dphpc
        # Matrix representations 
        matrices/Dense.hpp
        matrices/CSR.hpp
        matrices/COO.hpp
        matrices/triplet.h
        matrices/matrices.h

        # Benchmark code
        benchmark/benchmark.hpp
        benchmark/competitor.hpp
        benchmark/dataset.hpp

        # Competitors
        competitors/competitors.h

        # CPU Competitors
        competitors/cpu/cpu_basic.hpp

        # GPU Competitors
        competitors/gpu/gpu_basic.cu
        competitors/gpu/gpu_basic.h

        # Main
        main.cpp)