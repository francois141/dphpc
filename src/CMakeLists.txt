set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_BUILD_TYPE Debug) # Remove for measurements

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
        competitors/gpu/gpu_basic.hpp

        #Helpers
        utils/random_generator.hpp
        utils/helpers.hpp

        # Main
        main.cpp)

find_package(Torch REQUIRED)

target_link_libraries(dphpc "${TORCH_LIBRARIES}")