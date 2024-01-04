set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_BUILD_TYPE Debug) # Remove for measurements


set(SRC_FILES
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
        competitors/cpu/cpu_pytorch.hpp

        # GPU Competitors
        competitors/gpu/gpu_basic.cu
        competitors/gpu/gpu_basic.hpp
        competitors/gpu/gpu_pytorch.hpp
        competitors/gpu/gpu_shared.cu
        competitors/gpu/gpu_shared.hpp
        competitors/gpu/gpu_tiled.cu
        competitors/gpu/gpu_tiled.hpp
        competitors/gpu/gpu_thread_dispatcher.cu
        competitors/gpu/gpu_thread_dispatcher.hpp
        # competitors/gpu/gpu_tensor.cu
        # competitors/gpu/gpu_tensor.hpp
        competitors/gpu/gpu_convert.cu
        competitors/gpu/gpu_convert.hpp
        competitors/gpu/gpu_preprocessing.cu
        competitors/gpu/gpu_preprocessing.hpp
        competitors/gpu/gpu_cuSPARSE.hpp

        #Helpers
        utils/util.hpp
        utils/random_generator.hpp
        utils/helpers.hpp

        # Main
        main.cpp)

add_executable(dphpc ${SRC_FILES})

target_compile_options(dphpc PRIVATE "$<$<AND:$<CONFIG:Debug,RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-G>")

if(NOT MSVC)
  target_compile_options(dphpc PRIVATE -Wall)
endif()

if(MSVC)
    source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${SRC_FILES})
endif()

find_package(Torch REQUIRED)
target_link_libraries(dphpc "${TORCH_LIBRARIES}")
#set_property(TARGET dphpc PROPERTY CUDA_ARCHITECTURES 72)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")