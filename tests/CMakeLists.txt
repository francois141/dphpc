# Quickstart tutorial : http://google.github.io/googletest/quickstart-cmake.html
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#https://stackoverflow.com/questions/62693218/how-to-solve-gtest-and-libtorch-linkage-conflict
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(matrix_test 
        # Matrix representations 
        ../src/matrices/Dense.hpp
        ../src/matrices/CSR.hpp
        ../src/matrices/COO.hpp
        ../src/matrices/triplet.h
        ../src/matrices/matrices.h

        # Benchmark code
        ../src/benchmark/benchmark.hpp
        ../src/benchmark/competitor.hpp
        ../src/benchmark/dataset.hpp

        # Competitors
        ../src/competitors/competitors.h

        # CPU Competitors
        ../src/competitors/cpu/cpu_basic.hpp
        ../src/competitors/cpu/cpu_pytorch.hpp

        # GPU Competitors
        ../src/competitors/gpu/gpu_basic.cu
        ../src/competitors/gpu/gpu_basic.hpp
        ../src/competitors/gpu/gpu_tiled.cu
        ../src/competitors/gpu/gpu_tiled.hpp
        ../src/competitors/gpu/gpu_thread_dispatcher.cu
        ../src/competitors/gpu/gpu_thread_dispatcher.hpp
        ../src/competitors/gpu/gpu_shared.cu
        ../src/competitors/gpu/gpu_shared.hpp
		# ../src/competitors/gpu/gpu_tensor.cu
        # ../src/competitors/gpu/gpu_tensor.hpp
        ../src/competitors/gpu/gpu_convert.cu
        ../src/competitors/gpu/gpu_convert.hpp

        #Helpers
        ../src/utils/util.hpp
        ../src/utils/random_generator.hpp
        ../src/utils/helpers.hpp

        matrix_test.cc)

find_package(Torch REQUIRED)
target_link_libraries(
        matrix_test
        GTest::gtest_main
	    "${TORCH_LIBRARIES}")

set_property(TARGET matrix_test PROPERTY CUDA_ARCHITECTURES 72)

include(GoogleTest)
gtest_discover_tests(matrix_test)
