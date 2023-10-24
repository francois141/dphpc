# Quickstart tutorial : http://google.github.io/googletest/quickstart-cmake.html
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

        #Helpers
        ../src/utils/util.hpp
        ../src/utils/random_generator.hpp
        ../src/utils/helpers.hpp
        ../src/utils/file_writer.hpp

        matrix_test.cc)

target_link_libraries(
        matrix_test
        GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(matrix_test)