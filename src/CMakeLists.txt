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

        #Helpers
        utils/util.hpp
        utils/random_generator.hpp
        utils/helpers.hpp
        utils/file_writer.hpp

        # Main
        main.cpp)

add_executable(dphpc ${SRC_FILES})

if(MSVC)
  target_compile_options(dphpc PRIVATE /W4)
else()
  target_compile_options(dphpc PRIVATE -Wall)
endif()

if(MSVC)
    source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${SRC_FILES})
endif()

find_package(Torch REQUIRED)
target_link_libraries(dphpc "${TORCH_LIBRARIES}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET dphpc
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:dphpc>)
endif (MSVC)