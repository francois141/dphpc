cmake_minimum_required(VERSION 3.21)
project(dphpc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
include_directories(src)

add_subdirectory(src)
add_subdirectory(tests)
