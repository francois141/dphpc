cmake_minimum_required(VERSION 3.22)
project(dphpc)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
include_directories(src)

add_subdirectory(src)
add_subdirectory(tests)
