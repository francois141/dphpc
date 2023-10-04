cmake_minimum_required(VERSION 3.26)
project(dphpc)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(dphpc
        CSR.cpp
        CSR.h
        Dense.cpp
        Dense.h
        main.cpp)
