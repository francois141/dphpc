# Quickstart tutorial : http://google.github.io/googletest/quickstart-cmake.html
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
        matrix_test
        matrix_test.cc
)
target_link_libraries(
        matrix_test
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(matrix_test)