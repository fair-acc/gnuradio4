cmake_minimum_required(VERSION 3.19)

project(graph-prototype CXX)
set(CMAKE_CXX_STANDARD 20)

# Mainly for FMT
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

add_library(graph-prototype-options INTERFACE)
include(cmake/CompilerWarnings.cmake)
set_project_warnings(graph-prototype-options)

if (EMSCRIPTEN)
    set(CMAKE_EXECUTABLE_SUFFIX ".js")
    target_compile_options(graph-prototype-options INTERFACE
            -fwasm-exceptions
            -pthread
            )
    target_link_options(graph-prototype-options INTERFACE
            "SHELL:-s ALLOW_MEMORY_GROWTH=1"
            -fwasm-exceptions
            -pthread
            "SHELL:-s PTHREAD_POOL_SIZE=30"
            )
endif()


include(FetchContent)
FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 10.0.0
)

FetchContent_Declare(
        refl-cpp
        GIT_REPOSITORY https://github.com/veselink1/refl-cpp.git
        GIT_TAG 27fbd7d2e6d86bc135b87beef6b5f7ce53afd4fc # v0.12.3+4 11/2022
)

FetchContent_Declare(
        pmt
        GIT_REPOSITORY https://github.com/gnuradio/pmt.git
        GIT_TAG main
)

FetchContent_Declare(
        ut
        GIT_REPOSITORY https://github.com/boost-ext/ut.git
        GIT_TAG 696c047c368d3324e8e6fedb46d8eed4c9eeba7c# head as of 2023-05-25
)

FetchContent_MakeAvailable(fmt refl-cpp pmt ut)

add_library(pmtv INTERFACE)
target_include_directories(pmtv INTERFACE ${pmt_SOURCE_DIR}/include/)

add_subdirectory(include)
add_subdirectory(src)

option(ENABLE_TESTING "Enable Test Builds" ON)
if (ENABLE_TESTING AND UNIX AND NOT APPLE)
    list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")
    enable_testing()
    message("Building Tests.")
    add_subdirectory(test)
    message("Building Benchmarks.")
    add_subdirectory(bench)
endif ()
