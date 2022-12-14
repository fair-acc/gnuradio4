cmake_minimum_required(VERSION 3.19)

project(graph-prototype CXX)
set(CMAKE_CXX_STANDARD 20)

include(cmake/CompilerWarnings.cmake)
include(FetchContent)
FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 8.1.1
)
FetchContent_MakeAvailable(fmt)

add_executable(main ./main.cpp)
target_include_directories(main PUBLIC ./include)
target_link_libraries(main PUBLIC fmt::fmt)
set_project_warnings(main)
