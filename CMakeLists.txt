cmake_minimum_required(VERSION 3.19)

project(graph-prototype CXX)
set(CMAKE_CXX_STANDARD 20)

if (EMSCRIPTEN)
    set(CMAKE_EXECUTABLE_SUFFIX ".js")
endif()

include(cmake/CompilerWarnings.cmake)
include(FetchContent)
FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 8.1.1
)

FetchContent_Declare(
        refl-cpp
        GIT_REPOSITORY https://github.com/veselink1/refl-cpp.git
        GIT_TAG 27fbd7d2e6d86bc135b87beef6b5f7ce53afd4fc # v0.12.3+4 11/2022
)

FetchContent_Declare(
        ut
        GIT_REPOSITORY https://github.com/boost-ext/ut.git
        GIT_TAG 265199e173b16a75670fae62fc2446b9dffad39e # head as of 2022-12-19
)

FetchContent_MakeAvailable(fmt refl-cpp ut)

add_compile_options(-Wall -Wextra
        # -Werror # Make all warnings into errors.
        -Wshadow # warn the user if a variable declaration shadows one from a parent context
        -Wnon-virtual-dtor # warn the user if a class with virtual functions has a non-virtual destructor.
        -Wold-style-cast # warn for c-style casts
        -Wcast-align # warn for potential performance problem casts
        -Wunused # warn on anything being unused
        -Woverloaded-virtual # warn if you overload (not override) a virtual function
        -pedantic -Wpedantic # warn if non-standard C++ is used
        -Wconversion # warn on type conversions that may lose data
        -Wsign-conversion # warn on sign conversions
        -Wmisleading-indentation # warn if indentation implies blocks where blocks do not exist
        -Wduplicated-cond # warn if if / else chain has duplicated conditions
        -Wduplicated-branches # warn if if / else chain has duplicated conditions
        -Wlogical-op # warn about logical operations being used where bitwise were probably wanted
        -Wnull-dereference # warn if a null dereference is detected
        -Wuseless-cast # warn if you perform a cast to the same type
        -Wdouble-promotion # warn if float is implicitly promoted to double
        -Wformat=2 # warn on security issues around functions that format output (i.e., printf)
        -Wimplicit-fallthrough # Warns when case statements fall-through.
        )

option(ENABLE_TESTING "Enable Test Builds" ON)
if (ENABLE_TESTING AND UNIX AND NOT APPLE)
    enable_testing()
    message("Building Tests.")
    add_subdirectory(test)
    message("Building Benchmarks.")
    add_subdirectory(bench)
endif ()

add_subdirectory(include)
add_subdirectory(src)
target_include_directories(main PUBLIC ./include)
target_link_libraries(main PUBLIC fmt::fmt refl-cpp::refl-cpp)
set_project_warnings(main)
