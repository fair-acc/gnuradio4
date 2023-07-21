cmake_minimum_required(VERSION 3.22)

project(graph-prototype CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

if (CMAKE_CXX_COMPILER_ID MATCHES "(Clang|GNU|Intel)")
    # -Og is a much more reasonable default for debugging. Also enable gdb extensions.
    set(CMAKE_CXX_FLAGS_DEBUG "-Og -ggdb" CACHE INTERNAL
        "Flags used by the compiler during debug builds.")

    # Add a build type that keeps runtime checks enabled
    set(CMAKE_CXX_FLAGS_RELWITHASSERT "-O3" CACHE INTERNAL
        "Flags used by the compiler during release builds containing runtime checks.")

    # The default value is often an empty string, but this is usually not desirable and one of the
    # other standard build types is usually more appropriate.
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "RelWithAssert" CACHE STRING
            "Choose the type of build. Options are: None Debug Release RelWithAssert RelWithDebInfo MinSizeRel.\n\
 - None: no compiler flags, defaults and target-specific flags apply\n\
 - Debug: best/complete debugging experience; as optimized as reasonable\n\
 - Release: full optimization; some runtime checks disabled\n\
 - RelWithAssert: full optimization; runtime checks enabled\n\
 - RelWithDebInfo: optimized; debug info; some runtime checks disabled"
            FORCE)
    endif(NOT CMAKE_BUILD_TYPE)

    if (CMAKE_BUILD_TYPE STREQUAL "" AND NOT CMAKE_CXX_FLAGS MATCHES "-O[123gs]")
        message(WARNING "It seems you are compiling without optimization. Please set CMAKE_BUILD_TYPE or CMAKE_CXX_FLAGS.")
    endif ()
endif ()

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
endif ()


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

FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG yaml-cpp-0.7.0
)

FetchContent_MakeAvailable(fmt refl-cpp pmt ut yaml-cpp)

add_library(pmtv INTERFACE)
target_include_directories(pmtv INTERFACE ${pmt_SOURCE_DIR}/include/)

# FFTW3 is build 2 times for float and double precisions
SET(FFTW_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/fftw)
if (EMSCRIPTEN)
    SET(FFTW_CONFIG cd ${FFTW_PREFIX}/src/ && emconfigure ./configure --quiet --disable-fortran --prefix=${FFTW_PREFIX}/install)
    SET(FFTW_BUILD emmake make -j && emmake make install && emmake make clean)
else ()
    SET(FFTW_CONFIG ${FFTW_PREFIX}/src/configure --quiet --disable-fortran --prefix=${FFTW_PREFIX}/install)
    SET(FFTW_BUILD make -j && make install && make clean)
endif ()
SET(FFTW_INSTALL_COMMAND
        ${FFTW_CONFIG} && ${FFTW_BUILD} &&
        ${FFTW_CONFIG} --enable-float && ${FFTW_BUILD})

include(ExternalProject)
ExternalProject_Add(fftw_ext
        PREFIX ${FFTW_PREFIX}
        SOURCE_DIR ${FFTW_PREFIX}/src
        BINARY_DIR ${FFTW_PREFIX}/build
        INSTALL_DIR ${FFTW_INSTALL_DIR}
        STAMP_DIR ${FFTW_PREFIX}/stamp
        TMP_DIR ${FFTW_PREFIX}/tmp
        DOWNLOAD_DIR ${FFTW_PREFIX}/download
        LOG_DIR ${FFTW_PREFIX}/log
        URL "https://fftw.org/fftw-3.3.10.tar.gz"
        URL_MD5 8ccbf6a5ea78a16dbc3e1306e234cc5c
        CONFIGURE_COMMAND ${FFTW_INSTALL_COMMAND}
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        LOG_DOWNLOAD ON
        )

include_directories(${FFTW_PREFIX}/install/include)

add_library(fftw INTERFACE)
target_link_libraries(fftw INTERFACE fftw3 INTERFACE fftw3f INTERFACE m)
target_include_directories(fftw INTERFACE ${FFTW_PREFIX}/install/include)
target_link_directories(fftw INTERFACE ${FFTW_PREFIX}/install/lib)
add_dependencies(fftw fftw_ext)

add_subdirectory(include)
add_subdirectory(src)

option(ENABLE_TESTING "Enable Test Builds" ON)
if (ENABLE_TESTING AND UNIX AND NOT APPLE)
    list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")
    enable_testing()
    if (ENABLE_COVERAGE)
        message("Coverage reporting enabled")
        include(cmake/CodeCoverage.cmake) # https://github.com/bilke/cmake-modules/blob/master/CodeCoverage.cmake # (License: BSL-1.0)
        target_compile_options(graph-prototype-options INTERFACE --coverage -O0 -g -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0) # fortify_source is not possible without optimization
        target_link_libraries(graph-prototype-options INTERFACE --coverage)
        append_coverage_compiler_flags()
        setup_target_for_coverage_gcovr_xml(
                NAME coverage
                EXECUTABLE ctest
                DEPENDENCIES qa_buffer qa_data_sink qa_dynamic_port qa_dynamic_node qa_hier_node qa_filter qa_fft qa_settings qa_tags qa_scheduler qa_thread_pool qa_thread_affinity
                EXCLUDE "$CMAKE_BUILD_DIR/*" "concepts/.*" ".*/test/.*")
        setup_target_for_coverage_gcovr_html(
                NAME coverage_html
                EXECUTABLE ctest
                DEPENDENCIES qa_buffer qa_data_sink qa_dynamic_port qa_dynamic_node qa_hier_node qa_filter qa_fft qa_settings qa_tags qa_scheduler qa_thread_pool qa_thread_affinity
                EXCLUDE "$CMAKE_BUILD_DIR/*" "concepts/.*" ".*/test/.*")
    endif ()
    message("Building Tests.")
    add_subdirectory(test)
    message("Building Benchmarks.")
    add_subdirectory(bench)
endif ()
