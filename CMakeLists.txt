cmake_minimum_required(VERSION 3.25)

project(gnuradio CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

# Determine if fmt is built as a subproject (using add_subdirectory) or if it is the master project.
if (NOT DEFINED GR_TOPLEVEL_PROJECT)
    set(GR_TOPLEVEL_PROJECT OFF)
    if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
        set(GR_TOPLEVEL_PROJECT ON)
        message(STATUS "CMake version: ${CMAKE_VERSION}")
    endif ()
endif ()

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

# Initialize a variable to hold all the compiler flags -> exported into global config.h(.in)
if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(ALL_COMPILER_FLAGS "${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS}")
elseif(CMAKE_BUILD_TYPE MATCHES Release)
    set(ALL_COMPILER_FLAGS "${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS}")
elseif(CMAKE_BUILD_TYPE MATCHES RelWithAssert)
    set(ALL_COMPILER_FLAGS "${CMAKE_CXX_FLAGS_RELWITHASSERT} ${CMAKE_CXX_FLAGS}")
elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
    set(ALL_COMPILER_FLAGS "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${CMAKE_CXX_FLAGS}")
elseif(CMAKE_BUILD_TYPE MATCHES MinSizeRel)
    set(ALL_COMPILER_FLAGS "${CMAKE_CXX_FLAGS_MINSIZEREL} ${CMAKE_CXX_FLAGS}")
endif()
# Replace ; with space
string(REPLACE ";" " " ALL_COMPILER_FLAGS "${ALL_COMPILER_FLAGS}")

# Mainly for FMT
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

add_library(gnuradio-options INTERFACE)
include(cmake/CompilerWarnings.cmake)
set_project_warnings(gnuradio-options)

if(CMAKE_CXX_COMPILER MATCHES "/em\\+\\+(-[a-zA-Z0-9.])?$") # if this hasn't been set before via e.g. emcmake
    message(" Transpiling to WASM: using: Emscripten (${CMAKE_CXX_COMPILER})")
    set(EMSCRIPTEN true)
endif()

if (EMSCRIPTEN)
    set(CMAKE_EXECUTABLE_SUFFIX ".js")
    target_compile_options(gnuradio-options INTERFACE
            -fwasm-exceptions
            -pthread
    )
    target_link_options(gnuradio-options INTERFACE
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
        GIT_TAG 10.1.1
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
        GIT_TAG v2.0.0 # latest tag as of 2023-10-18
)

FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG yaml-cpp-0.7.0
)

FetchContent_Declare(
        vir-simd
        GIT_REPOSITORY https://github.com/mattkretz/vir-simd.git
        GIT_TAG v0.3.0
)

FetchContent_MakeAvailable(fmt refl-cpp pmt ut yaml-cpp vir-simd)

add_library(pmtv INTERFACE)
target_include_directories(pmtv INTERFACE ${pmt_SOURCE_DIR}/include/)

add_library(vir INTERFACE)
target_include_directories(vir INTERFACE ${vir-simd_SOURCE_DIR}/)

# FFTW3 is build 2 times for float and double precisions
SET(FFTW_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/fftw)
if (EMSCRIPTEN)
    SET(FFTW_CFLAGS "${CFLAGS} -fPIE -w")
    SET(FFTW_CONFIG cd ${FFTW_PREFIX}/src/ && emconfigure ./configure --enable-silent-rules --quiet --disable-fortran --prefix=${FFTW_PREFIX}/install)
    SET(FFTW_BUILD emmake make -j CFLAGS=${FFTW_CFLAGS} --silent V=0 && emmake make install --silent V=0 && emmake make clean --silent V=0)
else ()
    SET(FFTW_CFLAGS "${CFLAGS} -fPIE -w -O3 -march=native -mtune=native")
    SET(FFTW_CONFIG ${FFTW_PREFIX}/src/configure --enable-silent-rules --quiet --disable-fortran --prefix=${FFTW_PREFIX}/install)
    SET(FFTW_BUILD make -j CFLAGS=${FFTW_CFLAGS} --silent V=0 && make install --silent V=0 && make clean --silent V=0)
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

add_library(fftw INTERFACE)
target_link_libraries(fftw INTERFACE fftw3 INTERFACE fftw3f INTERFACE m)
target_include_directories(fftw INTERFACE ${FFTW_PREFIX}/install/include ${PROJECT_BINARY_DIR})
target_link_directories(fftw INTERFACE ${FFTW_PREFIX}/install/lib ${FFTW_PREFIX}/install/lib64)
add_dependencies(fftw fftw_ext)

option(ENABLE_EXAMPLES "Enable Example Builds" ${GR_TOPLEVEL_PROJECT})

option(ENABLE_TESTING "Enable Test Builds"  ${GR_TOPLEVEL_PROJECT})
if (ENABLE_TESTING AND UNIX AND NOT APPLE)
    list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")
    enable_testing()
    if (ENABLE_COVERAGE)
        message("Coverage reporting enabled")
        include(cmake/CodeCoverage.cmake) # https://github.com/bilke/cmake-modules/blob/master/CodeCoverage.cmake # (License: BSL-1.0)
        target_compile_options(gnuradio-options INTERFACE --coverage -O0 -g -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0) # fortify_source is not possible without optimization
        target_link_libraries(gnuradio-options INTERFACE --coverage)
        append_coverage_compiler_flags()
        setup_target_for_coverage_gcovr_xml(
                NAME coverage
                EXECUTABLE ctest
                EXECUTABLE_ARGS "--output-on-failure"
                DEPENDENCIES qa_buffer qa_DataSink qa_DynamicPort qa_DynamicBlock qa_HierBlock qa_filter qa_Settings qa_Tags qa_Scheduler qa_thread_pool qa_thread_affinity
                EXCLUDE "$CMAKE_BUILD_DIR/*")
        setup_target_for_coverage_gcovr_html(
                NAME coverage_html
                EXECUTABLE ctest
                EXECUTABLE_ARGS "--output-on-failure"
                DEPENDENCIES qa_buffer qa_DataSink qa_DynamicPort qa_DynamicBlock qa_HierBlock qa_filter qa_Settings qa_Tags qa_Scheduler qa_thread_pool qa_thread_affinity
                EXCLUDE "$CMAKE_BUILD_DIR/*")
    endif ()
    message("Building Tests and benchmarks.")
endif ()

add_subdirectory(bench) # custom ut addon for microbenchmarking

add_subdirectory(core)
add_subdirectory(meta)
add_subdirectory(algorithm)
add_subdirectory(blocks)
