cmake_minimum_required(VERSION 3.25)

project(gnuradio CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

add_library(gnuradio-options INTERFACE)

if (CMAKE_CXX_COMPILER MATCHES "/em\\+\\+(-[a-zA-Z0-9.])?$") # if this hasn't been set before via e.g. emcmake
    message(" Transpiling to WASM: using: Emscripten (${CMAKE_CXX_COMPILER})")
    set(EMSCRIPTEN ON)
endif ()

option(ENABLE_BLOCK_PLUGINS "Enable building the plugin system" ON)
option(ENABLE_BLOCK_REGISTRY "Enable building the block registry" ON)
option(EMBEDDED "Enable embedded mode" OFF)
option(TIMETRACE "Enable clang's -ftime-trace" OFF)
option(ADDRESS_SANITIZER "Enable address sanitizer" OFF)
option(UB_SANITIZER "Enable undefined behavior sanitizer" OFF)
option(THREAD_SANITIZER "Enable thread sanitizer" OFF)
option(ENABLE_TBB "Enable the TBB dependency for std::execution::par in gcc" OFF)

if (EMSCRIPTEN)
    set(ENABLE_BLOCK_PLUGINS OFF)
endif ()

message(STATUS "Is block registry enabled? (faster compile-times and when runtime or Python wrapping APIs are not required) ${ENABLE_BLOCK_REGISTRY}")
message(STATUS "Is plugin system enabled? ${ENABLE_BLOCK_PLUGINS}")

# Determine if fmt is built as a subproject (using add_subdirectory) or if it is the master project.
if (NOT DEFINED GR_TOPLEVEL_PROJECT)
    set(GR_TOPLEVEL_PROJECT OFF)
    if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
        set(GR_TOPLEVEL_PROJECT ON)
        message(STATUS "CMake version: ${CMAKE_VERSION}")
    endif ()
endif ()

# Use ccache if found and enabled
find_program(CCACHE_PROGRAM ccache)
option(USE_CCACHE "Use ccache if available" ON)
if (CCACHE_PROGRAM AND USE_CCACHE)
    message(STATUS "ccache found and will be used")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CCACHE_PROGRAM}")
else ()
    message(STATUS "ccache will not be used")
endif ()

set(CMAKE_EXT_DEP_WARNING_GUARD "")
if (DISABLE_EXTERNAL_DEPS_WARNINGS) # enable warnings for external dependencies
    set(CMAKE_EXT_DEP_WARNING_GUARD SYSTEM)
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
    if (NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "RelWithAssert" CACHE STRING
                "Choose the type of build. Options are: None Debug Release RelWithAssert RelWithDebInfo MinSizeRel.\n\
 - None: no compiler flags, defaults and target-specific flags apply\n\
 - Debug: best/complete debugging experience; as optimized as reasonable\n\
 - Release: full optimization; some runtime checks disabled\n\
 - RelWithAssert: full optimization; runtime checks enabled\n\
 - RelWithDebInfo: optimized; debug info; some runtime checks disabled"
                FORCE)
    endif (NOT CMAKE_BUILD_TYPE)

    if (CMAKE_BUILD_TYPE STREQUAL "" AND NOT CMAKE_CXX_FLAGS MATCHES "-O[123gs]")
        message(WARNING "It seems you are compiling without optimization. Please set CMAKE_BUILD_TYPE or CMAKE_CXX_FLAGS.")
    endif ()
endif ()

# Initialize a variable to hold all the compiler flags -> exported into global config.h(.in)
if (CMAKE_BUILD_TYPE MATCHES Debug)
    set(ALL_COMPILER_FLAGS "${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS}")
elseif (CMAKE_BUILD_TYPE MATCHES Release)
    set(ALL_COMPILER_FLAGS "${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS}")
elseif (CMAKE_BUILD_TYPE MATCHES RelWithAssert)
    set(ALL_COMPILER_FLAGS "${CMAKE_CXX_FLAGS_RELWITHASSERT} ${CMAKE_CXX_FLAGS}")
elseif (CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
    set(ALL_COMPILER_FLAGS "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${CMAKE_CXX_FLAGS}")
elseif (CMAKE_BUILD_TYPE MATCHES MinSizeRel)
    set(ALL_COMPILER_FLAGS "${CMAKE_CXX_FLAGS_MINSIZEREL} ${CMAKE_CXX_FLAGS}")
    set(EMBEDDED ON)
    # 'EMBEDDED is used to disable/minimise code features for embedded systems (e.g. code-size, console printouts, etc.)
    add_compile_definitions(EMBEDDED)
    message(STATUS "enable size-optimising core feature (e.g. suppressing self-documentation): ${EMBEDDED}")
endif ()
# Replace ; with space
string(REPLACE ";" " " ALL_COMPILER_FLAGS "${ALL_COMPILER_FLAGS}")

if (CMAKE_CXX_COMPILER_ID MATCHES ".*Clang") # set default C++ STL to Clang's libc++ when using Clang
    add_compile_options(-stdlib=libc++ -fcolor-diagnostics)
    if (TIMETRACE)
        add_compile_options(-ftime-trace)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -ftime-trace")
        message(STATUS "Enable TIMETRACE: ${TIMETRACE}")
    endif ()
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-fdiagnostics-color=always)
    if (ENABLE_TBB)
        find_package(TBB REQUIRED)
        target_link_libraries(gnuradio-options INTERFACE TBB::tbb)
    else ()
        target_compile_definitions(gnuradio-options INTERFACE _GLIBCXX_USE_TBB_PAR_BACKEND=0)
    endif ()
endif ()

if (CMAKE_CXX_COMPILER_ID MATCHES "(Clang|GNU)")
    # Validate that only one sanitizer option is enabled
    if ((ADDRESS_SANITIZER AND UB_SANITIZER) OR (ADDRESS_SANITIZER AND THREAD_SANITIZER) OR (UB_SANITIZER AND THREAD_SANITIZER))
        message(FATAL_ERROR "Only one of ADDRESS_SANITIZER, UB_SANITIZER, or THREAD_SANITIZER can be enabled at the same time.")
    endif ()

    if (ADDRESS_SANITIZER)
        add_compile_options(-fsanitize=address -fsanitize-address-use-after-scope -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address -fsanitize-address-use-after-scope -fno-omit-frame-pointer)
        message(STATUS "Enable ADDRESS_SANITIZER: ${ADDRESS_SANITIZER}")
    elseif (UB_SANITIZER)
        add_compile_options(-fsanitize=undefined)
        add_link_options(-fsanitize=undefined)
        message(STATUS "Enable UB_SANITIZER: ${UB_SANITIZER}")
    elseif (THREAD_SANITIZER)
        add_compile_options(-fsanitize=thread)
        add_link_options(-fsanitize=thread)
        message(STATUS "Enable THREAD_SANITIZER: ${THREAD_SANITIZER}")
    endif ()
endif ()

# Include What You Use tooling: https://github.com/include-what-you-use/include-what-you-use
find_program(INCLUDE_WHAT_YOU_USE_TOOL_PATH NAMES include-what-you-use iwyu)
if (INCLUDE_WHAT_YOU_USE_TOOL_PATH)
    message(" using 'Include What You Use' path: (${INCLUDE_WHAT_YOU_USE_TOOL_PATH})")
    set_property(GLOBAL PROPERTY CMAKE_CXX_INCLUDE_WHAT_YOU_USE ${INCLUDE_WHAT_YOU_USE_TOOL_PATH})
endif ()

# Mainly for FMT
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

include(cmake/CompilerWarnings.cmake)
set_project_warnings(gnuradio-options)

if (EMSCRIPTEN)
    set(CMAKE_EXECUTABLE_SUFFIX ".js")
    add_compile_options(
            -fwasm-exceptions
            -pthread
            -g
    )
    add_link_options(
            "SHELL:-s ALLOW_MEMORY_GROWTH=1"
            "SHELL:-s ASSERTIONS=1"
            "SHELL:-s INITIAL_MEMORY=256MB"
            # "SHELL:-s SAFE_HEAP=1" # additional for debug
            # "SHELL:-s ASSERTIONS=2" # additional for debug
            # "SHELL:-s STACK_OVERFLOW_CHECK=2" # additional for debug
            # "SHELL:-g" # additional for debug
            # "SHELL:-gsource-map" # additional for debug
            # "SHELL:--profiling-funcs" # additional for debug
            # "SHELL:--emit-symbol-map" # additional for debug
            -fwasm-exceptions
            -pthread
            "SHELL:-s PTHREAD_POOL_SIZE=60"
            "SHELL:-s FETCH=1"
            "SHELL:-s WASM=1" # output as web-assembly
    )
endif ()

# include header-only libraries that have been inlined to simplify builds w/o requiring access to the internet
if (NOT (TARGET refl-cpp))
    add_library(refl-cpp INTERFACE)
    target_include_directories(refl-cpp ${CMAKE_EXT_DEP_WARNING_GUARD} INTERFACE ${PROJECT_SOURCE_DIR}/third_party/refl-cpp/)
endif ()

if (NOT (TARGET magic_enum))
    add_library(magic_enum INTERFACE)
    target_include_directories(magic_enum ${CMAKE_EXT_DEP_WARNING_GUARD} INTERFACE ${PROJECT_SOURCE_DIR}/third_party/magic_enum/)
endif ()

include(FetchContent)
FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 10.2.1
)

FetchContent_Declare(
        pmt
        GIT_REPOSITORY https://github.com/gnuradio/pmt.git
        GIT_TAG main
)

FetchContent_Declare(
        ut
        GIT_REPOSITORY https://github.com/boost-ext/ut.git
        GIT_TAG v2.0.1 # latest tag as of 2023-12-18
)

FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG 0.8.0
        PATCH_COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/patches/yaml-cpp/0001-Avoid-static-reference-to-temporary.patch
        UPDATE_DISCONNECTED 1
)

FetchContent_Declare(
        vir-simd
        GIT_REPOSITORY https://github.com/mattkretz/vir-simd.git
        GIT_TAG v0.2.0
)

FetchContent_Declare(
        cpp-httplib
        GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
        GIT_TAG v0.15.3
)

FetchContent_MakeAvailable(fmt pmt ut yaml-cpp vir-simd cpp-httplib)

# Fetch SoapySDR -- needed since the distribution version is incompatible w.r.t. stdlibc++ vs. libc++
if (CMAKE_CXX_COMPILER_ID MATCHES "(GNU|Clang)") # WIP
    find_package(SoapySDR CONFIG)
endif ()

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

## check for CPython and Numpy dependencies
set(PYTHON_FORCE_INCLUDE OFF)
if (PYTHON_FORCE_INCLUDE)
    find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter Development NumPy)
else ()
    find_package(Python3 3.12 COMPONENTS Interpreter Development NumPy)
endif ()

set(PYTHON_AVAILABLE OFF)
if (Python3_FOUND AND NOT EMSCRIPTEN)
    execute_process(
            COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckNumPy.py"
            RESULT_VARIABLE NUMPY_NOT_FOUND
            OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # If NumPy is found, set PYTHON_AVAILABLE to ON
    if (NOT NUMPY_NOT_FOUND)
        set(PYTHON_AVAILABLE ON)
        include_directories(${Python3_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR})
        add_definitions(-DPYTHON_AVAILABLE)
        message(STATUS "Using Python Include Dirs: ${Python3_INCLUDE_DIRS} and ${NUMPY_INCLUDE_DIR}")
    else ()
        message(STATUS "Python and Numpy Include headers not found!!")
    endif ()
endif ()

option(ENABLE_EXAMPLES "Enable Example Builds" ${GR_TOPLEVEL_PROJECT})

option(ENABLE_TESTING "Enable Test Builds" ${GR_TOPLEVEL_PROJECT})
if (ENABLE_TESTING AND UNIX AND NOT APPLE)
    list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")
    enable_testing()
    if (ENABLE_COVERAGE)
        message("Coverage reporting enabled")
        include(cmake/CodeCoverage.cmake) # https://github.com/bilke/cmake-modules/blob/master/CodeCoverage.cmake # (License: BSL-1.0)
        target_compile_options(gnuradio-options INTERFACE --coverage -O0 -g -gz -gdwarf-2 -gstrict-dwarf -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0) # fortify_source is not possible without optimization
        target_link_libraries(gnuradio-options INTERFACE --coverage)
        append_coverage_compiler_flags()
        set(GCOVR_ADDITIONAL_ARGS "--merge-mode-functions=merge-use-line-min")
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
