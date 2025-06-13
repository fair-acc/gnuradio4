cmake_minimum_required(VERSION 3.27)

project(gnuradio CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# Mainly for FMT
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

if(CMAKE_CXX_COMPILER MATCHES "/em\\+\\+(-[a-zA-Z0-9.])?$") # if this hasn't been set before via e.g. emcmake
  message(" Transpiling to WASM: using: Emscripten (${CMAKE_CXX_COMPILER})")
  set(EMSCRIPTEN ON)
endif()

# default explicit compiler and linker flag optimisation per build type
if(NOT MSVC)
  # optional:  -fno-inline-functions -fvar-tracking-assignments
  set(CMAKE_CXX_FLAGS_DEBUG "-Og -g1 -DDEBUG -fno-omit-frame-pointer -ffunction-sections -fdata-sections")
  set(CMAKE_CXX_FLAGS_RELEASE "-O2 -g0 -DNDEBUG -ffunction-sections -fdata-sections")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g1 -gz -DNDEBUG -ffunction-sections -fdata-sections")
  set(CMAKE_CXX_FLAGS_RELWITHASSERT "-O2 -DASSERT_ENABLED -ffunction-sections -fdata-sections")
  # '-s': strip debug symbols, 'EMBEDDED is used to disable/minimise code features for embedded systems (e.g. code-size,
  # console printouts, etc.)
  set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -g0 -DNDEBUG -DEMBEDDED -ffunction-sections -fdata-sections")
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} -s")
  endif()

  set(_LD_GC_SECTIONS "-Wl,--gc-sections -ffunction-sections -fdata-sections")

  find_program(MOLD_BIN ld.mold)
  if(MOLD_BIN)
    set(_LD_GC_SECTIONS "${_LD_GC_SECTIONS} -fuse-ld=mold")
    message(STATUS "Using mold @ ${MOLD_BIN} as the linker.")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    find_program(LLD_BIN ld.lld)
    if(LLD_BIN)
      set(_LD_GC_SECTIONS "${_LD_GC_SECTIONS} -fuse-ld=lld")
      message(STATUS "Using lld @ ${LLD_BIN} as the linker.")
    endif()
  endif()
  set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${_LD_GC_SECTIONS}")
  set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${_LD_GC_SECTIONS}")
  set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${_LD_GC_SECTIONS}")
  set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${_LD_GC_SECTIONS}")
  set(CMAKE_EXE_LINKER_FLAGS_MINSIZEREL "${_LD_GC_SECTIONS}")
  set(CMAKE_SHARED_LINKER_FLAGS_MINSIZEREL "${_LD_GC_SECTIONS}")
elseif(MSVC)
  set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi /Zf /DDEBUG")
  set(CMAKE_CXX_FLAGS_RELEASE "/O2 /GL /DNDEBUG")
  set(CMAKE_CXX_FLAGS_RELWITHASSERT "/O2 /GL /DASSERT_ENABLED")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/O2 /GL /Zi /Zf /DNDEBUG")
  set(CMAKE_CXX_FLAGS_MINSIZEREL "/O1 /Os /GL /DNDEBUG")

  set(LINK_RELEASE "/LTCG /OPT:REF /OPT:ICF")
  set(LINK_RELASSERT "${LINK_RELEASE}")
  set(LINK_RELDEBINFO "${LINK_RELEASE}")
  set(LINK_MINSIZE "${LINK_RELEASE} /DEBUG:NONE") # strip PDB
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      "RelWithAssert"
      CACHE STRING "Default build type" FORCE)
endif()
set_property(
  CACHE CMAKE_BUILD_TYPE
  PROPERTY STRINGS
           None
           Debug
           Release
           RelWithAssert
           RelWithDebInfo
           MinSizeRel)

set(REGISTRY_DEFAULT ON)
if(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
  set(REGISTRY_DEFAULT OFF)
endif()

# Determine if GR4 is built as a subproject (using add_subdirectory) or if it is the master project.
if(NOT DEFINED GR_TOPLEVEL_PROJECT)
  set(GR_TOPLEVEL_PROJECT OFF)
  if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(GR_TOPLEVEL_PROJECT ON)
    message(STATUS "CMake version: ${CMAKE_VERSION}")
  endif()
endif()

# Option affects GR4 build, as well as the behaviour of BlockLib cmake functions that are used in out-of-tree modules
option(GR_ENABLE_BLOCK_REGISTRY "Enable building the block registry" ${REGISTRY_DEFAULT})
option(USE_CCACHE "Use ccache if available" ON)
option(INTERNAL_ENABLE_BLOCK_PLUGINS "Enable building the plugin system" ${REGISTRY_DEFAULT})
option(WARNINGS_AS_ERRORS "Enable -Werror flags" ON)
option(TIMETRACE "Enable clang's -ftime-trace" OFF)
option(ADDRESS_SANITIZER "Enable address sanitizer" OFF)
option(UB_SANITIZER "Enable undefined behavior sanitizer" OFF)
option(THREAD_SANITIZER "Enable thread sanitizer" OFF)
option(ENABLE_TBB "Enable the TBB dependency for std::execution::par in gcc" OFF)
option(ENABLE_EXAMPLES "Enable Example Builds" ${GR_TOPLEVEL_PROJECT})
option(ENABLE_TESTING "Enable Test Builds" ${GR_TOPLEVEL_PROJECT})

if(ENABLE_TESTING AND (UNIX OR APPLE))
  include(CTest)
  list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")
  enable_testing()
  message("Building Tests and benchmarks.")
endif()

# apply known LSAN suppressions to all tests
if(ADDRESS_SANITIZER OR UB_SANITIZER)
  set(ASAN_SUPPRESSION_FILE "${CMAKE_SOURCE_DIR}/cmake/asan_suppressions.supp")
  if(EXISTS "${ASAN_SUPPRESSION_FILE}")
    set(ENV_ASAN_OPTIONS "ASAN_OPTIONS=suppressions=${ASAN_SUPPRESSION_FILE}")
    message(STATUS "LeakSanitizer suppression enabled: ${ASAN_SUPPRESSION_FILE}")
  else()
    message(WARNING "AddressSanitizer suppression file not found: ${ASAN_SUPPRESSION_FILE}")
  endif()
  set(LSAN_SUPPRESSION_FILE "${CMAKE_SOURCE_DIR}/cmake/lsan_suppressions.supp")
  if(EXISTS "${LSAN_SUPPRESSION_FILE}")
    set(ENV_LSAN_OPTIONS "LSAN_OPTIONS=suppressions=${LSAN_SUPPRESSION_FILE}")
    message(STATUS "LeakSanitizer suppression enabled: ${LSAN_SUPPRESSION_FILE}")
  else()
    message(WARNING "LeakSanitizer suppression file not found: ${LSAN_SUPPRESSION_FILE}")
  endif()
  set_property(GLOBAL PROPERTY _GR_TEST_ENV "${ENV_ASAN_OPTIONS};${ENV_LSAN_OPTIONS}")
endif()

if(EMSCRIPTEN OR NOT GR_ENABLE_BLOCK_REGISTRY)
  set(INTERNAL_ENABLE_BLOCK_PLUGINS OFF)
endif()

if((ADDRESS_SANITIZER AND UB_SANITIZER)
   OR (ADDRESS_SANITIZER AND THREAD_SANITIZER)
   OR (UB_SANITIZER AND THREAD_SANITIZER))
  message(FATAL_ERROR "Only one sanitizer can be enabled at a time.")
endif()

include(cmake/CompilerWarnings.cmake)

# blocklib_generator is a build process dependency, add it before everything
add_subdirectory(blocklib_generator)

add_library(gnuradio-options INTERFACE)

message(
  STATUS
    "Is block registry enabled? (faster compile-times and when runtime or Python wrapping APIs are not required) ${GR_ENABLE_BLOCK_REGISTRY}"
)
message(STATUS "Is plugin system enabled? ${INTERNAL_ENABLE_BLOCK_PLUGINS}")

# clang-scan-deps in required by ccache
if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
  string(
    REGEX MATCH
          "^[0-9]+"
          CLANG_MAJOR
          "${CMAKE_CXX_COMPILER_VERSION}")
  find_program(CLANG_SCAN_DEPS_EXEC NAMES clang-scan-deps-${CLANG_MAJOR} clang-scan-deps)
  if(CLANG_SCAN_DEPS_EXEC)
    set(CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS
        "${CLANG_SCAN_DEPS_EXEC}"
        CACHE FILEPATH "" FORCE)
  else()
    message(WARNING "clang‑scan‑deps not found; module scanning will be disabled")
    set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
  endif()
endif()

# Use ccache if found and enabled
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM AND USE_CCACHE)
  message(STATUS "ccache found and will be used")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
else()
  message(STATUS "ccache will not be used")
endif()

# Prefer forced colored compiler output if there's a chance we'd otherwise get none at all. Ninja and ccache "consume"
# the compiler output, breaking the terminal detection of compilers. ccache tries to solve the problem, but can only do
# so if it determines that it's calling GCC or Clang. It uses a very lightweight heuristic, which breaks easily.
if((CMAKE_GENERATOR STREQUAL "Ninja" OR (CCACHE_PROGRAM AND USE_CCACHE)) AND NOT DEFINED CMAKE_COLOR_DIAGNOSTICS)
  message(
    STATUS
      "Forcing compiler color output due to the use of Ninja and/or ccache. Use -DCMAKE_COLOR_DIAGNOSTICS=OFF to turn it off."
  )
  set(CMAKE_COLOR_DIAGNOSTICS ON)
endif()

set(CMAKE_EXT_DEP_WARNING_GUARD "")
if(DISABLE_EXTERNAL_DEPS_WARNINGS) # enable warnings for external dependencies
  set(CMAKE_EXT_DEP_WARNING_GUARD SYSTEM)
endif()

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
string(
  REPLACE ";"
          " "
          ALL_COMPILER_FLAGS
          "${ALL_COMPILER_FLAGS}")

if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang") # set default C++ STL to Clang's libc++ when using Clang
  add_compile_options(-stdlib=libc++)
  if(TIMETRACE)
    add_compile_options(-ftime-trace)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -ftime-trace")
    message(STATUS "Enable TIMETRACE: ${TIMETRACE}")
  endif()
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libc++ -lc++")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(ENABLE_TBB)
    find_package(TBB REQUIRED)
    target_link_libraries(gnuradio-options INTERFACE TBB::tbb)
  else()
    target_compile_definitions(gnuradio-options INTERFACE _GLIBCXX_USE_TBB_PAR_BACKEND=0)
  endif()
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "(Clang|GNU)")
  if(ADDRESS_SANITIZER)
    set(SANITIZER_FLAGS
        -fsanitize=address
        -fsanitize-address-use-after-scope
        -fsanitize=leak
        -fno-omit-frame-pointer
        -fstack-protector-strong)
  elseif(UB_SANITIZER)
    set(SANITIZER_FLAGS
        -fsanitize=undefined
        -fsanitize-address-use-after-scope
        -fsanitize=leak
        -fno-omit-frame-pointer
        -fstack-protector-strong)
  elseif(THREAD_SANITIZER)
    set(SANITIZER_FLAGS -fsanitize=thread -fsanitize=thread)
  endif()

  if(DEFINED SANITIZER_FLAGS)
    add_compile_options(${SANITIZER_FLAGS} -fno-omit-frame-pointer -fstack-protector-strong)
    if(NOT EMSCRIPTEN)
      add_compile_options(${SANITIZER_FLAGS} -fstack-clash-protection)
    endif()
    add_link_options(${SANITIZER_FLAGS})
    message(STATUS "Enabled sanitizer flags: ${SANITIZER_FLAGS}")
  endif()
endif()

# Include What You Use tooling: https://github.com/include-what-you-use/include-what-you-use
find_program(INCLUDE_WHAT_YOU_USE_TOOL_PATH NAMES include-what-you-use iwyu)
if(INCLUDE_WHAT_YOU_USE_TOOL_PATH)
  message(" using 'Include What You Use' path: (${INCLUDE_WHAT_YOU_USE_TOOL_PATH})")
  set_property(GLOBAL PROPERTY CMAKE_CXX_INCLUDE_WHAT_YOU_USE ${INCLUDE_WHAT_YOU_USE_TOOL_PATH})
endif()

set_project_warnings(gnuradio-options)

if(EMSCRIPTEN)
  set(CMAKE_EXECUTABLE_SUFFIX ".js")
  add_compile_options(-fwasm-exceptions -pthread -g)
  add_link_options(
    "SHELL:-s ALLOW_MEMORY_GROWTH=1"
    "SHELL:-s ASSERTIONS=1"
    "SHELL:-s INITIAL_MEMORY=256MB"
    # "SHELL:-s SAFE_HEAP=1" # additional for debug "SHELL:-s ASSERTIONS=2" # additional for debug "SHELL:-s
    # STACK_OVERFLOW_CHECK=2" # additional for debug "SHELL:-g" # additional for debug "SHELL:-gsource-map" # additional
    # for debug "SHELL:--profiling-funcs" # additional for debug "SHELL:--emit-symbol-map" # additional for debug
    -fwasm-exceptions
    -pthread
    "SHELL:-s PTHREAD_POOL_SIZE=60"
    "SHELL:-s FETCH=1"
    "SHELL:-s WASM=1" # output as web-assembly
  )
endif()

# include header-only libraries that have been inlined to simplify builds w/o requiring access to the internet
if(NOT (TARGET magic_enum))
  add_library(magic_enum INTERFACE)
  target_include_directories(magic_enum ${CMAKE_EXT_DEP_WARNING_GUARD}
                             INTERFACE ${PROJECT_SOURCE_DIR}/third_party/magic_enum/)

  set(magic_enum_public_headers third_party/magic_enum/magic_enum.hpp third_party/magic_enum/magic_enum_utility.hpp)
  set_target_properties(magic_enum PROPERTIES PUBLIC_HEADER "${magic_enum_public_headers}")
  install(TARGETS magic_enum PUBLIC_HEADER DESTINATION include)
endif()

# include exprtk header-only libraries available as a statically linked library to simplify/speed-up builds
add_library(
  exprtk STATIC "${CMAKE_CURRENT_SOURCE_DIR}/third_party/exprtk.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/third_party/exprtk.cpp" # dummy source file
)
target_include_directories(exprtk PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
                                         "${CMAKE_CURRENT_SOURCE_DIR}/third_party/exprtk")
if(EMSCRIPTEN)
  target_compile_options(exprtk PRIVATE -O1 -g0)
elseif()
  target_compile_options(exprtk PRIVATE -O1 -g0 -fvisibility=hidden)
endif()
set_source_files_properties(third_party/exprtk.cpp PROPERTIES COMPILE_OPTIONS "-fvisibility=hidden")
# include exprtk header-only - END

include(FetchContent)
FetchContent_Declare(
  pmt
  GIT_REPOSITORY https://github.com/gnuradio/pmt.git
  GIT_TAG e1a46cb61decb044f6ab0a58a77211beb3630340 # latest as of 2023-12-06
)

set(BOOST_UT_DISABLE_MODULE
    ON
    CACHE BOOL "Disable UT Module Support" FORCE)
FetchContent_Declare(
  ut
  GIT_REPOSITORY https://github.com/boost-ext/ut.git
  GIT_TAG 53e17f25119598c6458d30351b260193096ba67e # latest tag as of 2023-04-02
)

FetchContent_Declare(
  vir-simd
  GIT_REPOSITORY https://github.com/mattkretz/vir-simd.git
  GIT_TAG v0.4.4)

FetchContent_Declare(
  cpp-httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG v0.18.1)

FetchContent_MakeAvailable(
  pmt
  ut
  vir-simd
  cpp-httplib)

# Fetch SoapySDR -- needed since the distribution version is incompatible w.r.t. stdlibc++ vs. libc++
if(CMAKE_CXX_COMPILER_ID MATCHES "(GNU|Clang)") # WIP
  find_package(SoapySDR CONFIG)
endif()

add_library(pmtv INTERFACE)
target_include_directories(pmtv INTERFACE ${pmt_SOURCE_DIR}/include/)
target_link_libraries(pmtv INTERFACE)

install(DIRECTORY ${pmt_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_library(vir INTERFACE)
target_include_directories(vir INTERFACE ${vir-simd_SOURCE_DIR}/)

# FFTW3 is build 2 times for float and double precisions
set(FFTW_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/fftw)
if(EMSCRIPTEN)
  set(FFTW_CFLAGS "${CFLAGS} -fPIC -w -O2 -g0 -ffunction-sections -fdata-sections")
  set(FFTW_CONFIG
      cd
      ${FFTW_PREFIX}/src/
      &&
      emconfigure
      ./configure
      --enable-silent-rules
      --quiet
      --disable-fortran
      --prefix=${FFTW_PREFIX}/install)
  set(FFTW_BUILD
      emmake
      make
      -j
      CFLAGS=${FFTW_CFLAGS}
      --silent
      V=0
      &&
      emmake
      make
      install
      --silent
      V=0
      &&
      emmake
      make
      clean
      --silent
      V=0)
else()
  set(FFTW_CFLAGS "${CFLAGS} -fPIC -w -O2 -g0 -march=native -mtune=native -ffunction-sections -fdata-sections")
  set(FFTW_CONFIG
      ${FFTW_PREFIX}/src/configure
      --enable-silent-rules
      --quiet
      --disable-fortran
      --prefix=${FFTW_PREFIX}/install)
  set(FFTW_BUILD
      make
      -j
      CFLAGS=${FFTW_CFLAGS}
      --silent
      V=0
      &&
      make
      install
      --silent
      V=0
      &&
      make
      clean
      --silent
      V=0)
endif()
set(FFTW_INSTALL_COMMAND ${FFTW_CONFIG} && ${FFTW_BUILD} && ${FFTW_CONFIG} --enable-float && ${FFTW_BUILD})

include(ExternalProject)
ExternalProject_Add(
  fftw_ext
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
  LOG_DOWNLOAD ON)

add_library(fftw INTERFACE)
target_link_libraries(
  fftw
  INTERFACE fftw3
  INTERFACE fftw3f
  INTERFACE m)
target_include_directories(fftw INTERFACE ${FFTW_PREFIX}/install/include ${PROJECT_BINARY_DIR})
target_link_directories(
  fftw
  INTERFACE
  ${FFTW_PREFIX}/install/lib
  ${FFTW_PREFIX}/install/lib64)
add_dependencies(fftw fftw_ext)

# check for CPython and Numpy dependencies
set(PYTHON_FORCE_INCLUDE OFF)
if(PYTHON_FORCE_INCLUDE)
  find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter Development NumPy)
else()
  find_package(Python3 3.12 COMPONENTS Interpreter Development NumPy)
endif()

set(PYTHON_AVAILABLE OFF)
if(Python3_FOUND AND NOT EMSCRIPTEN)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckNumPy.py"
    RESULT_VARIABLE NUMPY_NOT_FOUND
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  # If NumPy is found, set PYTHON_AVAILABLE to ON
  if(NOT NUMPY_NOT_FOUND)
    set(PYTHON_AVAILABLE ON)
    include_directories(${Python3_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR})
    add_definitions(-DPYTHON_AVAILABLE)
    message(STATUS "Using Python Include Dirs: ${Python3_INCLUDE_DIRS} and ${NUMPY_INCLUDE_DIR}")
  else()
    message(STATUS "Python and Numpy Include headers not found!!")
  endif()
endif()

add_subdirectory(bench) # custom ut addon for microbenchmarking

add_subdirectory(core)
add_subdirectory(meta)
add_subdirectory(algorithm)
add_subdirectory(blocks)

if(ENABLE_TESTING
   AND (UNIX OR APPLE)
   AND ENABLE_COVERAGE)
  message("Coverage reporting enabled")
  include(cmake/CodeCoverage.cmake) # https://github.com/bilke/cmake-modules/blob/master/CodeCoverage.cmake #
  # (License: BSL-1.0)
  target_compile_options(
    gnuradio-options
    INTERFACE --coverage
              -Og
              -g1
              -gz
    # -gdwarf-2 -gstrict-dwarf -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0
  ) # fortify_source is not possible without optimization
  target_link_libraries(gnuradio-options INTERFACE --coverage)
  append_coverage_compiler_flags()
  set(GCOVR_ADDITIONAL_ARGS "--merge-mode-functions=merge-use-line-min"
                            "--gcov-ignore-parse-errors=negative_hits.warn_once_per_file")
  setup_target_for_coverage_gcovr_xml(
    NAME
    coverage
    EXECUTABLE
    ctest
    EXECUTABLE_ARGS
    "--output-on-failure"
    DEPENDENCIES
    qa_buffer
    qa_DataSink
    qa_DynamicPort
    qa_DynamicBlock
    qa_HierBlock
    qa_filter
    qa_Settings
    qa_Tags
    qa_Scheduler
    qa_thread_pool
    qa_thread_affinity
    qa_YamlPmt
    EXCLUDE
    "$CMAKE_BUILD_DIR/*")
  setup_target_for_coverage_gcovr_html(
    NAME
    coverage_html
    EXECUTABLE
    ctest
    EXECUTABLE_ARGS
    "--output-on-failure"
    DEPENDENCIES
    qa_buffer
    qa_DataSink
    qa_DynamicPort
    qa_DynamicBlock
    qa_HierBlock
    qa_filter
    qa_Settings
    qa_Tags
    qa_Scheduler
    qa_thread_pool
    qa_thread_affinity
    qa_YamlPmt
    EXCLUDE
    "$CMAKE_BUILD_DIR/*")
endif()

# Configure and install the pkg-config file TODO: This should become more granular - .pc for each in-tree module
configure_file(cmake/gnuradio4.pc.in ${CMAKE_CURRENT_BINARY_DIR}/gnuradio4.pc @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/gnuradio4.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig # Standard location
                                                                                                     # for .pc files
)

# print effective options
message(STATUS "================================================")
message(STATUS "GR-4 build configuration")
message(STATUS "------------------------------------------------")
message(STATUS "GR_TOPLEVEL_PROJECT          : ${GR_TOPLEVEL_PROJECT}")
message(STATUS "GR_ENABLE_BLOCK_REGISTRY     : ${GR_ENABLE_BLOCK_REGISTRY}")
message(STATUS "INTERNAL_ENABLE_BLOCK_PLUGINS: ${INTERNAL_ENABLE_BLOCK_PLUGINS}")
message(STATUS "USE_CCACHE                   : ${USE_CCACHE}")
message(STATUS "WARNINGS_AS_ERRORS           : ${WARNINGS_AS_ERRORS}")
message(STATUS "TIMETRACE                    : ${TIMETRACE}")
message(STATUS "ADDRESS_SANITIZER            : ${ADDRESS_SANITIZER}")
message(STATUS "UB_SANITIZER                 : ${UB_SANITIZER}")
message(STATUS "THREAD_SANITIZER             : ${THREAD_SANITIZER}")
message(STATUS "ENABLE_TBB                   : ${ENABLE_TBB}")
message(STATUS "ENABLE_EXAMPLES              : ${ENABLE_EXAMPLES}")
message(STATUS "ENABLE_TESTING               : ${ENABLE_TESTING}")
message(STATUS "ENABLE_COVERAGE              : ${ENABLE_COVERAGE}")
message(STATUS "------------------------------------------------")

string(TOUPPER "${CMAKE_BUILD_TYPE}" _BT)
set(_config_cxx_flags "${CMAKE_CXX_FLAGS_${_BT}}")
set(_config_link_flags "${CMAKE_EXE_LINKER_FLAGS_${_BT}}")
set(_base_cxx_flags "${CMAKE_CXX_FLAGS}")
set(_base_link_flags "${CMAKE_EXE_LINKER_FLAGS}")
get_directory_property(_dir_compile_opts COMPILE_OPTIONS)
get_directory_property(_dir_link_opts LINK_OPTIONS)
get_target_property(_t_copts gnuradio-options INTERFACE_COMPILE_OPTIONS)
get_target_property(_t_lopts gnuradio-options INTERFACE_LINK_OPTIONS)
if(_t_lopts STREQUAL "_t_lopts-NOTFOUND")
  unset(_t_lopts)
endif()

string(
  JOIN
  " "
  _final_compile_opts
  ${_base_cxx_flags}
  ${_config_cxx_flags}
  ${_dir_compile_opts}
  ${_t_copts})
string(
  JOIN
  " "
  _final_link_opts
  ${_base_link_flags}
  ${_config_link_flags}
  ${_dir_link_opts}
  ${_t_lopts})

message(STATUS "Effective compiler flags : ${_final_compile_opts}")
message(STATUS "Effective linker flags   : ${_final_link_opts}")
message(STATUS "------------------------------------------------")
message(STATUS "Toolchain details")
message(STATUS "------------------------------------------------")
message(STATUS "C++ compiler ID          : ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "C++ compiler path        : ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ compiler version     : ${CMAKE_CXX_COMPILER_VERSION}")
string(
  REGEX MATCH
        "-fuse-ld=([a-zA-Z0-9._-]+)"
        _linker_backend
        "${_final_link_opts}")
if(_linker_backend MATCHES "-fuse-ld=([a-zA-Z0-9._-]+)")
  set(_detected_linker "${CMAKE_CXX_COMPILER} with ${CMAKE_MATCH_1}")
else()
  set(_detected_linker "${CMAKE_LINKER} (default/provided linker)")
endif()
message(STATUS "Linker                   : ${_detected_linker}")

if(CCACHE_PROGRAM)
  execute_process(
    COMMAND ${CCACHE_PROGRAM} --version
    OUTPUT_VARIABLE CCACHE_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  string(
    REGEX MATCH
          "[^\n]*"
          CCACHE_VERSION_LINE
          "${CCACHE_VERSION}")
  message(STATUS "ccache                   : ${CCACHE_VERSION_LINE}")
else()
  message(STATUS "ccache                   : not found or disabled")
endif()

message(STATUS "================================================")
