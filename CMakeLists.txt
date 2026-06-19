cmake_minimum_required(VERSION 3.27)

project(
  gnuradio
  VERSION 4.0.0
  LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(GR4_PKG_VERSION
    "${PROJECT_VERSION}-git"
    CACHE STRING "Version string used in gnuradio4.pc")

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

# limit parallel compile/link jobs to prevent OOM with template-heavy TUs override with -DGR_BUILD_PARALLEL_LEVEL=<N>
# (CI with more RAM can raise this)
set(GR_BUILD_PARALLEL_LEVEL
    "6"
    CACHE STRING "Maximum parallel compile/link jobs")
option(GR4_USE_THIN_ARCHIVES "Build static archives as thin archives (non-relocatable when installed)" OFF)
option(GR4_USE_LIBCXX "Force libc++ when compiling with Clang" ON)
if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL APPEND PROPERTY JOB_POOLS compile_pool=${GR_BUILD_PARALLEL_LEVEL} link_pool=2)
  set(CMAKE_JOB_POOL_COMPILE compile_pool)
  set(CMAKE_JOB_POOL_LINK link_pool)
endif()

if(CMAKE_CXX_COMPILER MATCHES "/em\\+\\+(-[a-zA-Z0-9.])?$") # if this hasn't been set before via e.g. emcmake
  message(" Transpiling to WASM: using: Emscripten (${CMAKE_CXX_COMPILER})")
  set(EMSCRIPTEN ON)
endif()

# default explicit compiler and linker flag optimisation per build type
if(NOT MSVC)
  # optional:  -fno-inline-functions -fvar-tracking-assignments
  set(CMAKE_CXX_FLAGS_DEBUG "-Og -g1 -gz -DDEBUG -fno-omit-frame-pointer -ffunction-sections -fdata-sections")
  set(CMAKE_CXX_FLAGS_RELEASE "-O2 -g0 -DNDEBUG -ffunction-sections -fdata-sections")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g1 -gz -DNDEBUG -ffunction-sections -fdata-sections")
  set(CMAKE_CXX_FLAGS_RELWITHASSERT "-O2 -DASSERT_ENABLED -ffunction-sections -fdata-sections")
  # '-s': strip debug symbols, 'EMBEDDED is used to disable/minimise code features for embedded systems (e.g. code-size,
  # console printouts, etc.)
  set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -g0 -DNDEBUG -DEMBEDDED -ffunction-sections -fdata-sections")
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} -s")
  endif()

  if(GR4_USE_THIN_ARCHIVES)
    set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> crT <TARGET> <LINK_FLAGS> <OBJECTS>")
    set(CMAKE_CXX_ARCHIVE_APPEND "<CMAKE_AR> rT <TARGET> <LINK_FLAGS> <OBJECTS>")
  else()
    set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> cr <TARGET> <LINK_FLAGS> <OBJECTS>")
    set(CMAKE_CXX_ARCHIVE_APPEND "<CMAKE_AR> r <TARGET> <LINK_FLAGS> <OBJECTS>")
  endif()

  set(_LD_GC_SECTIONS "-Wl,--gc-sections -ffunction-sections -fdata-sections")

  if(APPLE)
    # macOS Mach-O two-level namespace duplicates singletons (e.g. globalBlockRegistry()) when a static library is
    # linked into both the executable and shared plugins; -flat_namespace forces ELF-like single-level lookup ensuring
    # one instance. -stack_size 8 MiB matches the Linux default — macOS ARM64 non-main threads default to 512 KiB.
    string(APPEND _LD_GC_SECTIONS " -Wl,-flat_namespace")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-stack_size,0x800000")
  endif()

  # prefer mold > lld > system linker (Linux/non-Emscripten only) --icf=safe (Identical Code Folding) is disabled under
  # sanitizers to avoid merging functions that sanitiser instrumentation needs to track separately
  set(_USE_ICF TRUE)
  if(ADDRESS_SANITIZER
     OR UB_SANITIZER
     OR THREAD_SANITIZER)
    set(_USE_ICF FALSE)
  endif()

  # user-facing override: USE_MOLD_LINKER=OFF forces fall-through to lld/default even if mold is installed.
  option(USE_MOLD_LINKER "Use mold as the linker when available (falls back to lld, then default)" ON)

  if(NOT
     CMAKE_SYSTEM_NAME
     STREQUAL
     "Emscripten")
    set(_mold_usable FALSE)
    if(USE_MOLD_LINKER)
      find_program(MOLD_BIN ld.mold)
      if(MOLD_BIN)
        # mold < 3.0 does not parse the AS_NEEDED() directive used by some modern libatomic stubs (observed with GCC
        # 16's libatomic_asneeded.so; affects any compiler toolchain whose runtime ships an AS_NEEDED linker-script
        # stub). Auto-fall-through to lld / default in that case.
        execute_process(
          COMMAND "${MOLD_BIN}" --version
          OUTPUT_VARIABLE _mold_version_out
          OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(_mold_version_out MATCHES "mold ([0-9]+)\\.([0-9]+)")
          if(CMAKE_MATCH_1 VERSION_LESS 3)
            message(
              STATUS "Skipping mold ${CMAKE_MATCH_1}.${CMAKE_MATCH_2} (AS_NEEDED() support requires mold >= 3.0).")
          else()
            set(_mold_usable TRUE)
          endif()
        else()
          # unknown format — assume usable, user can override via -DUSE_MOLD_LINKER=OFF
          set(_mold_usable TRUE)
        endif()
      endif()
    endif()

    if(_mold_usable)
      set(_LD_GC_SECTIONS "${_LD_GC_SECTIONS} -fuse-ld=mold")
      if(_USE_ICF)
        string(APPEND _LD_GC_SECTIONS " -Wl,--icf=safe")
      endif()
      message(STATUS "Using mold @ ${MOLD_BIN} as the linker.")
    else()
      find_program(LLD_BIN ld.lld)
      if(LLD_BIN)
        set(_LD_GC_SECTIONS "${_LD_GC_SECTIONS} -fuse-ld=lld")
        if(_USE_ICF)
          string(APPEND _LD_GC_SECTIONS " -Wl,--icf=safe")
        endif()
        message(STATUS "Using lld @ ${LLD_BIN} as the linker.")
      else()
        message(STATUS "Using default system linker (mold and lld unavailable).")
      endif()
    endif()
  endif()
  set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${_LD_GC_SECTIONS}")
  set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${_LD_GC_SECTIONS}")
  set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${_LD_GC_SECTIONS}")
  set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${_LD_GC_SECTIONS}")
  set(CMAKE_EXE_LINKER_FLAGS_MINSIZEREL "${_LD_GC_SECTIONS}")
  set(CMAKE_SHARED_LINKER_FLAGS_MINSIZEREL "${_LD_GC_SECTIONS}")
  if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # we need to set -Wa,-mbig-obj or build fails with gcc
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wa,-mbig-obj")
    # on windows gcc requires stdc++exp to link for <format> and <print> functions
    set(CMAKE_CXX_STANDARD_LIBRARIES "${CMAKE_CXX_STANDARD_LIBRARIES} -lstdc++exp")
  endif()
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

# Determine if GR4 is built as a subproject (via add_subdirectory) or as the master project. Backed by CMake's built-in
# PROJECT_IS_TOP_LEVEL (3.21+); kept as named variable for downstream overrides and pre-existing references throughout
# the tree.
if(NOT DEFINED GR_TOPLEVEL_PROJECT)
  set(GR_TOPLEVEL_PROJECT ${PROJECT_IS_TOP_LEVEL})
  if(GR_TOPLEVEL_PROJECT)
    message(STATUS "CMake version: ${CMAKE_VERSION}")
  endif()
endif()

set(GR_USE_ADAPTIVE_CPP OFF) # Detect AdaptiveCpp via compiler executable name
if(CMAKE_CXX_COMPILER MATCHES ".*acpp.*" OR CMAKE_CXX_COMPILER MATCHES ".*adaptivecpp.*")
  set(GR_USE_ADAPTIVE_CPP ON)
  message(STATUS "Detected AdaptiveCpp based on compiler path: ${CMAKE_CXX_COMPILER}")
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
option(ENABLE_BENCHMARKS "Enable Benchmark Builds" ${GR_TOPLEVEL_PROJECT})
option(PERMISSIVE_LICENSED_ONLY "Exclude modules with strong-copyleft (e.g. GPLv3) licences" OFF)

if(NOT DEFINED GR_DATA_CACHE_DIR)
  set(GR_DATA_CACHE_DIR
      "${CMAKE_CURRENT_BINARY_DIR}/gnuradio_cache"
      CACHE PATH "Writeable system path where GR should store its cached data (for example, /var/lib/gnuradio4/cache)")
endif()

set(GR_MAX_WASM_THREAD_COUNT
    60
    CACHE STRING "Max number of threads for WASM pthread pool")
add_compile_definitions(GR_MAX_WASM_THREAD_COUNT=${GR_MAX_WASM_THREAD_COUNT}) # propagate to C++
set(GR_ENABLE_HTTP
    "ON"
    CACHE
      STRING
      "HTTP client support (libcurl wrapper): ON=enable and require libcurl (error if not found), OFF=disable, OPTIONAL=enable only if libcurl is found, otherwise disable"
)
set(GR4_PKGCONFIG_EXTRA_LIBS "")
set_property(
  CACHE GR_ENABLE_HTTP
  PROPERTY STRINGS
           ON
           OFF
           OPTIONAL)

if(ENABLE_TESTING
   AND (UNIX
        OR APPLE
        OR WIN32))
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

if(GR_USE_ADAPTIVE_CPP
   AND ((ADDRESS_SANITIZER
         OR UB_SANITIZER
         OR THREAD_SANITIZER)))
  message(
    FATAL_ERROR
      "SYCL is incompatible with ADDRESS_SANITIZER=${ADDRESS_SANITIZER} or UB_SANITIZER=${UB_SANITIZER} or THREAD_SANITIZER=${THREAD_SANITIZER}."
  )
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(cmake/CompilerWarnings.cmake)
include(cmake/Dependencies.cmake)

gr4_declare_dependency(
  NAME
  TBB
  FIND_PACKAGE_NAME
  TBB
  FIND_PACKAGE_TARGETS
  TBB::tbb
  MODES
  system
  TYPE
  OPTIONAL
  DESCRIPTION
  "Intel Threading Building Blocks"
  PURPOSE
  "std::execution::par backend with libstdc++ (ENABLE_TBB)"
  URL
  "https://github.com/uxlfoundation/oneTBB")

# Drift check between gr4_declare_dependency() registry and DEPENDENCIES.md. Runs on every build (lightweight regex scan
# of two files); top-level only, so consuming projects via add_subdirectory() are not affected.
if(GR_TOPLEVEL_PROJECT
   AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/lint_dependencies.cmake"
   AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/DEPENDENCIES.md")
  add_custom_target(
    lint-dependencies ALL
    COMMAND ${CMAKE_COMMAND} -DROOT=${CMAKE_CURRENT_SOURCE_DIR} -P
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/lint_dependencies.cmake
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Checking DEPENDENCIES.md vs gr4_declare_dependency() registry"
    VERBATIM)
endif()

# blocklib_generator is a build process dependency, add it before everything
add_subdirectory(blocklib_generator)

add_library(gnuradio-options INTERFACE)
add_library(gnuradio4::gnuradio-options ALIAS gnuradio-options)
install(TARGETS gnuradio-options EXPORT gnuradio4Targets)

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

if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
  if(NOT GR_USE_ADAPTIVE_CPP AND GR4_USE_LIBCXX)
    message(STATUS "Using plain Clang – enabling libc++ (GR4_USE_LIBCXX=ON)")
    add_compile_options(-stdlib=libc++ -Wno-unused-command-line-argument)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libc++ -lc++")
  elseif(NOT GR_USE_ADAPTIVE_CPP)
    message(STATUS "Using plain Clang – using default C++ standard library")
  else()
    message(STATUS "AdaptiveCpp detected – skipping libc++ settings")
  endif()
  if(TIMETRACE)
    add_compile_options(-ftime-trace)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -ftime-trace")
    message(STATUS "Enable TIMETRACE: ${TIMETRACE}")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(ENABLE_TBB)
    message(
      STATUS
        "ENABLE_TBB=ON requires system TBB. Install libtbb-dev/tbb-devel/tbb; use -DENABLE_TBB=OFF to disable the libstdc++ parallel STL backend."
    )
    gr4_resolve_dependency(TBB REQUIRED)
    target_link_libraries(gnuradio-options INTERFACE TBB::tbb)
    set(GR4_PKGCONFIG_EXTRA_LIBS "-ltbb")
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
  add_compile_options(-fexceptions -pthread -g)
  add_link_options(
    "SHELL:-s ALLOW_MEMORY_GROWTH=1"
    "SHELL:-s ASSERTIONS=1"
    "SHELL:-s INITIAL_MEMORY=256MB"
    "SHELL:-s STACK_SIZE=4194304"
    # "SHELL:-s SAFE_HEAP=1" # additional for debug "SHELL:-s ASSERTIONS=2" # additional for debug "SHELL:-s
    # STACK_OVERFLOW_CHECK=2" # additional for debug "SHELL:-g" # additional for debug "SHELL:-gsource-map" # additional
    # for debug "SHELL:--profiling-funcs" # additional for debug "SHELL:--emit-symbol-map" # additional for debug
    -fexceptions
    -pthread
    "SHELL:-s PTHREAD_POOL_SIZE=${GR_MAX_WASM_THREAD_COUNT}"
    "SHELL:-s PTHREAD_POOL_SIZE_STRICT=2"
    "SHELL:-s FETCH=1"
    "SHELL:-s WASM=1" # output as web-assembly
    "SHELL:-s EXPORTED_RUNTIME_METHODS=['ccall']"
    "SHELL:-s ENVIRONMENT=web,worker,node")
endif()

# include header-only libraries that have been inlined to simplify builds w/o requiring access to the internet
if(NOT (TARGET magic_enum))
  add_library(magic_enum INTERFACE)
  add_library(gnuradio4::magic_enum ALIAS magic_enum)
  target_include_directories(
    magic_enum ${CMAKE_EXT_DEP_WARNING_GUARD} INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/third_party/magic_enum/>
                                                        $<INSTALL_INTERFACE:include>)

  set(magic_enum_public_headers third_party/magic_enum/magic_enum.hpp third_party/magic_enum/magic_enum_utility.hpp)
  set_target_properties(magic_enum PROPERTIES PUBLIC_HEADER "${magic_enum_public_headers}")
  install(
    TARGETS magic_enum
    EXPORT gnuradio4Targets
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

# include exprtk header-only libraries available as a statically linked library to simplify/speed-up builds
add_library(
  exprtk STATIC "${CMAKE_CURRENT_SOURCE_DIR}/third_party/exprtk.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/third_party/exprtk.cpp" # dummy source file
)
add_library(gnuradio4::exprtk ALIAS exprtk)
target_include_directories(exprtk PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party>
                                         $<INSTALL_INTERFACE:include>)
if(EMSCRIPTEN)
  target_compile_options(exprtk PRIVATE -O1 -g0)
else()
  target_compile_options(exprtk PRIVATE -O1 -g0 -fvisibility=hidden)
endif()
set_source_files_properties(third_party/exprtk.cpp PROPERTIES COMPILE_OPTIONS "-fvisibility=hidden")
install(TARGETS exprtk EXPORT gnuradio4Targets)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/third_party/exprtk.hpp" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
# include exprtk header-only - END

# All external dependencies are declared centrally in cmake/Dependencies.cmake (already included). Resolution is
# mode-driven (system | fetch | system-or-fetch); subdirectories that own optional features call
# gr4_resolve_dependency() locally and self-skip when the dep is unavailable.

gr4_declare_dependency(
  NAME
  ut
  FIND_PACKAGE_NAME
  ut
  FETCH_GIT
  https://github.com/boost-ext/ut.git
  FETCH_TAG
  53e17f25119598c6458d30351b260193096ba67e
  FETCH_VARIABLES
  BOOST_UT_DISABLE_MODULE=ON
  TYPE
  OPTIONAL
  DESCRIPTION
  "Boost.UT C++20 unit-test framework"
  PURPOSE
  "GR4 unit tests and benchmarks (ENABLE_TESTING/ENABLE_BENCHMARKS)"
  URL
  "https://github.com/boost-ext/ut")

gr4_declare_dependency(
  NAME
  vir-simd
  FIND_PACKAGE_NAME
  vir-simd
  FIND_PACKAGE_ARGS
  CONFIG
  FIND_PACKAGE_TARGETS
  vir-simd::vir-simd
  FETCH_GIT
  https://github.com/mattkretz/vir-simd.git
  FETCH_TAG
  v0.4.4
  TYPE
  REQUIRED
  DESCRIPTION
  "C++ SIMD wrapper / std::simd shim"
  PURPOSE
  "core SIMD operations across all GR4 modules"
  URL
  "https://github.com/mattkretz/vir-simd")

gr4_declare_dependency(
  NAME
  cpp-httplib
  FIND_PACKAGE_NAME
  httplib
  FETCH_GIT
  https://github.com/yhirose/cpp-httplib.git
  FETCH_TAG
  v0.18.1
  TYPE
  OPTIONAL
  DESCRIPTION
  "Header-only HTTP client/server"
  PURPOSE
  "HTTP-based unit tests (ENABLE_TESTING)"
  URL
  "https://github.com/yhirose/cpp-httplib")

if(NOT EMSCRIPTEN)
  gr4_declare_dependency(
    NAME
    libsoundio
    FIND_PACKAGE_NAME
    LibSoundIo
    FIND_PACKAGE_TARGETS
    libsoundio_static
    FETCH_GIT
    https://github.com/andrewrk/libsoundio.git
    FETCH_TAG
    49a1f78b50eb0f5a49d096786a95a93874a2592a
    FETCH_PATCH_COMMAND
    git
    apply
    --ignore-space-change
    --ignore-whitespace
    "${CMAKE_CURRENT_SOURCE_DIR}/patches/libsoundio-cmake4.diff"
    ||
    true
    FETCH_VARIABLES
    BUILD_STATIC_LIBS=ON
    BUILD_DYNAMIC_LIBS=OFF
    BUILD_EXAMPLE_PROGRAMS=OFF
    BUILD_TESTS=OFF
    ENABLE_JACK=OFF
    TYPE
    OPTIONAL
    DESCRIPTION
    "Cross-platform audio I/O"
    PURPOSE
    "blocks/audio source/sink (system shared lib via cmake/FindLibSoundIo.cmake, or patched fetch)"
    URL
    "http://libsound.io")

  # cpr is fetch-only by deliberate choice: upstream cprConfigVersion.cmake uses ExactVersion policy, so a system
  # minimum-version probe rejects newer-and-thus-safe installs too; Debian/Ubuntu still ship pre-1.12.0 cpr (no
  # cpr::BodyView, used in algorithm/fileio/FileIo.hpp); and we patch cpr on the fetch path (see FETCH_PATCH_COMMAND)
  # plus force ABI-shared to avoid ODR violations across plugin shared libraries, neither of which a system build is
  # guaranteed to match.
  gr4_declare_dependency(
    NAME
    cpr
    FIND_PACKAGE_NAME
    cpr
    FIND_PACKAGE_ARGS
    CONFIG
    FIND_PACKAGE_TARGETS
    cpr::cpr
    MODES
    fetch
    FETCH_GIT
    https://github.com/libcpr/cpr.git
    FETCH_TAG
    1.14.1
    FETCH_PATCH_COMMAND
    git
    apply
    --ignore-space-change
    --ignore-whitespace
    "${CMAKE_CURRENT_SOURCE_DIR}/patches/cpr-disable-std-fs-test.diff"
    ||
    true
    FETCH_VARIABLES
    CPR_USE_SYSTEM_CURL=ON
    FETCH_INCLUDE_IN_ALL
    ON
    TYPE
    OPTIONAL
    DESCRIPTION
    "C++ HTTP/REST client (libcurl wrapper)"
    PURPOSE
    "native HTTP(S) for gnuradio-algorithm fileio; patched + ABI-shared (fetch-only); emits its own cprConfig for downstream find_dependency"
    URL
    "https://github.com/libcpr/cpr")

  gr4_declare_dependency(
    NAME
    SoapySDR
    FIND_PACKAGE_NAME
    SoapySDR
    FIND_PACKAGE_ARGS
    CONFIG
    FIND_PACKAGE_TARGETS
    SoapySDR
    MODES
    system
    TYPE
    OPTIONAL
    DESCRIPTION
    "SDR hardware abstraction"
    PURPOSE
    "blocks/sdr (system version may be incompatible with libc++)"
    URL
    "https://github.com/pothosware/SoapySDR")
endif()

# vir-simd is pulled in by every module — resolve here so vir-simd_SOURCE_DIR is known when we wrap it as gnuradio4::vir
# below.
gr4_resolve_dependency(vir-simd REQUIRED)

# ut is consumed by tests AND benchmarks (also by the public meta/UnitTestHelper.hpp and bench/benchmark.hpp
# infrastructure headers). Resolve here so the Emscripten fix-up below can patch its compile/link options before any
# consumer pulls it in.
if(ENABLE_TESTING OR ENABLE_BENCHMARKS)
  gr4_resolve_dependency(ut)

  if(EMSCRIPTEN AND TARGET ut)
    # Boost.UT upstream adds -fwasm-exceptions which conflicts with our -fexceptions (JS exceptions). JS-based
    # exceptions are required for Asyncify compatibility.
    get_target_property(_ut_copts ut INTERFACE_COMPILE_OPTIONS)
    get_target_property(_ut_lopts ut INTERFACE_LINK_OPTIONS)
    if(_ut_copts)
      list(REMOVE_ITEM _ut_copts -fwasm-exceptions)
      list(APPEND _ut_copts -fexceptions)
      set_target_properties(ut PROPERTIES INTERFACE_COMPILE_OPTIONS "${_ut_copts}")
    endif()
    if(_ut_lopts)
      list(REMOVE_ITEM _ut_lopts -fwasm-exceptions)
      list(APPEND _ut_lopts -fexceptions)
      set_target_properties(ut PROPERTIES INTERFACE_LINK_OPTIONS "${_ut_lopts}")
    endif()
  endif()
endif()

# cpp-httplib is purely test-only (mock HTTP servers / clients in qa_*.cpp).
if(ENABLE_TESTING)
  # a system httplib is typically built against libstdc++; linking it into a libc++ build mismatches the std:: ABI
  # (undefined httplib symbols at link, runtime corruption), so build it from source when using libc++.
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang"
     AND GR4_USE_LIBCXX
     AND NOT DEFINED GR4_DEP_CPP-HTTPLIB_MODE)
    set(GR4_DEP_CPP-HTTPLIB_MODE fetch)
  endif()
  gr4_resolve_dependency(cpp-httplib)
endif()

# vir-simd → gnuradio4::vir: stable internal alias regardless of which path satisfied the dep. When fetched: install the
# headers as part of the GR4 install so downstream consumers don't need vir-simd separately. When found via
# find_package: link the system imported target and let the Config template emit find_dependency(vir-simd) for
# downstreams.
add_library(vir INTERFACE)
add_library(gnuradio4::vir ALIAS vir)
if(TARGET vir-simd::vir-simd)
  set(GR4_VIR_SIMD_SYSTEM TRUE)
  target_link_libraries(vir INTERFACE vir-simd::vir-simd)
elseif(DEFINED vir-simd_SOURCE_DIR)
  set(GR4_VIR_SIMD_SYSTEM FALSE)
  target_include_directories(vir INTERFACE $<BUILD_INTERFACE:${vir-simd_SOURCE_DIR}/>
                                           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  install(DIRECTORY ${vir-simd_SOURCE_DIR}/vir DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
else()
  message(FATAL_ERROR "gr4 dep 'vir-simd': neither system imported target nor fetched source available")
endif()
install(TARGETS vir EXPORT gnuradio4Targets)

# the script configure won't run on win32 without asking bash to do it
if(WIN32)
  set(BASH "bash.exe")
else()
  set(BASH "")
endif()

# check for CPython and Numpy dependencies (Python integration is optional — only enables `setup_test()` linking and the
# NumPy Python check below).
find_package(Python3 3.12 COMPONENTS Interpreter Development NumPy)

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
   AND (UNIX
        OR APPLE
        OR WIN32)
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
    qa_UnmanagedSubGraph
    qa_ManagedSubGraph
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
    qa_UnmanagedSubGraph
    qa_ManagedSubGraph
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

set(GR_CMAKECONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/gnuradio4")

# Capture which optional system deps were actually consumed; gnuradio4Config.cmake.in uses these flags to emit
# conditional find_dependency() calls without leaking the build-time fetch policy to consumers.
get_property(GR4_HAS_SOAPYSDR GLOBAL PROPERTY GR4_HAS_SOAPYSDR)
if(NOT GR4_HAS_SOAPYSDR)
  set(GR4_HAS_SOAPYSDR FALSE)
endif()
get_property(GR4_HAS_SYSTEM_LIBSOUNDIO GLOBAL PROPERTY GR4_HAS_SYSTEM_LIBSOUNDIO)
if(NOT GR4_HAS_SYSTEM_LIBSOUNDIO)
  set(GR4_HAS_SYSTEM_LIBSOUNDIO FALSE)
endif()

configure_package_config_file(cmake/gnuradio4Config.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/gnuradio4Config.cmake"
                              INSTALL_DESTINATION "${GR_CMAKECONFIG_INSTALL_DIR}")

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/gnuradio4ConfigVersion.cmake"
  VERSION "${GR4_PKG_VERSION}"
  COMPATIBILITY SameMajorVersion)

install(
  EXPORT gnuradio4Targets
  FILE gnuradio4Targets.cmake
  NAMESPACE gnuradio4::
  DESTINATION "${GR_CMAKECONFIG_INSTALL_DIR}")

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/gnuradio4Config.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/gnuradio4ConfigVersion.cmake" DESTINATION "${GR_CMAKECONFIG_INSTALL_DIR}")

# When gnuradio4 was built against a system libsoundio, downstream consumers need our Find module so
# find_dependency(LibSoundIo) (in the Config template) can re-locate the IMPORTED target on their side.
if(GR4_HAS_SYSTEM_LIBSOUNDIO)
  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindLibSoundIo.cmake" DESTINATION "${GR_CMAKECONFIG_INSTALL_DIR}")
endif()

# End-to-end out-of-tree smoke test: install GR4 to a throw-away prefix and run a clean `find_package(gnuradio4)` from a
# tiny consumer project. Catches broken Config templates, missing transitive find_dependency() calls, and
# fetched-but-not-installed deps. Sanitizer and libc++ flags from the parent build are forwarded to the consumer's
# configure so its tiny executable can link against a static archive built with those instrumentation/stdlib choices —
# any real downstream of a sanitized or libc++-linked install would need the same flags too.
option(GR4_ENABLE_INSTALL_SMOKE_TEST "Add a CTest exercising the install + downstream find_package round-trip" ON)
set(_gr4SmokeFlags "")
if(ADDRESS_SANITIZER)
  string(APPEND _gr4SmokeFlags " -fsanitize=address")
elseif(UB_SANITIZER)
  string(APPEND _gr4SmokeFlags " -fsanitize=undefined")
elseif(THREAD_SANITIZER)
  string(APPEND _gr4SmokeFlags " -fsanitize=thread")
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang"
   AND NOT GR_USE_ADAPTIVE_CPP
   AND GR4_USE_LIBCXX)
  string(APPEND _gr4SmokeFlags " -stdlib=libc++")
endif()
string(STRIP "${_gr4SmokeFlags}" _gr4SmokeFlags)
if(ENABLE_TESTING
   AND GR_TOPLEVEL_PROJECT
   AND GR4_ENABLE_INSTALL_SMOKE_TEST
   AND NOT EMSCRIPTEN)
  add_test(
    NAME install_consumer_smoke
    COMMAND
      ${CMAKE_COMMAND} "-DGR4_BUILD_DIR=${CMAKE_BINARY_DIR}"
      "-DGR4_CONSUMER_SRC=${CMAKE_CURRENT_SOURCE_DIR}/cmake/install-consumer"
      "-DGR4_TEST_PREFIX=${CMAKE_BINARY_DIR}/install_consumer_smoke" "-DGR4_GENERATOR=${CMAKE_GENERATOR}"
      "-DGR4_BUILD_TYPE=${CMAKE_BUILD_TYPE}" "-DGR4_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
      "-DGR4_EXTRA_FLAGS=${_gr4SmokeFlags}" -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/install_consumer_smoke.cmake")
  set_tests_properties(install_consumer_smoke PROPERTIES LABELS "install-smoke")
endif()

# dependency status (system / fetch / not-found) via FeatureSummary
gr4_dependency_summary()

# print effective options
message(STATUS "================================================")
message(STATUS "GR-4 build configuration")
message(STATUS "------------------------------------------------")
message(STATUS "GR_TOPLEVEL_PROJECT          : ${GR_TOPLEVEL_PROJECT}")
message(STATUS "GR_DEPENDENCY_MODE           : ${GR4_DEPENDENCY_MODE}")
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
message(STATUS "ENABLE_BENCHMARKS            : ${ENABLE_BENCHMARKS}")
message(STATUS "ENABLE_COVERAGE              : ${ENABLE_COVERAGE}")
message(STATUS "GR_MAX_WASM_THREAD_COUNT     : ${GR_MAX_WASM_THREAD_COUNT}")
message(STATUS "GR_ENABLE_HTTP               : ${GR_ENABLE_HTTP}")
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
message(STATUS "GR_USE_ADAPTIVE_CPP?     : ${GR_USE_ADAPTIVE_CPP}")
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

include(cmake/CPackConfig.cmake)
