add_library(gnuradio-algorithm INTERFACE)
target_include_directories(gnuradio-algorithm INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                                        $<INSTALL_INTERFACE:include/>)

if(NOT EMSCRIPTEN)
  include(FetchContent)
  FetchContent_Declare(
    pffft
    GIT_REPOSITORY https://github.com/marton78/pffft.git
    GIT_TAG master # or a specific commit/tag
  )
  # prevent optional/test/benchmark targets
  set(PFFFT_USE_BENCH_FFTW
      OFF
      CACHE BOOL "" FORCE)
  set(PFFFT_USE_BENCH_GREEN
      OFF
      CACHE BOOL "" FORCE)
  set(PFFFT_USE_BENCH_KISS
      OFF
      CACHE BOOL "" FORCE)
  set(PFFFT_USE_BENCH_POCKET
      OFF
      CACHE BOOL "" FORCE)
  set(PFFFT_USE_BENCH_MKL
      OFF
      CACHE BOOL "" FORCE)
  set(PFFFT_USE_FFTPACK
      OFF
      CACHE BOOL "" FORCE)
  set(PFFFT_USE_DEBUG_ASAN
      OFF
      CACHE BOOL "" FORCE)
  set(PFFFT_USE_TYPE_FLOAT
      ON
      CACHE BOOL "" FORCE) # or OFF if you don’t need float
  set(PFFFT_USE_TYPE_DOUBLE
      ON
      CACHE BOOL "" FORCE)
  set(INSTALL_PFFFT
      OFF
      CACHE BOOL "" FORCE)
  set(INSTALL_PFDSP
      OFF
      CACHE BOOL "" FORCE)
  set(INSTALL_PFFASTCONV
      OFF
      CACHE BOOL "" FORCE)

  # prevent examples and test target inclusion
  set(BUILD_TESTING
      OFF
      CACHE BOOL "" FORCE)

  FetchContent_MakeAvailable(pffft)
  add_library(pffft-config INTERFACE)
  target_compile_definitions(pffft-config INTERFACE PFFFT_ENABLE_FLOAT PFFFT_ENABLE_DOUBLE)
  target_include_directories(pffft-config INTERFACE ${pffft_SOURCE_DIR})
  target_link_libraries(pffft-config INTERFACE PFFFT)
  target_compile_options(pffft-config INTERFACE -march=native)
else()
  add_library(pffft-config INTERFACE)
endif()

target_link_libraries(
  gnuradio-algorithm
  INTERFACE gnuradio-options
            gnuradio-meta
            vir
            fftw
            pffft-config
            magic_enum)

if(ENABLE_EXAMPLES)
  add_subdirectory(benchmarks)
  add_subdirectory(src)
endif()

if(ENABLE_TESTING)
  add_subdirectory(test)
endif()
