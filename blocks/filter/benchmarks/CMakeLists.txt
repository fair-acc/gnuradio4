add_benchmark(bm_Filter)
target_link_libraries(
  bm_Filter
  PRIVATE gnuradio-options
          gnuradio-core
          gnuradio-algorithm
          gr-filter
          gr-testing
          gnuradio-blocklib-core
          ut
          ut-benchmark)
target_include_directories(bm_Filter PRIVATE ${PROJECT_SOURCE_DIR}/core/test)
# a benchmark measures this machine: without an arch flag the host arm is held to the baseline ISA and the table would
# understate every tap-per-sample row
target_compile_options(bm_Filter PRIVATE -O3)
if(NOT EMSCRIPTEN AND NOT CMAKE_CROSSCOMPILING)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=native" GR_FILTER_BM_HAS_MARCH_NATIVE)
  if(GR_FILTER_BM_HAS_MARCH_NATIVE)
    target_compile_options(bm_Filter PRIVATE -march=native)
  endif()
endif()
