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
  check_cxx_compiler_flag("-march=${GR_BENCHMARK_ARCH}" GR_FILTER_BM_HAS_MARCH)
  if(GR_FILTER_BM_HAS_MARCH)
    target_compile_options(bm_Filter PRIVATE -march=${GR_BENCHMARK_ARCH})
  endif()
endif()

add_benchmark(bm_HierarchicalChannelizer)
target_link_libraries(
  bm_HierarchicalChannelizer
  PRIVATE gnuradio-options
          gnuradio-core
          gnuradio-algorithm
          gr-filter
          gr-testing
          gnuradio-blocklib-core
          ut)

foreach(_bm IN ITEMS bm_Delay bm_MovingAverage bm_Hilbert bm_PhaseUnwrap)
  add_benchmark(${_bm})
  target_link_libraries(
    ${_bm}
    PRIVATE gnuradio-options
            gnuradio-core
            gr-filter
            gr-testing
            gnuradio-blocklib-core
            ut
            ut-benchmark)
  target_compile_options(${_bm} PRIVATE -O3)
endforeach()
