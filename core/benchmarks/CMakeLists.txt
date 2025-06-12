if(NOT EMSCRIPTEN)
  function(add_gr_benchmark BM_NAME)
    add_benchmark(${BM_NAME})
    target_link_libraries(
      ${BM_NAME}
      PRIVATE gnuradio-options
              gnuradio-core
              gnuradio-blocklib-core
              gnuradio-plugin
              fftw
              gr-basic
              gr-fileio
              gr-math
              gr-testing
              gr-basic
              gr-filter
              ut
              ut-benchmark)
    if(EMSCRIPTEN) # performance benchmarks should be optimised
      target_compile_options(${BM_NAME} PRIVATE -O3)
    else()
      target_compile_options(${BM_NAME} PRIVATE -O3 -march=native)
    endif()
  endfunction()

  add_gr_benchmark(bm_Buffer)
  add_gr_benchmark(bm_HistoryBuffer)
  add_gr_benchmark(bm_Profiler)
  add_gr_benchmark(bm_Scheduler)
  add_gr_benchmark(bm-nosonar_node_api)
  add_gr_benchmark(bm_sync)
endif()
