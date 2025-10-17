function(add_gr_benchmark BM_NAME)
  add_benchmark(${BM_NAME})
  target_link_libraries(
    ${BM_NAME}
    PRIVATE gnuradio-options
            gnuradio-core
            gnuradio-algorithm
            ut
            ut-benchmark)

  if(EMSCRIPTEN)
    target_compile_options(${BM_NAME} PRIVATE -O2) # benchmarks should be optimised
  else()
    target_compile_options(${BM_NAME} PRIVATE -O2 -march=native) # benchmarks should be optimised
  endif()
endfunction()

add_gr_benchmark(bm_fft)
add_gr_benchmark(bm_pffft)
