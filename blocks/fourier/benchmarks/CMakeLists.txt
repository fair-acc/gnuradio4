function(add_gr_benchmark BM_NAME)
  add_benchmark(${BM_NAME})
  target_link_libraries(
    ${BM_NAME}
    PRIVATE gnuradio-options
            gnuradio-core
            gnuradio-algorithm
            gr-fourier
            ut
            ut-benchmark)
  target_compile_options(${BM_NAME} PRIVATE -O3) # performance related benchmarks should be optimised during
                                                 # compile-time
endfunction()

add_gr_benchmark(bm_fft_block)
