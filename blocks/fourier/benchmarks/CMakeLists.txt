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
  target_compile_options(${BM_NAME} PRIVATE -O3)
endfunction()

add_gr_benchmark(bm_fft_block)
add_gr_benchmark(bm_fft_subgraph)
target_link_libraries(bm_fft_subgraph PRIVATE gr-testing gnuradio-blocklib-core)
add_gr_benchmark(bm_fft_stream_graph)
target_link_libraries(bm_fft_stream_graph PRIVATE gr-testing gnuradio-blocklib-core)
# FFT backend benchmark: algorithm/benchmarks/bm_FFT_backends.cpp
