add_benchmark(bm_ArrayChain)
target_link_libraries(
  bm_ArrayChain
  PRIVATE gnuradio-options
          gnuradio-core
          gr-beamforming
          gr-localisation
          gr-basic
          gr-fourier
          gr-math
          gr-testing
          gnuradio-blocklib-core
          ut
          ut-benchmark)
target_compile_options(bm_ArrayChain PRIVATE -O3)
