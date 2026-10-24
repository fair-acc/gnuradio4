foreach(_bm IN ITEMS bm_Clamp bm_Comparator)
  add_benchmark(${_bm})
  target_link_libraries(
    ${_bm}
    PRIVATE gnuradio-options
            gnuradio-core
            gr-math
            gr-testing
            gnuradio-blocklib-core
            ut
            ut-benchmark)
  target_compile_options(${_bm} PRIVATE -O3)
endforeach()
