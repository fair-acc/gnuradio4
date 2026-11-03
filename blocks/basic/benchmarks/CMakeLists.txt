foreach(
  _bm IN
  ITEMS bm_Mux
        bm_Repeat
        bm_StreamToTensor
        bm_SwapEndianness)
  add_benchmark(${_bm})
  target_link_libraries(
    ${_bm}
    PRIVATE gnuradio-options
            gnuradio-core
            gr-basic
            gr-testing
            gnuradio-blocklib-core
            ut
            ut-benchmark)
  target_compile_options(${_bm} PRIVATE -O3)
endforeach()
