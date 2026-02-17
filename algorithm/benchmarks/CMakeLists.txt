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
    # WebAssembly SIMD (128-bit) + optimisation for benches
    target_compile_options(${BM_NAME} PRIVATE -O2 -msimd128)
    target_link_options(${BM_NAME} PRIVATE -msimd128)

    # (optional) Relaxed SIMD if your toolchain supports it
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mrelaxed-simd" HAS_RELAXED_SIMD)
    if(HAS_RELAXED_SIMD)
      target_compile_options(${BM_NAME} PRIVATE -mrelaxed-simd)
      target_link_options(${BM_NAME} PRIVATE -mrelaxed-simd)
    endif()
  else()
    target_compile_options(${BM_NAME} PRIVATE -O2 -march=native)
  endif()
endfunction()

add_gr_benchmark(bm_fft)
add_gr_benchmark(bm_pffft)
add_gr_benchmark(bm_svd)
add_gr_benchmark(bm_signal_generators)
