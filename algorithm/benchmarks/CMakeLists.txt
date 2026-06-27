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

# FFT backend comparison benchmark (SimdFFT + all available DeviceContext backends).
# Under acpp: SHARED library required — nd_range kernel HCF lookup fails when boost/ut.hpp
# is in the same binary. See G10 in claude_wip.md and AdaptiveCpp#2042.
add_gr_benchmark(bm_FFT_backends)
if (GR_USE_ADAPTIVE_CPP)
  add_library(bm_FFT_backends_helpers SHARED bm_FFT_backends_helpers.cpp)
  set_target_properties(bm_FFT_backends_helpers PROPERTIES CXX_VISIBILITY_PRESET default)
else ()
  add_library(bm_FFT_backends_helpers STATIC bm_FFT_backends_helpers.cpp)
endif ()
target_link_libraries(bm_FFT_backends_helpers PUBLIC gnuradio-options gnuradio-core gnuradio-algorithm)
target_link_libraries(bm_FFT_backends PRIVATE bm_FFT_backends_helpers)
if (GR_HAS_GL_COMPUTE)
  target_link_libraries(bm_FFT_backends PRIVATE PkgConfig::GR_EGL PkgConfig::GR_GL)
  target_link_libraries(bm_FFT_backends_helpers PUBLIC PkgConfig::GR_EGL PkgConfig::GR_GL)
endif ()
