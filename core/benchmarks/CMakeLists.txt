if(NOT EMSCRIPTEN OR GR_ENABLE_WEBGPU)
  function(add_gr_benchmark BM_NAME)
    add_benchmark(${BM_NAME})
    target_link_libraries(
      ${BM_NAME}
      PRIVATE gnuradio-options
              gnuradio-core
              gnuradio-blocklib-core
              gnuradio-plugin
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
      target_compile_options(${BM_NAME} PRIVATE -O3 -march=${GR_BENCHMARK_ARCH})
    endif()

    target_include_directories(${BM_NAME} PRIVATE "../test")
  endfunction()

  if(GR_ENABLE_WEBGPU)
    add_gr_benchmark(bm_DeviceDispatch) # run in a browser: wasm_test.html?test=bm_DeviceDispatch
    gr_translate_webgpu_kernels(bm_DeviceDispatch)
    return() # the other benchmarks stay native, as in a plain Emscripten build
  endif()
  if(GR_USE_ADAPTIVE_CPP)
    add_gr_benchmark(bm_DeviceDispatch)
    add_gr_benchmark(bm_DeviceDspChain)
  endif()

  add_gr_benchmark(bm_Buffer)
  add_gr_benchmark(bm_CRC)
  add_gr_benchmark(bm_HistoryBuffer)
  add_gr_benchmark(bm_Profiler)
  add_gr_benchmark(bm_Scheduler)
  add_gr_benchmark(bm_tensor)
  add_gr_benchmark(bm_MergeApi)
  add_gr_benchmark(bm_sync)
  add_gr_benchmark(bm_TagForwarding)
  add_gr_benchmark(bm_PerformanceMonitor)
  add_gr_benchmark(bm_TensorPort)
  add_gr_benchmark(bm_portLimits)

  if(NOT THREAD_SANITIZER)
    add_gr_benchmark(bm_ValueMap)
  else()
    message("Disabling bm_ValueMap under TSAN. It depends on defining new/delete which conflicts with TSAN")
  endif()
endif()
