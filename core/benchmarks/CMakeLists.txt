function (add_gr_benchmark BM_NAME)
    add_benchmark(${BM_NAME})
    target_link_libraries(${BM_NAME} PRIVATE gnuradio-core refl-cpp fmt gr-basic gr-fileio gr-math gr-testing)
    target_compile_options(${BM_NAME} PRIVATE -O3) # performance related benchmarks should be optimised during compile-time
endfunction()

add_gr_benchmark(bm_Buffer)
add_gr_benchmark(bm_HistoryBuffer)
add_gr_benchmark(bm_Profiler)
add_gr_benchmark(bm_Scheduler)
add_gr_benchmark(bm-nosonar_node_api)
add_gr_benchmark(bm_fft)
target_link_libraries(bm_fft PRIVATE gr-fourier)
