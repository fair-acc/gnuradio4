add_ut_test(bench_pffft bench_pffft.cpp pffft.h) # pffft.cpp
target_link_libraries(bench_pffft PUBLIC gnuradio-core vir)
#target_compile_definitions(bench_pffft PRIVATE _USE_MATH_DEFINES)
#target_compile_definitions(bench_pffft PRIVATE PFFFT_ENABLE_FLOAT)
