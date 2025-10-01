add_ut_test(bench_pffft bench_pffft.cpp SimdFFT.hpp) # pffft.cpp
target_link_libraries(bench_pffft PUBLIC gnuradio-core vir)
if(CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_options(bench_pffft PRIVATE -O0 -g)
elseif(CMAKE_BUILD_TYPE MATCHES Release)
    target_compile_options(bench_pffft PRIVATE -O3 -march=native)
endif()
