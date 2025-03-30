add_library(gr-fourier INTERFACE)
target_link_libraries(gr-fourier INTERFACE gnuradio-core gnuradio-algorithm)
target_include_directories(gr-fourier INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/> $<INSTALL_INTERFACE:include/>)

# An OBJECT library as the generated files have namespace-scope variables with dynamic C++ initialization
add_library(GrFourierBlocks SHARED)
gr_generate_block_instantiations(GrFourierBlocks HEADERS include/gnuradio-4.0/fourier/fft.hpp)
target_link_libraries(
  GrFourierBlocks
  PUBLIC gnuradio-core
         gnuradio-blocklib-core
         gr-basic
         gr-fourier)

if (ENABLE_TESTING)
    add_subdirectory(test)
endif ()
