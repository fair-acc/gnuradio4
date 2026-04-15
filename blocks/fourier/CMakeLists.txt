add_library(gr-fourier INTERFACE)
add_library(gnuradio4::gr-fourier ALIAS gr-fourier)
target_link_libraries(gr-fourier INTERFACE gnuradio4::gnuradio-core gnuradio4::gnuradio-algorithm)
target_include_directories(gr-fourier INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                                $<INSTALL_INTERFACE:include/>)
install(TARGETS gr-fourier EXPORT gnuradio4Targets)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/gnuradio-4.0/fourier"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/gnuradio-4.0")

gr_add_block_library(
  GrFourierBlocks
  MAKE_SHARED_LIBRARY
  HEADERS
  include/gnuradio-4.0/fourier/fft.hpp
  LINK_LIBRARIES
  gnuradio4::gr-basic
  gr-fourier)

if(TARGET gnuradio4::GrFourierBlocksShared AND ENABLE_TESTING)
  add_subdirectory(test)
endif()

if(ENABLE_EXAMPLES)
  add_subdirectory(benchmarks)
endif()
