add_library(gr-fourier INTERFACE)
target_link_libraries(gr-fourier INTERFACE gnuradio-core gnuradio-algorithm)
target_include_directories(gr-fourier INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                                $<INSTALL_INTERFACE:include/>)
install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/gnuradio-4.0/fourier"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/gnuradio-4.0")

gr_add_block_library(
  GrFourierBlocks
  MAKE_SHARED_LIBRARY
  HEADERS
  include/gnuradio-4.0/fourier/fft.hpp
  LINK_LIBRARIES
  gr-basic
  gr-fourier)

if(TARGET GrFourierBlocksShared AND ENABLE_TESTING)
  add_subdirectory(test)
endif()

if(ENABLE_EXAMPLES)
  add_subdirectory(benchmarks)
endif()
