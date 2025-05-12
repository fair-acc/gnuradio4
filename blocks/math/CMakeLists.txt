add_library(gr-math INTERFACE)
target_link_libraries(gr-math INTERFACE gnuradio-core gnuradio-algorithm exprtk)
target_include_directories(gr-math INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                             $<INSTALL_INTERFACE:include/>)

gr_add_block_library(
  GrMathBlocks
  MAKE_SHARED_LIBRARY
  HEADERS
  include/gnuradio-4.0/math/ExpressionBlocks.hpp
  include/gnuradio-4.0/math/Math.hpp
  include/gnuradio-4.0/math/Rotator.hpp
  LINK_LIBRARIES
  gr-math)

if(TARGET GrMathBlocksShared AND ENABLE_TESTING)
  add_subdirectory(test)
endif()
