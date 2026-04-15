add_library(gr-math INTERFACE)
add_library(gnuradio4::gr-math ALIAS gr-math)
target_link_libraries(gr-math INTERFACE gnuradio4::gnuradio-core gnuradio4::gnuradio-algorithm gnuradio4::exprtk)
target_include_directories(gr-math INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                             $<INSTALL_INTERFACE:include/>)
install(TARGETS gr-math EXPORT gnuradio4Targets)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/gnuradio-4.0/math"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/gnuradio-4.0")

gr_add_block_library(
  GrMathBlocks
  MAKE_SHARED_LIBRARY
  HEADERS
  include/gnuradio-4.0/math/ExpressionBlocks.hpp
  include/gnuradio-4.0/math/Math.hpp
  include/gnuradio-4.0/math/Rotator.hpp
  LINK_LIBRARIES
  gr-math)

if(TARGET gnuradio4::GrMathBlocksShared AND ENABLE_TESTING)
  add_subdirectory(test)
endif()
