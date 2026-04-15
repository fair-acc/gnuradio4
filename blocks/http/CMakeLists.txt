add_library(gr-http INTERFACE)
add_library(gnuradio4::gr-http ALIAS gr-http)
target_link_libraries(gr-http INTERFACE gnuradio4::gnuradio-core gnuradio4::gnuradio-algorithm)
if(APPLE)
  target_link_libraries(gr-http INTERFACE "-framework CFNetwork")
endif()
target_include_directories(gr-http INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                             $<INSTALL_INTERFACE:include/>)
install(TARGETS gr-http EXPORT gnuradio4Targets)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/gnuradio-4.0/http"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/gnuradio-4.0")

gr_add_block_library(
  GrHttpBlocks
  MAKE_SHARED_LIBRARY
  HEADERS
  include/gnuradio-4.0/http/HttpBlock.hpp
  LINK_LIBRARIES
  gnuradio4::gr-http)

if(TARGET gnuradio4::GrHttpBlocksShared AND ENABLE_TESTING)
  add_subdirectory(test)
endif()
