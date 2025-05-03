add_library(gr-http INTERFACE)
target_link_libraries(gr-http INTERFACE gnuradio-core gnuradio-algorithm httplib::httplib)
target_include_directories(gr-http INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                             $<INSTALL_INTERFACE:include/>)
if(WIN32)
  target_compile_definitions(gr-http INTERFACE BUILDING_MYPLUGIN)
endif()

gr_add_block_library(
  GrHttpBlocks
  MAKE_SHARED_LIBRARY
  HEADERS
  include/gnuradio-4.0/http/HttpBlock.hpp
  LINK_LIBRARIES
  gr-http)

if(TARGET GrHttpBlocksShared AND ENABLE_TESTING)
  add_subdirectory(test)
endif()
