add_library(gr-fileio INTERFACE)
target_link_libraries(gr-fileio INTERFACE gnuradio-core)
target_include_directories(gr-fileio INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                               $<INSTALL_INTERFACE:include/>)
if(WIN32)
  target_compile_definitions(gr-fileio INTERFACE BUILDING_MYPLUGIN)
endif()

gr_add_block_library(
  GrFileIoBlocks
  MAKE_SHARED_LIBRARY
  SPLIT_BLOCK_INSTANTIATIONS
  HEADERS
  include/gnuradio-4.0/fileio/BasicFileIo.hpp
  LINK_LIBRARIES
  gr-fileio)

if(TARGET GrFileIoBlocksShared AND ENABLE_TESTING)
  add_subdirectory(test)
endif()
