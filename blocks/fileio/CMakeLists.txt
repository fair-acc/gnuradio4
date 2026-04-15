add_library(gr-fileio INTERFACE)
add_library(gnuradio4::gr-fileio ALIAS gr-fileio)
target_link_libraries(gr-fileio INTERFACE gnuradio4::gnuradio-core gnuradio4::gnuradio-algorithm)
target_include_directories(gr-fileio INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                               $<INSTALL_INTERFACE:include/>)
install(TARGETS gr-fileio EXPORT gnuradio4Targets)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/gnuradio-4.0/fileio"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/gnuradio-4.0")

gr_add_block_library(
  GrFileIoBlocks
  MAKE_SHARED_LIBRARY
  SPLIT_BLOCK_INSTANTIATIONS
  HEADERS
  include/gnuradio-4.0/fileio/BasicFileIo.hpp
  include/gnuradio-4.0/fileio/WavBlocks.hpp
  LINK_LIBRARIES
  gnuradio4::gr-fileio)

if(TARGET gnuradio4::GrFileIoBlocksShared AND ENABLE_TESTING)
  add_subdirectory(test)
endif()
