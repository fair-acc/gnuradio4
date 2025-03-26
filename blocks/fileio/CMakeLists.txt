add_library(gr-fileio INTERFACE)
target_link_libraries(gr-fileio INTERFACE gnuradio-core)
target_include_directories(gr-fileio INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                               $<INSTALL_INTERFACE:include/>)

# An OBJECT library as the generated files have namespace-scope variables with dynamic C++ initialization
add_library(GrFileIoBlocks OBJECT)
gr_generate_block_instantiations(GrFileIoBlocks HEADERS include/gnuradio-4.0/fileio/BasicFileIo.hpp)
target_link_libraries(GrFileIoBlocks PUBLIC gr-fileio)

if(ENABLE_TESTING)
  add_subdirectory(test)
endif()
