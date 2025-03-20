add_library(gr-http INTERFACE)
target_link_libraries(gr-http INTERFACE gnuradio-core gnuradio-algorithm httplib::httplib)
target_include_directories(gr-http INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/> $<INSTALL_INTERFACE:include/>)

add_library(GrHttpBlocks)
gr_generate_block_instantiations(
    GrHttpBlocks
    HEADERS include/gnuradio-4.0/http/HttpBlock.hpp
)
target_link_libraries(GrHttpBlocks PUBLIC gr-http)

if (ENABLE_TESTING)
    add_subdirectory(test)
endif ()
