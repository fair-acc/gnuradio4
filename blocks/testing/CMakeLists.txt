add_library(gr-testing INTERFACE)
target_link_libraries(gr-testing INTERFACE gnuradio-core)
target_include_directories(gr-testing INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/> $<INSTALL_INTERFACE:include/>)
