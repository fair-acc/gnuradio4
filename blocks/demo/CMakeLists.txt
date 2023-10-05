add_library(gr-demo INTERFACE)
target_link_libraries(gr-demo INTERFACE gnuradio-core)
target_include_directories(gr-demo INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/> $<INSTALL_INTERFACE:include/>)
