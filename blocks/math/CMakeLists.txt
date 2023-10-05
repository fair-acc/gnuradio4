add_library(gr-math INTERFACE)
target_link_libraries(gr-math INTERFACE gnuradio-core)
target_include_directories(gr-math INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/> $<INSTALL_INTERFACE:include/>)
