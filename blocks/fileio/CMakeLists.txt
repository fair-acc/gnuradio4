add_library(gr-fileio INTERFACE)
target_link_libraries(gr-fileio INTERFACE gnuradio-core)
target_include_directories(gr-fileio INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/> $<INSTALL_INTERFACE:include/>)

if (ENABLE_TESTING)
    add_subdirectory(test)
endif ()
