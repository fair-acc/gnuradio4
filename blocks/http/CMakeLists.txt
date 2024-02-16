add_library(gr-http INTERFACE)
target_link_libraries(gr-http INTERFACE gnuradio-core gnuradio-algorithm httplib::httplib)
target_include_directories(gr-http INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/> $<INSTALL_INTERFACE:include/>)

if (ENABLE_TESTING)
    add_subdirectory(test)
endif ()
