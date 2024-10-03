add_library(gnuradio-meta INTERFACE)
target_include_directories(gnuradio-meta INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/> $<INSTALL_INTERFACE:include/>)
target_link_libraries(gnuradio-meta INTERFACE gnuradio-options vir)

if (ENABLE_TESTING)
    add_subdirectory(test)
endif ()
