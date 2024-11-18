add_library(gr-electrical INTERFACE)
target_link_libraries(gr-electrical INTERFACE gnuradio-core gnuradio-algorithm)
target_include_directories(gr-electrical INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                                   $<INSTALL_INTERFACE:include/>)

if(ENABLE_TESTING)
  add_subdirectory(test)
endif()
