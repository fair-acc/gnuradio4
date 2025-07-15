
add_library(gr-timing INTERFACE)
target_include_directories(gr-timing INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                                $<INSTALL_INTERFACE:include/>)
target_link_libraries(gr-timing INTERFACE gnuradio-core)

if(ENABLE_EXAMPLES)
  add_subdirectory(src)
endif()

if(ENABLE_TESTING)
  add_subdirectory(test)
endif()
