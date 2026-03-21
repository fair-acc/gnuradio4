add_library(gr-sdr INTERFACE)
target_include_directories(gr-sdr INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                            $<INSTALL_INTERFACE:include/>)
target_link_libraries(gr-sdr INTERFACE gnuradio-core gr-common)

install(FILES include/gnuradio-4.0/sdr/RTL2832Device.hpp include/gnuradio-4.0/sdr/RTL2832Source.hpp
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/gnuradio-4.0/sdr")

if(ENABLE_TESTING)
  add_subdirectory(test)
endif()

if(ENABLE_EXAMPLES)
  add_subdirectory(src)
endif()
