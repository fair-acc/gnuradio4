add_library(gr-common INTERFACE)
add_library(gnuradio4::gr-common ALIAS gr-common)
target_include_directories(gr-common INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                               $<INSTALL_INTERFACE:include/>)
target_link_libraries(gr-common INTERFACE gnuradio4::gnuradio-core)
install(TARGETS gr-common EXPORT gnuradio4Targets)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/gnuradio-4.0/common"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/gnuradio-4.0")
