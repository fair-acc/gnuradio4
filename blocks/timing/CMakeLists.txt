add_library(gr-timing INTERFACE)
target_include_directories(gr-timing INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                               $<INSTALL_INTERFACE:include/>)
target_link_libraries(gr-timing INTERFACE gnuradio-core gr-common)

install(
  FILES include/gnuradio-4.0/GpsSource.hpp
        include/gnuradio-4.0/NMEADevice.hpp
        include/gnuradio-4.0/NMEAParser.hpp
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/gnuradio-4.0")

if(ENABLE_EXAMPLES)
  add_subdirectory(src)
endif()

if(ENABLE_TESTING AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/CMakeLists.txt")
  add_subdirectory(test)
endif()
