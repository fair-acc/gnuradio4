add_library(gr-analog INTERFACE)
add_library(gnuradio4::gr-analog ALIAS gr-analog)
target_link_libraries(gr-analog INTERFACE gnuradio4::gnuradio-core gnuradio4::gnuradio-algorithm)
target_include_directories(gr-analog INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                               $<INSTALL_INTERFACE:include/>)
install(TARGETS gr-analog EXPORT gnuradio4Targets)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/gnuradio-4.0/analog"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/gnuradio-4.0")

gr_add_block_library(
  GrAnalogBlocks
  MAKE_SHARED_LIBRARY
  MAKE_PLUGIN
  HEADERS
  include/gnuradio-4.0/analog/AMModulation.hpp
  include/gnuradio-4.0/analog/AutomaticGainControl.hpp
  include/gnuradio-4.0/analog/FMEmphasis.hpp
  include/gnuradio-4.0/analog/FMModulation.hpp
  include/gnuradio-4.0/analog/FMStereoDecoder.hpp
  include/gnuradio-4.0/analog/PhaseLockedLoop.hpp
  include/gnuradio-4.0/analog/PMModulation.hpp
  include/gnuradio-4.0/analog/PowerSquelch.hpp
  LINK_LIBRARIES
  gr-analog)

if(ENABLE_TESTING)
  add_subdirectory(test)
endif()
