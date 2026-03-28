add_library(gr-sdr INTERFACE)
target_include_directories(gr-sdr INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                            $<INSTALL_INTERFACE:include/>)
target_link_libraries(gr-sdr INTERFACE gnuradio-core gr-common gnuradio-algorithm)

install(FILES include/gnuradio-4.0/sdr/RTL2832Device.hpp include/gnuradio-4.0/sdr/RTL2832Source.hpp
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/gnuradio-4.0/sdr")

if(TARGET SoapySDR)
  target_link_libraries(gr-sdr INTERFACE SoapySDR)
  install(FILES include/gnuradio-4.0/sdr/SoapyRaiiWrapper.hpp include/gnuradio-4.0/sdr/SoapySource.hpp
          DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/gnuradio-4.0/sdr")

  # loopback device: static library for tests + optional shared module for system-wide use
  add_library(gr-sdr-loopback-static STATIC src/SoapyLoopbackModule.cpp)
  target_link_libraries(gr-sdr-loopback-static PUBLIC gr-sdr SoapySDR)
  set_target_properties(gr-sdr-loopback-static PROPERTIES POSITION_INDEPENDENT_CODE ON)

  add_library(gr-sdr-loopback MODULE src/SoapyLoopbackModule.cpp)
  target_link_libraries(gr-sdr-loopback PRIVATE gr-sdr SoapySDR)
  set_target_properties(gr-sdr-loopback PROPERTIES PREFIX "") # SoapySDR expects no lib prefix
endif()

if(ENABLE_TESTING)
  add_subdirectory(test)
endif()

if(ENABLE_EXAMPLES)
  add_subdirectory(src)
endif()
