if(EMSCRIPTEN)
  # WASM: WebUSB via thin EM_JS shims in RtlSdrDevice.hpp (no native USB library)
  add_library(gr-rtlsdr INTERFACE)
  target_include_directories(gr-rtlsdr INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                                 $<INSTALL_INTERFACE:include/>)
  target_link_libraries(gr-rtlsdr INTERFACE gnuradio-core)

else()
  # native: link against libusb-1.0 directly (LGPL-2.1)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(LIBUSB IMPORTED_TARGET libusb-1.0)
  endif()

  if(NOT LIBUSB_FOUND)
    message(STATUS "libusb-1.0 not found — skipping RTL-SDR support")
    return()
  endif()

  add_library(gr-rtlsdr INTERFACE)
  target_include_directories(gr-rtlsdr INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                                 $<INSTALL_INTERFACE:include/>)
  target_link_libraries(gr-rtlsdr INTERFACE gnuradio-core PkgConfig::LIBUSB)
endif()

install(FILES include/gnuradio-4.0/rtlsdr/RtlSdrDevice.hpp include/gnuradio-4.0/rtlsdr/RtlSdrSource.hpp
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/gnuradio-4.0/rtlsdr")

if(ENABLE_TESTING AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/CMakeLists.txt")
  add_subdirectory(test)
endif()

if(ENABLE_EXAMPLES AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/CMakeLists.txt")
  add_subdirectory(src)
endif()
