# Check for optional SoapySDR local dependency https://github.com/pothosware/SoapySDR
if(TARGET SoapySDR)
  add_library(gr-soapy INTERFACE)
  target_include_directories(gr-soapy INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                                $<INSTALL_INTERFACE:include/>)
  target_link_libraries(gr-soapy INTERFACE gnuradio-core SoapySDR)

  if(ENABLE_EXAMPLES)
    add_subdirectory(src)
  endif()

  if(ENABLE_TESTING)
    add_subdirectory(test)
  endif()
else()
  message(WARNING "SoapySDR development files not found: ${SoapySDR_FOUND} - skipping SoapySDR support")
endif()
