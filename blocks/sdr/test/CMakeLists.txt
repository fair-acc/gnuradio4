add_executable(qa_RTL2832Source qa_RTL2832Source.cpp)
target_link_libraries(
  qa_RTL2832Source
  PRIVATE gnuradio-options
          gnuradio-core
          gnuradio-blocklib-core
          gr-sdr
          gr-timing
          gr-testing
          ut)
add_test(NAME qa_RTL2832Source COMMAND ${CMAKE_CROSSCOMPILING_EMULATOR} ${CMAKE_CURRENT_BINARY_DIR}/qa_RTL2832Source)

if(TARGET SoapySDR)
  # detect ABI compatibility: tests using SoapySDR's C++ API (std::vector, std::string in ArgInfo/Kwargs) require the
  # same stdlib as the system SoapySDR library. The C API wrapper (SoapyRaiiWrapper) is ABI-safe and works with either
  # stdlib.
  set(SOAPY_CXX_ABI_COMPATIBLE TRUE)
  get_target_property(_soapy_imported_configs SoapySDR IMPORTED_CONFIGURATIONS)
  if(_soapy_imported_configs)
    list(
      GET
      _soapy_imported_configs
      0
      _soapy_first_config)
    get_target_property(_soapy_lib SoapySDR IMPORTED_LOCATION_${_soapy_first_config})
  endif()
  if(NOT _soapy_lib)
    get_target_property(_soapy_lib SoapySDR IMPORTED_LOCATION)
  endif()
  if(_soapy_lib AND NOT CMAKE_CROSSCOMPILING)
    execute_process(
      COMMAND ldd "${_soapy_lib}"
      OUTPUT_VARIABLE _soapy_ldd
      ERROR_QUIET)
    string(FIND "${_soapy_ldd}" "libc++.so" _soapy_has_libcxx)
    string(FIND "${_soapy_ldd}" "libstdc++.so" _soapy_has_libstdcxx)
    if(_soapy_has_libcxx GREATER_EQUAL 0)
      set(_soapy_stdlib "libc++")
    elseif(_soapy_has_libstdcxx GREATER_EQUAL 0)
      set(_soapy_stdlib "libstdc++")
    else()
      set(_soapy_stdlib "unknown")
    endif()
    # determine our stdlib
    if((CMAKE_CXX_COMPILER_ID MATCHES "Clang") AND GR4_USE_LIBCXX)
      set(_our_stdlib "libc++")
    else()
      set(_our_stdlib "libstdc++")
    endif()
    if(NOT
       _soapy_stdlib
       STREQUAL
       _our_stdlib
       AND NOT
           _soapy_stdlib
           STREQUAL
           "unknown")
      set(SOAPY_CXX_ABI_COMPATIBLE FALSE)
      message(
        STATUS
          "SoapySDR stdlib (${_soapy_stdlib}) != project stdlib (${_our_stdlib}) — C++ API tests disabled (ABI mismatch)"
      )
    endif()
  endif()

  add_executable(qa_SoapySource qa_SoapySource.cpp)
  target_link_libraries(
    qa_SoapySource
    PRIVATE gnuradio-options
            gnuradio-core
            gnuradio-blocklib-core
            gr-sdr
            gr-common
            gr-testing
            ut)
  add_test(NAME qa_SoapySource COMMAND ${CMAKE_CROSSCOMPILING_EMULATOR} ${CMAKE_CURRENT_BINARY_DIR}/qa_SoapySource)
  # skip LSAN: 168-byte leak inside SoapySDR's dynamically loaded driver modules (not our code)
  set_tests_properties(qa_SoapySource PROPERTIES ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

  if(SOAPY_CXX_ABI_COMPATIBLE)
    # C++ API test: directly uses SoapySDR::Device, ArgInfo, Kwargs — requires matching stdlib
    add_executable(qa_SoapyLoopback qa_SoapyLoopback.cpp)
    target_link_libraries(
      qa_SoapyLoopback
      PRIVATE gnuradio-options
              gnuradio-core
              gnuradio-blocklib-core
              gr-sdr
              gr-sdr-loopback-static
              gr-testing
              ut)
    add_test(NAME qa_SoapyLoopback COMMAND ${CMAKE_CROSSCOMPILING_EMULATOR}
                                           ${CMAKE_CURRENT_BINARY_DIR}/qa_SoapyLoopback)
    set_tests_properties(qa_SoapyLoopback PROPERTIES ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
  else()
    message(STATUS "qa_SoapyLoopback SKIPPED (SoapySDR C++ ABI mismatch)")
  endif()

  # C API wrapper test: ABI-safe, works with any stdlib combination. The .so module must be built with the same compiler
  # as SoapySDR (uses C++ API internally), so we only build this test when the .so module can be loaded by the system
  # SoapySDR.
  if(SOAPY_CXX_ABI_COMPATIBLE)
    add_executable(qa_SoapyRaiiWrapper qa_SoapyRaiiWrapper.cpp)
    target_link_libraries(
      qa_SoapyRaiiWrapper
      PRIVATE gnuradio-options
              gnuradio-core
              gnuradio-blocklib-core
              gr-sdr
              gr-testing
              ut)
    add_dependencies(qa_SoapyRaiiWrapper gr-sdr-loopback)
    set(SOAPY_MODULE_DIR ${CMAKE_CURRENT_BINARY_DIR}/soapy_modules)
    add_custom_command(
      TARGET qa_SoapyRaiiWrapper
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory ${SOAPY_MODULE_DIR}
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:gr-sdr-loopback> ${SOAPY_MODULE_DIR}/
      COMMENT "Copying gr-sdr-loopback.so for wrapper test")
    add_test(NAME qa_SoapyRaiiWrapper COMMAND ${CMAKE_CROSSCOMPILING_EMULATOR}
                                              ${CMAKE_CURRENT_BINARY_DIR}/qa_SoapyRaiiWrapper)
    set_tests_properties(qa_SoapyRaiiWrapper
                         PROPERTIES ENVIRONMENT "SOAPY_SDR_PLUGIN_PATH=${SOAPY_MODULE_DIR};ASAN_OPTIONS=detect_leaks=0")
    # GR4 integration test: SoapySource + loopback through scheduler
    add_executable(qa_SoapyIntegration qa_SoapyIntegration.cpp)
    target_link_libraries(
      qa_SoapyIntegration
      PRIVATE gnuradio-options
              gnuradio-core
              gnuradio-blocklib-core
              gr-sdr
              gr-basic
              gr-testing
              ut)
    add_dependencies(qa_SoapyIntegration gr-sdr-loopback)
    add_custom_command(
      TARGET qa_SoapyIntegration
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory ${SOAPY_MODULE_DIR}
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:gr-sdr-loopback> ${SOAPY_MODULE_DIR}/
      COMMENT "Copying gr-sdr-loopback.so for integration test")
    add_test(NAME qa_SoapyIntegration COMMAND ${CMAKE_CROSSCOMPILING_EMULATOR}
                                              ${CMAKE_CURRENT_BINARY_DIR}/qa_SoapyIntegration)
    set_tests_properties(qa_SoapyIntegration
                         PROPERTIES ENVIRONMENT "SOAPY_SDR_PLUGIN_PATH=${SOAPY_MODULE_DIR};ASAN_OPTIONS=detect_leaks=0")
  else()
    message(STATUS "qa_SoapyRaiiWrapper SKIPPED (loopback .so module requires matching SoapySDR ABI)")
  endif()
endif()
