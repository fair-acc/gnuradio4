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
endif()
