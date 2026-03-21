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
