add_executable(qa_RtlSdrSource qa_RtlSdrSource.cpp)
target_include_directories(qa_RtlSdrSource PRIVATE ${CMAKE_BINARY_DIR}/include)
target_link_libraries(
  qa_RtlSdrSource
  PRIVATE gnuradio-options
          gnuradio-core
          gnuradio-blocklib-core
          gr-rtlsdr
          gr-testing
          ut)
add_test(NAME qa_RtlSdrSource COMMAND ${CMAKE_CROSSCOMPILING_EMULATOR} ${CMAKE_CURRENT_BINARY_DIR}/qa_RtlSdrSource)
