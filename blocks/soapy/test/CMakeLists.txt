add_executable(qa_Soapy qa_Soapy.cpp)
target_include_directories(qa_Soapy PRIVATE ${CMAKE_BINARY_DIR}/include ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(
  qa_Soapy
  PRIVATE gnuradio-options
          gnuradio-core
          gnuradio-blocklib-core
          fftw
          gr-soapy
          gr-testing
          ut)
add_test(NAME qa_Soapy COMMAND ${CMAKE_CROSSCOMPILING_EMULATOR} ${CMAKE_CURRENT_BINARY_DIR}/qa_Soapy)
# skip LSAN due to a known S
set_tests_properties(qa_Soapy PROPERTIES ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
