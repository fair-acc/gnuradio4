function(add_timing_test TEST_NAME)
  add_executable(${TEST_NAME} ${TEST_NAME}.cpp)
  target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_BINARY_DIR}/include)
  target_link_libraries(
    ${TEST_NAME}
    PRIVATE gnuradio-options
            gnuradio-core
            gnuradio-blocklib-core
            gr-timing
            gr-testing
            ut)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(${TEST_NAME} PRIVATE util)
  endif()
  add_test(NAME ${TEST_NAME} COMMAND ${CMAKE_CROSSCOMPILING_EMULATOR} ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME})
endfunction()

add_timing_test(qa_NMEAParser)
add_timing_test(qa_NMEADevice)

if(NOT EMSCRIPTEN)
  add_timing_test(qa_GpsSource)
endif()
