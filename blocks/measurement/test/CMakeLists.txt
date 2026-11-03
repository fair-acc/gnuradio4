function(add_measurement_test TEST_NAME)
  add_executable(${TEST_NAME} ${TEST_NAME}.cpp)
  target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_BINARY_DIR}/include ${PROJECT_SOURCE_DIR}/blocks/trigger/test)
  target_link_libraries(
    ${TEST_NAME}
    PRIVATE gnuradio-options
            gnuradio-core
            gnuradio-blocklib-core
            gr-measurement
            gr-trigger
            gr-basic
            gr-testing
            ut)
  add_test(NAME ${TEST_NAME} COMMAND ${CMAKE_CROSSCOMPILING_EMULATOR} ${TEST_NAME})
endfunction()

add_measurement_test(qa_Measure)
