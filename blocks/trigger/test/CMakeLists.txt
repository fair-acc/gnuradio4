function(add_trigger_test TEST_NAME)
  add_executable(${TEST_NAME} ${TEST_NAME}.cpp)
  target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_BINARY_DIR}/include)
  target_link_libraries(
    ${TEST_NAME}
    PRIVATE gnuradio-options
            gnuradio-core
            gnuradio-blocklib-core
            gr-trigger
            gr-basic
            gr-math
            gr-testing
            ut)
  add_test(NAME ${TEST_NAME} COMMAND ${CMAKE_CROSSCOMPILING_EMULATOR} ${TEST_NAME})
endfunction()

add_trigger_test(qa_EventStore)
add_trigger_test(qa_Marble)
add_trigger_test(qa_SegmentCollector)
add_trigger_test(qa_TimeBase)
add_trigger_test(qa_WindowCollector)
