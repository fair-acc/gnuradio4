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

add_trigger_test(qa_EventReduce)
add_trigger_test(qa_Coincidence)
add_trigger_test(qa_EventShaping)
add_trigger_test(qa_EventFilter)
add_trigger_test(qa_EventSchema)
add_trigger_test(qa_EventStore)
add_trigger_test(qa_Gate)
add_trigger_test(qa_Marble)
add_trigger_test(qa_TagBridge)
add_trigger_test(qa_MultiChannelRecorder)
add_trigger_test(qa_PatternTrigger)
add_trigger_test(qa_Predicate)
add_trigger_test(qa_SampleAndHold)
add_trigger_test(qa_SchmittTriggerBlock)
add_trigger_test(qa_SchmittTriggerDevice)
add_trigger_test(qa_SegmentCollector)
add_trigger_test(qa_TakeSkip)
add_trigger_test(qa_TimeBase)
add_trigger_test(qa_TimeInterval)
add_trigger_test(qa_UseCase1_MultiChannelTrigger)
add_trigger_test(qa_TriggerWatchdog)
add_trigger_test(qa_ValueTrigger)
add_trigger_test(qa_WindowCollector)
