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

add_trigger_test(qa_Acceptance)
add_trigger_test(qa_EventAlgebra)
add_trigger_test(qa_EventCorrelation)
add_trigger_test(qa_Gating)
add_trigger_test(qa_Marble)
add_trigger_test(qa_MultiChannelRecorder)
add_trigger_test(qa_PatternTriggers)
add_trigger_test(qa_Routing)
add_trigger_test(qa_SchmittTriggerBlock)
add_trigger_test(qa_SchmittTriggerDevice)
add_trigger_test(qa_StreamOps)
add_trigger_test(qa_TagBridge)
add_trigger_test(qa_TriggerDevice)
add_trigger_test(qa_TriggerPrimitives)
add_trigger_test(qa_UseCases)
add_trigger_test(qa_ValueTrigger)
add_trigger_test(qa_Windows)

# the shared block library is what this test loads, and an Emscripten plugins package does not build one
if(GR_ENABLE_BLOCK_REGISTRY
   AND INTERNAL_ENABLE_BLOCK_PLUGINS
   AND TARGET gnuradio4::GrTriggerBlocksShared)
  add_trigger_test(qa_TriggerGrc)
  target_link_libraries(qa_TriggerGrc PRIVATE gnuradio4::GrTriggerBlocksShared)
endif()
