add_ut_test(qa_Onnx)

add_ut_test(qa_OnnxInstallationTest)
target_link_libraries(qa_OnnxInstallationTest PRIVATE gr-onnx gr-testing)

if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    # WASM tests need node.js runner
    set_tests_properties(qa_OnnxInstallationTest PROPERTIES CROSSCOMPILING_EMULATOR "${CMAKE_CROSSCOMPILING_EMULATOR}" )
endif()
