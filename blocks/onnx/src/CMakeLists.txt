add_executable(onnx_example0 onnx_example0.cpp)
target_link_libraries(onnx_example0 PRIVATE gr-onnx)

add_custom_command(TARGET onnx_example0 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/ex0_python/peak_detector.onnx
        $<TARGET_FILE_DIR:onnx_example0>
)
