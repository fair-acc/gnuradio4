add_executable(onnx_example0 onnx_example0.cpp)
target_link_libraries(onnx_example0 PRIVATE gr-onnx gnuradio-blocklib-core)

add_executable(onnx_example1 onnx_example1.cpp)
target_link_libraries(onnx_example1 PRIVATE gr-onnx gr-testing gnuradio-blocklib-core)

set(_onnx_models_dir "${CMAKE_CURRENT_SOURCE_DIR}/../models")

if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    # no model path for WASM examples (model must be passed as argument or embedded)
else ()
    if (_ort_minimal)
        set(_model_ext ".ort")
    else ()
        set(_model_ext ".onnx")
    endif ()
    target_compile_definitions(onnx_example0
            PRIVATE MODEL_N1024_PATH="${_onnx_models_dir}/peak_detector_N1024${_model_ext}")
    target_compile_definitions(onnx_example1
            PRIVATE MODEL_N1024_PATH="${_onnx_models_dir}/peak_detector_N1024${_model_ext}")
endif ()
