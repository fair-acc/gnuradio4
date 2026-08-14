add_executable(ex05_ml_peak_detector ex05_ml_peak_detector.cpp)
target_link_libraries(ex05_ml_peak_detector PRIVATE gr-onnx gnuradio-blocklib-core)

add_executable(ex06_ml_vs_classical ex06_ml_vs_classical.cpp)
target_link_libraries(ex06_ml_vs_classical PRIVATE gr-onnx gr-testing gnuradio-blocklib-core)

add_executable(onnx_litmus onnx_litmus.cpp)
target_link_libraries(onnx_litmus PRIVATE gr-onnx gnuradio-blocklib-core)

# one graph-wired example per documented OnnxInference use-case
foreach(
  _example
  ex00_identity
  ex01_elementwise
  ex02_decimating
  ex03_sliding_window
  ex04_multi_output)
  add_executable(${_example} ${_example}.cpp)
  target_link_libraries(${_example} PRIVATE gr-onnx gnuradio-blocklib-core)
endforeach()

set(_onnx_models_dir "${CMAKE_CURRENT_SOURCE_DIR}/../models")

if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  # no model path for WASM examples (model must be passed as argument or embedded)
else()
  if(_ort_minimal)
    set(_model_ext ".ort")
  else()
    set(_model_ext ".onnx")
  endif()
  target_compile_definitions(ex06_ml_vs_classical
                             PRIVATE MODEL_PEAKS_FIXTURE_PATH="${_onnx_models_dir}/peaks_fixture_N1024${_model_ext}")
  target_compile_definitions(ex05_ml_peak_detector
                             PRIVATE MODEL_PEAKS_FIXTURE_PATH="${_onnx_models_dir}/peaks_fixture_N1024${_model_ext}")
  target_compile_definitions(ex00_identity
                             PRIVATE MODEL_IDENTITY_N64_PATH="${_onnx_models_dir}/identity_N64${_model_ext}")
  target_compile_definitions(ex01_elementwise
                             PRIVATE MODEL_AFFINE_N64_PATH="${_onnx_models_dir}/affine_N64${_model_ext}")
  target_compile_definitions(ex02_decimating
                             PRIVATE MODEL_FRAME_MEAN_N64_M4_PATH="${_onnx_models_dir}/frame_mean_N64_M4${_model_ext}")
  target_compile_definitions(
    ex03_sliding_window
    PRIVATE MODEL_FRAME_DELTA_N64_M8_PATH="${_onnx_models_dir}/frame_delta_N64_M8${_model_ext}")
  target_compile_definitions(ex04_multi_output
                             PRIVATE MODEL_MEAN_RMS_N64_PATH="${_onnx_models_dir}/mean_rms_N64${_model_ext}")
endif()
