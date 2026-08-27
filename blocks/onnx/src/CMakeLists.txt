# gnuradio-core before gnuradio-blocklib-core, matching blocks/timing and blocks/sdr: a coverage build keeps
# PluginLoader's references to the registries alive, and they resolve only if the shared blocklib follows it
add_executable(ex05_ml_peak_detector ex05_ml_peak_detector.cpp)
target_link_libraries(
  ex05_ml_peak_detector
  PRIVATE gnuradio-options
          gnuradio-core
          gnuradio-blocklib-core
          gr-onnx)

# one graph-wired example per documented OnnxInference use-case
foreach(
  _example
  ex00_identity
  ex01_elementwise
  ex02_decimating
  ex03_sliding_window
  ex04_multi_output)
  add_executable(${_example} ${_example}.cpp)
  target_link_libraries(
    ${_example}
    PRIVATE gnuradio-options
            gnuradio-core
            gnuradio-blocklib-core
            gr-onnx)
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
  target_compile_definitions(ex05_ml_peak_detector
                             PRIVATE MODEL_PEAKS_FIXTURE_PATH="${_onnx_models_dir}/peaks_fixture_N1024${_model_ext}")
  target_compile_definitions(ex00_identity
                             PRIVATE MODEL_IDENTITY_N64_PATH="${_onnx_models_dir}/identity_N64${_model_ext}")
  target_compile_definitions(ex01_elementwise
                             PRIVATE MODEL_AFFINE_N64_PATH="${_onnx_models_dir}/affine_N64${_model_ext}")
  target_compile_definitions(ex02_decimating
                             PRIVATE MODEL_FRAME_MEAN_N64_M4_PATH="${_onnx_models_dir}/frame_mean_N64_M4${_model_ext}")
  target_compile_definitions(
    ex03_sliding_window PRIVATE MODEL_FRAME_DELTA_N64_M8_PATH="${_onnx_models_dir}/frame_delta_N64_M8${_model_ext}")
  target_compile_definitions(ex04_multi_output
                             PRIVATE MODEL_MEAN_RMS_N64_PATH="${_onnx_models_dir}/mean_rms_N64${_model_ext}")

endif()

# WASM has no filesystem: bake the deliverables into the VFS at the /data path deliverableModelPath() reports. Only .ort
# -- a minimal ONNX Runtime rejects the .onnx protobuf.
if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  set(_onnx_models_dir "${CMAKE_CURRENT_SOURCE_DIR}/../models")
  foreach(_deliverable ex05_peak_detector_single_stage ex05_peak_detector_cascaded)
    target_link_options(ex05_ml_peak_detector PRIVATE
                        "SHELL:--embed-file ${_onnx_models_dir}/${_deliverable}.ort.gz@/data/${_deliverable}.ort.gz")
  endforeach()
endif()
