# gnuradio-core before gnuradio-blocklib-core, matching blocks/timing and blocks/sdr: a coverage build keeps
# PluginLoader's references to the registries alive, and they resolve only if the shared blocklib follows it
add_executable(ex05_ml_peak_detector ex05_ml_peak_detector.cpp)
target_link_libraries(
  ex05_ml_peak_detector
  PRIVATE gnuradio-options
          gnuradio-core
          gnuradio-blocklib-core
          gr-onnx)

add_executable(ex06_ml_vs_classical ex06_ml_vs_classical.cpp)
target_link_libraries(
  ex06_ml_vs_classical
  PRIVATE gnuradio-options
          gnuradio-core
          gnuradio-blocklib-core
          gr-onnx
          gr-testing)

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

# WASM has no filesystem: bake the deliverables into the VFS at the /data path deliverableModelPath() reports. Only .ort
# -- a minimal ONNX Runtime rejects the .onnx protobuf.
if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  set(_onnx_models_dir "${CMAKE_CURRENT_SOURCE_DIR}/../models")
  foreach(_deliverable ex05_peak_detector_single_stage ex05_peak_detector_cascaded)
    target_link_options(ex05_ml_peak_detector PRIVATE
                        "SHELL:--embed-file ${_onnx_models_dir}/${_deliverable}.ort.gz@/data/${_deliverable}.ort.gz")
  endforeach()
endif()
