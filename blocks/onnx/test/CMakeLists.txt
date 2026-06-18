add_ut_test(qa_OnnxInstallationTest)
target_link_libraries(qa_OnnxInstallationTest PRIVATE gr-onnx gr-testing)

# Model paths
set(_onnx_models_dir "${CMAKE_CURRENT_SOURCE_DIR}/../models")

# qa_OnnxPreprocess — preprocessing utility tests (normalisation modes + ExprTk)
add_ut_test(qa_OnnxPreprocess)
target_link_libraries(qa_OnnxPreprocess PRIVATE gr-onnx gr-testing)

# qa_OnnxSession — ORT session lifecycle + inference tests
add_ut_test(qa_OnnxSession)
target_link_libraries(qa_OnnxSession PRIVATE gr-onnx gr-testing)

# qa_OnnxInference — generic inference block tests
add_ut_test(qa_OnnxInference)
target_link_libraries(qa_OnnxInference PRIVATE gr-onnx gr-testing)

# qa_OnnxInferenceUseCases — the four documented rate configurations on analytic fixtures
add_ut_test(qa_OnnxInferenceUseCases)
target_link_libraries(qa_OnnxInferenceUseCases PRIVATE gr-onnx gr-testing)

# qa_PeakDetector — classical peak detection tests: moved to blocks/fourier/test/CMakeLists.txt
# (the classical detector is a generic spectrum-domain block, no ORT dependency)

if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  # WASM: embed model into virtual filesystem, always use .ort format
  set(_model_n1024 "${_onnx_models_dir}/peak_detector_N1024.ort")
  set(_vfs_model_n1024 "/data/peak_detector_N1024.ort")
  set(_model_n64_m4 "${_onnx_models_dir}/identity_N64_M4.ort")
  set(_vfs_model_n64_m4 "/data/identity_N64_M4.ort")
  set(_model_peaks_fixture "${_onnx_models_dir}/peaks_fixture_N1024.ort")
  set(_vfs_model_peaks_fixture "/data/peaks_fixture_N1024.ort")
  set(_model_peaks_m16_fixture "${_onnx_models_dir}/peaks_fixture_M16_N1024.ort")
  set(_vfs_model_peaks_m16_fixture "/data/peaks_fixture_M16_N1024.ort")

  target_compile_definitions(qa_OnnxSession PRIVATE MODEL_N1024_PATH="${_vfs_model_n1024}"
                                                    MODEL_N64_M4_PATH="${_vfs_model_n64_m4}")
  target_link_options(
    qa_OnnxSession
    PRIVATE
    "SHELL:--embed-file ${_model_n1024}@${_vfs_model_n1024}"
    "SHELL:--embed-file ${_model_n64_m4}@${_vfs_model_n64_m4}")

  target_compile_definitions(
    qa_OnnxInference PRIVATE MODEL_N1024_PATH="${_vfs_model_n1024}" MODEL_N64_M4_PATH="${_vfs_model_n64_m4}"
                             MODEL_PEAKS_FIXTURE_PATH="${_vfs_model_peaks_fixture}")
  target_link_options(
    qa_OnnxInference
    PRIVATE
    "SHELL:--embed-file ${_model_n1024}@${_vfs_model_n1024}"
    "SHELL:--embed-file ${_model_n64_m4}@${_vfs_model_n64_m4}"
    "SHELL:--embed-file ${_model_peaks_fixture}@${_vfs_model_peaks_fixture}")

  set(_model_affine "${_onnx_models_dir}/affine_N64.ort")
  set(_model_frame_mean "${_onnx_models_dir}/frame_mean_N64_M4.ort")
  set(_model_frame_delta "${_onnx_models_dir}/frame_delta_N64_M8.ort")
  set(_model_mean_rms "${_onnx_models_dir}/mean_rms_N64.ort")
  target_compile_definitions(
    qa_OnnxInferenceUseCases
    PRIVATE MODEL_AFFINE_N64_PATH="/data/affine_N64.ort"
            MODEL_FRAME_MEAN_N64_M4_PATH="/data/frame_mean_N64_M4.ort"
            MODEL_FRAME_DELTA_N64_M8_PATH="/data/frame_delta_N64_M8.ort"
            MODEL_MEAN_RMS_N64_PATH="/data/mean_rms_N64.ort")
  target_link_options(
    qa_OnnxInferenceUseCases
    PRIVATE
    "SHELL:--embed-file ${_model_affine}@/data/affine_N64.ort"
    "SHELL:--embed-file ${_model_frame_mean}@/data/frame_mean_N64_M4.ort"
    "SHELL:--embed-file ${_model_frame_delta}@/data/frame_delta_N64_M8.ort"
    "SHELL:--embed-file ${_model_mean_rms}@/data/mean_rms_N64.ort")

  # node.js runner for all WASM tests
  foreach(
    _test
    qa_OnnxInstallationTest
    qa_OnnxPreprocess
    qa_OnnxSession
    qa_OnnxInference
    qa_OnnxInferenceUseCases)
    set_tests_properties(${_test} PROPERTIES CROSSCOMPILING_EMULATOR "${CMAKE_CROSSCOMPILING_EMULATOR}")
  endforeach()
else()
  # Native: use filesystem paths directly
  if(_ort_minimal)
    set(_model_ext ".ort")
  else()
    set(_model_ext ".onnx")
  endif()

  target_compile_definitions(
    qa_OnnxSession
    PRIVATE MODEL_N1024_PATH="${_onnx_models_dir}/peak_detector_N1024${_model_ext}"
            MODEL_N64_M4_PATH="${_onnx_models_dir}/identity_N64_M4${_model_ext}"
            MODEL_HISTORY_N1024_M16_PATH="${_onnx_models_dir}/peak_detector_history_N1024_M16${_model_ext}")
  target_compile_definitions(
    qa_OnnxInference
    PRIVATE MODEL_N1024_PATH="${_onnx_models_dir}/peak_detector_N1024${_model_ext}"
            MODEL_N4096_PATH="${_onnx_models_dir}/peak_detector_N4096${_model_ext}"
            MODEL_N64_M4_PATH="${_onnx_models_dir}/identity_N64_M4${_model_ext}"
            MODEL_HISTORY_N1024_M16_PATH="${_onnx_models_dir}/peak_detector_history_N1024_M16${_model_ext}"
            MODEL_PEAKS_FIXTURE_PATH="${_onnx_models_dir}/peaks_fixture_N1024${_model_ext}")
  target_compile_definitions(
    qa_OnnxInferenceUseCases
    PRIVATE MODEL_AFFINE_N64_PATH="${_onnx_models_dir}/affine_N64${_model_ext}"
            MODEL_FRAME_MEAN_N64_M4_PATH="${_onnx_models_dir}/frame_mean_N64_M4${_model_ext}"
            MODEL_FRAME_DELTA_N64_M8_PATH="${_onnx_models_dir}/frame_delta_N64_M8${_model_ext}"
            MODEL_MEAN_RMS_N64_PATH="${_onnx_models_dir}/mean_rms_N64${_model_ext}")
endif()
