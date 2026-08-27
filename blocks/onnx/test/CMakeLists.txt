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

# qa_PeakDetector — classical peak detection tests: moved to blocks/fourier/test/CMakeLists.txt (the classical detector
# is a generic spectrum-domain block, no ORT dependency)

if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  # WASM has no host filesystem: these lists must mirror the modelPath() calls in each source
  function(embed_onnx_models _target)
    foreach(_model IN LISTS ARGN)
      target_link_options(${_target} PRIVATE "SHELL:--embed-file ${_onnx_models_dir}/${_model}@/data/${_model}")
    endforeach()
  endfunction()

  embed_onnx_models(
    qa_OnnxSession
    identity_N64.ort
    identity_N64.ort.gz
    identity_N64_M4.ort.gz
    peaks_fixture_N1024.ort.gz
    peaks_fixture_N1024_M16.ort.gz
    rank1_scale_N16.ort.gz
    rank2_scale_N16.ort.gz
    rank2_dynbatch_scale_N16.ort.gz
    rank4_scale_N16.ort.gz)

  embed_onnx_models(
    qa_OnnxInference
    identity_N64_M4.ort.gz
    peaks_fixture_N1024.ort.gz
    peaks_fixture_N1024_M16.ort.gz
    peaks_fixture_N512.ort.gz)

  embed_onnx_models(
    qa_OnnxInferenceUseCases
    affine_N64.ort.gz
    frame_delta_N64_M8.ort.gz
    frame_mean_N64_M4.ort.gz
    mean_rms_N64.ort.gz)

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
endif()
