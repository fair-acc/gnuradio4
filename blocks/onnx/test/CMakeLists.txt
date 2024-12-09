add_ut_test(qa_OnnxInstallationTest)
target_link_libraries(qa_OnnxInstallationTest PRIVATE gr-onnx gr-testing)

# Model paths
set(_onnx_models_dir "${CMAKE_CURRENT_SOURCE_DIR}/../models")

# qa_OnnxUtils — C++ utility tests (normalise, resample, NMS)
add_ut_test(qa_OnnxUtils)
target_link_libraries(qa_OnnxUtils PRIVATE gr-onnx gr-testing)

# qa_OnnxSession — ORT session lifecycle + inference tests
add_ut_test(qa_OnnxSession)
target_link_libraries(qa_OnnxSession PRIVATE gr-onnx gr-testing)

# qa_OnnxInference — generic inference block tests
add_ut_test(qa_OnnxInference)
target_link_libraries(qa_OnnxInference PRIVATE gr-onnx gr-testing)

# qa_OnnxPeakDetector — ML peak detection block tests
add_ut_test(qa_OnnxPeakDetector)
target_link_libraries(qa_OnnxPeakDetector PRIVATE gr-onnx gr-testing)

# qa_PeakDetector — classical peak detection tests (no ORT dependency)
add_ut_test(qa_PeakDetector)
target_link_libraries(qa_PeakDetector PRIVATE gr-onnx gr-testing)

if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    # WASM: embed model into virtual filesystem, always use .ort format
    set(_model_n1024 "${_onnx_models_dir}/peak_detector_N1024.ort")
    set(_vfs_model "/data/peak_detector_N1024.ort")

    target_compile_definitions(qa_OnnxSession PRIVATE MODEL_N1024_PATH="${_vfs_model}")
    target_link_options(qa_OnnxSession PRIVATE "SHELL:--embed-file ${_model_n1024}@${_vfs_model}")

    target_compile_definitions(qa_OnnxInference PRIVATE MODEL_N1024_PATH="${_vfs_model}")
    target_link_options(qa_OnnxInference PRIVATE "SHELL:--embed-file ${_model_n1024}@${_vfs_model}")

    target_compile_definitions(qa_OnnxPeakDetector PRIVATE MODEL_N1024_PATH="${_vfs_model}")
    target_link_options(qa_OnnxPeakDetector PRIVATE "SHELL:--embed-file ${_model_n1024}@${_vfs_model}")

    # node.js runner for all WASM tests
    foreach (
            _test
            qa_OnnxInstallationTest
            qa_OnnxUtils
            qa_OnnxSession
            qa_OnnxInference
            qa_OnnxPeakDetector
            qa_PeakDetector)
        set_tests_properties(${_test} PROPERTIES CROSSCOMPILING_EMULATOR "${CMAKE_CROSSCOMPILING_EMULATOR}")
    endforeach ()
else ()
    # Native: use filesystem paths directly
    if (_ort_minimal)
        set(_model_ext ".ort")
    else ()
        set(_model_ext ".onnx")
    endif ()

    target_compile_definitions(
            qa_OnnxSession PRIVATE MODEL_N1024_PATH="${_onnx_models_dir}/peak_detector_N1024${_model_ext}")
    target_compile_definitions(qa_OnnxInference
            PRIVATE MODEL_N1024_PATH="${_onnx_models_dir}/peak_detector_N1024${_model_ext}")
    target_compile_definitions(qa_OnnxPeakDetector
            PRIVATE MODEL_N1024_PATH="${_onnx_models_dir}/peak_detector_N1024${_model_ext}")
endif()
