add_ut_test(qa_algorithm_fourier)
add_ut_test(qa_DataSetEstimators)
add_ut_test(qa_FilterTool)
add_ut_test(qa_ImCanvas)
add_ut_test(qa_ImChart)
add_ut_test(qa_ImGraph)
add_ut_test(qa_SchmittTrigger)
add_ut_test(qa_SimdFFT)
add_ut_test(qa_FileIo)
add_ut_test(qa_FileIoHelpers)

if(EMSCRIPTEN)
  # Simple HTTP server used only for unit tests.
  target_link_options(
    qa_FileIo
    PRIVATE
    --pre-js=${CMAKE_CURRENT_SOURCE_DIR}/fileio_pre.js
    --emrun)
endif()

if(EMSCRIPTEN)
  # FileIo wasm test application for manual test in a web browser.
  add_subdirectory(FileioTestApp)
endif()

target_link_libraries(qa_algorithm_fourier PRIVATE gnuradio-algorithm)
target_link_libraries(qa_DataSetEstimators PRIVATE gnuradio-algorithm)
target_link_libraries(qa_FilterTool PRIVATE gnuradio-algorithm)
target_link_libraries(qa_ImCanvas PRIVATE gnuradio-algorithm)
target_link_libraries(qa_ImChart PRIVATE gnuradio-algorithm)
target_link_libraries(qa_ImGraph PRIVATE gnuradio-algorithm)
target_link_libraries(qa_SchmittTrigger PRIVATE gnuradio-algorithm)
target_link_libraries(qa_SimdFFT PRIVATE gnuradio-algorithm)
target_link_libraries(qa_FileIo PRIVATE gnuradio-algorithm)
target_link_libraries(qa_FileIoHelpers PRIVATE gnuradio-algorithm)

add_executable(example_ImChart example_ImChart.cpp)
target_link_libraries(example_ImChart PRIVATE gnuradio-algorithm)
