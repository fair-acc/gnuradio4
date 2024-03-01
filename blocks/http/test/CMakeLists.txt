add_ut_test(qa_HttpBlock)
target_link_libraries(qa_HttpBlock PRIVATE gr-http)

if (EMSCRIPTEN)
    target_link_options(qa_HttpBlock PRIVATE --pre-js=${CMAKE_CURRENT_SOURCE_DIR}/pre.js --emrun)
endif ()
