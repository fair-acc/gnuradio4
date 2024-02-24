add_ut_test(qa_HttpBlock)
target_link_libraries(qa_HttpBlock PRIVATE gr-http)

add_ut_test(qa_HttpKnownBlocks)
target_link_libraries(qa_HttpKnownBlocks PRIVATE gr-http)

if (EMSCRIPTEN)
    target_link_options(qa_HttpBlock PRIVATE --pre-js=${CMAKE_CURRENT_SOURCE_DIR}/pre.js --emrun)
    target_link_options(qa_HttpKnownBlocks PRIVATE --pre-js=${CMAKE_CURRENT_SOURCE_DIR}/pre.js --emrun)
endif ()
