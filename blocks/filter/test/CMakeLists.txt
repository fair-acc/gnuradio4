add_ut_test(qa_filter)
target_link_libraries(qa_filter PRIVATE gr-filter)

add_ut_test(qa_FilterKnownBlocks)
target_link_libraries(qa_FilterKnownBlocks PRIVATE gr-filter)
