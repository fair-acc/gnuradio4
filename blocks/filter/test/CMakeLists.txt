add_ut_test(qa_filter)
target_link_libraries(qa_filter PRIVATE gr-filter)

add_ut_test(qa_FrequencyEstimator)
target_link_libraries(qa_FrequencyEstimator PRIVATE gr-filter)
