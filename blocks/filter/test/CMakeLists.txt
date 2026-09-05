add_ut_test(qa_filter)
target_link_libraries(qa_filter PRIVATE gr-filter gr-testing)

add_ut_test(qa_FrequencyEstimator)
target_link_libraries(qa_FrequencyEstimator PRIVATE gr-filter)

add_ut_test(qa_SvdDenoiser)
target_link_libraries(qa_SvdDenoiser PRIVATE gr-filter)

add_ut_test(qa_SavitzkyGolayFilter)
add_ut_test(qa_FastConvolutionFilter)
target_include_directories(qa_FastConvolutionFilter PRIVATE ${PROJECT_SOURCE_DIR}/core/test)
target_link_libraries(qa_SavitzkyGolayFilter PRIVATE gr-filter)
