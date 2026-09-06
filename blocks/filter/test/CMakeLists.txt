add_ut_test(qa_filter)
target_link_libraries(qa_filter PRIVATE gr-filter)

add_ut_test(qa_FrequencyEstimator)
target_link_libraries(qa_FrequencyEstimator PRIVATE gr-filter)

add_ut_test(qa_SvdDenoiser)
target_link_libraries(qa_SvdDenoiser PRIVATE gr-filter)

add_ut_test(qa_SavitzkyGolayFilter)

add_ut_test(qa_RationalResampler)

add_ut_test(qa_FastConvolutionFilter)

add_ut_test(qa_DriftResampler)

add_ut_test(qa_Correlator)

add_ut_test(qa_FrameStatistics)
target_link_libraries(qa_SavitzkyGolayFilter PRIVATE gr-filter)
