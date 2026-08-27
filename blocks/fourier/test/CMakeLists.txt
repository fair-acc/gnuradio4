add_ut_test(qa_fourier)
target_link_libraries(qa_fourier PRIVATE gr-fourier)

add_ut_test(qa_PeakDetector)
target_link_libraries(qa_PeakDetector PRIVATE gr-fourier gr-testing)
