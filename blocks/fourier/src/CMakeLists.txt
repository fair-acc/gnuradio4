# needs gr-testing for SyntheticPeakSpectrum
add_executable(ex0_peak_detector_classical ex0_peak_detector_classical.cpp)
target_link_libraries(ex0_peak_detector_classical PRIVATE gr-fourier gr-testing gnuradio-blocklib-core)
