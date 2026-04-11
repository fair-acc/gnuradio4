add_ut_test(qa_fourier)
target_link_libraries(qa_fourier PRIVATE gr-fourier)

add_ut_test(qa_FFT2)
target_link_libraries(qa_FFT2 PRIVATE gr-fourier gr-basic gr-testing)
if (GR_HAS_GL_COMPUTE)
    target_link_libraries(qa_FFT2 PRIVATE PkgConfig::GR_EGL PkgConfig::GR_GL)
endif ()
