add_ut_test(qa_fourier)
target_link_libraries(qa_fourier PRIVATE gr-fourier)

add_ut_test(qa_FFT2)
target_link_libraries(qa_FFT2 PRIVATE gr-fourier gr-basic gr-testing)
if (GR_HAS_GL_COMPUTE)
    target_link_libraries(qa_FFT2 PRIVATE PkgConfig::GR_EGL PkgConfig::GR_GL)
endif ()

if (GR_USE_ADAPTIVE_CPP)
    add_executable(qa_FFT2Device qa_FFT2.cpp)
    target_compile_definitions(qa_FFT2Device PRIVATE GR_FFT2_DEVICE_TEST)
    target_link_libraries(qa_FFT2Device PRIVATE gnuradio-core gr-fourier)
    add_test(NAME qa_FFT2Device COMMAND ${CMAKE_CROSSCOMPILING_EMULATOR} $<TARGET_FILE:qa_FFT2Device>)
    set_tests_properties(qa_FFT2Device PROPERTIES LABELS "sycl;adaptivecpp" TIMEOUT 90 SKIP_RETURN_CODE 77
                                                   ENVIRONMENT "ACPP_APPDB_DIR=/tmp/gnuradio4-acpp-appdb;XDG_DATA_HOME=/tmp/gnuradio4-acpp-data")
endif ()
