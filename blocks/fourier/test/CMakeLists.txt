add_ut_test(qa_FFT)
target_link_libraries(qa_FFT PRIVATE gr-fourier gr-testing)

add_ut_test(qa_PeakDetector)
target_link_libraries(qa_PeakDetector PRIVATE gr-fourier gr-testing)

add_ut_test(qa_FFTDevice)
target_link_libraries(qa_FFTDevice PRIVATE gr-fourier gr-basic gr-testing)

add_ut_test(qa_FFTSubGraph)
target_link_libraries(qa_FFTSubGraph PRIVATE gr-fourier gr-testing)
if (GR_USE_ADAPTIVE_CPP)
    add_ut_test(qa_FFTPerformance)
    target_link_libraries(qa_FFTPerformance PRIVATE gnuradio-core gr-fourier gr-basic gr-testing)
    set_tests_properties(qa_FFTPerformance PROPERTIES LABELS "sycl;adaptivecpp" TIMEOUT 120
                                                      ENVIRONMENT "ACPP_APPDB_DIR=/tmp/gnuradio4-acpp-appdb;XDG_DATA_HOME=/tmp/gnuradio4-acpp-data")

    add_executable(qa_FFTDeviceSycl qa_FFTDevice.cpp)
    target_compile_definitions(qa_FFTDeviceSycl PRIVATE GR_FFT_DEVICE_TEST)
    target_link_libraries(qa_FFTDeviceSycl PRIVATE gnuradio-core gr-fourier)
    add_test(NAME qa_FFTDeviceSycl COMMAND ${CMAKE_CROSSCOMPILING_EMULATOR} $<TARGET_FILE:qa_FFTDeviceSycl>)
    set_tests_properties(qa_FFTDeviceSycl PROPERTIES LABELS "sycl;adaptivecpp" TIMEOUT 90 SKIP_RETURN_CODE 77
                                                     ENVIRONMENT "ACPP_APPDB_DIR=/tmp/gnuradio4-acpp-appdb;XDG_DATA_HOME=/tmp/gnuradio4-acpp-data")

    add_executable(qa_SyclFFTSweep qa_SyclFFTSweep.cpp)
    target_link_libraries(qa_SyclFFTSweep PRIVATE gnuradio-core gr-fourier)
    add_test(NAME qa_SyclFFTSweep COMMAND ${CMAKE_CROSSCOMPILING_EMULATOR} $<TARGET_FILE:qa_SyclFFTSweep>)
    set_tests_properties(qa_SyclFFTSweep PROPERTIES LABELS "sycl;adaptivecpp" TIMEOUT 120 SKIP_RETURN_CODE 77
                                                    ENVIRONMENT "ACPP_APPDB_DIR=/tmp/gnuradio4-acpp-appdb;XDG_DATA_HOME=/tmp/gnuradio4-acpp-data")
endif ()
