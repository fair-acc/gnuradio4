# blocks/analog/test/CMakeLists.txt
add_ut_test(qa_FrequencyMod)             
target_link_libraries(qa_FrequencyMod PRIVATE gr-analog)

add_ut_test(qa_QuadratureDemod)          
target_link_libraries(qa_QuadratureDemod PRIVATE gr-analog)

add_ut_test(qa_FmDet)
target_link_libraries(qa_FmDet PRIVATE gr-analog)
target_compile_options(qa_FmDet PRIVATE -Wno-conversion -Wno-error=conversion)

add_ut_test(qa_AmDemod)
target_link_libraries(qa_AmDemod PRIVATE gr-analog)
target_compile_options(qa_AmDemod PRIVATE -Wno-conversion -Wno-error=conversion)

add_ut_test(qa_PhaseModulator)
target_link_libraries(qa_PhaseModulator PRIVATE gr-analog)

add_ut_test(qa_Agc)
target_link_libraries(qa_Agc PRIVATE gr-analog)
target_compile_options(qa_Agc PRIVATE -Wno-conversion -Wno-error=conversion)

add_ut_test(qa_Agc2)
target_link_libraries(qa_Agc2 PRIVATE gr-analog)
target_compile_options(qa_Agc2 PRIVATE -Wno-conversion -Wno-error=conversion)