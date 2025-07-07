# blocks/analog/test/CMakeLists.txt
add_ut_test(qa_Analog)                       # helper macro defined by GNURadio-4
target_link_libraries(qa_Analog PRIVATE gr-analog)
