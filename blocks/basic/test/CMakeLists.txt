add_ut_test(qa_Selector)
add_ut_test(qa_sources)
add_ut_test(qa_DataSink)
add_ut_test(qa_BasicKnownBlocks)

message(STATUS "###Python Include Dirs: ${Python3_INCLUDE_DIRS}")
if(PYTHON_AVAILABLE)
    add_ut_test(qa_PythonBlock)
endif()
