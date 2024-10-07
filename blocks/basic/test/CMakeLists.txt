add_ut_test(qa_Converter)
add_ut_test(qa_Selector)
add_ut_test(qa_sources)
add_ut_test(qa_DataSink)

if (ENABLE_BLOCK_REGISTRY AND ENABLE_BLOCK_PLUGINS)
    add_ut_test(qa_BasicKnownBlocks)
endif ()
add_ut_test(qa_StreamToDataSet)

message(STATUS "###Python Include Dirs: ${Python3_INCLUDE_DIRS}")
if (PYTHON_AVAILABL AND ENABLE_BLOCK_REGISTRY AND ENABLE_BLOCK_PLUGINS)
    add_ut_test(qa_PythonBlock)
endif ()
