add_ut_test(qa_Converter)
add_ut_test(qa_Selector)
add_ut_test(qa_sources)
add_ut_test(qa_DataSink)
add_ut_test(qa_TriggerBlocks)

if(ENABLE_BLOCK_REGISTRY AND ENABLE_BLOCK_PLUGINS)
  add_ut_test(qa_BasicKnownBlocks)
  target_link_libraries(qa_BasicKnownBlocks PRIVATE GrBasicBlocks)
endif()
add_ut_test(qa_StreamToDataSet)
add_ut_test(qa_SyncBlock)

message(STATUS "###Python Include Dirs: ${Python3_INCLUDE_DIRS}")
if(PYTHON_AVAILABLE
   AND ENABLE_BLOCK_REGISTRY
   AND ENABLE_BLOCK_PLUGINS)
  add_ut_test(qa_PythonBlock)
endif()
