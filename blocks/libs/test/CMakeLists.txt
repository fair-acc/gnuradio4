if(NOT EMSCRIPTEN AND TARGET GrBasicBlocksShared)
  add_ut_test(qa_KnownSharedLibBlocks)
  target_link_libraries(
    qa_KnownSharedLibBlocks
    PRIVATE gnuradio-core
            GrTestingBlocksShared
            GrFileIoBlocksShared
            GrBasicBlocksShared
            GrElectricalBlocksShared
            GrHttpBlocksShared
            GrFilterBlocksShared
            GrFourierBlocksShared)

  add_executable(qa_apptest_KnownStaticLibBlocks qa_apptest_KnownStaticLibBlocks.cpp)
  setup_test(qa_apptest_KnownStaticLibBlocks)
  target_link_libraries(qa_apptest_KnownStaticLibBlocks PRIVATE gnuradio-core GrBasicBlocksStatic)

  add_executable(qa_apptest_LoadingPlainBlocklibs qa_apptest_LoadingPlainBlocklibs.cpp)
  setup_test(qa_apptest_LoadingPlainBlocklibs)
  target_link_libraries(qa_apptest_KnownStaticLibBlocks PRIVATE gnuradio-core gnuradio-blocklib-core)
  set_property(
    TEST qa_apptest_LoadingPlainBlocklibs
    PROPERTY ENVIRONMENT_MODIFICATION
             "GNURADIO4_PLUGIN_DIRECTORIES=set:${CMAKE_CURRENT_BINARY_DIR}/../../../blocks/basic")

endif()
