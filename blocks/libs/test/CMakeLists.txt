if(NOT EMSCRIPTEN AND TARGET gnuradio4::GrBasicBlocksShared)
  add_ut_test(qa_KnownSharedLibBlocks)
  target_link_libraries(
    qa_KnownSharedLibBlocks
    PRIVATE gnuradio-core
            gnuradio4::GrAudioBlocksShared
            gnuradio4::GrTestingBlocksShared
            gnuradio4::GrFileIoBlocksShared
            gnuradio4::GrBasicBlocksShared
            gnuradio4::GrElectricalBlocksShared
            gnuradio4::GrHttpBlocksShared
            gnuradio4::GrFilterBlocksShared
            gnuradio4::GrFourierBlocksShared)

  add_executable(qa_apptest_KnownStaticLibBlocks qa_apptest_KnownStaticLibBlocks.cpp)
  setup_test(qa_apptest_KnownStaticLibBlocks)
  target_link_libraries(qa_apptest_KnownStaticLibBlocks PRIVATE gnuradio-core gnuradio4::GrBasicBlocksStatic)

  add_executable(qa_apptest_LoadingPlainBlocklibs qa_apptest_LoadingPlainBlocklibs.cpp)
  setup_test(qa_apptest_LoadingPlainBlocklibs)
  target_link_libraries(qa_apptest_LoadingPlainBlocklibs PRIVATE gnuradio-core gnuradio-blocklib-core)
  set_property(
    TEST qa_apptest_LoadingPlainBlocklibs
    PROPERTY ENVIRONMENT_MODIFICATION
             "GNURADIO4_PLUGIN_DIRECTORIES=set:${CMAKE_CURRENT_BINARY_DIR}/../../../blocks/basic")

endif()
