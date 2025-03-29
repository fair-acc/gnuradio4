if(NOT EMSCRIPTEN)
  add_ut_test(qa_KnownLibBlocks)
  target_link_libraries(qa_KnownLibBlocks PRIVATE gnuradio-core GnuRadioBlockLib)

  add_executable(qa_apptest_LoadingPlainBlocklibs qa_apptest_LoadingPlainBlocklibs.cpp)
  setup_test(qa_apptest_LoadingPlainBlocklibs)
  # target_link_libraries(qa_apptest_LoadingPlainBlocklibs PUBLIC gnuradio-core gnuradio-blocklib-core)
  set_property(
    TEST qa_apptest_LoadingPlainBlocklibs
    PROPERTY ENVIRONMENT_MODIFICATION
             "GNURADIO4_PLUGIN_DIRECTORIES=set:${CMAKE_CURRENT_BINARY_DIR}/../../../blocks/basic")
endif()
