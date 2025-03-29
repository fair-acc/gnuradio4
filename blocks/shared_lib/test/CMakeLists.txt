if(NOT EMSCRIPTEN)
  add_ut_test(qa_KnownLibBlocks)
  target_link_libraries(qa_KnownLibBlocks PRIVATE gnuradio-core GnuRadioBlockLib)
endif()
