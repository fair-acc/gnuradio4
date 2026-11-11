add_ut_test(qa_BasicFileIo)
add_ut_test(qa_WavFile)
target_link_libraries(
  qa_WavFile
  PRIVATE gr-fileio
          gr-basic
          gr-trigger
          gr-testing
          httplib::httplib)
