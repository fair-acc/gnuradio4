if(NOT APPLE)
  add_ut_test(qa_HttpBlock)
  target_link_libraries(qa_HttpBlock PRIVATE gr-http gr-testing httplib::httplib)

  # binds a local test server, so it must not run beside another test that binds one
  set_tests_properties(qa_HttpBlock PROPERTIES RESOURCE_LOCK http_test_server)

  if(EMSCRIPTEN)
    target_link_options(
      qa_HttpBlock
      PRIVATE
      --pre-js=${CMAKE_CURRENT_SOURCE_DIR}/pre.js
      --emrun)
  endif()
endif()
