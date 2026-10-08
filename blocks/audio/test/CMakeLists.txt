add_ut_test(qa_Audio)
target_link_libraries(qa_Audio PRIVATE gr-audio gr-testing httplib::httplib)

if(EMSCRIPTEN)
  add_executable(WebAudioIntegrationProbe WebAudioIntegrationProbe.cpp)
  target_link_libraries(WebAudioIntegrationProbe PRIVATE gr-audio)
  target_link_options(
    WebAudioIntegrationProbe
    PRIVATE
    "SHELL:--shell-file ${CMAKE_CURRENT_SOURCE_DIR}/WebAudioIntegrationProbe.html"
    "SHELL:-s EXPORTED_RUNTIME_METHODS=['ccall']"
    "SHELL:-s EXIT_RUNTIME=0")
  set_target_properties(WebAudioIntegrationProbe PROPERTIES SUFFIX ".html")
  set_property(
    TARGET WebAudioIntegrationProbe
    APPEND
    PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/WebAudioIntegrationProbe.html")
endif()
