set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_executable(AudioDemo main.cpp)

target_link_libraries(
  AudioDemo
  PRIVATE gnuradio-options
          gnuradio-core
          gnuradio-blocklib-core
          gr-audio)

set(AUDIO_DEMO_SHELL "${CMAKE_CURRENT_SOURCE_DIR}/index.html")

target_link_options(
  AudioDemo
  PRIVATE
  "SHELL:--shell-file ${AUDIO_DEMO_SHELL}"
  "SHELL:-s WASM=1"
  "SHELL:-s EXPORTED_RUNTIME_METHODS=['ccall','FS']"
  "SHELL:-s EXPORTED_FUNCTIONS=['_main','_play_audio_from_uri','_stop_playback','_audio_playback_is_running','_start_mic','_stop_mic','_mic_is_running','_gr_requestAllPermissions','_gr_isAudioGranted','_gr_resumeAudioContexts','_malloc','_free']"
  "SHELL:-s ASSERTIONS=2"
  "SHELL:-s EXIT_RUNTIME=0"
  "SHELL:-s ALLOW_MEMORY_GROWTH=1")
set_target_properties(AudioDemo PROPERTIES SUFFIX ".html")

add_custom_target(
  serve_audio
  COMMAND ${CMAKE_COMMAND} -E echo ""
  COMMAND ${CMAKE_COMMAND} -E echo "  GR4 Audio Demo served at:"
  COMMAND ${CMAKE_COMMAND} -E echo ""
  COMMAND ${CMAKE_COMMAND} -E echo "    http://localhost:8080/AudioDemo.html"
  COMMAND ${CMAKE_COMMAND} -E echo ""
  COMMAND ${CMAKE_COMMAND} -E echo "  Press Ctrl+C to stop the server."
  COMMAND ${CMAKE_COMMAND} -E echo ""
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/serve.py 8080 ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS AudioDemo
  COMMENT "Serving Audio Demo on http://localhost:8080/"
  VERBATIM USES_TERMINAL)
