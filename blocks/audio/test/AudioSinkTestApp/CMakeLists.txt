set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_executable(AudioSinkTestApp main.cpp)

target_link_libraries(
  AudioSinkTestApp
  PRIVATE gnuradio-options
          gnuradio-core
          gnuradio-blocklib-core
          gr-audio)

target_link_options(
  AudioSinkTestApp
  PRIVATE
  "SHELL:-s WASM=1"
  "SHELL:-s EXPORTED_RUNTIME_METHODS=['ccall','FS']"
  "SHELL:-s EXPORTED_FUNCTIONS=['_main','_play_audio_from_uri','_audio_playback_is_done','_finish_audio_playback']"
  "SHELL:-s ASSERTIONS=2"
  "SHELL:-s EXIT_RUNTIME=0"
  "SHELL:-s ALLOW_MEMORY_GROWTH=1")

add_custom_command(
  TARGET AudioSinkTestApp
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/index.html
          $<TARGET_FILE_DIR:AudioSinkTestApp>/index.html)
