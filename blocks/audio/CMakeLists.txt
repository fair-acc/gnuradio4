add_library(gr-audio INTERFACE)
add_library(gnuradio4::gr-audio ALIAS gr-audio)
target_link_libraries(
  gr-audio
  INTERFACE gnuradio4::gnuradio-core
            gnuradio4::gnuradio-algorithm
            gnuradio4::gr-common
            gnuradio4::gr-fileio)
if(NOT EMSCRIPTEN)
  target_link_libraries(gr-audio INTERFACE gnuradio4::gr-libsoundio)
endif()
if(EMSCRIPTEN)
  target_link_options(
    gr-audio
    INTERFACE
    "SHELL:-s AUDIO_WORKLET=1"
    "SHELL:-s WASM_WORKERS=1")
endif()
target_include_directories(gr-audio INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                              $<INSTALL_INTERFACE:include/>)
install(TARGETS gr-audio EXPORT gnuradio4Targets)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/gnuradio-4.0/audio"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/gnuradio-4.0")

gr_add_block_library(
  GrAudioBlocks
  MAKE_SHARED_LIBRARY
  HEADERS
  include/gnuradio-4.0/audio/AudioBackends.hpp
  include/gnuradio-4.0/audio/AudioBlocks.hpp
  include/gnuradio-4.0/audio/EmscriptenAudioBackend.hpp
  include/gnuradio-4.0/audio/SoundIoBackend.hpp
  LINK_LIBRARIES
  gnuradio4::gr-audio)

if(TARGET gnuradio4::GrAudioBlocksShared AND ENABLE_TESTING)
  add_subdirectory(test)
endif()

if(EMSCRIPTEN)
  add_subdirectory(src/AudioDemo)
endif()
