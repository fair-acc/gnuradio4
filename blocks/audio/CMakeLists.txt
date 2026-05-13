add_library(gr-audio INTERFACE)
add_library(gnuradio4::gr-audio ALIAS gr-audio)
target_link_libraries(
  gr-audio
  INTERFACE gnuradio4::gnuradio-core
            gnuradio4::gnuradio-algorithm
            gnuradio4::gr-common
            gnuradio4::gr-fileio)
target_include_directories(gr-audio INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                              $<INSTALL_INTERFACE:include/>)

if(EMSCRIPTEN)
  target_link_options(
    gr-audio
    INTERFACE
    "SHELL:-s AUDIO_WORKLET=1"
    "SHELL:-s WASM_WORKERS=1")
else()
  gr4_resolve_dependency(libsoundio)
  if(NOT libsoundio_FOUND)
    message(STATUS "blocks/audio: libsoundio not available — skipping")
    return()
  endif()

  # IMPORTED targets come from FindLibSoundIo.cmake (system path) and are already wired with INCLUDE_DIRECTORIES +
  # IMPORTED_LOCATION; the platform libraries (ALSA / PulseAudio / CoreAudio etc.) are baked into the system .so via its
  # own SONAME deps, so we only do post-build wiring on the fetched-and-built target.
  get_target_property(_libsoundio_imported libsoundio_static IMPORTED)

  if(NOT _libsoundio_imported)
    target_include_directories(
      libsoundio_static ${CMAKE_EXT_DEP_WARNING_GUARD} PUBLIC $<BUILD_INTERFACE:${libsoundio_SOURCE_DIR}>
                                                              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
    if(APPLE)
      find_library(GR_LIBSOUNDIO_COREAUDIO_LIBRARY CoreAudio REQUIRED)
      find_library(GR_LIBSOUNDIO_COREFOUNDATION_LIBRARY CoreFoundation REQUIRED)
      find_library(GR_LIBSOUNDIO_AUDIOUNIT_LIBRARY AudioUnit REQUIRED)
      target_link_libraries(
        libsoundio_static PUBLIC ${GR_LIBSOUNDIO_COREAUDIO_LIBRARY} ${GR_LIBSOUNDIO_COREFOUNDATION_LIBRARY}
                                 ${GR_LIBSOUNDIO_AUDIOUNIT_LIBRARY})
    endif()
    if(DEFINED ALSA_LIBRARY
       AND NOT
           ALSA_LIBRARY
           MATCHES
           "NOTFOUND")
      target_link_libraries(libsoundio_static PUBLIC ${ALSA_LIBRARY})
    endif()
    if(DEFINED PULSEAUDIO_LIBRARY
       AND NOT
           PULSEAUDIO_LIBRARY
           MATCHES
           "NOTFOUND")
      target_link_libraries(libsoundio_static PUBLIC ${PULSEAUDIO_LIBRARY})
    endif()
    install(TARGETS libsoundio_static EXPORT gnuradio4Targets)
    install(DIRECTORY "${libsoundio_SOURCE_DIR}/soundio" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
    add_library(gnuradio4::libsoundio_static ALIAS libsoundio_static)
  else()
    set_property(GLOBAL PROPERTY GR4_HAS_SYSTEM_LIBSOUNDIO TRUE)
  endif()

  add_library(gr-libsoundio INTERFACE)
  add_library(gnuradio4::gr-libsoundio ALIAS gr-libsoundio)
  target_link_libraries(gr-libsoundio INTERFACE libsoundio_static)
  install(TARGETS gr-libsoundio EXPORT gnuradio4Targets)

  target_link_libraries(gr-audio INTERFACE gnuradio4::gr-libsoundio)
endif()

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
