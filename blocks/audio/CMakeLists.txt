add_library(gr-audio INTERFACE)
target_link_libraries(gr-audio INTERFACE gnuradio-core gnuradio-algorithm gr-miniaudio)
target_include_directories(gr-audio INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                              $<INSTALL_INTERFACE:include/>)
install(TARGETS gr-audio EXPORT gnuradio4Targets)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/gnuradio-4.0/audio"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/gnuradio-4.0")

gr_add_block_library(
  GrAudioBlocks
  MAKE_SHARED_LIBRARY
  HEADERS
  include/gnuradio-4.0/audio/AudioCommon.hpp
  include/gnuradio-4.0/audio/AudioBlocks.hpp
  include/gnuradio-4.0/audio/AudioFileSource.hpp
  include/gnuradio-4.0/audio/WavSource.hpp
  LINK_LIBRARIES
  gr-audio)

if(TARGET GrAudioBlocksShared AND ENABLE_TESTING)
  add_subdirectory(test)
endif()

if(ENABLE_TESTING AND EMSCRIPTEN)
  add_subdirectory(test/AudioSinkTestApp)
endif()
