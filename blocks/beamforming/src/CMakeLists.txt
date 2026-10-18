foreach(example ex_array_localise_report ex0_beam_display)
  add_executable(${example} ${example}.cpp)
  # gnuradio-core before gnuradio-blocklib-core, as in blocks/fourier/src
  target_link_libraries(
    ${example}
    PRIVATE gnuradio-options
            gnuradio-core
            gnuradio-blocklib-core
            gr-beamforming
            gr-localisation
            gr-basic
            gr-fourier
            gr-math
            gr-testing)
  # gr-audio on the include path switches ArrayScene.hpp to live capture; the synthetic scene runs everywhere
  if(TARGET gr-libsoundio AND NOT EMSCRIPTEN)
    target_link_libraries(${example} PRIVATE gr-audio)
  endif()
endforeach()
