set(localisation_examples
    ex1_pair_delay_report
    ex2_steered_power_projections
    ex3_located_source_report
    ex4_range_heatmap
    ex5_line_waterfall
    ex6_line_bands)
foreach(example IN LISTS localisation_examples)
  add_executable(${example} ${example}.cpp)
  # gnuradio-core before gnuradio-blocklib-core, as in blocks/fourier/src
  target_link_libraries(
    ${example}
    PRIVATE gnuradio-options
            gnuradio-core
            gnuradio-blocklib-core
            gr-localisation
            gr-basic
            gr-fourier
            gr-math
            gr-testing)
  if(TARGET gr-libsoundio AND NOT EMSCRIPTEN)
    target_link_libraries(${example} PRIVATE gr-audio)
  endif()
endforeach()
