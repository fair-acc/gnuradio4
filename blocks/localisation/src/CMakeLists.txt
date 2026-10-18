add_library(
  gr-localisation-map-example STATIC
  SteeredPowerMapExample.cpp
  SteeredPowerMapExample_history.cpp
  SteeredPowerMapExample_angles.cpp
  SteeredPowerMapExample_bands.cpp
  SteeredPowerRange.cpp)
set(localisation_examples
    ex1_pair_delays
    ex2_steered_power_map
    ex3_source_from_delays
    ex4_range_heatmap
    ex5_line_waterfall)
foreach(example IN LISTS localisation_examples)
  add_executable(${example} ${example}.cpp)
endforeach()
# the example library first, so the registry references it pulls in from gnuradio-core precede gnuradio-blocklib-core
foreach(
  example
  ex2_steered_power_map
  ex3_source_from_delays
  ex4_range_heatmap
  ex5_line_waterfall)
  target_link_libraries(${example} PRIVATE gr-localisation-map-example)
endforeach()
foreach(target IN ITEMS ${localisation_examples} gr-localisation-map-example)
  # gnuradio-core before gnuradio-blocklib-core, as in blocks/fourier/src
  target_link_libraries(
    ${target}
    PRIVATE gnuradio-options
            gnuradio-core
            gnuradio-blocklib-core
            gr-localisation
            gr-basic
            gr-fourier
            gr-math
            gr-testing)
  if(TARGET gr-libsoundio AND NOT EMSCRIPTEN)
    target_link_libraries(${target} PRIVATE gr-audio)
  endif()
endforeach()
