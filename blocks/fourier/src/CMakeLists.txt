# needs gr-testing for SyntheticPeakSpectrum
add_executable(ex0_peak_detector_classical ex0_peak_detector_classical.cpp)
# gnuradio-core before gnuradio-blocklib-core, matching blocks/timing and blocks/sdr: a coverage build keeps
# PluginLoader's references to the registries alive, and they resolve only if the shared blocklib follows it
target_link_libraries(
  ex0_peak_detector_classical
  PRIVATE gnuradio-options
          gnuradio-core
          gnuradio-blocklib-core
          gr-fourier
          gr-testing)
