add_library(gr-testing INTERFACE)
target_link_libraries(gr-testing INTERFACE gnuradio-core ut-benchmark)
target_include_directories(gr-testing INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                                $<INSTALL_INTERFACE:include/>)

# An OBJECT library as the generated files have namespace-scope variables with dynamic C++ initialization
add_library(GrTestingBlocks SHARED)
gr_generate_block_instantiations(
  GrTestingBlocks
  HEADERS
  include/gnuradio-4.0/testing/NullSources.hpp
  include/gnuradio-4.0/testing/ImChartMonitor.hpp
  include/gnuradio-4.0/testing/TagMonitors.hpp
  include/gnuradio-4.0/testing/Delay.hpp
  include/gnuradio-4.0/testing/PerformanceMonitor.hpp
  include/gnuradio-4.0/testing/SettingsChangeRecorder.hpp)
target_link_libraries(
  GrTestingBlocks
  PUBLIC gnuradio-core
         gnuradio-blocklib-core
         gr-testing
         gnuradio-algorithm)

if(ENABLE_TESTING)
  add_subdirectory(test)
endif()
