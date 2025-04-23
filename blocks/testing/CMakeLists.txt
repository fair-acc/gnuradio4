add_library(gr-testing INTERFACE)
target_link_libraries(gr-testing INTERFACE gnuradio-core ut-benchmark)
target_include_directories(gr-testing INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                                $<INSTALL_INTERFACE:include/>)
if(WIN32)
  target_compile_definitions(gr-testing INTERFACE BUILDING_MYPLUGIN)
endif()

gr_add_block_library(
  GrTestingBlocks
  MAKE_SHARED_LIBRARY
  HEADERS
  include/gnuradio-4.0/testing/NullSources.hpp
  include/gnuradio-4.0/testing/ImChartMonitor.hpp
  include/gnuradio-4.0/testing/TagMonitors.hpp
  include/gnuradio-4.0/testing/Delay.hpp
  include/gnuradio-4.0/testing/PerformanceMonitor.hpp
  include/gnuradio-4.0/testing/SettingsChangeRecorder.hpp
  LINK_LIBRARIES
  gr-testing
  gnuradio-algorithm)

if(TARGET GrTestingBlocksShared AND ENABLE_TESTING)
  add_subdirectory(test)
endif()
