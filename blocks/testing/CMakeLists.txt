add_library(gr-testing INTERFACE)
add_library(gnuradio4::gr-testing ALIAS gr-testing)
target_link_libraries(gr-testing INTERFACE gnuradio4::gnuradio-core gnuradio4::ut-benchmark)
target_include_directories(gr-testing INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                                $<INSTALL_INTERFACE:include/>)
install(TARGETS gr-testing EXPORT gnuradio4Targets)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/gnuradio-4.0/testing"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/gnuradio-4.0")

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
  gnuradio4::gr-testing
  gnuradio4::gnuradio-algorithm)

if(TARGET gnuradio4::GrTestingBlocksShared AND ENABLE_TESTING)
  add_subdirectory(test)
endif()
