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
  MAKE_PLUGIN
  HEADERS
  include/gnuradio-4.0/testing/NullSources.hpp
  include/gnuradio-4.0/testing/ImChartMonitor.hpp
  include/gnuradio-4.0/testing/TagMonitors.hpp
  include/gnuradio-4.0/testing/Delay.hpp
  include/gnuradio-4.0/testing/PerformanceMonitor.hpp
  include/gnuradio-4.0/testing/SettingsChangeRecorder.hpp
  include/gnuradio-4.0/testing/TestSpectrumGenerator.hpp
  include/gnuradio-4.0/testing/SyntheticPeakSpectrum.hpp
  LINK_LIBRARIES
  gnuradio4::gr-testing
  gnuradio4::gnuradio-algorithm)

# the tests use the header-only interface target, not the shared block library, so they must build wherever testing is
# on -- including a SYCL configuration, which is BUILD_SHARED_LIBS=OFF and so has no `...Shared` target. Gating them on
# it meant any device case was only ever compiled where SYCL was absent, and a device case that always skips asserts
# nothing at all.
if(ENABLE_TESTING)
  add_subdirectory(test)
endif()
