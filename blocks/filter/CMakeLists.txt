add_library(gr-filter INTERFACE)
add_library(gnuradio4::gr-filter ALIAS gr-filter)
target_link_libraries(gr-filter INTERFACE gnuradio4::gnuradio-core)
target_include_directories(gr-filter INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                               $<INSTALL_INTERFACE:include/>)
install(TARGETS gr-filter EXPORT gnuradio4Targets)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/gnuradio-4.0/filter"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/gnuradio-4.0")

gr_add_block_library(
  GrFilterBlocks
  MAKE_SHARED_LIBRARY
  MAKE_PLUGIN
  HEADERS
  include/gnuradio-4.0/filter/DriftResampler.hpp
  include/gnuradio-4.0/filter/FrequencyEstimator.hpp
  include/gnuradio-4.0/filter/FrequencyXlatingFilter.hpp
  include/gnuradio-4.0/filter/HierarchicalPolyphaseChannelizer.hpp
  include/gnuradio-4.0/filter/InterleavedStreamDecimator.hpp
  include/gnuradio-4.0/filter/PolyphaseArbitraryResampler.hpp
  include/gnuradio-4.0/filter/PolyphaseChannelizer.hpp
  include/gnuradio-4.0/filter/PolyphaseSynthesizer.hpp
  include/gnuradio-4.0/filter/RationalResampler.hpp
  include/gnuradio-4.0/filter/SavitzkyGolayFilter.hpp
  include/gnuradio-4.0/filter/SvdDenoiser.hpp
  include/gnuradio-4.0/filter/time_domain_filter.hpp
  LINK_LIBRARIES
  gr-filter
  gnuradio4::gnuradio-algorithm)

# the tests use the header-only `gr-filter` interface target, not the shared block library, so they must build wherever
# testing is on -- including a SYCL configuration, which is BUILD_SHARED_LIBS=OFF and so has no `...Shared` target.
# Gating them on it meant the device cases were only ever compiled where SYCL was absent, and a device case that always
# skips asserts nothing at all.
if(ENABLE_TESTING)
  add_subdirectory(test)
endif()

if(ENABLE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
