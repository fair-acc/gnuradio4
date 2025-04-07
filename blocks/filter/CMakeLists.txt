add_library(gr-filter INTERFACE)
target_link_libraries(gr-filter INTERFACE gnuradio-core)
target_include_directories(gr-filter INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                               $<INSTALL_INTERFACE:include/>)

gr_add_block_library(
  GrFilterBlocks
  MAKE_SHARED_LIBRARY
  HEADERS
  include/gnuradio-4.0/filter/FrequencyEstimator.hpp
  include/gnuradio-4.0/filter/time_domain_filter.hpp
  LINK_LIBRARIES
  gr-filter
  gnuradio-algorithm)

if(TARGET GrFilterBlocksShared AND ENABLE_TESTING)
  add_subdirectory(test)
endif()
