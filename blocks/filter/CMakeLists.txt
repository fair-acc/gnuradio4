add_library(gr-filter INTERFACE)
target_link_libraries(gr-filter INTERFACE gnuradio-core)
target_include_directories(gr-filter INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                               $<INSTALL_INTERFACE:include/>)

# An OBJECT library as the generated files have namespace-scope variables with dynamic C++ initialization
add_library(GrFilterBlocks SHARED)
gr_generate_block_instantiations(
  GrFilterBlocks
  HEADERS
  include/gnuradio-4.0/filter/FrequencyEstimator.hpp
  include/gnuradio-4.0/filter/time_domain_filter.hpp)
target_link_libraries(
  GrFilterBlocks
  PUBLIC gnuradio-core
         gnuradio-blocklib-core
         gr-filter
         gnuradio-algorithm)

if(ENABLE_TESTING)
  add_subdirectory(test)
endif()
