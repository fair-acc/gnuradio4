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
  HEADERS
  include/gnuradio-4.0/filter/FrequencyEstimator.hpp
  include/gnuradio-4.0/filter/SavitzkyGolayFilter.hpp
  include/gnuradio-4.0/filter/SvdDenoiser.hpp
  include/gnuradio-4.0/filter/time_domain_filter.hpp
  LINK_LIBRARIES
  gr-filter
  gnuradio4::gnuradio-algorithm)

if(TARGET gnuradio4::GrFilterBlocksShared AND ENABLE_TESTING)
  add_subdirectory(test)
endif()
