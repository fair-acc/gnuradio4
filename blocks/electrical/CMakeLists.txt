add_library(gr-electrical INTERFACE)
target_link_libraries(gr-electrical INTERFACE gnuradio-core gnuradio-algorithm)
target_include_directories(gr-electrical INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                                   $<INSTALL_INTERFACE:include/>)

gr_add_block_library(GrElectricalBlocks
  MAKE_SHARED_LIBRARY MAKE_STATIC_LIBRARY
  HEADERS include/gnuradio-4.0/electrical/PowerEstimators.hpp
  LINK_LIBRARIES gr-basic gr-electrical)

if(ENABLE_TESTING)
  add_subdirectory(test)
endif()
