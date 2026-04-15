add_library(gr-electrical INTERFACE)
add_library(gnuradio4::gr-electrical ALIAS gr-electrical)
target_link_libraries(gr-electrical INTERFACE gnuradio4::gnuradio-core gnuradio4::gnuradio-algorithm)
target_include_directories(gr-electrical INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                                   $<INSTALL_INTERFACE:include/>)
install(TARGETS gr-electrical EXPORT gnuradio4Targets)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/gnuradio-4.0/electrical"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/gnuradio-4.0")

gr_add_block_library(
  GrElectricalBlocks
  MAKE_SHARED_LIBRARY
  HEADERS
  include/gnuradio-4.0/electrical/PowerEstimators.hpp
  LINK_LIBRARIES
  gnuradio4::gr-basic
  gr-electrical)

if(TARGET gnuradio4::GrElectricalBlocksShared AND ENABLE_TESTING)
  add_subdirectory(test)
endif()
