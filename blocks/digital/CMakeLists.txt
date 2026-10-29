add_library(gr-digital INTERFACE)
add_library(gnuradio4::gr-digital ALIAS gr-digital)
target_link_libraries(gr-digital INTERFACE gnuradio4::gnuradio-core gnuradio4::gnuradio-algorithm)
target_include_directories(gr-digital INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                                $<INSTALL_INTERFACE:include/>)
install(TARGETS gr-digital EXPORT gnuradio4Targets)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/gnuradio-4.0/digital"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/gnuradio-4.0")

gr_add_block_library(
  GrDigitalBlocks
  MAKE_SHARED_LIBRARY
  MAKE_PLUGIN
  HEADERS
  include/gnuradio-4.0/digital/Constellation.hpp
  include/gnuradio-4.0/digital/CostasLoop.hpp
  include/gnuradio-4.0/digital/DifferentialCoding.hpp
  include/gnuradio-4.0/digital/ManchesterCoding.hpp
  include/gnuradio-4.0/digital/RDSDecoder.hpp
  include/gnuradio-4.0/digital/RootRaisedCosineFilter.hpp
  include/gnuradio-4.0/digital/SymbolSync.hpp
  LINK_LIBRARIES
  gr-digital)

if(ENABLE_TESTING)
  add_subdirectory(test)
endif()
