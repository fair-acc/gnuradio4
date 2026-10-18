add_library(gr-localisation INTERFACE)
add_library(gnuradio4::gr-localisation ALIAS gr-localisation)
target_link_libraries(gr-localisation INTERFACE gnuradio4::gnuradio-core gnuradio4::gnuradio-algorithm)
target_include_directories(gr-localisation INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                                     $<INSTALL_INTERFACE:include/>)
install(TARGETS gr-localisation EXPORT gnuradio4Targets)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/gnuradio-4.0/localisation"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/gnuradio-4.0")

gr_add_block_library(
  GrLocalisationBlocks
  MAKE_SHARED_LIBRARY
  MAKE_PLUGIN
  HEADERS
  include/gnuradio-4.0/localisation/PairCrossSpectra.hpp
  include/gnuradio-4.0/localisation/SourceLocation.hpp
  LINK_LIBRARIES
  gr-localisation)

if(ENABLE_TESTING)
  add_subdirectory(test)
endif()

if(ENABLE_EXAMPLES)
  add_subdirectory(src)
endif()
