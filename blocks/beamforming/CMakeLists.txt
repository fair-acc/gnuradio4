add_library(gr-beamforming INTERFACE)
add_library(gnuradio4::gr-beamforming ALIAS gr-beamforming)
target_link_libraries(gr-beamforming INTERFACE gnuradio4::gnuradio-core gnuradio4::gnuradio-algorithm)
target_include_directories(gr-beamforming INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                                    $<INSTALL_INTERFACE:include/>)
install(TARGETS gr-beamforming EXPORT gnuradio4Targets)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/gnuradio-4.0/beamforming"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/gnuradio-4.0")

gr_add_block_library(
  GrBeamformingBlocks MAKE_SHARED_LIBRARY MAKE_PLUGIN HEADERS include/gnuradio-4.0/beamforming/Beamformer.hpp
                                                              include/gnuradio-4.0/beamforming/CrossSpectralMatrix.hpp
  LINK_LIBRARIES gr-beamforming)

if(ENABLE_TESTING)
  add_subdirectory(test)
endif()

if(ENABLE_EXAMPLES)
  add_subdirectory(src)
endif()

if(ENABLE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
