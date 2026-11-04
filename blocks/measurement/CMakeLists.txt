add_library(gr-measurement INTERFACE)
add_library(gnuradio4::gr-measurement ALIAS gr-measurement)
target_include_directories(gr-measurement INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                                    $<INSTALL_INTERFACE:include/>)
target_link_libraries(gr-measurement INTERFACE gnuradio-core gnuradio-algorithm gr-trigger)

install(TARGETS gr-measurement EXPORT gnuradio4Targets)
install(FILES include/gnuradio-4.0/measurement/EventStatistics.hpp include/gnuradio-4.0/measurement/Measure.hpp
              include/gnuradio-4.0/measurement/MeasurementCondition.hpp
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/gnuradio-4.0/measurement")

# On Emscripten, MAKE_SHARED_LIBRARY already produces a .a; Static would be a duplicate.
set(_measurement_static_opt)
if(NOT EMSCRIPTEN)
  list(APPEND _measurement_static_opt MAKE_STATIC_LIBRARY)
endif()

gr_add_block_library(
  GrMeasurementBlocks
  MAKE_SHARED_LIBRARY
  MAKE_PLUGIN
  ${_measurement_static_opt}
  HEADERS
  include/gnuradio-4.0/measurement/EventStatistics.hpp
  include/gnuradio-4.0/measurement/Measure.hpp
  include/gnuradio-4.0/measurement/MeasurementCondition.hpp
  LINK_LIBRARIES
  gnuradio4::gr-measurement
  gnuradio4::gnuradio-algorithm)

if(ENABLE_TESTING)
  add_subdirectory(test)
endif()
