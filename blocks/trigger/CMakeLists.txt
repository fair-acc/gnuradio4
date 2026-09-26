add_library(gr-trigger INTERFACE)
add_library(gnuradio4::gr-trigger ALIAS gr-trigger)
target_include_directories(gr-trigger INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                                $<INSTALL_INTERFACE:include/>)
target_link_libraries(gr-trigger INTERFACE gnuradio-core gr-common gnuradio4::exprtk)

install(TARGETS gr-trigger EXPORT gnuradio4Targets)
install(
  FILES include/gnuradio-4.0/trigger/ConditionSource.hpp
        include/gnuradio-4.0/trigger/DigitalLine.hpp
        include/gnuradio-4.0/trigger/EventStore.hpp
        include/gnuradio-4.0/trigger/Events.hpp
        include/gnuradio-4.0/trigger/Marble.hpp
        include/gnuradio-4.0/trigger/MonotonicClock.hpp
        include/gnuradio-4.0/trigger/PatternTrigger.hpp
        include/gnuradio-4.0/trigger/SamplePredicate.hpp
        include/gnuradio-4.0/trigger/SampleTest.hpp
        include/gnuradio-4.0/trigger/SchmittTrigger.hpp
        include/gnuradio-4.0/trigger/SegmentCollector.hpp
        include/gnuradio-4.0/trigger/Segments.hpp
        include/gnuradio-4.0/trigger/TimeBase.hpp
        include/gnuradio-4.0/trigger/ValueTrigger.hpp
        include/gnuradio-4.0/trigger/WindowCollector.hpp
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/gnuradio-4.0/trigger")

# On Emscripten, MAKE_SHARED_LIBRARY already produces a .a; Static would be a duplicate.
set(_trigger_static_opt)
if(NOT EMSCRIPTEN)
  list(APPEND _trigger_static_opt MAKE_STATIC_LIBRARY)
endif()

gr_add_block_library(
  GrTriggerBlocks
  MAKE_SHARED_LIBRARY
  MAKE_PLUGIN
  ${_trigger_static_opt}
  HEADERS
  include/gnuradio-4.0/trigger/Marble.hpp
  include/gnuradio-4.0/trigger/PatternTrigger.hpp
  include/gnuradio-4.0/trigger/SchmittTrigger.hpp
  include/gnuradio-4.0/trigger/ValueTrigger.hpp
  LINK_LIBRARIES
  gnuradio4::gr-trigger
  gnuradio4::gnuradio-algorithm)

if(ENABLE_TESTING)
  add_subdirectory(test)
endif()
