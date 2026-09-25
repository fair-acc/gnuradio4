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
        include/gnuradio-4.0/trigger/MonotonicClock.hpp
        include/gnuradio-4.0/trigger/SamplePredicate.hpp
        include/gnuradio-4.0/trigger/SampleTest.hpp
        include/gnuradio-4.0/trigger/SegmentCollector.hpp
        include/gnuradio-4.0/trigger/Segments.hpp
        include/gnuradio-4.0/trigger/TimeBase.hpp
        include/gnuradio-4.0/trigger/WindowCollector.hpp
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/gnuradio-4.0/trigger")

if(ENABLE_TESTING)
  add_subdirectory(test)
endif()
