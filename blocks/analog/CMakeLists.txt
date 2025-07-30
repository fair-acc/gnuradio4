cmake_minimum_required(VERSION 3.18)

add_library(gr-analog INTERFACE)
target_link_libraries(gr-analog INTERFACE gnuradio-core)
target_include_directories(
    gr-analog INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
    $<INSTALL_INTERFACE:include/>
)

gr_add_block_library(
    GrAnalogBlocks
    MAKE_SHARED_LIBRARY
    HEADERS
        include/gnuradio-4.0/analog/FrequencyMod.hpp
        include/gnuradio-4.0/analog/QuadratureDemod.hpp
        include/gnuradio-4.0/analog/FmDet.hpp     
        include/gnuradio-4.0/analog/AmDemod.hpp
        include/gnuradio-4.0/analog/PhaseModulator.hpp
        include/gnuradio-4.0/analog/Agc.hpp
        include/gnuradio-4.0/analog/Agc2.hpp
    LINK_LIBRARIES
        gr-basic
        gr-analog)

if (TARGET GrAnalogBlocksShared AND ENABLE_TESTING)
    add_subdirectory(test)
endif()
