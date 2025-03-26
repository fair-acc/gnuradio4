add_library(gr-electrical INTERFACE)
target_link_libraries(gr-electrical INTERFACE gnuradio-core gnuradio-algorithm)
target_include_directories(gr-electrical INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                                   $<INSTALL_INTERFACE:include/>)

# An OBJECT library as the generated files have namespace-scope variables with dynamic C++ initialization
add_library(GrElectricalBlocks OBJECT)
gr_generate_block_instantiations(GrElectricalBlocks HEADERS include/gnuradio-4.0/electrical/PowerEstimators.hpp)
target_link_libraries(GrElectricalBlocks PUBLIC gr-basic gr-electrical)

if(ENABLE_TESTING)
  add_subdirectory(test)
endif()
