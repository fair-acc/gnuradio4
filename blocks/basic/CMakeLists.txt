add_library(
  gr-basic
  INTERFACE
  include/gnuradio-4.0/basic/ClockSource.hpp
  include/gnuradio-4.0/basic/CommonBlocks.hpp
  include/gnuradio-4.0/basic/ConverterBlocks.hpp
  include/gnuradio-4.0/basic/DataSink.hpp
  include/gnuradio-4.0/basic/FunctionGenerator.hpp
  include/gnuradio-4.0/basic/PythonBlock.hpp
  include/gnuradio-4.0/basic/PythonInterpreter.hpp
  include/gnuradio-4.0/basic/Selector.hpp
  include/gnuradio-4.0/basic/SignalGenerator.hpp
  include/gnuradio-4.0/basic/StreamToDataSet.hpp
  include/gnuradio-4.0/basic/SyncBlock.hpp
  include/gnuradio-4.0/basic/Trigger.hpp)
target_link_libraries(gr-basic INTERFACE gnuradio-core gnuradio-algorithm)
target_include_directories(gr-basic INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                              $<INSTALL_INTERFACE:include/>)

# An OBJECT library as the generated files have namespace-scope variables with dynamic C++ initialization
add_library(GrBasicBlocks OBJECT)
gr_generate_block_instantiations(
  GrBasicBlocks
  HEADERS
  include/gnuradio-4.0/basic/ClockSource.hpp
  include/gnuradio-4.0/basic/CommonBlocks.hpp
  include/gnuradio-4.0/basic/ConverterBlocks.hpp
  include/gnuradio-4.0/basic/DataSink.hpp
  include/gnuradio-4.0/basic/FunctionGenerator.hpp
  include/gnuradio-4.0/basic/Selector.hpp
  include/gnuradio-4.0/basic/SignalGenerator.hpp
  include/gnuradio-4.0/basic/StreamToDataSet.hpp
  include/gnuradio-4.0/basic/SyncBlock.hpp
  include/gnuradio-4.0/basic/Trigger.hpp)
target_link_libraries(GrBasicBlocks PUBLIC gr-basic gr-testing)

if(PYTHON_AVAILABLE)
  gr_generate_block_instantiations(GrBasicBlocks HEADERS include/gnuradio-4.0/basic/PythonBlock.hpp)
  target_include_directories(GrBasicBlocks PRIVATE ${Python3_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR})
  target_link_libraries(GrBasicBlocks PRIVATE ${Python3_LIBRARIES})
endif()

if(ENABLE_TESTING)
  add_subdirectory(test)
endif()
