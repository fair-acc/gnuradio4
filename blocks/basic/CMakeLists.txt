add_library(
  gr-basic
  INTERFACE
  include/gnuradio-4.0/basic/clock_source.hpp
  include/gnuradio-4.0/basic/common_blocks.hpp
  include/gnuradio-4.0/basic/ConverterBlocks.hpp
  include/gnuradio-4.0/basic/DataSink.hpp
  include/gnuradio-4.0/basic/function_generator.hpp
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

if(ENABLE_TESTING)
  add_subdirectory(test)
endif()
