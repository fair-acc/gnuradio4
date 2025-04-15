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
<<<<<<< HEAD
  include/gnuradio-4.0/basic/Trigger.hpp)
=======
  include/gnuradio-4.0/basic/Trigger.hpp
  include/gnuradio-4.0/basic/And.hpp
  include/gnuradio-4.0/basic/AndConst.hpp
  include/gnuradio-4.0/basic/Not.hpp
  include/gnuradio-4.0/basic/Or.hpp
  include/gnuradio-4.0/basic/Xor.hpp
  )
>>>>>>> 0c1c5cd (Update restyled And block implementation.)
target_link_libraries(gr-basic INTERFACE gnuradio-core gnuradio-algorithm)
target_include_directories(gr-basic INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                              $<INSTALL_INTERFACE:include/>)

gr_add_block_library(
  GrBasicBlocks
  MAKE_SHARED_LIBRARY
  MAKE_STATIC_LIBRARY
  HEADERS
  ${GrBasicBlocks_HDRS}
  LINK_LIBRARIES
  ${GrBasicBlocks_LIBS})

if(TARGET GrBasicBlocksShared AND ENABLE_TESTING)
  add_subdirectory(test)
endif()
