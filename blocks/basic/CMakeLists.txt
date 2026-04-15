set(GrBasicBlocks_HDRS
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

set(GrBasicBlocks_LIBS gnuradio4::gr-basic gnuradio4::gr-testing)

if(PYTHON_AVAILABLE)
  list(
    APPEND
    GrBasicBlocks_HDRS
    include/gnuradio-4.0/basic/PythonBlock.hpp
    include/gnuradio-4.0/basic/PythonInterpreter.hpp)
  list(
    APPEND
    GrBasicBlocks_LIBS
    Python3::Python
    Python3::NumPy)
endif()

add_library(gr-basic INTERFACE ${GrBasicBlocks_HDRS})
add_library(gnuradio4::gr-basic ALIAS gr-basic)
target_link_libraries(gr-basic INTERFACE gnuradio4::gnuradio-core gnuradio4::gnuradio-algorithm)
target_include_directories(gr-basic INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                              $<INSTALL_INTERFACE:include/>)
install(TARGETS gr-basic EXPORT gnuradio4Targets)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/gnuradio-4.0/basic"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/gnuradio-4.0")

gr_add_block_library(
  GrBasicBlocks
  MAKE_SHARED_LIBRARY
  MAKE_STATIC_LIBRARY
  HEADERS
  ${GrBasicBlocks_HDRS}
  LINK_LIBRARIES
  ${GrBasicBlocks_LIBS})

if(TARGET gnuradio4::GrBasicBlocksShared AND ENABLE_TESTING)
  add_subdirectory(test)
endif()
