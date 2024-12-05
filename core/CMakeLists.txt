add_library(
  gnuradio-core
  INTERFACE
  include/gnuradio-4.0/Settings.hpp
  include/gnuradio-4.0/annotated.hpp
  include/gnuradio-4.0/AtomicBitset.hpp
  include/gnuradio-4.0/Block.hpp
  include/gnuradio-4.0/BlockModel.hpp
  include/gnuradio-4.0/BlockRegistry.hpp
  include/gnuradio-4.0/BlockTraits.hpp
  include/gnuradio-4.0/Buffer.hpp
  include/gnuradio-4.0/BufferSkeleton.hpp
  include/gnuradio-4.0/CircularBuffer.hpp
  include/gnuradio-4.0/ClaimStrategy.hpp
  include/gnuradio-4.0/DataSet.hpp
  include/gnuradio-4.0/Graph_yaml_importer.hpp
  include/gnuradio-4.0/Graph.hpp
  include/gnuradio-4.0/HistoryBuffer.hpp
  include/gnuradio-4.0/LifeCycle.hpp
  include/gnuradio-4.0/Message.hpp
  include/gnuradio-4.0/plugin.hpp
  include/gnuradio-4.0/PluginLoader.hpp
  include/gnuradio-4.0/Port.hpp
  include/gnuradio-4.0/PortTraits.hpp
  include/gnuradio-4.0/Profiler.hpp
  include/gnuradio-4.0/reader_writer_lock.hpp
  include/gnuradio-4.0/Scheduler.hpp
  include/gnuradio-4.0/Sequence.hpp
  include/gnuradio-4.0/Settings.hpp
  include/gnuradio-4.0/Tag.hpp
  include/gnuradio-4.0/TriggerMatcher.hpp
  include/gnuradio-4.0/WaitStrategy.hpp
  include/gnuradio-4.0/YamlPmt.hpp)
target_include_directories(
  gnuradio-core
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/src>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
            $<INSTALL_INTERFACE:include/>)
target_link_libraries(
  gnuradio-core
  INTERFACE gnuradio-options
            gnuradio-meta
            magic_enum
            pmtv
            vir)

# configure a header file to pass the CMake settings to the source code
configure_file("${PROJECT_SOURCE_DIR}/cmake/config.hpp.in"
               "${CMAKE_CURRENT_BINARY_DIR}/include/gnuradio-4.0/config.hpp" @ONLY)
# TODO: install configure file... but not really meaningful for header only library, since compile flags are defined by
# the user...

install(
  TARGETS gnuradio-core
  EXPORT graphTargets
  PUBLIC_HEADER DESTINATION include/)

add_subdirectory(src)

if(ENABLE_TESTING)
  add_subdirectory(test)
  add_subdirectory(benchmarks)
endif()
