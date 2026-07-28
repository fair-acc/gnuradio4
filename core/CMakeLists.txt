set(gnuradio_core_sources
    src/Value.cpp
    src/ValueHelper.cpp
    src/Block.cpp
    src/BlockModel.cpp
    src/Logger.cpp
    src/Settings.cpp
    src/Graph.cpp
    src/PluginLoader.cpp
    src/PmtTypeHelpers.cpp
    src/SharedLibrary.cpp)

set(public_headers
    include/gnuradio-4.0/annotated.hpp
    include/gnuradio-4.0/AtomicBitset.hpp
    include/gnuradio-4.0/Block.hpp
    include/gnuradio-4.0/BlockMerging.hpp
    include/gnuradio-4.0/BlockModel.hpp
    include/gnuradio-4.0/BlockTraits.hpp
    include/gnuradio-4.0/Buffer.hpp
    include/gnuradio-4.0/BufferSkeleton.hpp
    include/gnuradio-4.0/CircularBuffer.hpp
    include/gnuradio-4.0/ClaimStrategy.hpp
    include/gnuradio-4.0/BlockingSync.hpp
    include/gnuradio-4.0/DataSet.hpp
    include/gnuradio-4.0/Graph_yaml_importer.hpp
    include/gnuradio-4.0/Graph.hpp
    include/gnuradio-4.0/HistoryBuffer.hpp
    include/gnuradio-4.0/LifeCycle.hpp
    include/gnuradio-4.0/Logger.hpp
    include/gnuradio-4.0/MemoryAllocators.hpp
    include/gnuradio-4.0/Message.hpp
    include/gnuradio-4.0/PluginLoader.hpp
    include/gnuradio-4.0/PmtCollections.hpp
    include/gnuradio-4.0/Port.hpp
    include/gnuradio-4.0/PortTraits.hpp
    include/gnuradio-4.0/Profiler.hpp
    include/gnuradio-4.0/reader_writer_lock.hpp
    include/gnuradio-4.0/Scheduler.hpp
    include/gnuradio-4.0/Sequence.hpp
    include/gnuradio-4.0/Settings.hpp
    include/gnuradio-4.0/SharedLibrary.hpp
    include/gnuradio-4.0/Tag.hpp
    include/gnuradio-4.0/Tensor.hpp
    include/gnuradio-4.0/TensorMath.hpp
    include/gnuradio-4.0/TriggerMatcher.hpp
    include/gnuradio-4.0/Value.hpp
    include/gnuradio-4.0/ValueHelper.hpp
    include/gnuradio-4.0/formatter/ValueFormatter.hpp
    include/gnuradio-4.0/formatter/ValueHelperFormatter.hpp
    include/gnuradio-4.0/WaitStrategy.hpp
    include/gnuradio-4.0/YamlPmt.hpp)

if(INTERNAL_ENABLE_BLOCK_PLUGINS)
  # provide definitions for Plugin.hpp which is only used by PluginLoader.hpp if INTERNAL_ENABLE_BLOCK_PLUGINS is on
  list(APPEND gnuradio_core_sources src/plugin.cpp)
  list(APPEND public_headers include/gnuradio-4.0/Plugin.hpp)
endif()

add_library(gnuradio-core STATIC ${gnuradio_core_sources})

set_target_properties(gnuradio-core PROPERTIES PUBLIC_HEADER "${public_headers}")
add_library(gnuradio4::gnuradio-core ALIAS gnuradio-core)
target_include_directories(
  gnuradio-core
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/src>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
         $<INSTALL_INTERFACE:include/>)

target_link_libraries(
  gnuradio-core
  PUBLIC gnuradio-options
         gnuradio-meta
         gnuradio-algorithm
         magic_enum
         vir)

# gnuradio-plugin is legacy and has since been merged into core
add_library(gnuradio-plugin ALIAS gnuradio-core)
add_library(gnuradio4::gnuradio-plugin ALIAS gnuradio-core)

# ── gnuradio-core-dynload: core + optional runtime plugin loading support ──
# Apps that load block plugins at runtime (especially WASM side modules) should
# link gnuradio4::gnuradio-core-dynload instead of gnuradio-core alone.
#
# On Emscripten it:
#   • Compiles SharedLibraryEmscriptenDlopen.cpp (emscripten_dlopen) into a static
#     archive that is whole-archived so the strong open hook overrides the weak
#     null fallback in SharedLibrary.cpp.
#   • Propagates -sMAIN_MODULE=2 to the consumer's linker via INTERFACE options.
#
# On all other platforms gnuradio-core-dynload is identical to gnuradio-core.
if(EMSCRIPTEN)
  add_library(gnuradio-core-dynload-impl STATIC src/SharedLibraryEmscriptenDlopen.cpp)
  target_link_libraries(gnuradio-core-dynload-impl PUBLIC gnuradio-core)
  set_target_properties(gnuradio-core-dynload-impl PROPERTIES POSITION_INDEPENDENT_CODE ON)

  add_library(gnuradio-core-dynload INTERFACE)
  # WHOLE_ARCHIVE so the strong s_emscriptenAsyncOpen overrides the weak null fallback.
  target_link_libraries(gnuradio-core-dynload INTERFACE gnuradio-core "$<LINK_LIBRARY:WHOLE_ARCHIVE,gnuradio-core-dynload-impl>")
  target_link_options(gnuradio-core-dynload INTERFACE "-sMAIN_MODULE=2")
else()
  add_library(gnuradio-core-dynload INTERFACE)
  target_link_libraries(gnuradio-core-dynload INTERFACE gnuradio-core)
endif()
add_library(gnuradio4::gnuradio-core-dynload ALIAS gnuradio-core-dynload)

# libdl for SharedLibrary / plugins on POSIX (not Windows, WASI, or Emscripten)
if(NOT WIN32
   AND NOT CMAKE_SYSTEM_NAME STREQUAL "WASI"
   AND NOT EMSCRIPTEN)
  target_link_libraries(gnuradio-core PUBLIC ${CMAKE_DL_LIBS})
endif()

add_library(gnuradio-blocklib-core SHARED src/BlockRegistry.cpp)
set(blocklib_public_headers include/gnuradio-4.0/BlockRegistry.hpp)
set_target_properties(gnuradio-blocklib-core PROPERTIES PUBLIC_HEADER "${blocklib_public_headers}")
add_library(gnuradio4::gnuradio-blocklib-core ALIAS gnuradio-blocklib-core)
message(INFO "PATH include " $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>)
target_include_directories(gnuradio-blocklib-core PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/>
                                                         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(gnuradio-blocklib-core PUBLIC gnuradio-core)
if(WIN32)
  target_compile_definitions(gnuradio-core PUBLIC BUILDING_MYPLUGIN)
  target_link_libraries(gnuradio-blocklib-core INTERFACE dl)
endif()

# configure a header file to pass the CMake settings to the source code
configure_file("${PROJECT_SOURCE_DIR}/cmake/config.hpp.in"
               "${CMAKE_CURRENT_BINARY_DIR}/include/gnuradio-4.0/config.hpp" @ONLY)
# TODO: install configure file... but not really meaningful for header only library, since compile flags are defined by
# the user...
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/gnuradio-4.0/config.hpp"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/gnuradio-4.0")

set(_gr_core_install_targets gnuradio-core gnuradio-blocklib-core gnuradio-core-dynload)
if(EMSCRIPTEN)
  list(APPEND _gr_core_install_targets gnuradio-core-dynload-impl)
endif()
install(
  TARGETS ${_gr_core_install_targets}
  EXPORT gnuradio4Targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  PUBLIC_HEADER DESTINATION include/gnuradio-4.0)

# Just install the entire include directory
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_subdirectory(src)

if(ENABLE_TESTING)
  add_subdirectory(test)
endif()
if(ENABLE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
