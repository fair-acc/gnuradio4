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
    include/gnuradio-4.0/Compression.hpp
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

if(GR_ENABLE_WEBGPU)
  list(APPEND gnuradio_core_sources src/DeviceContextWebGpu.cpp)
endif()

add_library(gnuradio-core STATIC ${gnuradio_core_sources})

set_target_properties(gnuradio-core PROPERTIES PUBLIC_HEADER "${public_headers}")
add_library(gnuradio4::gnuradio-core ALIAS gnuradio-core)
target_include_directories(
  gnuradio-core
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/> $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/src>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include> $<INSTALL_INTERFACE:include/>)

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

# ── gnuradio-core-dynload: core + optional runtime plugin loading support ── Apps that load block plugins at runtime
# (especially WASM side modules) should link gnuradio4::gnuradio-core-dynload instead of gnuradio-core alone.
#
# On Emscripten it: • Compiles SharedLibraryEmscriptenDlopen.cpp (emscripten_dlopen) into a static archive that is
# whole-archived so the strong open hook overrides the weak null fallback in SharedLibrary.cpp. • WHOLE_ARCHIVEs
# gnuradio-core so SIDE_MODULE GOT imports (RTTI, pmt::Value, CtxSettingsBase, …) exist in the MAIN_MODULE even when the
# host never references them directly. • Propagates -sMAIN_MODULE=1, -Wl,--export-all, and -Wl,--no-gc-sections. Mode 2
# + Release -O3 dead-strips relocatable env imports (notably __stack_pointer as a WebAssembly.Global), so runtime
# SIDE_MODULE dlopen fails with "imported mutable global must be a WebAssembly.Global object". Mode 1 keeps those
# imports; --export-all exports hidden-visibility definitions that --export-dynamic would skip. • --post-js
# gnuradio_dynlink_post.js publishes __stack_pointer into wasmImports and patches leftover GOT slots (install path wired
# in gnuradio4Config.cmake for find_package consumers).
#
# On all other platforms gnuradio-core-dynload is identical to gnuradio-core.
if(EMSCRIPTEN)
  add_library(gnuradio-core-dynload-impl STATIC src/SharedLibraryEmscriptenDlopen.cpp)
  # PRIVATE: host WHOLE_ARCHIVEs gnuradio-core via the INTERFACE target below.
  target_link_libraries(gnuradio-core-dynload-impl PRIVATE gnuradio-core)
  set_target_properties(gnuradio-core-dynload-impl PROPERTIES POSITION_INDEPENDENT_CODE ON)
  # In-tree alias matching the installed imported name (NAMESPACE gnuradio4::).
  add_library(gnuradio4::gnuradio-core-dynload-impl ALIAS gnuradio-core-dynload-impl)

  set(_gr_dynlink_post_js "${CMAKE_CURRENT_SOURCE_DIR}/emscripten/gnuradio_dynlink_post.js")

  add_library(gnuradio-core-dynload INTERFACE)
  # WHOLE_ARCHIVE so the strong s_emscriptenAsyncOpen overrides the weak null fallback, and every core .a member is
  # available to SIDE_MODULE imports. CMake does not rewrite bare target names nested inside $<LINK_LIBRARY:...> on
  # export, so use the build target in BUILD_INTERFACE and the namespaced imported name for installs (otherwise
  # find_package consumers get a bare -l… with no search path).
  target_link_libraries(
    gnuradio-core-dynload
    INTERFACE "$<BUILD_INTERFACE:$<LINK_LIBRARY:WHOLE_ARCHIVE,gnuradio-core>>"
              "$<INSTALL_INTERFACE:$<LINK_LIBRARY:WHOLE_ARCHIVE,gnuradio4::gnuradio-core>>"
              "$<BUILD_INTERFACE:$<LINK_LIBRARY:WHOLE_ARCHIVE,gnuradio-core-dynload-impl>>"
              "$<INSTALL_INTERFACE:$<LINK_LIBRARY:WHOLE_ARCHIVE,gnuradio4::gnuradio-core-dynload-impl>>")
  target_link_options(
    gnuradio-core-dynload
    INTERFACE
    "-sMAIN_MODULE=1"
    "-Wl,--export-all"
    "-Wl,--no-gc-sections"
    "$<BUILD_INTERFACE:SHELL:--post-js=${_gr_dynlink_post_js}>")

  install(FILES "${_gr_dynlink_post_js}" DESTINATION "${CMAKE_INSTALL_DATADIR}/gnuradio4")
else()
  add_library(gnuradio-core-dynload INTERFACE)
  target_link_libraries(gnuradio-core-dynload INTERFACE gnuradio-core)
endif()
add_library(gnuradio4::gnuradio-core-dynload ALIAS gnuradio-core-dynload)

# libdl for SharedLibrary / plugins on POSIX (not Windows, WASI, or Emscripten)
if(NOT WIN32 AND NOT CMAKE_SYSTEM_NAME STREQUAL "WASI" AND NOT EMSCRIPTEN)
  target_link_libraries(gnuradio-core PUBLIC ${CMAKE_DL_LIBS})
endif()

# SHARED collapses to a SIDE_MODULE .wasm once TARGET_SUPPORTS_SHARED_LIBS is enabled for block plugins; keep a normal
# static archive for Emscripten hosts.
if(EMSCRIPTEN)
  add_library(gnuradio-blocklib-core STATIC src/BlockRegistry.cpp)
else()
  add_library(gnuradio-blocklib-core SHARED src/BlockRegistry.cpp)
endif()
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

if(GR_ENABLE_WEBGPU)
  # The kernel translator runs on the build host after each link, so it is built with a native compiler in a separate
  # directory, in the shape of blocklib_generator's parser tool (whose host-compiler setting it shares).
  set(_wasm2wgsl_build_dir "${PROJECT_BINARY_DIR}/wasm2wgsl_build")
  set(_wasm2wgsl_cmake_args
      -S
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/wasm2wgsl
      -B
      ${_wasm2wgsl_build_dir}
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_TOOLCHAIN_FILE=
      -UCMAKE_SYSROOT
      -UCMAKE_C_COMPILER_TARGET
      -UCMAKE_CXX_COMPILER_TARGET)
  # the host compiler blocklib_generator uses; its legacy spelling is what CI passes, mapped only in that directory
  set(_host_cxx ${GNURADIO_PARSE_REGISTRATIONS_TOOL_CXX_COMPILER})
  if(NOT _host_cxx)
    set(_host_cxx ${GNURADIO_PARSE_REGISTRATIONS_TOOL_CXX_COMPLILER})
  endif()
  if(_host_cxx)
    list(APPEND _wasm2wgsl_cmake_args -DCMAKE_CXX_COMPILER=${_host_cxx})
  endif()
  set(_wasm2wgsl_env_cmd
      ${CMAKE_COMMAND}
      -E
      env
      --unset=CXX
      --unset=CC
      --unset=CXXFLAGS
      --unset=CFLAGS
      --unset=LDFLAGS
      --unset=CPPFLAGS
      --unset=CMAKE_TOOLCHAIN_FILE)
  if(NOT EXISTS "${_wasm2wgsl_build_dir}/CMakeCache.txt")
    execute_process(COMMAND ${_wasm2wgsl_env_cmd} ${CMAKE_COMMAND} ${_wasm2wgsl_cmake_args} RESULT_VARIABLE _result
                    ERROR_VARIABLE _error)
    if(NOT _result EQUAL 0)
      message(FATAL_ERROR "Failed to configure wasm2wgsl: ${_error}")
    endif()
  endif()
  execute_process(COMMAND ${_wasm2wgsl_env_cmd} ${CMAKE_COMMAND} --build ${_wasm2wgsl_build_dir} RESULT_VARIABLE _result
                  OUTPUT_VARIABLE _output ERROR_VARIABLE _error)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR "Failed to build wasm2wgsl: ${_error} ${_output}")
  endif()
  set(GR_WASM2WGSL_EXECUTABLE "${_wasm2wgsl_build_dir}/wasm2wgsl" CACHE INTERNAL "build-host WebGPU kernel translator")
  # built while configuring, so an edit to its sources (C++, CMake or the embedded WGSL) configures again
  file(GLOB _wasm2wgsl_sources ${CMAKE_CURRENT_SOURCE_DIR}/tools/wasm2wgsl/*)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_wasm2wgsl_sources})

  # Tint validates every translated kernel; one it rejects is refused by name rather than failing at run time. It is a
  # build tool like the compiler: on PATH or named here (docker/Dockerfile builds it from Dawn).
  find_program(GR_TINT_EXECUTABLE tint DOC "Tint, Dawn's WGSL compiler: validates the translated kernels")
  if(NOT GR_TINT_EXECUTABLE)
    message(FATAL_ERROR "GR_ENABLE_WEBGPU needs Tint to validate the translated kernels: put 'tint' on PATH or set "
                        "GR_TINT_EXECUTABLE (docker/Dockerfile shows how it is built from Dawn)")
  endif()

  # writes <target>.kernels.json beside <target>.wasm: the WGSL pipelines DeviceContextWebGpu.js loads at start-up
  function(gr_translate_webgpu_kernels target)
    # the runtime is linked in by `--js-library` and `--pre-js`, which CMake does not track as inputs of the link, and a
    # new translator must translate again
    set_property(
      TARGET ${target} APPEND PROPERTY LINK_DEPENDS ${PROJECT_SOURCE_DIR}/core/src/DeviceContextWebGpu.js
                                       ${PROJECT_SOURCE_DIR}/core/src/BlockingNodeStdio.js ${GR_WASM2WGSL_EXECUTABLE})
    add_custom_command(
      TARGET ${target} POST_BUILD
      COMMAND ${GR_WASM2WGSL_EXECUTABLE} $<TARGET_FILE_DIR:${target}>/$<TARGET_FILE_BASE_NAME:${target}>.wasm
              $<TARGET_FILE_DIR:${target}>/$<TARGET_FILE_BASE_NAME:${target}>.kernels.json --tint ${GR_TINT_EXECUTABLE}
      COMMENT "Translating the WebGPU kernels of ${target}")
  endfunction()
endif()

if(ENABLE_TESTING)
  add_subdirectory(test)
endif()
if(ENABLE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
