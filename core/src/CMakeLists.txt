if(ENABLE_TESTS)
  add_subdirectory(test)
  add_subdirectory(benchmarks)
endif()

if(NOT EMSCRIPTEN)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
  add_library(gnuradio-plugin SHARED plugin.cpp)
  target_include_directories(
    gnuradio-plugin PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR};${CMAKE_CURRENT_BINARY_DIR}>
                           $<INSTALL_INTERFACE:include/>)
  target_link_libraries(
    gnuradio-plugin
    PUBLIC gnuradio-core
           gnuradio-blocklib-core
           gnuradio-options
           pmtv)

  install(
    TARGETS gnuradio-plugin
    EXPORT gnuradio4PluginTargets
    PUBLIC_HEADER DESTINATION include/gnuradio-4.0)
endif()

if(ENABLE_EXAMPLES)
  add_executable(main main.cpp)
  target_link_libraries(main PUBLIC gnuradio-options gnuradio-core magic_enum)
endif()
