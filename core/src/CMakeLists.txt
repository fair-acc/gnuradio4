if(ENABLE_TESTS)
  add_subdirectory(test)
  add_subdirectory(benchmarks)
endif()

if(NOT EMSCRIPTEN)
  add_library(gnuradio-plugin SHARED plugin.cpp)
  add_library(gnuradio4::gnuradio-plugin ALIAS gnuradio-plugin)
  set_target_properties(gnuradio-plugin PROPERTIES BUILD_WITH_INSTALL_RPATH ON)
  target_include_directories(
    gnuradio-plugin PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}> $<INSTALL_INTERFACE:include/>)
  target_link_libraries(gnuradio-plugin PUBLIC gnuradio-core gnuradio-blocklib-core gnuradio-options)

  install(
    TARGETS gnuradio-plugin
    EXPORT gnuradio4PluginTargets
    PUBLIC_HEADER DESTINATION include/gnuradio-4.0)
endif()

if(ENABLE_EXAMPLES)
  add_executable(main main.cpp)
  target_link_libraries(main PUBLIC gnuradio-options gnuradio-core magic_enum)
endif()

add_non_blocklib_plugin(SimpleSchedulersPlugin)
