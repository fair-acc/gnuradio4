if(ENABLE_TESTS)
  add_subdirectory(test)
  add_subdirectory(benchmarks)
endif()

if(NOT EMSCRIPTEN)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
  add_library(gnuradio-plugin SHARED plugin.cpp)
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


  if(GR_USE_ADAPTIVE_CPP)
  find_package(AdaptiveCpp CONFIG REQUIRED)

  add_executable(main_sycl main_sycl.cpp)
  target_link_libraries(main_sycl PUBLIC gnuradio-options gnuradio-core magic_enum cuda)
  target_compile_options(main_sycl PRIVATE -Wno-unknown-cuda-version)
  #  add_sycl_to_target(TARGET main_sycl)

  add_executable(main_sycl_fft main_sycl_fft.cpp)
  target_link_libraries(main_sycl_fft PUBLIC gnuradio-core gnuradio-algorithm cuda)
  target_compile_options(main_sycl_fft PRIVATE -Wno-unknown-cuda-version)

  add_executable(fft_test fft_test.cpp)
  target_link_libraries(fft_test PUBLIC cuda)
  target_compile_options(fft_test PRIVATE -Wno-unknown-cuda-version)
  endif()
endif()
