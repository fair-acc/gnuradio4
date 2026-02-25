add_library(gnuradio-algorithm INTERFACE)
target_include_directories(gnuradio-algorithm INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                                        $<INSTALL_INTERFACE:include/>)

target_link_libraries(
  gnuradio-algorithm
  INTERFACE gnuradio-options
            gnuradio-meta
            vir
            magic_enum)

if(GR_HTTP_ENABLED AND NOT EMSCRIPTEN)
  target_link_libraries(gnuradio-algorithm INTERFACE cpr::cpr)
endif()
target_compile_definitions(gnuradio-algorithm INTERFACE GR_HTTP_ENABLED=$<BOOL:${GR_HTTP_ENABLED}>)

if(ENABLE_EXAMPLES)
  add_subdirectory(benchmarks)
  add_subdirectory(src)
endif()

if(ENABLE_TESTING)
  add_subdirectory(test)
endif()
