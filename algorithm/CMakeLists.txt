add_library(gnuradio-algorithm INTERFACE)
add_library(gnuradio4::gnuradio-algorithm ALIAS gnuradio-algorithm)
target_include_directories(gnuradio-algorithm INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                                        $<INSTALL_INTERFACE:include/>)
install(TARGETS gnuradio-algorithm EXPORT gnuradio4Targets)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/gnuradio-4.0/algorithm"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/gnuradio-4.0")

target_link_libraries(
  gnuradio-algorithm
  INTERFACE gnuradio4::gnuradio-options
            gnuradio4::gnuradio-meta
            gnuradio4::vir
            gnuradio4::magic_enum)

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
