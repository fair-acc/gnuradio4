add_library(gnuradio-algorithm INTERFACE)
target_link_libraries(gnuradio-algorithm INTERFACE httplib::httplib)
target_include_directories(gnuradio-algorithm INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                                        $<INSTALL_INTERFACE:include/>)

target_link_libraries(
  gnuradio-algorithm
  INTERFACE gnuradio-options
            gnuradio-meta
            vir
            magic_enum)

if(NOT EMSCRIPTEN)
  # find_package(libCurl) if (LIB_CURL_FOUND)

  include(FetchContent)
  FetchContent_Declare(
    cpr
    GIT_REPOSITORY https://github.com/libcpr/cpr.git
    GIT_TAG 1.12.0)
  FetchContent_MakeAvailable(cpr)
  target_link_libraries(gnuradio-algorithm INTERFACE cpr::cpr)
endif()

if(ENABLE_EXAMPLES)
  add_subdirectory(benchmarks)
  add_subdirectory(src)
endif()

if(ENABLE_TESTING)
  add_subdirectory(test)
endif()
