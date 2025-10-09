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

set(GR_FILEIO_CPR_AVAILABLE FALSE)
if(NOT EMSCRIPTEN)

  if(NOT
     GR_INSTALL_CPR
     STREQUAL
     "OFF")

    find_package(CURL QUIET)

    # ON -> always install CPR; OPTIONAL -> only install CPR if system libcurl is available
    if(GR_INSTALL_CPR STREQUAL "ON" OR (GR_INSTALL_CPR STREQUAL "OPTIONAL" AND CURL_FOUND))

      include(FetchContent)

      if(CURL_FOUND)
        set(CPR_USE_SYSTEM_CURL
            ON
            CACHE BOOL "Use system libcurl in CPR" FORCE)
      endif()

      FetchContent_Declare(
        cpr
        GIT_REPOSITORY https://github.com/libcpr/cpr.git
        GIT_TAG 1.12.0)
      FetchContent_MakeAvailable(cpr)

      target_link_libraries(gnuradio-algorithm INTERFACE cpr::cpr)

      set(GR_FILEIO_CPR_AVAILABLE TRUE)
    endif()
  endif()
endif()
message(STATUS "[CPR] GR_FILEIO_CPR_AVAILABLE=${GR_FILEIO_CPR_AVAILABLE}")

if(GR_FILEIO_CPR_AVAILABLE)
  target_compile_definitions(gnuradio-algorithm INTERFACE GR_FILEIO_CPR_AVAILABLE=1)
else()
  target_compile_definitions(gnuradio-algorithm INTERFACE GR_FILEIO_CPR_AVAILABLE=0)
endif()

if(ENABLE_EXAMPLES)
  add_subdirectory(benchmarks)
  add_subdirectory(src)
endif()

if(ENABLE_TESTING)
  add_subdirectory(test)
endif()
