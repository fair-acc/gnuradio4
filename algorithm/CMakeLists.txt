add_library(gnuradio-algorithm INTERFACE)
target_include_directories(gnuradio-algorithm INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                                        $<INSTALL_INTERFACE:include/>)

if(NOT EMSCRIPTEN)
  include(FetchContent)
  FetchContent_Declare(
    pffft
    GIT_REPOSITORY https://github.com/fair-acc/pffft.git
    GIT_TAG cmakeTestFlag # feature branch which adds better compatibility with CMake's add_subdirectory/FetchContent
    EXCLUDE_FROM_ALL)
  FetchContent_MakeAvailable(pffft)
  target_link_libraries(gnuradio-algorithm INTERFACE PFFFT)
endif()

target_link_libraries(
  gnuradio-algorithm
  INTERFACE gnuradio-options
            gnuradio-meta
            vir
            fftw
            magic_enum)

if(ENABLE_EXAMPLES)
  add_subdirectory(benchmarks)
  add_subdirectory(src)
endif()

if(ENABLE_TESTING)
  add_subdirectory(test)
endif()
