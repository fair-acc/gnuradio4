add_library(gnuradio-core INTERFACE)
target_include_directories(gnuradio-core INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/> $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/src> $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include> $<INSTALL_INTERFACE:include/>)
target_link_libraries(gnuradio-core INTERFACE gnuradio-options gnuradio-meta refl-cpp::refl-cpp pmtv vir)

# configure a header file to pass the CMake settings to the source code
configure_file("${PROJECT_SOURCE_DIR}/cmake/config.h.in"  "${CMAKE_CURRENT_BINARY_DIR}/include/gnuradio-4.0/config.h" @ONLY)
# TODO: install configure file... but not really meaningful for header only library, since compile flags are defined by the user...

install(
        TARGETS gnuradio-core
        EXPORT graphTargets
        PUBLIC_HEADER DESTINATION include/
)

add_subdirectory(src)

if (ENABLE_TESTING)
    add_subdirectory(test)
    add_subdirectory(benchmarks)
endif ()
