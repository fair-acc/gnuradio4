add_library(gnuradio-core INTERFACE)
target_include_directories(gnuradio-core INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/> $<INSTALL_INTERFACE:include/>)
target_link_libraries(gnuradio-core INTERFACE gnuradio-options refl-cpp::refl-cpp pmtv vir)

install(
        TARGETS gnuradio-core
        EXPORT graphTargets
        PUBLIC_HEADER DESTINATION include/
)

add_subdirectory(src)

if (ENABLE_TESTING)
    add_subdirectory(test)
endif ()
