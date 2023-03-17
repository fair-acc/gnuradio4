# setup header only library
add_library(graph-prototype INTERFACE)
target_include_directories(graph-prototype INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> $<INSTALL_INTERFACE:include/>)
target_link_libraries(graph-prototype INTERFACE graph-prototype-options refl-cpp::refl-cpp pmtv)

install(
        TARGETS graph-prototype
        EXPORT graphTargets
        PUBLIC_HEADER DESTINATION include/
)
