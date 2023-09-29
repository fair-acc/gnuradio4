add_library(graph-prototype-algorithm INTERFACE)
target_include_directories(graph-prototype-algorithm INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/> $<INSTALL_INTERFACE:include/>)

install(
        TARGETS graph-prototype-algorithm
        EXPORT graphTargets
        PUBLIC_HEADER DESTINATION include/
)