# setup header only library
add_library(graph-prototype INTERFACE)
target_include_directories(graph-prototype INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> $<INSTALL_INTERFACE:include/>)
target_link_libraries(graph-prototype INTERFACE graph-prototype-options refl-cpp::refl-cpp pmtv)

install(
        TARGETS graph-prototype
        EXPORT graphTargets
        PUBLIC_HEADER DESTINATION include/
)

if (NOT EMSCRIPTEN)
    add_library(graph-prototype-plugin SHARED plugin.cpp)
    target_include_directories(graph-prototype-plugin PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> $<INSTALL_INTERFACE:include/>)
    target_include_directories(graph-prototype-plugin PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(graph-prototype-plugin PUBLIC graph-prototype graph-prototype-options pmtv fmt)

    include(GenerateExportHeader)
    generate_export_header(graph-prototype-plugin)
    # TODO install exported headers

    install(
            TARGETS graph-prototype-plugin
            EXPORT graphPluginTargets
            PUBLIC_HEADER DESTINATION include/
    )
endif ()
