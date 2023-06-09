function(add_gp_plugin PLUGIN_NAME)
    add_library(${PLUGIN_NAME} MODULE ${PLUGIN_NAME}.cpp)

    target_compile_options(${PLUGIN_NAME} PRIVATE -fsanitize=address -Wall)
    # target_compile_options(${PLUGIN_NAME} PRIVATE -fsanitize=address -Wall -fvisibility=hidden -fvisibility-inlines-hidden)
    target_link_options(${PLUGIN_NAME} PRIVATE -fsanitize=address -Wall)

    target_link_libraries(${PLUGIN_NAME} PUBLIC graph-prototype-options graph-prototype refl-cpp fmt)

    add_custom_command(
        TARGET "${PLUGIN_NAME}" POST_BUILD
        DEPENDS "${PLUGIN_NAME}"
        COMMAND "${CMAKE_STRIP}"
        ARGS --strip-all -K gp_plugin_make -K gp_plugin_free $<TARGET_FILE:${PLUGIN_NAME}>
    )
endfunction()

add_gp_plugin(good_math_plugin)
add_gp_plugin(good_base_plugin)
add_gp_plugin(bad_plugin)
