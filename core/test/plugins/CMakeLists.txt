function(add_gp_plugin PLUGIN_NAME)
    add_library(${PLUGIN_NAME} MODULE ${PLUGIN_NAME}.cpp)

    # target_compile_options(${PLUGIN_NAME} PRIVATE -fsanitize=address -Wall)
    target_compile_options(${PLUGIN_NAME} PRIVATE -Wall)

    target_include_directories(${PLUGIN_NAME} PUBLIC ${CMAKE_BINARY_DIR}/include)
    target_link_libraries(${PLUGIN_NAME} PUBLIC gnuradio-options gnuradio-core gnuradio-plugin refl-cpp fmt)
endfunction()

add_gp_plugin(good_math_plugin)
add_gp_plugin(good_base_plugin)
add_gp_plugin(good_conversion_plugin)
add_gp_plugin(bad_plugin)
