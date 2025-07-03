# blocks/analog/CMakeLists.txt
# ================================================================
cmake_minimum_required(VERSION 3.18)

# ----------------------------------------------------------------
# 1. Public headers that belong to the analog block-library
# ----------------------------------------------------------------
set(GrAnalogBlocks_HDRS
    include/gnuradio-4.0/analog/Analog.hpp)

# ----------------------------------------------------------------
# 2. Interface lib so other block-libs can link against "gr-analog"
# ----------------------------------------------------------------
add_library(gr-analog INTERFACE)
target_link_libraries(gr-analog INTERFACE
    gnuradio-core
    gnuradio-algorithm)
target_include_directories(gr-analog INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
    $<INSTALL_INTERFACE:include/>)

# ----------------------------------------------------------------
# 3. Main block-library – triggers the header-parser
# ----------------------------------------------------------------
gr_add_block_library(
    GrAnalogBlocks
    MAKE_SHARED_LIBRARY
    HEADERS         ${GrAnalogBlocks_HDRS}
    LINK_LIBRARIES  gr-analog)

# -------- make sure the object lib sees the generated headers ----
target_include_directories(GrAnalogBlocksObject PRIVATE
    ${CMAKE_BINARY_DIR}/generated_plugins/GrAnalogBlocks)

# ----------------------------------------------------------------
# 4.  Stub "declarations.hpp/raw_calls.hpp" if the parser produced
#     none (otherwise the #include in integrator.cpp is empty).
# ----------------------------------------------------------------
set(_analog_plugin_dir "${CMAKE_BINARY_DIR}/generated_plugins/GrAnalogBlocks")
file(MAKE_DIRECTORY "${_analog_plugin_dir}")

# --- declarations.hpp (touches 'registry' to silence unused param)
set(_decl_file "${_analog_plugin_dir}/declarations.hpp")
if(NOT EXISTS "${_decl_file}")
    file(WRITE "${_decl_file}"
"// stub created by CMake because parser emitted no declarations\n"
"(void)registry; // mark parameter as used\n")
endif()

# --- raw_calls.hpp – empty stub, harmless if overwritten later
set(_raw_file "${_analog_plugin_dir}/raw_calls.hpp")
if(NOT EXISTS "${_raw_file}")
    file(WRITE "${_raw_file}"
"// stub: no raw-call wrappers generated for this block-lib\n")
endif()

# --- 5. Silence the "unused-parameter" warning only for this block-lib ------
foreach(_tgt IN ITEMS GrAnalogBlocksObject GrAnalogBlocksShared)
    if(TARGET ${_tgt})
        target_compile_options(${_tgt} PRIVATE
            # Clang
            $<$<CXX_COMPILER_ID:Clang>:-Wno-unused-parameter -Wno-error=unused-parameter>
            # GCC
            $<$<CXX_COMPILER_ID:GNU>:-Wno-unused-parameter -Wno-error=unused-parameter>)
    endif()
endforeach()

# ----------------------------------------------------------------
# 6. Unit-tests
# ----------------------------------------------------------------
if(TARGET GrAnalogBlocksShared AND ENABLE_TESTING)
    add_subdirectory(test)
endif()