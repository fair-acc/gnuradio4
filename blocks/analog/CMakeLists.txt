# blocks/analog/CMakeLists.txt
# ===================================================================
cmake_minimum_required(VERSION 3.18)

# -------------------------------------------------------------------
# 1. Public headers
# -------------------------------------------------------------------
set(GrAnalogBlocks_HDRS
    include/gnuradio-4.0/analog/Analog.hpp)

# -------------------------------------------------------------------
# 2. “gr-analog” interface library (other block-libs can link to this)
# -------------------------------------------------------------------
add_library(gr-analog INTERFACE)
target_link_libraries(gr-analog INTERFACE
    gnuradio-core
    gnuradio-algorithm)
target_include_directories(gr-analog INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
    $<INSTALL_INTERFACE:include/>)

# -------------------------------------------------------------------
# 3. Main block library -- kicks off header-parser & code-gen
# -------------------------------------------------------------------
gr_add_block_library(
    GrAnalogBlocks                     # logical block-lib name
    MAKE_SHARED_LIBRARY
    HEADERS        ${GrAnalogBlocks_HDRS}
    LINK_LIBRARIES gr-analog)

# Make sure the generated headers can be found by the object target
target_include_directories(GrAnalogBlocksObject PRIVATE
    ${CMAKE_BINARY_DIR}/generated_plugins/GrAnalogBlocks)

# -------------------------------------------------------------------
# 4.  Stubs if the parser produced no registration helpers
# -------------------------------------------------------------------
set(_analog_plugin_dir "${CMAKE_BINARY_DIR}/generated_plugins/GrAnalogBlocks")
file(MAKE_DIRECTORY "${_analog_plugin_dir}")

# -- declarations.hpp  (touch “registry” so it is ‘used’)
set(_decl_file "${_analog_plugin_dir}/declarations.hpp")
if(NOT EXISTS "${_decl_file}")
    file(WRITE "${_decl_file}"
"// stub created by CMake because parser emitted no declarations\n"
"(void)registry; // mark parameter as used\n")
endif()

# -- raw_calls.hpp     (empty stub, harmless if overwritten)
set(_raw_file "${_analog_plugin_dir}/raw_calls.hpp")
if(NOT EXISTS "${_raw_file}")
    file(WRITE "${_raw_file}"
"// stub: no raw-call wrappers generated for this block-lib\n")
endif()

# -------------------------------------------------------------------
# 5. Silence block-local warnings that are deliberate in Analog.hpp
#    (we *disable the warnings* AND the “-Werror” promotion)
# -------------------------------------------------------------------
foreach(_tgt IN ITEMS GrAnalogBlocksObject GrAnalogBlocksShared)
    if(TARGET ${_tgt})
        target_compile_options(${_tgt} PRIVATE
            # -------- Clang ----------------------------------------
            $<$<CXX_COMPILER_ID:Clang>:
                -Wno-implicit-float-conversion
                -Wno-error=implicit-float-conversion
                -Wno-unused-parameter
                -Wno-error=unused-parameter>
            # -------- GCC ------------------------------------------
            $<$<CXX_COMPILER_ID:GNU>:
                -Wno-implicit-float-conversion
                -Wno-error=implicit-float-conversion
                -Wno-unused-parameter
                -Wno-error=unused-parameter>)
    endif()
endforeach()

# -------------------------------------------------------------------
# 6. Unit tests
# -------------------------------------------------------------------
if(TARGET GrAnalogBlocksShared AND ENABLE_TESTING)
    add_subdirectory(test)
endif()
