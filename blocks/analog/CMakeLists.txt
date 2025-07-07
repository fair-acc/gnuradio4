cmake_minimum_required(VERSION 3.18)

# ──────────────────────────────────────────────────────────
# 1) Interface target that exposes the public headers
# ──────────────────────────────────────────────────────────
add_library(gr-analog INTERFACE)
target_link_libraries(gr-analog INTERFACE gnuradio-core)
target_include_directories(
    gr-analog INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
    $<INSTALL_INTERFACE:include/>
)

# ──────────────────────────────────────────────────────────
# 2) Main block library – parsed by the header-parser
# ──────────────────────────────────────────────────────────
gr_add_block_library(
    GrAnalogBlocks
    MAKE_SHARED_LIBRARY
    HEADERS         include/gnuradio-4.0/analog/Analog.hpp
    LINK_LIBRARIES  gr-basic          # helpers from gr-basic
                    gr-analog         # header-only interface we just made
)

# ──────────────────────────────────────────────────────────
# 3) Drive the unit-tests only when the shared lib exists
# ──────────────────────────────────────────────────────────
if (TARGET GrAnalogBlocksShared AND ENABLE_TESTING)
    add_subdirectory(test)
endif()
