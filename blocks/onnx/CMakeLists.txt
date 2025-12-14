# blocks/onnx/CMakeLists.txt
#
# ONNX Runtime GR4 integration
#
# Modes:
#   off - Disable ONNX integration entirely
#   opt - Use system-installed onnxruntime (supports ROCm/CUDA, full .onnx format)
#   on  - Use bundled static libraries (minimal, .ort format only)
#
# Bundled libraries are stored in Git LFS and fetched on-demand when mode=on.
# To rebuild bundled libraries: ./lib/rebuild_static_libs.sh
#

if(TARGET gr-onnx)
  return()
endif()

set(ENABLE_ONNX_INTEGRATION "on" CACHE STRING "off | opt | on")
set_property(CACHE ENABLE_ONNX_INTEGRATION PROPERTY STRINGS off opt on)

if(ENABLE_ONNX_INTEGRATION STREQUAL "off")
  message(STATUS "gr-onnx: disabled")
  return()
endif()

set(_have_ort FALSE)
set(_ort_minimal FALSE)
set(_ort_has_gpu FALSE)

# Bundled library version (update when rebuilding libs)
set(GR_ONNX_BUNDLED_VERSION "1.23.2")

# =============================================================================
# Helper: Ensure Git LFS files are present
# =============================================================================
function(gr_onnx_ensure_lfs_files lib_subdir)
  set(_lib_path "${CMAKE_CURRENT_SOURCE_DIR}/lib/${lib_subdir}")

  # Check if this looks like an LFS pointer (small text file starting with "version")
  if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    set(_check_file "${_lib_path}/libonnxruntime_webassembly.a")
  else()
    set(_check_file "${_lib_path}/libonnxruntime.a")
  endif()

  if(NOT EXISTS "${_check_file}")
    message(FATAL_ERROR
            "gr-onnx: bundled library not found: ${_check_file}\n"
            "This may be a Git LFS issue. Try:\n"
            "  cd ${CMAKE_SOURCE_DIR} && git lfs pull --include=\"blocks/onnx/lib/${lib_subdir}/*\"\n"
            "Or use ENABLE_ONNX_INTEGRATION=opt with a system-installed onnxruntime.")
  endif()

  # Check file size - LFS pointers are ~130 bytes, real libs are megabytes
  file(SIZE "${_check_file}" _file_size)
  if(_file_size LESS 10000)
    # Likely an LFS pointer, try to fetch
    message(STATUS "gr-onnx: fetching LFS files for ${lib_subdir}...")

    find_package(Git QUIET)
    if(Git_FOUND)
      execute_process(
              COMMAND ${GIT_EXECUTABLE} lfs pull --include "blocks/onnx/lib/${lib_subdir}/*"
              WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
              RESULT_VARIABLE _lfs_result
              OUTPUT_VARIABLE _lfs_output
              ERROR_VARIABLE _lfs_error
      )

      if(NOT _lfs_result EQUAL 0)
        message(FATAL_ERROR
                "gr-onnx: failed to fetch LFS files.\n"
                "Error: ${_lfs_error}\n"
                "Ensure git-lfs is installed: apt install git-lfs\n"
                "Then run: cd ${CMAKE_SOURCE_DIR} && git lfs install && git lfs pull")
      endif()

      # Verify file size again
      file(SIZE "${_check_file}" _file_size)
      if(_file_size LESS 10000)
        message(FATAL_ERROR
                "gr-onnx: LFS fetch succeeded but file is still a pointer.\n"
                "Try: cd ${CMAKE_SOURCE_DIR} && git lfs fetch --all && git lfs checkout")
      endif()
    else()
      message(FATAL_ERROR
              "gr-onnx: bundled library appears to be an LFS pointer but git not found.\n"
              "Install git and git-lfs, then run: git lfs pull")
    endif()
  endif()

  message(STATUS "gr-onnx: bundled library verified (${_file_size} bytes)")
endfunction()

# =============================================================================
# MODE: opt - Use system packages (full runtime, GPU support possible)
# =============================================================================
if(ENABLE_ONNX_INTEGRATION STREQUAL "opt")
  if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    message(STATUS "gr-onnx: 'opt' mode not available for WASM (no system packages)")
    message(STATUS "gr-onnx: use ENABLE_ONNX_INTEGRATION=on for WASM builds")
    return()
  endif()

  # Try CMake config first (vcpkg, proper installs)
  find_package(onnxruntime CONFIG QUIET)
  if(TARGET onnxruntime::onnxruntime)
    set(_have_ort TRUE)
    message(STATUS "gr-onnx: found system onnxruntime via CMake config")
  elseif(TARGET onnxruntime)
    add_library(onnxruntime::onnxruntime ALIAS onnxruntime)
    set(_have_ort TRUE)
    message(STATUS "gr-onnx: found system onnxruntime (aliased)")
  endif()

  # Try pkg-config
  if(NOT _have_ort)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
      pkg_check_modules(ORT QUIET libonnxruntime)
      if(ORT_FOUND)
        add_library(onnxruntime::onnxruntime INTERFACE IMPORTED)
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIRS}"
                INTERFACE_LINK_LIBRARIES "${ORT_LIBRARIES}"
        )
        set(_have_ort TRUE)
        message(STATUS "gr-onnx: found system onnxruntime via pkg-config")
      endif()
    endif()
  endif()

  # Try standard paths (manual install to /usr/local)
  if(NOT _have_ort)
    find_library(ORT_LIBRARY NAMES onnxruntime
            HINTS /usr/local/lib /opt/onnxruntime/lib
            PATH_SUFFIXES lib lib64
    )
    find_path(ORT_INCLUDE_DIR NAMES onnxruntime_cxx_api.h
            HINTS /usr/local/include /opt/onnxruntime/include
            PATH_SUFFIXES include onnxruntime
    )
    if(ORT_LIBRARY AND ORT_INCLUDE_DIR)
      add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)
      set_target_properties(onnxruntime::onnxruntime PROPERTIES
              IMPORTED_LOCATION "${ORT_LIBRARY}"
              INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIR}"
      )
      set(_have_ort TRUE)
      message(STATUS "gr-onnx: found system onnxruntime at ${ORT_LIBRARY}")

      # Check for GPU providers
      get_filename_component(_ort_lib_dir "${ORT_LIBRARY}" DIRECTORY)
      if(EXISTS "${_ort_lib_dir}/libonnxruntime_providers_rocm.so" OR
              EXISTS "${_ort_lib_dir}/libonnxruntime_providers_cuda.so")
        set(_ort_has_gpu TRUE)
        message(STATUS "gr-onnx: GPU execution provider detected")
      endif()
    endif()
  endif()

  if(NOT _have_ort)
    message(STATUS "gr-onnx: system onnxruntime not found")
    message(STATUS "gr-onnx: install via package manager, or use ENABLE_ONNX_INTEGRATION=on")
    return()
  endif()
endif()

# =============================================================================
# MODE: on - Use bundled static libraries (fetched via Git LFS)
# =============================================================================
if(ENABLE_ONNX_INTEGRATION STREQUAL "on")
  # Determine platform-specific library subdirectory
  if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    # WASM build - prefer WebGPU variant, fall back to CPU
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lib/wasm-webgpu/libonnxruntime_webassembly.a")
      set(_bundled_subdir "wasm-webgpu")
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lib/wasm-cpu/libonnxruntime_webassembly.a")
      set(_bundled_subdir "wasm-cpu")
    else()
      set(_bundled_subdir "wasm-webgpu")  # Will trigger LFS fetch
    endif()
    set(_bundled_lib_name "libonnxruntime_webassembly.a")

    # Allow override
    option(GR_ONNX_WASM_CPU_ONLY "Use CPU-only WASM build instead of WebGPU" OFF)
    if(GR_ONNX_WASM_CPU_ONLY)
      set(_bundled_subdir "wasm-cpu")
    endif()
  else()
    # Native build - detect architecture
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
      set(_bundled_subdir "native-linux-x86_64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
      set(_bundled_subdir "native-linux-aarch64")
    else()
      message(FATAL_ERROR
              "gr-onnx: no bundled library for ${CMAKE_SYSTEM_PROCESSOR}\n"
              "Use ENABLE_ONNX_INTEGRATION=opt with a system-installed onnxruntime,\n"
              "or rebuild libraries with: ./blocks/onnx/lib/rebuild_static_libs.sh")
    endif()
    set(_bundled_lib_name "libonnxruntime.a")
  endif()

  # Ensure LFS files are fetched
  gr_onnx_ensure_lfs_files(${_bundled_subdir})

  set(_bundled_lib "${CMAKE_CURRENT_SOURCE_DIR}/lib/${_bundled_subdir}/${_bundled_lib_name}")
  set(_bundled_include "${CMAKE_CURRENT_SOURCE_DIR}/lib/${_bundled_subdir}/include")

  if(NOT EXISTS "${_bundled_lib}")
    message(FATAL_ERROR "gr-onnx: bundled library not found: ${_bundled_lib}")
  endif()

  # Create imported target
  # ===========================================================================
  # IMPORTANT: Use --whole-archive for static ONNX Runtime libraries
  # ===========================================================================
  # ONNX Runtime static libraries contain template instantiations (like
  # GetOpSchema<MaxPool_Onnx_ver1>) that aren't directly referenced by user
  # code. Without --whole-archive, the linker drops these objects and you
  # get undefined symbol errors at link time.
  # ===========================================================================
  add_library(onnxruntime::onnxruntime INTERFACE IMPORTED GLOBAL)

  # Use INTERFACE library with explicit linker flags for --whole-archive
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
    # Modern CMake approach using WHOLE_ARCHIVE feature
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${_bundled_include}"
            INTERFACE_LINK_LIBRARIES "$<LINK_LIBRARY:WHOLE_ARCHIVE,${_bundled_lib}>"
    )
    message(STATUS "gr-onnx: using CMake 3.24+ WHOLE_ARCHIVE feature")
  else()
    # Legacy approach for older CMake versions
    if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
      # Emscripten doesn't need --whole-archive (single WASM output)
      set_target_properties(onnxruntime::onnxruntime PROPERTIES
              INTERFACE_INCLUDE_DIRECTORIES "${_bundled_include}"
              INTERFACE_LINK_LIBRARIES "${_bundled_lib}"
      )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      set_target_properties(onnxruntime::onnxruntime PROPERTIES
              INTERFACE_INCLUDE_DIRECTORIES "${_bundled_include}"
              INTERFACE_LINK_LIBRARIES "-Wl,--whole-archive;${_bundled_lib};-Wl,--no-whole-archive"
      )
      message(STATUS "gr-onnx: using --whole-archive for static linking")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
      set_target_properties(onnxruntime::onnxruntime PROPERTIES
              INTERFACE_INCLUDE_DIRECTORIES "${_bundled_include}"
              INTERFACE_LINK_LIBRARIES "-Wl,-force_load,${_bundled_lib}"
      )
      message(STATUS "gr-onnx: using -force_load for static linking")
    else()
      # Fallback - might have linking issues
      message(WARNING "gr-onnx: unknown compiler, static linking may have undefined symbols")
      set_target_properties(onnxruntime::onnxruntime PROPERTIES
              INTERFACE_INCLUDE_DIRECTORIES "${_bundled_include}"
              INTERFACE_LINK_LIBRARIES "${_bundled_lib}"
      )
    endif()
  endif()

  # Add required system libraries for native builds
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    target_link_libraries(onnxruntime::onnxruntime INTERFACE pthread dl)
  endif()

  set(_have_ort TRUE)
  set(_ort_minimal TRUE)

  message(STATUS "gr-onnx: using bundled ${_bundled_subdir} v${GR_ONNX_BUNDLED_VERSION}")
endif()

# =============================================================================
# Create gr-onnx target
# =============================================================================
if(NOT _have_ort)
  message(FATAL_ERROR "gr-onnx: internal error - no onnxruntime target created")
endif()

add_library(gr-onnx INTERFACE)
target_link_libraries(gr-onnx INTERFACE
        gnuradio-options
        gnuradio-meta
        gnuradio-core
        gnuradio-algorithm
        onnxruntime::onnxruntime
)
target_include_directories(gr-onnx INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
        $<INSTALL_INTERFACE:include/>
)

# Compile definitions for runtime feature detection
if(_ort_minimal)
  target_compile_definitions(gr-onnx INTERFACE GR_ONNX_MINIMAL_BUILD=1)
else()
  target_compile_definitions(gr-onnx INTERFACE GR_ONNX_MINIMAL_BUILD=0)
endif()

if(_ort_has_gpu)
  target_compile_definitions(gr-onnx INTERFACE GR_ONNX_HAS_GPU=1)
else()
  target_compile_definitions(gr-onnx INTERFACE GR_ONNX_HAS_GPU=0)
endif()

# Subdirectories
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src)
  add_subdirectory(src)
endif()
if(ENABLE_TESTING AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test)
  add_subdirectory(test)
endif()

# Summary
message(STATUS "gr-onnx: enabled (mode=${ENABLE_ONNX_INTEGRATION}, minimal=${_ort_minimal}, gpu=${_ort_has_gpu})")
