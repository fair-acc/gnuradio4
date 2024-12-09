# blocks/onnx/CMakeLists.txt

if(TARGET gr-onnx)
  return()
endif()

set(ENABLE_ONNX_INTEGRATION "opt" CACHE STRING "off | opt | on")
set_property(CACHE ENABLE_ONNX_INTEGRATION PROPERTY STRINGS off opt on)

if(ENABLE_ONNX_INTEGRATION STREQUAL "off")
  message(STATUS "gr-onnx: disabled")
  return()
endif()

set(ONNXRUNTIME_VERSION "1.20.1" CACHE STRING "ONNX Runtime version for 'on' mode")
set(_have_ort FALSE)
set(_ort_minimal FALSE)

# =============================================================================
# MODE: opt - use system packages (supports .onnx and .ort)
# =============================================================================
if(ENABLE_ONNX_INTEGRATION STREQUAL "opt")
  if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    message(STATUS "gr-onnx: 'opt' mode not available for WASM (no system packages)")
    message(STATUS "gr-onnx: use ENABLE_ONNX_INTEGRATION=on for WASM builds")
    return()
  endif()

  find_package(onnxruntime CONFIG QUIET)
  if(TARGET onnxruntime::onnxruntime)
    set(_have_ort TRUE)
    message(STATUS "gr-onnx: using system onnxruntime (full, .onnx + .ort)")
  elseif(TARGET onnxruntime)
    add_library(onnxruntime::onnxruntime ALIAS onnxruntime)
    set(_have_ort TRUE)
    message(STATUS "gr-onnx: using system onnxruntime (aliased)")
  else()
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
      pkg_check_modules(ORT QUIET libonnxruntime)
      if(ORT_FOUND)
        add_library(onnxruntime::onnxruntime INTERFACE IMPORTED)
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIRS}"
                INTERFACE_LINK_LIBRARIES "${ORT_LIBRARIES}")
        set(_have_ort TRUE)
        message(STATUS "gr-onnx: using system onnxruntime (pkg-config)")
      endif()
    endif()
  endif()

  if(NOT _have_ort)
    message(STATUS "gr-onnx: system onnxruntime not found, skipping")
    message(STATUS "gr-onnx: install via: vcpkg install onnxruntime, or use ENABLE_ONNX_INTEGRATION=on")
    return()
  endif()
endif()

# =============================================================================
# MODE: on - build minimal static lib from source (.ort only, smaller binary)
# =============================================================================
if(ENABLE_ONNX_INTEGRATION STREQUAL "on")
  include(ExternalProject)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  set(ORT_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime)
  set(ORT_INSTALL_DIR ${ORT_PREFIX}/install)

  # Determine output library path and build arguments based on platform
  if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    set(_ort_lib_name "libonnxruntime_webassembly.a")
    set(_ort_platform_args "--build_wasm_static_lib --enable_wasm_simd --enable_wasm_threads")

    # Optional WebGPU support
    option(ONNX_ENABLE_WEBGPU "Enable WebGPU execution provider in WASM build" OFF)
    if(ONNX_ENABLE_WEBGPU)
      string(APPEND _ort_platform_args " --use_jsep --use_webgpu")
    endif()
  else()
    set(_ort_lib_name "libonnxruntime.a")
    set(_ort_platform_args "")

    # Optional CUDA support for native builds
    option(ONNX_ENABLE_CUDA "Enable CUDA execution provider" OFF)
    if(ONNX_ENABLE_CUDA)
      set(_ort_platform_args "--use_cuda")
      if(DEFINED CUDA_HOME)
        string(APPEND _ort_platform_args " --cuda_home ${CUDA_HOME} --cudnn_home ${CUDA_HOME}")
      endif()
    endif()
  endif()

  set(_ort_lib "${ORT_INSTALL_DIR}/lib/${_ort_lib_name}")

  # Ensure prefix directory exists for scripts
  file(MAKE_DIRECTORY ${ORT_PREFIX})

  # Patch script - fix Eigen hash mismatch (GitLab regenerates archives with different hashes)
  # This patches cmake/deps.txt after clone
  set(ORT_PATCH_SCRIPT "${ORT_PREFIX}/patch_ort.sh")
  file(WRITE ${ORT_PATCH_SCRIPT}
          "#!/bin/bash
set -e
cd \"${ORT_PREFIX}/src\"
echo \"Patching ONNX Runtime for CMake 4.x and Eigen hash fix...\"

# Fix Eigen hash mismatch - GitLab regenerates zip archives with different hashes
# Old hash: be8be39fdbc6e60e94fa7870b280707069b5b81a
# New hash: 32b145f525a8308d7ab1c09388b2e288312d8eba
if [ -f cmake/deps.txt ]; then
  sed -i 's/be8be39fdbc6e60e94fa7870b280707069b5b81a/32b145f525a8308d7ab1c09388b2e288312d8eba/g' cmake/deps.txt
  echo \"  - Patched Eigen hash in cmake/deps.txt\"
fi

echo \"Patch complete\"
")
  file(CHMOD ${ORT_PATCH_SCRIPT} PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

  # Build script - must run from source directory
  # build.py handles configure+build in one step
  # Note: CMAKE_POLICY_VERSION_MINIMUM=3.5 required for CMake 4.x compatibility
  #       with older dependencies (google_nsync, etc.)
  set(ORT_BUILD_SCRIPT "${ORT_PREFIX}/build_ort.sh")
  file(WRITE ${ORT_BUILD_SCRIPT}
          "#!/bin/bash
set -e
cd \"${ORT_PREFIX}/src\"
echo \"Building ONNX Runtime v${ONNXRUNTIME_VERSION} minimal from: $(pwd)\"
\"${Python3_EXECUTABLE}\" tools/ci_build/build.py \\
  --build_dir \"${ORT_PREFIX}/build\" \\
  --config Release \\
  --cmake_extra_defines CMAKE_POLICY_VERSION_MINIMUM=3.5 \\
  --update \\
  --build \\
  --minimal_build extended \\
  --disable_ml_ops \\
  --disable_rtti \\
  --skip_tests \\
  --parallel \\
  ${_ort_platform_args}
echo \"ONNX Runtime build completed successfully\"
")
  file(CHMOD ${ORT_BUILD_SCRIPT} PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

  # Install script - copy artifacts to install dir
  set(ORT_INSTALL_SCRIPT "${ORT_PREFIX}/install_ort.sh")
  file(WRITE ${ORT_INSTALL_SCRIPT}
          "#!/bin/bash
set -e
echo \"Installing ONNX Runtime artifacts...\"
mkdir -p \"${ORT_INSTALL_DIR}/lib\" \"${ORT_INSTALL_DIR}/include\"
cp \"${ORT_PREFIX}/build/Release/${_ort_lib_name}\" \"${ORT_INSTALL_DIR}/lib/\"
cp -r \"${ORT_PREFIX}/src/include/onnxruntime/core/session/\"* \"${ORT_INSTALL_DIR}/include/\"
echo \"ONNX Runtime installed to: ${ORT_INSTALL_DIR}\"
")
  file(CHMOD ${ORT_INSTALL_SCRIPT} PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

  ExternalProject_Add(onnxruntime_ext
          PREFIX          ${ORT_PREFIX}
          SOURCE_DIR      ${ORT_PREFIX}/src
          BINARY_DIR      ${ORT_PREFIX}/build
          INSTALL_DIR     ${ORT_INSTALL_DIR}
          STAMP_DIR       ${ORT_PREFIX}/stamp
          TMP_DIR         ${ORT_PREFIX}/tmp
          DOWNLOAD_DIR    ${ORT_PREFIX}/download
          LOG_DIR         ${ORT_PREFIX}/log
          GIT_REPOSITORY  https://github.com/microsoft/onnxruntime.git
          GIT_TAG         v${ONNXRUNTIME_VERSION}
          GIT_SHALLOW     TRUE
          GIT_PROGRESS    TRUE
          PATCH_COMMAND   ${ORT_PATCH_SCRIPT}
          CONFIGURE_COMMAND ${ORT_BUILD_SCRIPT}
          BUILD_COMMAND   ""
          INSTALL_COMMAND ${ORT_INSTALL_SCRIPT}
          BUILD_BYPRODUCTS ${_ort_lib}
          LOG_DOWNLOAD    ON
          LOG_PATCH       ON
          LOG_CONFIGURE   ON
          LOG_INSTALL     ON
  )

  # Create imported target
  add_library(onnxruntime::onnxruntime STATIC IMPORTED)
  set_target_properties(onnxruntime::onnxruntime PROPERTIES
          IMPORTED_LOCATION "${_ort_lib}"
          INTERFACE_INCLUDE_DIRECTORIES "${ORT_INSTALL_DIR}/include"
  )
  add_dependencies(onnxruntime::onnxruntime onnxruntime_ext)

  set(_have_ort TRUE)
  set(_ort_minimal TRUE)

  if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    message(STATUS "gr-onnx: building minimal WASM onnxruntime v${ONNXRUNTIME_VERSION} (.ort only)")
  else()
    message(STATUS "gr-onnx: building minimal native onnxruntime v${ONNXRUNTIME_VERSION} (.ort only)")
  endif()
endif()

# =============================================================================
# Create gr-onnx target
# =============================================================================
if(NOT _have_ort)
  message(FATAL_ERROR "gr-onnx: internal error - _have_ort not set")
endif()

add_library(gr-onnx INTERFACE)
target_link_libraries(gr-onnx INTERFACE
        gnuradio-options gnuradio-meta gnuradio-core gnuradio-algorithm
        onnxruntime::onnxruntime)
target_include_directories(gr-onnx INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
        $<INSTALL_INTERFACE:include/>)

# Expose whether this is a minimal build (for runtime checks)
if(_ort_minimal)
  target_compile_definitions(gr-onnx INTERFACE GR_ONNX_MINIMAL_BUILD=1)
else()
  target_compile_definitions(gr-onnx INTERFACE GR_ONNX_MINIMAL_BUILD=0)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src)
  add_subdirectory(src)
endif()
if(ENABLE_TESTING AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test)
  add_subdirectory(test)
endif()

message(STATUS "gr-onnx: enabled (mode=${ENABLE_ONNX_INTEGRATION}, minimal=${_ort_minimal})")
