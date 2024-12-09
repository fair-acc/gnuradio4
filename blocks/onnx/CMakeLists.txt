# blocks/onnx/CMakeLists.txt
#
# ONNX Runtime GR4 integration
#
# Modes:
#   off - disable ONNX integration entirely
#   opt - (default) system .so if available, otherwise bundled static lib
#   on  - build minimal ORT from source via ExternalProject
#
# Resolution order for 'opt':
#   1. System shared library (.so) via find_package / pkg-config / manual search
#      → full runtime, GPU support, uses .onnx models
#   2. Bundled static library from ORT_STATIC_LIB_DIR (set by Docker or user)
#      → minimal runtime, uses .ort models
#   3. Not found → skip gracefully

if(TARGET gr-onnx)
  return()
endif()

set(ENABLE_ONNX_INTEGRATION "opt" CACHE STRING "off | opt | on")
set_property(CACHE ENABLE_ONNX_INTEGRATION PROPERTY STRINGS off opt on)

if(ENABLE_ONNX_INTEGRATION STREQUAL "off")
  message(STATUS "gr-onnx: disabled")
  return()
endif()

set(ONNXRUNTIME_VERSION "1.21.0" CACHE STRING "ONNX Runtime version for 'on' mode")
set(GR_ONNX_HOST_CXX_COMPILER "" CACHE STRING "host C++ compiler for building ORT from source (use when cross-compiling or with AdaptiveCpp)")
if (NOT GR_ONNX_HOST_CXX_COMPILER)
  if (CMAKE_CROSSCOMPILING OR CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER MATCHES "acpp")
    find_program(_host_gxx NAMES g++-15 g++-14 g++)
    if (_host_gxx)
      set(GR_ONNX_HOST_CXX_COMPILER "${_host_gxx}")
    else ()
      set(GR_ONNX_HOST_CXX_COMPILER "${CMAKE_CXX_COMPILER}")
    endif ()
  else ()
    set(GR_ONNX_HOST_CXX_COMPILER "${CMAKE_CXX_COMPILER}")
  endif ()
endif ()

set(_have_ort FALSE)
set(_ort_minimal FALSE)
set(_ort_has_gpu FALSE)

# hint path for pre-built static libraries (set in Docker image or by user)
set(ORT_STATIC_LIB_DIR "/opt/onnxruntime" CACHE PATH "path to pre-built ORT static libs")

# =============================================================================
# MODE: opt — system .so first, then bundled static lib fallback
# =============================================================================
if(ENABLE_ONNX_INTEGRATION STREQUAL "opt")

  # --- try system shared library (native only) ---
  if (NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    find_package(onnxruntime CONFIG QUIET)
    if (TARGET onnxruntime::onnxruntime)
      set(_have_ort TRUE)
      message(STATUS "gr-onnx: found system onnxruntime via CMake config")
    elseif (TARGET onnxruntime)
      add_library(onnxruntime::onnxruntime ALIAS onnxruntime)
      set(_have_ort TRUE)
      message(STATUS "gr-onnx: found system onnxruntime (aliased)")
    endif ()

    if (NOT _have_ort)
      find_package(PkgConfig QUIET)
      if (PkgConfig_FOUND)
        pkg_check_modules(ORT QUIET libonnxruntime)
        if (ORT_FOUND)
          add_library(onnxruntime::onnxruntime INTERFACE IMPORTED)
          set_target_properties(onnxruntime::onnxruntime PROPERTIES
                  INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIRS}"
                  INTERFACE_LINK_LIBRARIES "${ORT_LIBRARIES}")
          set(_have_ort TRUE)
          message(STATUS "gr-onnx: found system onnxruntime via pkg-config")
        endif ()
      endif ()
    endif ()

    if (NOT _have_ort)
      find_library(ORT_LIBRARY NAMES onnxruntime
              HINTS /usr/local/lib /opt/onnxruntime/lib
              PATH_SUFFIXES lib lib64)
      find_path(ORT_INCLUDE_DIR NAMES onnxruntime_cxx_api.h
              HINTS /usr/local/include /opt/onnxruntime/include
              PATH_SUFFIXES include onnxruntime)
      if (ORT_LIBRARY AND ORT_INCLUDE_DIR)
        add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
                IMPORTED_LOCATION "${ORT_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIR}")
        set(_have_ort TRUE)
        message(STATUS "gr-onnx: found system onnxruntime at ${ORT_LIBRARY}")

        get_filename_component(_ort_lib_dir "${ORT_LIBRARY}" DIRECTORY)
        if (EXISTS "${_ort_lib_dir}/libonnxruntime_providers_rocm.so" OR
                EXISTS "${_ort_lib_dir}/libonnxruntime_providers_cuda.so")
          set(_ort_has_gpu TRUE)
          message(STATUS "gr-onnx: GPU execution provider detected")
        endif ()
      endif()
    endif()
  endif()

  # --- fallback: bundled static library ---
  if(NOT _have_ort)
    if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
      set(_bundled_lib "${ORT_STATIC_LIB_DIR}/wasm/lib/libonnxruntime_webassembly.a")
      set(_bundled_inc "${ORT_STATIC_LIB_DIR}/wasm/include")
    else ()
      set(_bundled_lib "${ORT_STATIC_LIB_DIR}/native/lib/libonnxruntime.a")
      set(_bundled_inc "${ORT_STATIC_LIB_DIR}/native/include")
    endif ()

    if (EXISTS "${_bundled_lib}" AND IS_DIRECTORY "${_bundled_inc}")
      add_library(onnxruntime::onnxruntime INTERFACE IMPORTED GLOBAL)
      if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${_bundled_inc}"
                INTERFACE_LINK_LIBRARIES "${_bundled_lib}")
      else ()
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${_bundled_inc}"
                INTERFACE_LINK_LIBRARIES "$<LINK_LIBRARY:WHOLE_ARCHIVE,${_bundled_lib}>")
        find_package(ZLIB REQUIRED)
        target_link_libraries(onnxruntime::onnxruntime INTERFACE pthread dl ZLIB::ZLIB)
      endif ()
      set(_have_ort TRUE)
      set(_ort_minimal TRUE)
      message(STATUS "gr-onnx: using bundled static library at ${_bundled_lib}")
    endif ()
  endif ()

  if (NOT _have_ort)
    message(STATUS "gr-onnx: no system onnxruntime or bundled static lib found")
    message(STATUS "gr-onnx: install via package manager, set ORT_STATIC_LIB_DIR, or use ENABLE_ONNX_INTEGRATION=on")
    return()
  endif()
endif()

# =============================================================================
# MODE: on — build minimal ORT from source via ExternalProject
# =============================================================================
if(ENABLE_ONNX_INTEGRATION STREQUAL "on")
  include(ExternalProject)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  set(ORT_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime")
  set(ORT_INSTALL_DIR "${ORT_PREFIX}/install")
  set(ORT_SRC_DIR "${ORT_PREFIX}/src")
  set(GR_ONNX_BUILD_JOBS "${GR_BUILD_PARALLEL_LEVEL}" CACHE STRING "parallel jobs for ORT build")
  if (NOT GR_ONNX_BUILD_JOBS)
    set(GR_ONNX_BUILD_JOBS 6)
  endif ()

  file(MAKE_DIRECTORY "${ORT_PREFIX}")

  if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    set(_ort_lib_name "libonnxruntime_webassembly.a")
    set(_ort_platform_args
            "--build_wasm_static_lib --enable_wasm_simd --enable_wasm_threads"
            " --disable_wasm_exception_catching --enable_wasm_api_exception_catching"
            " --cmake_extra_defines onnxruntime_ENABLE_LTO=OFF"
            " --cmake_extra_defines CMAKE_CXX_FLAGS=-fwasm-exceptions"
            " --cmake_extra_defines CMAKE_C_FLAGS=-fwasm-exceptions")
  else ()
    set(_ort_lib_name "libonnxruntime.a")
    set(_ort_platform_args "")
  endif ()

  set(_ort_lib "${ORT_INSTALL_DIR}/lib/${_ort_lib_name}")

  # --- patch script (GCC 15 compatibility + Eigen hash fixes) ---
  set(_patch_script "${ORT_PREFIX}/patch_ort.sh")
  file(WRITE "${_patch_script}"
          [=[#!/bin/bash
set -e
cd "$1"

# Eigen hash fixes (GitLab regenerates archives with different hashes)
if [ -f cmake/deps.txt ]; then
  sed -i 's/be8be39fdbc6e60e94fa7870b280707069b5b81a/32b145f525a8308d7ab1c09388b2e288312d8eba/g' cmake/deps.txt
  sed -i 's/5ea4d05e62d7f954a46b3213f9b2535bdd866803/51982be81bbe52572b54180454df11a3ece9a934/g' cmake/deps.txt
fi

# GCC 15: optimizer_api.h missing <cstdint> (uint8_t not in scope)
OPT_API_H="onnxruntime/core/optimizer/transpose_optimization/optimizer_api.h"
if [ -f "$OPT_API_H" ] && ! grep -q '<cstdint>' "$OPT_API_H"; then
  sed -i '/#pragma once/a #include <cstdint>' "$OPT_API_H"
fi

# GCC 15: semver.h missing <cstdint> and uses POSIX-reserved names major/minor/patch
SEMVER_H="onnxruntime/core/common/semver.h"
if [ -f "$SEMVER_H" ] && ! grep -q '<cstdint>' "$SEMVER_H"; then
  sed -i '/#pragma once/a #include <cstdint>' "$SEMVER_H"
  sed -i 's/uint32_t major{}/uint32_t ver_major{}/g' "$SEMVER_H"
  sed -i 's/uint32_t minor{}/uint32_t ver_minor{}/g' "$SEMVER_H"
  sed -i 's/uint32_t patch{}/uint32_t ver_patch{}/g' "$SEMVER_H"
fi
SEMVER_CC="onnxruntime/core/common/semver.cc"
if [ -f "$SEMVER_CC" ] && grep -qE '\.(major|minor|patch)' "$SEMVER_CC"; then
  sed -i 's/->major/->ver_major/g; s/->minor/->ver_minor/g; s/->patch/->ver_patch/g' "$SEMVER_CC"
  sed -i 's/\.major/\.ver_major/g; s/\.minor/\.ver_minor/g; s/\.patch/\.ver_patch/g' "$SEMVER_CC"
fi
]=])
  file(CHMOD "${_patch_script}" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)

  # --- build script (configure + compile via ORT's build.py) ---
  set(_build_script "${ORT_PREFIX}/build_ort.sh")
  message(STATUS "gr-onnx: ORT will be compiled with ${GR_ONNX_HOST_CXX_COMPILER}")
  file(WRITE "${_build_script}"
          "#!/bin/bash
set -e
cd \"${ORT_SRC_DIR}\"
export CC=\"$(echo ${GR_ONNX_HOST_CXX_COMPILER} | sed 's/g++/gcc/;s/clang++/clang/')\"
export CXX=\"${GR_ONNX_HOST_CXX_COMPILER}\"
\"${Python3_EXECUTABLE}\" tools/ci_build/build.py \
  --build_dir \"${ORT_PREFIX}/build\" \
  --config Release \
  --cmake_extra_defines CMAKE_POLICY_VERSION_MINIMUM=3.5 \
  --cmake_extra_defines CMAKE_C_COMPILER=$CC \
  --cmake_extra_defines CMAKE_CXX_COMPILER=$CXX \
  --cmake_extra_defines FETCHCONTENT_QUIET=OFF \
  --cmake_extra_defines CMAKE_DISABLE_FIND_PACKAGE_Protobuf=TRUE \
  --cmake_extra_defines CMAKE_DISABLE_FIND_PACKAGE_flatbuffers=TRUE \
  --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
  --update --build \
  --minimal_build extended \
  --disable_ml_ops \
  --disable_contrib_ops \
  --compile_no_warning_as_error \
  --skip_tests \
  --parallel ${GR_ONNX_BUILD_JOBS} \
  ${_ort_platform_args}
          ")
  file(CHMOD "${_build_script}" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)

  # --- install script (combine archives for native, copy for WASM) ---
  set(_install_script "${ORT_PREFIX}/install_ort.sh")
  if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    file(WRITE "${_install_script}"
            "#!/bin/bash
set -e
mkdir -p \"${ORT_INSTALL_DIR}/lib\" \"${ORT_INSTALL_DIR}/include\"
BUILD=\"${ORT_PREFIX}/build/Release\"
if [ -f \"$BUILD/libonnxruntime_webassembly.a\" ]; then
  cp \"$BUILD/libonnxruntime_webassembly.a\" \"${ORT_INSTALL_DIR}/lib/\"
elif [ -f \"$BUILD/onnxruntime_webassembly.a\" ]; then
  cp \"$BUILD/onnxruntime_webassembly.a\" \"${ORT_INSTALL_DIR}/lib/libonnxruntime_webassembly.a\"
fi
cp -r \"${ORT_SRC_DIR}/include/onnxruntime/core/session/\"* \"${ORT_INSTALL_DIR}/include/\"
")
  else()
    file(WRITE "${_install_script}"
            [=[#!/bin/bash
set -e
BUILD="]=] "${ORT_PREFIX}/build/Release" [=["
INSTALL_DIR="]=] "${ORT_INSTALL_DIR}" [=["
SRC_DIR="]=] "${ORT_SRC_DIR}" [=["
mkdir -p "$INSTALL_DIR/lib" "$INSTALL_DIR/include"

MRI=$(mktemp)
echo "CREATE $INSTALL_DIR/lib/libonnxruntime.a" > "$MRI"

for lib in libonnxruntime_common.a libonnxruntime_flatbuffers.a libonnxruntime_framework.a \
           libonnxruntime_graph.a libonnxruntime_lora.a libonnxruntime_mlas.a \
           libonnxruntime_optimizer.a libonnxruntime_providers.a libonnxruntime_session.a \
           libonnxruntime_util.a; do
  [ -f "$BUILD/$lib" ] && echo "ADDLIB $BUILD/$lib" >> "$MRI"
done

for name in libonnx.a libonnx_proto.a; do
  for loc in "$BUILD/_deps/onnx-build/$name" "$BUILD/$name"; do
    [ -f "$loc" ] && { echo "ADDLIB $loc" >> "$MRI"; break; }
  done
done

for lib in "$BUILD/_deps/protobuf-build/libprotobuf.a" "$BUILD/_deps/protobuf-build/libprotobuf-lite.a"; do
  [ -f "$lib" ] && { echo "ADDLIB $lib" >> "$MRI"; break; }
done

[ -f "$BUILD/_deps/flatbuffers-build/libflatbuffers.a" ] && \
  echo "ADDLIB $BUILD/_deps/flatbuffers-build/libflatbuffers.a" >> "$MRI"

find "$BUILD/_deps/abseil_cpp-build" -name "libabsl_*.a" -type f 2>/dev/null | sort | \
  while read -r lib; do echo "ADDLIB $lib" >> "$MRI"; done

[ -f "$BUILD/_deps/pytorch_cpuinfo-build/libcpuinfo.a" ] && \
  echo "ADDLIB $BUILD/_deps/pytorch_cpuinfo-build/libcpuinfo.a" >> "$MRI"

# re2 (needed by RegexFullMatch kernel in CPU execution provider)
# ORT fetches re2 source but may not build it — build if needed
if [ -d "$BUILD/_deps/re2-build" ] && [ ! -f "$BUILD/_deps/re2-build/libre2.a" ]; then
  cmake --build "$BUILD" --target re2 -j "${GR_ONNX_BUILD_JOBS:-6}" 2>/dev/null || true
fi
for re2 in "$BUILD/_deps/re2-build/libre2.a" \
           /usr/local/lib64/libre2.a \
           /usr/lib/x86_64-linux-gnu/libre2.a \
           /usr/lib64/libre2.a; do
  [ -f "$re2" ] && { echo "ADDLIB $re2" >> "$MRI"; break; }
done

echo "SAVE" >> "$MRI"
echo "END" >> "$MRI"
ar -M < "$MRI"
rm -f "$MRI"

strip --strip-debug "$INSTALL_DIR/lib/libonnxruntime.a" 2>/dev/null || true
cp -r "$SRC_DIR/include/onnxruntime/core/session/"* "$INSTALL_DIR/include/"
]=])
  endif()
  file(CHMOD "${_install_script}" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)

  ExternalProject_Add(onnxruntime_ext
          PREFIX "${ORT_PREFIX}"
          SOURCE_DIR "${ORT_SRC_DIR}"
          BINARY_DIR "${ORT_PREFIX}/build"
          INSTALL_DIR "${ORT_INSTALL_DIR}"
          STAMP_DIR "${ORT_PREFIX}/stamp"
          TMP_DIR "${ORT_PREFIX}/tmp"
          DOWNLOAD_DIR "${ORT_PREFIX}/download"
          LOG_DIR "${ORT_PREFIX}/log"
          GIT_REPOSITORY https://github.com/microsoft/onnxruntime.git
          GIT_TAG "v${ONNXRUNTIME_VERSION}"
          GIT_SHALLOW TRUE
          GIT_PROGRESS TRUE
          PATCH_COMMAND bash "${_patch_script}" "${ORT_SRC_DIR}"
          CONFIGURE_COMMAND bash "${_build_script}"
          BUILD_COMMAND ""
          INSTALL_COMMAND bash "${_install_script}"
          BUILD_BYPRODUCTS "${_ort_lib}"
          LOG_DOWNLOAD ON
          LOG_PATCH ON
          LOG_CONFIGURE ON
          LOG_INSTALL ON
  )

  add_library(onnxruntime::onnxruntime INTERFACE IMPORTED GLOBAL)
  if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${ORT_INSTALL_DIR}/include"
            INTERFACE_LINK_LIBRARIES "${_ort_lib}")
  else ()
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${ORT_INSTALL_DIR}/include"
            INTERFACE_LINK_LIBRARIES "$<LINK_LIBRARY:WHOLE_ARCHIVE,${_ort_lib}>")
    find_package(ZLIB REQUIRED)
    target_link_libraries(onnxruntime::onnxruntime INTERFACE pthread dl ZLIB::ZLIB)
  endif()
  add_dependencies(onnxruntime::onnxruntime onnxruntime_ext)

  set(_have_ort TRUE)
  set(_ort_minimal TRUE)

  message(STATUS "gr-onnx: building minimal onnxruntime v${ONNXRUNTIME_VERSION} from source")
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
        exprtk
        onnxruntime::onnxruntime)
target_include_directories(gr-onnx INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
        $<INSTALL_INTERFACE:include/>)

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

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src)
  add_subdirectory(src)
endif()
if(ENABLE_TESTING AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test)
  add_subdirectory(test)
endif()

message(STATUS "gr-onnx: enabled (mode=${ENABLE_ONNX_INTEGRATION}, minimal=${_ort_minimal}, gpu=${_ort_has_gpu})")
