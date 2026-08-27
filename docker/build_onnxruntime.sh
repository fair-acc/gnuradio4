#!/bin/sh
# Builds the native and WebAssembly ONNX Runtime static libraries gr-onnx links against. The
# Dockerfile runs this; run it directly to try a change without paying for a CI image rebuild.
#
#   ORT_STATIC_LIB_DIR=$HOME/ort EMSDK_HOME=$HOME/git/emsdk ./docker/build_onnxruntime.sh
#   cmake -DORT_STATIC_LIB_DIR=$HOME/ort ...
set -eu

ONNXRUNTIME_VERSION="${ONNXRUNTIME_VERSION:-1.21.0}"
ORT_STATIC_LIB_DIR="${ORT_STATIC_LIB_DIR:-/opt/onnxruntime}"
EMSDK_HOME="${EMSDK_HOME:-/opt/emsdk}"
WORK="${ORT_BUILD_WORKDIR:-/tmp}"
# capped: a few translation units peak near 4 GB, so more jobs than this OOMs a 16 GB runner
JOBS="${ORT_BUILD_JOBS:-$(nproc)}"
if [ "$JOBS" -gt 4 ]; then JOBS=4; fi

SRC="$WORK/ort-src"
NATIVE_BUILD="$WORK/ort-build-native"
WASM_BUILD="$WORK/ort-build-wasm"
ORT_EMSDK="$WORK/ort-emsdk"

rm -rf "$SRC" "$NATIVE_BUILD" "$WASM_BUILD" "$ORT_EMSDK"
mkdir -p "$ORT_STATIC_LIB_DIR/wasm/lib" "$ORT_STATIC_LIB_DIR/wasm/include" \
         "$ORT_STATIC_LIB_DIR/native/lib" "$ORT_STATIC_LIB_DIR/native/include"

git clone --depth 1 --branch "v${ONNXRUNTIME_VERSION}" https://github.com/microsoft/onnxruntime.git "$SRC"
cd "$SRC"

# GCC 15: optimizer_api.h uses uint8_t without including <cstdint>
OPT_API_H=onnxruntime/core/optimizer/transpose_optimization/optimizer_api.h
if [ -f "$OPT_API_H" ] && ! grep -q '<cstdint>' "$OPT_API_H"; then
  sed -i '/#pragma once/a #include <cstdint>' "$OPT_API_H"
fi

# GCC 15: semver.h needs <cstdint>, and its major/minor/patch members collide with glibc macros
SEMVER_H=onnxruntime/core/common/semver.h
if [ -f "$SEMVER_H" ] && ! grep -q '<cstdint>' "$SEMVER_H"; then
  sed -i '/#pragma once/a #include <cstdint>' "$SEMVER_H"
  sed -i 's/uint32_t major{}/uint32_t ver_major{}/g; s/uint32_t minor{}/uint32_t ver_minor{}/g; s/uint32_t patch{}/uint32_t ver_patch{}/g' "$SEMVER_H"
fi
SEMVER_CC=onnxruntime/core/common/semver.cc
if [ -f "$SEMVER_CC" ] && grep -qE '\.(major|minor|patch)' "$SEMVER_CC"; then
  sed -i 's/->major/->ver_major/g; s/->minor/->ver_minor/g; s/->patch/->ver_patch/g' "$SEMVER_CC"
  sed -i 's/\.major/\.ver_major/g; s/\.minor/\.ver_minor/g; s/\.patch/\.ver_patch/g' "$SEMVER_CC"
fi

# the pinned Eigen revisions 404 on the mirror ORT lists; these are the surviving ones
if [ -f cmake/deps.txt ]; then
  sed -i 's/be8be39fdbc6e60e94fa7870b280707069b5b81a/32b145f525a8308d7ab1c09388b2e288312d8eba/g; s/5ea4d05e62d7f954a46b3213f9b2535bdd866803/51982be81bbe52572b54180454df11a3ece9a934/g' cmake/deps.txt
fi

# ORT appends -flto unconditionally, which leaves the archive as LLVM bitcode rather than wasm
# objects and crashes wasm-ld on any consumer with a different LLVM. -fno-lto loses: ORT appends last.
sed -i 's/^\( *\)string(APPEND CMAKE_C_FLAGS " -flto")/\1# -flto removed: it makes the static lib bitcode, see docker\/build_onnxruntime.sh/;
        s/^\( *\)string(APPEND CMAKE_CXX_FLAGS " -flto")/\1# -flto removed: it makes the static lib bitcode, see docker\/build_onnxruntime.sh/' \
    cmake/adjust_global_compile_flags.cmake
if grep -qE '^\s*string\(APPEND CMAKE_(C|CXX)_FLAGS " -flto"\)' cmake/adjust_global_compile_flags.cmake; then
  echo "ERROR: failed to remove ORT's hard-coded -flto; the wasm archive would be bitcode" >&2
  exit 1
fi

python3 tools/ci_build/build.py \
  --build_dir "$NATIVE_BUILD" --config Release \
  --cmake_extra_defines CMAKE_POLICY_VERSION_MINIMUM=3.5 \
  --cmake_extra_defines FETCHCONTENT_QUIET=OFF \
  --cmake_extra_defines CMAKE_DISABLE_FIND_PACKAGE_Protobuf=TRUE \
  --cmake_extra_defines CMAKE_DISABLE_FIND_PACKAGE_flatbuffers=TRUE \
  --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
  --update --build --minimal_build extended \
  --disable_ml_ops --disable_contrib_ops \
  --compile_no_warning_as_error --skip_tests --parallel "$JOBS" --allow_running_as_root

# bundle ORT's dozen archives and vendored deps into the single libonnxruntime.a CMake expects
B="$NATIVE_BUILD/Release"
MRI=$(mktemp)
echo "CREATE $ORT_STATIC_LIB_DIR/native/lib/libonnxruntime.a" > "$MRI"
addlib() { [ -f "$1" ] || return 1; echo "ADDLIB $1" >> "$MRI"; }
# required: a missing one silently yields an archive that links OrtGetApiBase but not much else
requirelib() { addlib "$1" || { echo "ERROR: $1 not found; the bundled archive would be incomplete" >&2; exit 1; }; }
for lib in libonnxruntime_common.a libonnxruntime_flatbuffers.a libonnxruntime_framework.a \
           libonnxruntime_graph.a libonnxruntime_lora.a libonnxruntime_mlas.a \
           libonnxruntime_optimizer.a libonnxruntime_providers.a libonnxruntime_session.a \
           libonnxruntime_util.a; do
  requirelib "$B/$lib"
done
# next to ORT's own archives, not under _deps/onnx-build; these carry the generated onnx::*Proto code
requirelib "$B/libonnx.a"
requirelib "$B/libonnx_proto.a"
for lib in "$B/_deps/protobuf-build/libprotobuf.a" "$B/_deps/protobuf-build/libprotobuf-lite.a"; do
  if addlib "$lib"; then break; fi
done
addlib "$B/_deps/flatbuffers-build/libflatbuffers.a" || true
find "$B/_deps/abseil_cpp-build" -name "libabsl_*.a" -type f 2>/dev/null | sort |
  while read -r lib; do echo "ADDLIB $lib" >> "$MRI"; done
addlib "$B/_deps/pytorch_cpuinfo-build/libcpuinfo.a" || true
for re2 in "$B/_deps/re2-build/libre2.a" /usr/local/lib64/libre2.a /usr/lib/x86_64-linux-gnu/libre2.a; do
  if addlib "$re2"; then break; fi
done
printf 'SAVE\nEND\n' >> "$MRI"
ar -M < "$MRI"
rm -f "$MRI"
strip --strip-debug "$ORT_STATIC_LIB_DIR/native/lib/libonnxruntime.a" 2>/dev/null || true
cp -r "$SRC"/include/onnxruntime/core/session/* "$ORT_STATIC_LIB_DIR/native/include/"

# build.py installs its own pinned emsdk into whatever tree it is given, so give it a copy --
# pointed at $EMSDK_HOME it uninstalls ${EMSDK_VERSION} and every later emsdk activate fails
cp -a "$EMSDK_HOME" "$ORT_EMSDK"
rm -rf cmake/external/emsdk
ln -s "$ORT_EMSDK" cmake/external/emsdk
# source in place: emsdk_env.sh resolves relative to its own directory, and it CLEARS EMSDK_HOME --
# nothing below may use that variable
cd "$ORT_EMSDK"
# shellcheck source=/dev/null
. ./emsdk_env.sh
cd "$SRC"

# -fexceptions, not -fwasm-exceptions: GR4 compiles and links its WASM targets with JS-based
# exceptions (see the Boost.UT flag scrubbing in the top-level CMakeLists.txt), and mixing the two
# leaves __cpp_exception / __wasm_lpad_context / _Unwind_CallPersonality undefined at link time.
python3 tools/ci_build/build.py \
  --build_dir "$WASM_BUILD" --config Release \
  --cmake_extra_defines CMAKE_POLICY_VERSION_MINIMUM=3.5 \
  --cmake_extra_defines FETCHCONTENT_QUIET=OFF \
  --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
  --cmake_extra_defines onnxruntime_ENABLE_LTO=OFF \
  --cmake_extra_defines "CMAKE_CXX_FLAGS=-fexceptions" \
  --cmake_extra_defines "CMAKE_C_FLAGS=-fexceptions" \
  --update --build --minimal_build extended \
  --disable_ml_ops --disable_contrib_ops \
  --compile_no_warning_as_error --skip_tests --parallel "$JOBS" --allow_running_as_root \
  --skip_submodule_sync \
  --build_wasm_static_lib --enable_wasm_simd --enable_wasm_threads \
  --disable_wasm_exception_catching --enable_wasm_api_exception_catching

WB="$WASM_BUILD/Release"
if [ -f "$WB/libonnxruntime_webassembly.a" ]; then
  cp "$WB/libonnxruntime_webassembly.a" "$ORT_STATIC_LIB_DIR/wasm/lib/"
elif [ -f "$WB/onnxruntime_webassembly.a" ]; then
  cp "$WB/onnxruntime_webassembly.a" "$ORT_STATIC_LIB_DIR/wasm/lib/libonnxruntime_webassembly.a"
fi
cp -r "$SRC"/include/onnxruntime/core/session/* "$ORT_STATIC_LIB_DIR/wasm/include/"

# the archive must hold wasm objects; bitcode members mean the -flto removal silently stopped working
WASM_LIB="$ORT_STATIC_LIB_DIR/wasm/lib/libonnxruntime_webassembly.a"
FIRST_MEMBER=$(ar t "$WASM_LIB" | head -1)
if [ "$(ar p "$WASM_LIB" "$FIRST_MEMBER" | head -c 4 | od -An -tx1 | tr -d ' \n')" = "4243c0de" ]; then
  echo "ERROR: $WASM_LIB contains LLVM bitcode, not wasm objects -- wasm-ld will crash on it" >&2
  exit 1
fi

[ -n "${ORT_KEEP_BUILD:-}" ] || rm -rf "$SRC" "$NATIVE_BUILD" "$WASM_BUILD" "$ORT_EMSDK"
