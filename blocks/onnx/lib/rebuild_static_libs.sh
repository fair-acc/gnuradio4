#!/bin/bash
#
# rebuild_static_libs.sh
#
# Rebuilds minimal ONNX Runtime static libraries for GNU Radio 4.
# These libraries are stored in Git LFS and bundled with the repository.
#
# Usage:
#   ./rebuild_static_libs.sh [--native] [--wasm] [--wasm-cpu] [--all]
#
# Output structure:
#   native-linux-x86_64/
#   ├── libonnxruntime.a
#   └── include/
#   wasm-webgpu/
#   ├── libonnxruntime_webassembly.a
#   └── include/
#   wasm-cpu/
#   ├── libonnxruntime_webassembly.a
#   └── include/
#
# After rebuilding, commit the updated libraries:
#   git add -A lib/
#   git commit -m "Update ONNX Runtime static libs to vX.Y.Z"
#   git push
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONNXRUNTIME_VERSION="${ONNXRUNTIME_VERSION:-1.23.2}"
BUILD_BASE="${BUILD_BASE:-/tmp/onnxruntime-rebuild}"
JOBS="${JOBS:-$(nproc)}"

# Build targets
BUILD_NATIVE=false
BUILD_WASM_WEBGPU=false
BUILD_WASM_CPU=false
FORCE_REBUILD=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info()  { echo -e "${GREEN}==>${NC} $*"; }
log_warn()  { echo -e "${YELLOW}Warning:${NC} $*"; }
log_error() { echo -e "${RED}Error:${NC} $*" >&2; }

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --native)      BUILD_NATIVE=true; shift ;;
    --wasm)        BUILD_WASM_WEBGPU=true; shift ;;
    --wasm-webgpu) BUILD_WASM_WEBGPU=true; shift ;;
    --wasm-cpu)    BUILD_WASM_CPU=true; shift ;;
    --all)         BUILD_NATIVE=true; BUILD_WASM_WEBGPU=true; BUILD_WASM_CPU=true; shift ;;
    --force|-f)    FORCE_REBUILD=true; shift ;;
    --version)     ONNXRUNTIME_VERSION="$2"; shift 2 ;;
    --jobs|-j)     JOBS="$2"; shift 2 ;;
    --help|-h)
      cat << EOF
Usage: $0 [OPTIONS]

Rebuilds minimal ONNX Runtime static libraries for GR4 bundling.

Prerequisites:
  - python3, cmake
  - emscripten (for WASM builds only)

Targets:
  --native       Build native Linux x86_64 static library (CPU only)
  --wasm-webgpu  Build WASM static library with WebGPU (EXPERIMENTAL)
  --wasm-cpu     Build WASM static library (CPU only, recommended)
  --all          Build all targets

Options:
  --version VER  ONNX Runtime version (default: $ONNXRUNTIME_VERSION)
  --jobs N       Parallel build jobs (default: $JOBS)
  --force, -f    Force rebuild even if build artifacts exist
  --help         Show this help

Environment:
  BUILD_BASE     Build directory (default: /tmp/onnxruntime-rebuild)
  EMSDK          Emscripten SDK path (for WASM builds)

Build notes:
  - Native and wasm-cpu builds are the most reliable
  - wasm-webgpu has known Dawn/Emscripten linking issues
  - Contrib ops (tokenizer, regex) are disabled to avoid RE2 dependency
  - RTTI is enabled (required for CPU kernel template instantiation)

Examples:
  $0 --native              # Rebuild native library only
  $0 --wasm-cpu            # Rebuild WASM CPU-only
  $0 --native --force      # Force full rebuild
EOF
      exit 0
      ;;
    *)
      log_error "Unknown option: $1"
      exit 1
      ;;
  esac
done

if ! $BUILD_NATIVE && ! $BUILD_WASM_WEBGPU && ! $BUILD_WASM_CPU; then
  log_error "Specify at least one target: --native, --wasm-cpu, --wasm-webgpu, or --all"
  exit 1
fi

# Verify Python is available
if ! command -v python3 &> /dev/null; then
  log_error "python3 is required but not found"
  exit 1
fi

# Verify Emscripten for WASM builds
if $BUILD_WASM_WEBGPU || $BUILD_WASM_CPU; then
  if ! command -v emcc &> /dev/null; then
    log_error "Emscripten not found in PATH"
    echo ""
    echo "Install Emscripten:"
    echo "  git clone https://github.com/emscripten-core/emsdk.git"
    echo "  cd emsdk && ./emsdk install latest && ./emsdk activate latest"
    echo "  source ./emsdk_env.sh"
    exit 1
  fi
  log_info "Emscripten: $(emcc --version | head -n1)"
fi

# =============================================================================
# Clone/update ONNX Runtime source
# =============================================================================
SRC_DIR="$BUILD_BASE/onnxruntime-$ONNXRUNTIME_VERSION"

clone_source() {
  if [ -d "$SRC_DIR/.git" ]; then
    log_info "Source exists at $SRC_DIR"
    cd "$SRC_DIR"
    CURRENT_TAG=$(git describe --tags --exact-match 2>/dev/null || echo "unknown")
    if [ "$CURRENT_TAG" != "v$ONNXRUNTIME_VERSION" ]; then
      log_warn "Source is $CURRENT_TAG, expected v$ONNXRUNTIME_VERSION"
      read -p "Re-clone? [y/N] " -n 1 -r
      echo
      if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd ..
        rm -rf "$SRC_DIR"
        clone_source
        return
      fi
    fi
  else
    log_info "Cloning ONNX Runtime v$ONNXRUNTIME_VERSION..."
    mkdir -p "$(dirname "$SRC_DIR")"
    git clone --depth 1 --branch "v$ONNXRUNTIME_VERSION" \
      https://github.com/microsoft/onnxruntime.git "$SRC_DIR"
  fi
}

clone_source

# =============================================================================
# Apply patches (Eigen hash fixes for GitLab archive regeneration)
# =============================================================================
log_info "Checking for required patches..."
cd "$SRC_DIR"

if [ -f cmake/deps.txt ]; then
  # v1.20.x hash fix
  if grep -q "be8be39fdbc6e60e94fa7870b280707069b5b81a" cmake/deps.txt; then
    sed -i 's/be8be39fdbc6e60e94fa7870b280707069b5b81a/32b145f525a8308d7ab1c09388b2e288312d8eba/g' cmake/deps.txt
    log_info "  Patched Eigen hash (v1.20.x)"
  fi
  # v1.21.x hash fix
  if grep -q "5ea4d05e62d7f954a46b3213f9b2535bdd866803" cmake/deps.txt; then
    sed -i 's/5ea4d05e62d7f954a46b3213f9b2535bdd866803/51982be81bbe52572b54180454df11a3ece9a934/g' cmake/deps.txt
    log_info "  Patched Eigen hash (v1.21.x)"
  fi
fi

# =============================================================================
# Helper: Combine multiple static libraries using MRI script
# Avoids name collisions when archives contain duplicate object base names
# =============================================================================
combine_archives() {
  local output="$1"
  shift
  local libs=("$@")

  local mri_script=$(mktemp)
  echo "CREATE $output" > "$mri_script"
  for lib in "${libs[@]}"; do
    if [ -f "$lib" ]; then
      echo "ADDLIB $lib" >> "$mri_script"
      local obj_count=$(ar t "$lib" 2>/dev/null | wc -l)
      log_info "    Adding: $(basename "$lib") ($obj_count objects)"
    fi
  done
  echo "SAVE" >> "$mri_script"
  echo "END" >> "$mri_script"

  ar -M < "$mri_script"
  rm -f "$mri_script"
}

# =============================================================================
# Native Linux x86_64 build
# =============================================================================
build_native() {
  log_info "Building native-linux-x86_64 (CPU only)..."

  local BUILD_DIR="$BUILD_BASE/build-native"
  local OUT_DIR="$SCRIPT_DIR/native-linux-x86_64"
  local NEED_BUILD=false
  local NEED_COMBINE=false

  # Force rebuild if requested
  if $FORCE_REBUILD; then
    log_warn "Force rebuild requested - cleaning build directory..."
    rm -rf "$BUILD_DIR"
    NEED_BUILD=true
    NEED_COMBINE=true
  else
    # Check if we need to build
    if [ ! -f "$BUILD_DIR/Release/libonnxruntime_session.a" ]; then
      log_info "  Build artifacts not found - will build"
      NEED_BUILD=true
    fi

    # Check if we need to combine
    if [ ! -f "$OUT_DIR/libonnxruntime.a" ]; then
      log_info "  Output library not found - will combine"
      NEED_COMBINE=true
    elif [ -f "$BUILD_DIR/Release/libonnxruntime_session.a" ] && \
         [ "$BUILD_DIR/Release/libonnxruntime_session.a" -nt "$OUT_DIR/libonnxruntime.a" ]; then
      log_info "  Build is newer than output - will recombine"
      NEED_COMBINE=true
    fi

    # Clean previous failed build
    if [ -d "$BUILD_DIR" ] && [ ! -f "$BUILD_DIR/Release/libonnxruntime_session.a" ]; then
      log_warn "Cleaning previous failed build..."
      rm -rf "$BUILD_DIR"
      NEED_BUILD=true
    fi
  fi

  # Run build if needed
  if $NEED_BUILD; then
    cd "$SRC_DIR"

    # Build configuration:
    # - Full (non-minimal) build for reliability
    # - Disable system protobuf/flatbuffers to avoid conflicts
    # - Disable contrib_ops to avoid RE2 dependency (tokenizer/regex ops)
    # - RTTI enabled (required for BuildKernelCreateInfo<> template instantiation)
    log_info "Running ONNX Runtime build (this may take 20-40 minutes)..."
    python3 tools/ci_build/build.py \
      --build_dir "$BUILD_DIR" \
      --config Release \
      --cmake_extra_defines CMAKE_POLICY_VERSION_MINIMUM=3.5 \
      --cmake_extra_defines FETCHCONTENT_QUIET=OFF \
      --cmake_extra_defines CMAKE_DISABLE_FIND_PACKAGE_Protobuf=TRUE \
      --cmake_extra_defines CMAKE_DISABLE_FIND_PACKAGE_flatbuffers=TRUE \
      --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
      --update \
      --build \
      --disable_ml_ops \
      --disable_contrib_ops \
      --skip_tests \
      --parallel "$JOBS"

    NEED_COMBINE=true
  else
    log_info "  Build artifacts exist, skipping build step"
  fi

  # Check build succeeded
  if [ ! -f "$BUILD_DIR/Release/libonnxruntime_session.a" ]; then
    log_error "Build failed - libonnxruntime_session.a not found"
    return 1
  fi

  # Skip combining if not needed
  if ! $NEED_COMBINE; then
    log_info "  Output library is up-to-date, skipping combine step"
    local SIZE=$(du -h "$OUT_DIR/libonnxruntime.a" | cut -f1)
    log_info "native-linux-x86_64: $SIZE (existing)"
    return 0
  fi

  # Combine all static libraries into one
  log_info "Combining static libraries..."
  mkdir -p "$OUT_DIR/include"

  local ALL_LIBS=()

  # Core ONNX Runtime libraries
  log_info "  Collecting core ONNX Runtime libraries..."
  for lib in libonnxruntime_common.a libonnxruntime_flatbuffers.a libonnxruntime_framework.a \
             libonnxruntime_graph.a libonnxruntime_lora.a libonnxruntime_mlas.a \
             libonnxruntime_optimizer.a libonnxruntime_providers.a libonnxruntime_session.a \
             libonnxruntime_util.a; do
    [ -f "$BUILD_DIR/Release/$lib" ] && ALL_LIBS+=("$BUILD_DIR/Release/$lib")
  done

  # ONNX libraries
  log_info "  Collecting ONNX libraries..."
  local ONNX_DIR="$BUILD_DIR/Release/_deps/onnx-build"
  for lib in libonnx.a libonnx_proto.a; do
    if [ -f "$ONNX_DIR/$lib" ]; then
      ALL_LIBS+=("$ONNX_DIR/$lib")
    else
      log_error "    MISSING: $lib - this will cause linker errors!"
    fi
  done

  # Protobuf
  log_info "  Collecting Protobuf..."
  local PROTOBUF_DIR="$BUILD_DIR/Release/_deps/protobuf-build"
  if [ -f "$PROTOBUF_DIR/libprotobuf.a" ]; then
    ALL_LIBS+=("$PROTOBUF_DIR/libprotobuf.a")
  elif [ -f "$PROTOBUF_DIR/libprotobuf-lite.a" ]; then
    log_warn "    Using protobuf-lite (full protobuf not found)"
    ALL_LIBS+=("$PROTOBUF_DIR/libprotobuf-lite.a")
  fi

  # Flatbuffers
  log_info "  Collecting Flatbuffers..."
  local FB_DIR="$BUILD_DIR/Release/_deps/flatbuffers-build"
  [ -f "$FB_DIR/libflatbuffers.a" ] && ALL_LIBS+=("$FB_DIR/libflatbuffers.a")

  # Abseil libraries
  log_info "  Collecting Abseil libraries..."
  local ABSEIL_COUNT=0
  while IFS= read -r lib; do
    ALL_LIBS+=("$lib")
    ABSEIL_COUNT=$((ABSEIL_COUNT + 1))
  done < <(find "$BUILD_DIR/Release/_deps/abseil_cpp-build" -name "libabsl_*.a" -type f 2>/dev/null | sort)
  log_info "    Found $ABSEIL_COUNT Abseil libraries"

  # cpuinfo
  log_info "  Collecting cpuinfo..."
  local CPUINFO_DIR="$BUILD_DIR/Release/_deps/pytorch_cpuinfo-build"
  [ -f "$CPUINFO_DIR/libcpuinfo.a" ] && ALL_LIBS+=("$CPUINFO_DIR/libcpuinfo.a")

  # Create combined archive
  log_info "  Creating combined libonnxruntime.a (${#ALL_LIBS[@]} archives)..."
  combine_archives "$OUT_DIR/libonnxruntime.a" "${ALL_LIBS[@]}"

  # Copy headers
  cp -r "$SRC_DIR/include/onnxruntime/core/session/"* "$OUT_DIR/include/"

  # Strip debug symbols
  strip --strip-debug "$OUT_DIR/libonnxruntime.a" 2>/dev/null || true

  # Verify
  local OBJ_COUNT=$(ar t "$OUT_DIR/libonnxruntime.a" 2>/dev/null | wc -l)
  log_info "  Verifying symbols..."

  nm "$OUT_DIR/libonnxruntime.a" 2>/dev/null | grep -q "GetOpSchema" \
    && log_info "    ✓ ONNX operator schemas present" \
    || log_error "    ✗ ONNX operator schemas MISSING!"

  nm "$OUT_DIR/libonnxruntime.a" 2>/dev/null | grep -q "OrtGetApiBase" \
    && log_info "    ✓ ONNX Runtime C API present" \
    || log_error "    ✗ ONNX Runtime C API MISSING!"

  local DEFINED_KERNELS=$(nm "$OUT_DIR/libonnxruntime.a" 2>/dev/null | grep -c " T.*BuildKernelCreateInfo" || echo 0)
  local UNDEFINED_KERNELS=$(nm "$OUT_DIR/libonnxruntime.a" 2>/dev/null | grep -c " U.*BuildKernelCreateInfo" || echo 0)

  if [ "$DEFINED_KERNELS" -gt 0 ]; then
    log_info "    ✓ CPU kernel implementations ($DEFINED_KERNELS defined, $UNDEFINED_KERNELS undefined)"
  else
    log_error "    ✗ CPU kernel implementations MISSING!"
  fi

  local SIZE=$(du -h "$OUT_DIR/libonnxruntime.a" | cut -f1)
  log_info "native-linux-x86_64: $SIZE ($OBJ_COUNT objects)"
}

# =============================================================================
# WASM WebGPU build (EXPERIMENTAL)
# =============================================================================
build_wasm_webgpu() {
  log_warn "Building wasm-webgpu (EXPERIMENTAL - may have linking issues)..."
  log_warn "Consider using wasm-cpu for reliable WASM builds"

  local BUILD_DIR="$BUILD_BASE/build-wasm-webgpu"
  local OUT_DIR="$SCRIPT_DIR/wasm-webgpu"

  if [ -d "$BUILD_DIR" ] && [ ! -f "$BUILD_DIR/Release/libonnxruntime_webassembly.a" ]; then
    log_warn "Cleaning previous failed build..."
    rm -rf "$BUILD_DIR"
  fi

  cd "$SRC_DIR"

  python3 tools/ci_build/build.py \
    --build_dir "$BUILD_DIR" \
    --config Release \
    --cmake_extra_defines CMAKE_POLICY_VERSION_MINIMUM=3.5 \
    --cmake_extra_defines FETCHCONTENT_QUIET=OFF \
    --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
    --cmake_extra_defines onnxruntime_ENABLE_LTO=OFF \
    --build_wasm_static_lib \
    --enable_wasm_simd \
    --enable_wasm_threads \
    --use_webgpu \
    --minimal_build extended \
    --disable_ml_ops \
    --disable_contrib_ops \
    --skip_tests \
    --parallel "$JOBS" \
  || {
    if [ -f "$BUILD_DIR/Release/libonnxruntime_webassembly.a" ]; then
      log_warn "Build had errors but library was created"
    else
      log_error "Build failed - library not found"
      return 1
    fi
  }

  mkdir -p "$OUT_DIR/include"

  if [ -f "$BUILD_DIR/Release/libonnxruntime_webassembly.a" ]; then
    cp "$BUILD_DIR/Release/libonnxruntime_webassembly.a" "$OUT_DIR/"
  elif [ -f "$BUILD_DIR/Release/onnxruntime_webassembly.a" ]; then
    cp "$BUILD_DIR/Release/onnxruntime_webassembly.a" "$OUT_DIR/libonnxruntime_webassembly.a"
  else
    log_error "Library not found"
    return 1
  fi

  cp -r "$SRC_DIR/include/onnxruntime/core/session/"* "$OUT_DIR/include/"

  local SIZE=$(du -h "$OUT_DIR/libonnxruntime_webassembly.a" | cut -f1)
  log_info "wasm-webgpu: $SIZE"
}

# =============================================================================
# WASM CPU-only build (recommended)
# =============================================================================
build_wasm_cpu() {
  log_info "Building wasm-cpu (CPU only)..."

  local BUILD_DIR="$BUILD_BASE/build-wasm-cpu"
  local OUT_DIR="$SCRIPT_DIR/wasm-cpu"

  if [ -d "$BUILD_DIR" ] && [ ! -f "$BUILD_DIR/Release/libonnxruntime_webassembly.a" ]; then
    log_warn "Cleaning previous failed build..."
    rm -rf "$BUILD_DIR"
  fi

  cd "$SRC_DIR"

  python3 tools/ci_build/build.py \
    --build_dir "$BUILD_DIR" \
    --config Release \
    --cmake_extra_defines CMAKE_POLICY_VERSION_MINIMUM=3.5 \
    --cmake_extra_defines FETCHCONTENT_QUIET=OFF \
    --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
    --cmake_extra_defines onnxruntime_ENABLE_LTO=OFF \
    --build_wasm_static_lib \
    --enable_wasm_simd \
    --enable_wasm_threads \
    --minimal_build extended \
    --disable_ml_ops \
    --disable_contrib_ops \
    --skip_tests \
    --parallel "$JOBS" \
  || {
    if [ -f "$BUILD_DIR/Release/libonnxruntime_webassembly.a" ]; then
      log_warn "Build had errors but library was created"
    else
      log_error "Build failed - library not found"
      return 1
    fi
  }

  mkdir -p "$OUT_DIR/include"

  if [ -f "$BUILD_DIR/Release/libonnxruntime_webassembly.a" ]; then
    cp "$BUILD_DIR/Release/libonnxruntime_webassembly.a" "$OUT_DIR/"
  elif [ -f "$BUILD_DIR/Release/onnxruntime_webassembly.a" ]; then
    cp "$BUILD_DIR/Release/onnxruntime_webassembly.a" "$OUT_DIR/libonnxruntime_webassembly.a"
  else
    log_error "Library not found"
    return 1
  fi

  cp -r "$SRC_DIR/include/onnxruntime/core/session/"* "$OUT_DIR/include/"

  local SIZE=$(du -h "$OUT_DIR/libonnxruntime_webassembly.a" | cut -f1)
  log_info "wasm-cpu: $SIZE"
}

# =============================================================================
# Execute builds
# =============================================================================
echo ""
echo "=============================================="
echo "ONNX Runtime Static Library Builder"
echo "=============================================="
echo "Version:    $ONNXRUNTIME_VERSION"
echo "Source:     $SRC_DIR"
echo "Output:     $SCRIPT_DIR"
echo "Parallel:   $JOBS jobs"
echo "Targets:    native=$BUILD_NATIVE wasm-webgpu=$BUILD_WASM_WEBGPU wasm-cpu=$BUILD_WASM_CPU"
echo "=============================================="
echo ""

$BUILD_NATIVE && build_native
$BUILD_WASM_WEBGPU && build_wasm_webgpu
$BUILD_WASM_CPU && build_wasm_cpu

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "Build complete!"
echo "=============================================="
echo ""

echo "Output libraries:"
for dir in native-linux-x86_64 wasm-webgpu wasm-cpu; do
  if [ -d "$SCRIPT_DIR/$dir" ]; then
    LIB=$(ls "$SCRIPT_DIR/$dir/"*.a 2>/dev/null | head -1)
    if [ -n "$LIB" ]; then
      LIB_SIZE=$(du -h "$LIB" | cut -f1)
      echo "  $dir: $LIB_SIZE"
    fi
  fi
done

echo ""
echo "Update version in CMakeLists.txt:"
echo "  set(GR_ONNX_BUNDLED_VERSION \"$ONNXRUNTIME_VERSION\")"
echo ""
echo "Commit and push:"
echo "  git add -A lib/"
echo "  git commit -m \"chore(onnx): update bundled libs to v$ONNXRUNTIME_VERSION\""
echo "  git push"
