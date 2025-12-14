#!/bin/bash
#
# install_onnxruntime_rocm.sh
#
# Builds and installs ONNX Runtime with ROCm (AMD GPU) support system-wide.
# Use this for ENABLE_ONNX_INTEGRATION=opt with GPU acceleration.
#
# Prerequisites:
#   - ROCm 6.0+ installed (see below)
#   - Python 3.8+
#   - CMake 3.26+
#   - 32+ GB RAM recommended (build is memory-intensive)
#   - ~50 GB disk space for build
#
set -e

ONNXRUNTIME_VERSION="${ONNXRUNTIME_VERSION:-1.21.1}"
ROCM_HOME="${ROCM_HOME:-/opt/rocm}"
INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local}"
BUILD_DIR="${BUILD_DIR:-/tmp/onnxruntime-rocm-build}"
JOBS="${JOBS:-$(nproc)}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}==>${NC} $*"; }
log_warn()  { echo -e "${YELLOW}Warning:${NC} $*"; }
log_error() { echo -e "${RED}Error:${NC} $*" >&2; }

while [[ $# -gt 0 ]]; do
  case $1 in
    --version)      ONNXRUNTIME_VERSION="$2"; shift 2 ;;
    --rocm-home)    ROCM_HOME="$2"; shift 2 ;;
    --prefix)       INSTALL_PREFIX="$2"; shift 2 ;;
    --jobs|-j)      JOBS="$2"; shift 2 ;;
    --help|-h)
      cat << 'EOF'
Usage: ./install_onnxruntime_rocm.sh [OPTIONS]

Builds and installs ONNX Runtime with ROCm (AMD GPU) support.

Options:
  --version VER    ONNX Runtime version (default: 1.20.1)
  --rocm-home DIR  ROCm installation path (default: /opt/rocm)
  --prefix DIR     Installation prefix (default: /usr/local)
  --jobs N         Parallel jobs (default: nproc)

Prerequisites - Install ROCm 6.x first:
  wget https://repo.radeon.com/amdgpu-install/6.3.1/ubuntu/noble/amdgpu-install_6.3.60301-1_all.deb
  sudo apt install ./amdgpu-install_6.3.60301-1_all.deb
  sudo amdgpu-install --usecase=rocm
  sudo usermod -aG render,video $USER
  # logout and login, then verify with: rocminfo
EOF
      exit 0
      ;;
    *) log_error "Unknown option: $1"; exit 1 ;;
  esac
done

echo "=============================================="
echo "ONNX Runtime ROCm Installation"
echo "=============================================="
echo "Version:      $ONNXRUNTIME_VERSION"
echo "ROCm:         $ROCM_HOME"
echo "Install to:   $INSTALL_PREFIX"
echo "Parallel:     $JOBS jobs"
echo "=============================================="

# Verify ROCm
if [ ! -d "$ROCM_HOME" ]; then
  log_error "ROCm not found at $ROCM_HOME"
  echo "Install ROCm first - see --help for instructions"
  exit 1
fi

export PATH="$ROCM_HOME/bin:$PATH"
if ! command -v hipcc &> /dev/null; then
  log_error "hipcc not found"
  exit 1
fi

log_info "ROCm: $(cat "$ROCM_HOME/.info/version" 2>/dev/null || echo "found")"

# Clone
mkdir -p "$BUILD_DIR"
SRC_DIR="$BUILD_DIR/onnxruntime"
if [ ! -d "$SRC_DIR/.git" ]; then
  log_info "Cloning ONNX Runtime v$ONNXRUNTIME_VERSION..."
  git clone --depth 1 --branch "v$ONNXRUNTIME_VERSION" \
    https://github.com/microsoft/onnxruntime.git "$SRC_DIR"
fi
cd "$SRC_DIR"

# Patch Eigen hash
if [ -f cmake/deps.txt ]; then
  sed -i 's/be8be39fdbc6e60e94fa7870b280707069b5b81a/32b145f525a8308d7ab1c09388b2e288312d8eba/g' cmake/deps.txt
fi

# Build
log_info "Building (this takes 30-90 minutes)..."
python3 tools/ci_build/build.py \
  --build_dir "$BUILD_DIR/build" \
  --config Release \
  --cmake_extra_defines CMAKE_POLICY_VERSION_MINIMUM=3.5 \
  --build_shared_lib \
  --use_rocm \
  --rocm_home "$ROCM_HOME" \
  --parallel "$JOBS" \
  --skip_tests \
  --update \
  --build

# Install
log_info "Installing to $INSTALL_PREFIX..."
sudo cmake --install "$BUILD_DIR/build/Release" --prefix "$INSTALL_PREFIX"

# pkg-config
sudo mkdir -p "$INSTALL_PREFIX/lib/pkgconfig"
sudo tee "$INSTALL_PREFIX/lib/pkgconfig/libonnxruntime.pc" > /dev/null << EOF
prefix=$INSTALL_PREFIX
libdir=\${prefix}/lib
includedir=\${prefix}/include
Name: ONNX Runtime
Description: ONNX Runtime with ROCm
Version: $ONNXRUNTIME_VERSION
Libs: -L\${libdir} -lonnxruntime
Cflags: -I\${includedir}
EOF

sudo ldconfig

echo ""
log_info "Installation complete!"
echo "Library: $INSTALL_PREFIX/lib/libonnxruntime.so"
echo ""
echo "Use with GR4: cmake -DENABLE_ONNX_INTEGRATION=opt .."
