#!/bin/bash
#
# install_onnxruntime_cuda.sh
#
# Builds and installs ONNX Runtime with CUDA (NVIDIA GPU) support system-wide.
# Use this for ENABLE_ONNX_INTEGRATION=opt with GPU acceleration.
#
# Prerequisites:
#   - CUDA Toolkit 11.8+ or 12.x installed
#   - cuDNN 8.x or 9.x matching your CUDA version
#   - Python 3.8+
#   - CMake 3.26+
#   - 32+ GB RAM recommended (build is memory-intensive)
#   - ~50 GB disk space for build
#
# Usage:
#   ./install_onnxruntime_cuda.sh
#   ./install_onnxruntime_cuda.sh --version 1.21.0
#   CUDA_HOME=/usr/local/cuda-12.4 ./install_onnxruntime_cuda.sh
#
set -e

ONNXRUNTIME_VERSION="${ONNXRUNTIME_VERSION:-1.21.1}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDNN_HOME="${CUDNN_HOME:-$CUDA_HOME}"
INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local}"
BUILD_DIR="${BUILD_DIR:-/tmp/onnxruntime-cuda-build}"
JOBS="${JOBS:-$(nproc)}"

# TensorRT is optional but recommended for inference optimization
USE_TENSORRT="${USE_TENSORRT:-OFF}"
TENSORRT_HOME="${TENSORRT_HOME:-/usr/local/tensorrt}"

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
    --cuda-home)    CUDA_HOME="$2"; shift 2 ;;
    --cudnn-home)   CUDNN_HOME="$2"; shift 2 ;;
    --prefix)       INSTALL_PREFIX="$2"; shift 2 ;;
    --jobs|-j)      JOBS="$2"; shift 2 ;;
    --with-tensorrt) USE_TENSORRT="ON"; shift ;;
    --tensorrt-home) TENSORRT_HOME="$2"; USE_TENSORRT="ON"; shift 2 ;;
    --help|-h)
      cat << 'EOF'
Usage: ./install_onnxruntime_cuda.sh [OPTIONS]

Builds and installs ONNX Runtime with CUDA (NVIDIA GPU) support.

Options:
  --version VER      ONNX Runtime version (default: 1.20.1)
  --cuda-home DIR    CUDA installation path (default: /usr/local/cuda)
  --cudnn-home DIR   cuDNN installation path (default: same as CUDA_HOME)
  --prefix DIR       Installation prefix (default: /usr/local)
  --jobs N           Parallel jobs (default: nproc)
  --with-tensorrt    Enable TensorRT execution provider
  --tensorrt-home    TensorRT installation path (implies --with-tensorrt)

Prerequisites:
  1. Install CUDA Toolkit:
     # Ubuntu - using NVIDIA's repository
     wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
     sudo dpkg -i cuda-keyring_1.1-1_all.deb
     sudo apt update
     sudo apt install cuda-toolkit-12-4

  2. Install cuDNN:
     # Download from https://developer.nvidia.com/cudnn (requires NVIDIA account)
     # Or via apt if using NVIDIA repo:
     sudo apt install libcudnn8 libcudnn8-dev

  3. Set up environment:
     export CUDA_HOME=/usr/local/cuda
     export PATH=$CUDA_HOME/bin:$PATH
     export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

  4. Verify:
     nvcc --version
     nvidia-smi

After installation:
  Use with GR4: cmake -DENABLE_ONNX_INTEGRATION=opt ..
EOF
      exit 0
      ;;
    *) log_error "Unknown option: $1"; exit 1 ;;
  esac
done

echo "=============================================="
echo "ONNX Runtime CUDA Installation"
echo "=============================================="
echo "Version:      $ONNXRUNTIME_VERSION"
echo "CUDA:         $CUDA_HOME"
echo "cuDNN:        $CUDNN_HOME"
echo "TensorRT:     $USE_TENSORRT"
echo "Install to:   $INSTALL_PREFIX"
echo "Parallel:     $JOBS jobs"
echo "=============================================="
echo ""

# =============================================================================
# Verify CUDA installation
# =============================================================================
if [ ! -d "$CUDA_HOME" ]; then
  log_error "CUDA not found at $CUDA_HOME"
  echo ""
  echo "Install CUDA Toolkit first. For Ubuntu 24.04:"
  echo ""
  echo "  # Add NVIDIA repository"
  echo "  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"
  echo "  sudo dpkg -i cuda-keyring_1.1-1_all.deb"
  echo "  sudo apt update"
  echo ""
  echo "  # Install CUDA (choose your version)"
  echo "  sudo apt install cuda-toolkit-12-4"
  echo ""
  echo "  # Set environment"
  echo "  export CUDA_HOME=/usr/local/cuda-12.4"
  echo "  export PATH=\$CUDA_HOME/bin:\$PATH"
  echo ""
  echo "For other distributions, see:"
  echo "  https://developer.nvidia.com/cuda-downloads"
  exit 1
fi

# Add CUDA to PATH if needed
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

if ! command -v nvcc &> /dev/null; then
  log_error "nvcc not found even after adding CUDA to PATH"
  exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
log_info "CUDA version: $CUDA_VERSION"
log_info "nvcc: $(which nvcc)"

# Verify cuDNN
CUDNN_HEADER="$CUDNN_HOME/include/cudnn.h"
CUDNN_HEADER_V8="$CUDNN_HOME/include/cudnn_version.h"
if [ -f "$CUDNN_HEADER_V8" ]; then
  CUDNN_MAJOR=$(grep "#define CUDNN_MAJOR" "$CUDNN_HEADER_V8" | awk '{print $3}')
  CUDNN_MINOR=$(grep "#define CUDNN_MINOR" "$CUDNN_HEADER_V8" | awk '{print $3}')
  log_info "cuDNN version: $CUDNN_MAJOR.$CUDNN_MINOR"
elif [ -f "$CUDNN_HEADER" ]; then
  log_info "cuDNN found at $CUDNN_HOME"
else
  log_error "cuDNN not found at $CUDNN_HOME"
  echo ""
  echo "Install cuDNN:"
  echo "  # Option 1: Via apt (if using NVIDIA repository)"
  echo "  sudo apt install libcudnn8 libcudnn8-dev"
  echo ""
  echo "  # Option 2: Download from NVIDIA (requires account)"
  echo "  https://developer.nvidia.com/cudnn"
  echo ""
  echo "  # After installing, cuDNN should be at:"
  echo "  # $CUDA_HOME/include/cudnn.h"
  echo "  # $CUDA_HOME/lib64/libcudnn.so"
  exit 1
fi

# Verify GPU is accessible
if ! nvidia-smi &>/dev/null; then
  log_warn "nvidia-smi failed - GPU may not be accessible"
  log_warn "Check that NVIDIA drivers are installed: nvidia-driver-XXX"
else
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
  log_info "GPU: $GPU_NAME ($GPU_MEMORY)"
fi

# Check TensorRT if enabled
if [ "$USE_TENSORRT" = "ON" ]; then
  if [ ! -d "$TENSORRT_HOME" ]; then
    log_warn "TensorRT not found at $TENSORRT_HOME"
    log_warn "Disabling TensorRT support"
    USE_TENSORRT="OFF"
  else
    TRT_VERSION=$(cat "$TENSORRT_HOME/include/NvInferVersion.h" 2>/dev/null | grep "#define NV_TENSORRT_MAJOR" | awk '{print $3}' || echo "unknown")
    log_info "TensorRT version: $TRT_VERSION.x"
  fi
fi

# Check available memory
AVAILABLE_MEM=$(free -g | awk '/^Mem:/{print $7}')
if [ "$AVAILABLE_MEM" -lt 24 ]; then
  log_warn "Only ${AVAILABLE_MEM}GB RAM available. Build may fail or be slow."
  log_warn "CUDA build typically needs 24-32 GB RAM. Consider reducing --jobs."
fi

# =============================================================================
# Clone source
# =============================================================================
mkdir -p "$BUILD_DIR"
SRC_DIR="$BUILD_DIR/onnxruntime"

if [ ! -d "$SRC_DIR/.git" ]; then
  log_info "Cloning ONNX Runtime v$ONNXRUNTIME_VERSION..."
  git clone --depth 1 --branch "v$ONNXRUNTIME_VERSION" \
    https://github.com/microsoft/onnxruntime.git "$SRC_DIR"
else
  log_info "Using existing source at $SRC_DIR"
  cd "$SRC_DIR"
  CURRENT_TAG=$(git describe --tags --exact-match 2>/dev/null || echo "unknown")
  if [ "$CURRENT_TAG" != "v$ONNXRUNTIME_VERSION" ]; then
    log_warn "Source is $CURRENT_TAG, expected v$ONNXRUNTIME_VERSION"
  fi
fi

cd "$SRC_DIR"

# =============================================================================
# Apply patches
# =============================================================================
log_info "Applying patches..."

# Eigen hash fix
if [ -f cmake/deps.txt ]; then
  if grep -q "be8be39fdbc6e60e94fa7870b280707069b5b81a" cmake/deps.txt; then
    sed -i 's/be8be39fdbc6e60e94fa7870b280707069b5b81a/32b145f525a8308d7ab1c09388b2e288312d8eba/g' cmake/deps.txt
    echo "  - Patched Eigen hash"
  fi
fi

# =============================================================================
# Build
# =============================================================================
log_info "Building ONNX Runtime with CUDA..."
log_info "This will take 30-90 minutes depending on your system."
echo ""

BUILD_OUT="$BUILD_DIR/build"

# Build arguments
BUILD_ARGS=(
  --build_dir "$BUILD_OUT"
  --config Release
  --cmake_extra_defines CMAKE_POLICY_VERSION_MINIMUM=3.5
  --build_shared_lib
  --use_cuda
  --cuda_home "$CUDA_HOME"
  --cudnn_home "$CUDNN_HOME"
  --parallel "$JOBS"
  --skip_tests
  --update
  --build
)

# Add TensorRT if enabled
if [ "$USE_TENSORRT" = "ON" ]; then
  BUILD_ARGS+=(--use_tensorrt --tensorrt_home "$TENSORRT_HOME")
fi

python3 tools/ci_build/build.py "${BUILD_ARGS[@]}"

# =============================================================================
# Install
# =============================================================================
log_info "Installing to $INSTALL_PREFIX (may require sudo)..."

if [ -w "$INSTALL_PREFIX" ]; then
  cmake --install "$BUILD_OUT/Release" --prefix "$INSTALL_PREFIX"
else
  sudo cmake --install "$BUILD_OUT/Release" --prefix "$INSTALL_PREFIX"
fi

# Create pkg-config file
PKG_CONFIG_DIR="$INSTALL_PREFIX/lib/pkgconfig"

PKG_CONFIG_CONTENT="prefix=$INSTALL_PREFIX
exec_prefix=\${prefix}
libdir=\${prefix}/lib
includedir=\${prefix}/include

Name: ONNX Runtime
Description: ONNX Runtime with CUDA GPU support
Version: $ONNXRUNTIME_VERSION
Libs: -L\${libdir} -lonnxruntime
Cflags: -I\${includedir}
"

if [ -w "$PKG_CONFIG_DIR" ] || [ -w "$(dirname "$PKG_CONFIG_DIR")" ]; then
  mkdir -p "$PKG_CONFIG_DIR"
  echo "$PKG_CONFIG_CONTENT" > "$PKG_CONFIG_DIR/libonnxruntime.pc"
else
  sudo mkdir -p "$PKG_CONFIG_DIR"
  echo "$PKG_CONFIG_CONTENT" | sudo tee "$PKG_CONFIG_DIR/libonnxruntime.pc" > /dev/null
fi

# Update library cache
if command -v ldconfig &> /dev/null; then
  sudo ldconfig 2>/dev/null || true
fi

# =============================================================================
# Verification
# =============================================================================
echo ""
echo "=============================================="
echo "Installation complete!"
echo "=============================================="
echo ""
echo "Library:     $INSTALL_PREFIX/lib/libonnxruntime.so"
echo "Headers:     $INSTALL_PREFIX/include/"
echo "pkg-config:  $PKG_CONFIG_DIR/libonnxruntime.pc"
echo ""

# Verify pkg-config
if pkg-config --exists libonnxruntime 2>/dev/null; then
  log_info "pkg-config verification: OK"
  echo "  $(pkg-config --libs libonnxruntime)"
else
  log_warn "pkg-config cannot find libonnxruntime"
  echo "  You may need to add $PKG_CONFIG_DIR to PKG_CONFIG_PATH"
fi

# Test with Python if available
if python3 -c "import onnxruntime" 2>/dev/null; then
  PROVIDERS=$(python3 -c "import onnxruntime; print(onnxruntime.get_available_providers())")
  log_info "Python onnxruntime providers: $PROVIDERS"
  if echo "$PROVIDERS" | grep -q "CUDAExecutionProvider"; then
    log_info "CUDA execution provider: AVAILABLE"
  fi
  if echo "$PROVIDERS" | grep -q "TensorrtExecutionProvider"; then
    log_info "TensorRT execution provider: AVAILABLE"
  fi
else
  log_info "Python onnxruntime not installed (optional)"
fi

echo ""
echo "Usage with GNU Radio 4:"
echo "  cmake -DENABLE_ONNX_INTEGRATION=opt .."
echo ""
echo "The CUDAExecutionProvider will be used automatically when available."
if [ "$USE_TENSORRT" = "ON" ]; then
  echo "TensorRT will be used for optimized inference when possible."
fi

echo ""
echo "Runtime environment (add to ~/.bashrc):"
echo "  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
