# ONNX Runtime GR4 Integration

## Quick Start

### Option 1: Use bundled static libraries (`on` mode)
```bash
# Clone with LFS (fetches ~30 MB of static libs on demand)
git clone https://github.com/fair-acc/gnuradio4.git

# Build - LFS files are auto-fetched when needed
cmake -DENABLE_ONNX_INTEGRATION=on ..
make
```

### Option 2: Use system library with GPU (`opt` mode)
```bash
# For AMD GPU - build and install ONNX Runtime with ROCm
./install_onnxruntime_rocm.sh

# For NVIDIA GPU - build and install with CUDA
./install_onnxruntime_cuda.sh

# Then build GR4
cmake -DENABLE_ONNX_INTEGRATION=opt ..
make
```

### Option 3: No ONNX (`off` mode)
```bash
# Fast clone without LFS objects
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/fair-acc/gnuradio4.git

cmake -DENABLE_ONNX_INTEGRATION=off ..
make
```

## Files Provided

```
blocks-onnx/                    # Drop into blocks/onnx/
├── CMakeLists.txt              # Main CMake configuration
└── lib/
    ├── .gitattributes          # Git LFS tracking config
    ├── README.md               # Library documentation
    ├── rebuild_static_libs.sh  # Rebuild script for maintainers
    ├── native-linux-x86_64/    # Linux static lib (~20 MB)
    ├── wasm-webgpu/            # WASM with WebGPU (~15 MB)
    └── wasm-cpu/               # WASM CPU-only (~10 MB)

install_onnxruntime_rocm.sh     # System-wide ROCm installation
install_onnxruntime_cuda.sh     # System-wide CUDA installation
```

## Git LFS Setup (Repository Maintainers)

One-time setup for the repository:
```bash
git lfs install
git lfs track "blocks/onnx/lib/**/*.a"
git add blocks/onnx/lib/.gitattributes
git commit -m "Track ONNX static libs with Git LFS"
```

Build and add the libraries:
```bash
cd blocks/onnx/lib

# First, clean any old build artifacts
rm -rf /tmp/onnxruntime-rebuild

# Build all targets (uses v1.21.1 by default)
./rebuild_static_libs.sh --all

# Commit
git add -A .
git commit -m "Add ONNX Runtime v1.21.1 static libraries"
git push
```

## Mode Comparison

| Mode | LFS Download | GPU Support | Model Format | Use Case |
|------|--------------|-------------|--------------|----------|
| `off` | No | - | - | No ML features |
| `opt` | No | Yes (ROCm/CUDA) | .onnx + .ort | Development with GPU |
| `on` | Yes (~30 MB) | WebGPU only | .ort only | Distribution, CI |

## Version Notes

**v1.21.1** is used by default because v1.20.x has a CMake bug where `flatbuffers::flatbuffers`
target is not created properly in minimal builds with newer CMake versions.

If you need to use v1.20.1, you may need to install flatbuffers system-wide first:
```bash
# Debian/Ubuntu
sudo apt install libflatbuffers-dev flatbuffers-compiler
```
