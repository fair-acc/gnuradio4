# ONNX Runtime Integration for GNU Radio 4

## Build Modes

| Mode | Format Support | Binary Size | Use Case                               |
|------|---------------|-------------|----------------------------------------|
| `off` | — | — | disable ONNX entirely                  |
| `on` | `.ort` only | ~3-5 MB | E<br/>embedded/WASM, minimal footprint |
| `opt` | `.onnx` + `.ort` | System | full features, development             |

## Configuration

```bash
# Use system packages (native only, full format support)
cmake -DENABLE_ONNX_INTEGRATION=opt ..

# Build minimal embedded library (native + WASM)
cmake -DENABLE_ONNX_INTEGRATION=on ..

# Disable ONNX support
cmake -DENABLE_ONNX_INTEGRATION=off ..
```

## Mode Details

### `opt` — System Packages (Native Only)

Uses pre-installed ONNX Runtime. Supports both `.onnx` and `.ort` formats.

**Install options:**
```bash
# vcpkg (recommended)
vcpkg install onnxruntime

# Ubuntu/Debian
apt install libonnxruntime-dev

# From GitHub releases
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz
tar xzf onnxruntime-linux-x64-1.20.1.tgz
export CMAKE_PREFIX_PATH="$(pwd)/onnxruntime-linux-x64-1.20.1:$CMAKE_PREFIX_PATH"
```

### `on` — Embedded Minimal Build (Native + WASM)

CMake automatically builds ONNX Runtime v1.20.1 via `ExternalProject`. First build takes 15-30 minutes (cached afterward).

**What gets built:**
- Minimal build with `extended` runtime optimisations
- ORT format only (no ONNX parser = smaller binary)
- RTTI disabled, ML ops disabled

**Optional GPU support:**
```bash
# Native with CUDA
cmake -DENABLE_ONNX_INTEGRATION=on -DONNX_ENABLE_CUDA=ON -DCUDA_HOME=/usr/local/cuda ..

# WASM with WebGPU
cmake -DENABLE_ONNX_INTEGRATION=on -DONNX_ENABLE_WEBGPU=ON ..
```

## Converting Models to ORT Format

Minimal builds (`on` mode) require `.ort` format:

```bash
pip install onnxruntime

# Basic conversion
python -m onnxruntime.tools.convert_onnx_models_to_ort model.onnx

# With optimization
python -m onnxruntime.tools.convert_onnx_models_to_ort \
    --optimization_level extended \
    model.onnx
```

Output: `model.ort` in same directory.

## Runtime Detection

```cpp
#include <gnuradio-4.0/onnx/OnnxHelper.hpp>

// Compile-time check
if constexpr (gr::onnx::isMinimalBuild()) {
    // Only .ort files supported
}

// Validate before loading
gr::onnx::validateModelPath("model.ort");  // OK in all modes
gr::onnx::validateModelPath("model.onnx"); // throws in minimal build
```

## Troubleshooting

**Build fails with memory error:**
```bash
export NODE_OPTIONS="--max-old-space-size=8192"
```

**Missing Python dependencies:**
```bash
pip install numpy packaging wheel
```

**"ORT format required" at runtime:**
Your model is `.onnx` but build is minimal. Convert to `.ort` format.