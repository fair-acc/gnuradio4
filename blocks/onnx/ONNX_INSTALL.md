# ONNX Runtime — build and installation

## Build modes

| Mode  | Format support   | Description                                                                         |
|-------|------------------|-------------------------------------------------------------------------------------|
| `off` | —                | disable ONNX integration entirely                                                   |
| `opt` | `.onnx` + `.ort` | use system ORT package; full format support, recommended for development            |
| `on`  | `.ort` only      | build ORT v1.21.0 from source (minimal build); for cross-compile, WASM, AdaptiveCpp |

```bash
cmake -DENABLE_ONNX_INTEGRATION=opt ..   # recommended
cmake -DENABLE_ONNX_INTEGRATION=on  ..
cmake -DENABLE_ONNX_INTEGRATION=off ..
```

## `opt` — system packages

Uses a pre-installed ONNX Runtime shared library. Resolution order: system `.so`
→ bundled static → skip gracefully.

```bash
# vcpkg
vcpkg install onnxruntime

# Ubuntu / Debian
apt install libonnxruntime-dev

# openSUSE Tumbleweed
zypper install onnxruntime-devel

# manual: download from https://github.com/microsoft/onnxruntime/releases
# and set CMAKE_PREFIX_PATH to the extracted directory
```

## `on` — build from source

CMake builds ORT v1.21.0 via `ExternalProject`. First build takes 15–30 minutes
(cached afterwards). Produces a minimal static library supporting `.ort` format only.

Optional GPU execution providers:

```bash
# CUDA
cmake -DENABLE_ONNX_INTEGRATION=on -DONNX_ENABLE_CUDA=ON -DCUDA_HOME=/usr/local/cuda ..

# ROCm
cmake -DENABLE_ONNX_INTEGRATION=on -DONNX_ENABLE_ROCM=ON ..

# WebGPU (Emscripten)
cmake -DENABLE_ONNX_INTEGRATION=on -DONNX_ENABLE_WEBGPU=ON ..
```

## Converting `.onnx` to `.ort`

Minimal builds (`on` mode) and WASM targets require `.ort` format:

```bash
pip install onnxruntime
python -m onnxruntime.tools.convert_onnx_models_to_ort model.onnx
```

Output: `model.ort` in the same directory.

## Runtime format detection

```cpp
#include <gnuradio-4.0/onnx/OnnxHelper.hpp>

if constexpr (gr::onnx::isMinimalBuild()) {
    // only .ort files supported
}

// validateModelPath returns std::expected — no exceptions
auto result = gr::onnx::validateModelPath("model.onnx");
if (!result) {
    // result.error().message describes the issue
}
```

In practice, `OnnxSession::load()` calls `validateModelPath` internally — use
that directly rather than validating separately.

## Troubleshooting

**ORT source build fails with out-of-memory:**
Limit parallelism: `cmake --build build -j4` (not `-j` unbounded).

**"unsupported model format '.onnx'" at runtime:**
The build is minimal (`.ort` only). Convert with:
`python -m onnxruntime.tools.convert_onnx_models_to_ort model.onnx`

**Missing Python dependencies for ORT conversion:**
`pip install numpy packaging wheel onnxruntime`
