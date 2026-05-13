# ONNX Runtime — build and installation

## Build modes

| Mode  | Format support   | Description                                                                         |
|-------|------------------|-------------------------------------------------------------------------------------|
| `off` | —                | disable ONNX integration entirely                                                   |
| `opt` | `.onnx` + `.ort` | use system ORT shared library; full format support, recommended for development     |
| `on`  | `.ort` only      | build ORT from source (minimal build); for cross-compile, WASM, AdaptiveCpp         |

The ORT version for `on` mode is the `ONNXRUNTIME_VERSION` cache variable
(default 1.21.0). If `opt` falls back to a bundled static library (see below),
the build is minimal, i.e. `.ort` only.

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

CMake builds ORT via `ExternalProject`. First build takes 15–30 minutes (cached
afterwards). Produces a minimal static library supporting `.ort` format only.
Parallelism is capped by the `GR_ONNX_BUILD_JOBS` cache variable (default 6).

The source build compiles the CPU execution provider only. GPU execution
providers (CUDA/TensorRT/ROCm) require a system ORT built with the desired
provider; select them per block via the `execution_provider` setting, which is
validated against the linked runtime at model load — an unavailable provider
fails the load with a descriptive error rather than silently falling back to CPU.

## Converting `.onnx` to `.ort`

Minimal builds (`on` mode) and WASM targets require `.ort` format:

```bash
pip install onnxruntime
python -m onnxruntime.tools.convert_onnx_models_to_ort model.onnx
```

Output: `model.ort` in the same directory.

## Runtime format detection

The build defines `GR_ONNX_MINIMAL_BUILD` as `1` (minimal, `.ort` only) or `0`
(full). Model paths are checked against it before loading:

```cpp
#include <gnuradio-4.0/onnx/OnnxHelper.hpp>

// returns std::expected — no exceptions
auto result = gr::blocks::onnx::validateModelPath("model.onnx");
if (!result) {
    // result.error().message describes the issue
}
```

In practice, `OnnxSession::load()` calls `validateModelPath` internally — use
that directly rather than validating separately.

## Troubleshooting

**Tests or examples fail to load with unresolved `VERS_…` symbols:**
Not a build break. When more than one ONNX Runtime is installed and
`ld.so.conf` puts a different one ahead of the library the build linked
(here: `/opt/onnxruntime/gpu/lib` 1.23 shadows the system 1.27 in
`/usr/lib64`), binaries link fine but fail to load. Point the loader at the
linked library explicitly:

```bash
LD_LIBRARY_PATH=/usr/lib64 ctest --test-dir build -R "qa_(Onnx|Peak)" --output-on-failure -j6
```

**A fresh build directory fails at configure**
("Failed to build gnuradio_4_0_parse_registrations", no Makefile generated —
IDEs then report "does not compile"): also not a build break. A fresh configure
needs

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER=g++-15 -DCMAKE_C_COMPILER=gcc-15 \
  -DGNURADIO_PARSE_REGISTRATIONS_TOOL_CXX_COMPLILER=/usr/bin/g++-15
```

The third flag is load-bearing, and `COMPLILER` is misspelled in the project
source (`blocklib_generator/CMakeLists.txt`) — type it exactly as shown.
Configure build directories one at a time; several back-to-back configures fail
transiently.

**ORT source build fails with out-of-memory:**
Limit parallelism: set `-DGR_ONNX_BUILD_JOBS=4` (the ORT external build ignores
the outer `cmake --build -j` value).

**"unsupported model format '.onnx'" at runtime:**
The build is minimal (`.ort` only). Convert with:
`python -m onnxruntime.tools.convert_onnx_models_to_ort model.onnx`

**Missing Python dependencies for ORT conversion:**
`pip install numpy packaging wheel onnxruntime`
