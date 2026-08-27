# ONNX Runtime — build and installation

## Git LFS

The models under `models/` are Git LFS objects. Install `git-lfs` and run `git lfs install` before
cloning, or repair an existing checkout with `git lfs install && git lfs pull`. A pointer file left
by a missing LFS setup starts with `version https://git-lfs.github.com/spec/v1` and makes every ONNX
test fail with `Failed to load model because protobuf parsing failed`.

## Build modes

| Mode  | Format support   | Description                                                                     |
| ----- | ---------------- | ------------------------------------------------------------------------------- |
| `off` | —                | disable ONNX integration entirely                                               |
| `opt` | `.onnx` + `.ort` | use system ORT shared library; full format support, recommended for development |
| `on`  | `.ort` only      | build ORT from source (minimal build); for cross-compile, WASM, AdaptiveCpp     |

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

Model format is decided by ONNX Runtime, not by the file name. `OnnxSession::load()` fetches the
bytes and passes them to the runtime, which reports through `std::expected` whatever it cannot
parse — a minimal build refuses `.onnx` protobuf there, with no compile-time flag and no
name-based pre-check involved. File names may therefore carry any extension, several
(`model.ort.gz`), or none.

## Model formats in this module

Deployment artefacts are `.ort.gz`: `fileio` inflates a `.gz` source on read, and `validateModelPath`
checks the format underneath the suffix, so nothing in the block gates a load on the file name. The
`.ort` form is a packaging step over the fp32 `.onnx` master (`src/pack_models.py`), not a separate model.

Note that ORT session construction scales roughly quadratically with graph node count, and `.ort`
does not shorten it — the cost is graph construction rather than optimisation. A model built by
unrolling repeated stages should express the repetition as an ONNX `Loop` instead; `Loop` is available
at opset 17, survives `.ort` conversion, and appears in the generated required-operators config.

## WebAssembly / Emscripten

The container image pre-builds an ORT WASM static library at `${ORT_STATIC_LIB_DIR}/wasm`, so an
`emcmake` build finds it and `gr-onnx` enables itself. To reproduce that build outside the image:

```
ORT_STATIC_LIB_DIR=$HOME/ort EMSDK_HOME=$HOME/git/emsdk ./docker/build_onnxruntime.sh
emcmake cmake -S . -B ../build-wasm -DORT_STATIC_LIB_DIR=$HOME/ort
```

`docker/build_onnxruntime.sh` is the same script the Dockerfile runs, so a change can be tried here
before it costs a full image rebuild in CI.

Three traps the script exists to avoid — each was a day of CI archaeology:

- **ORT hard-codes `-flto` for WASM.** `cmake/adjust_global_compile_flags.cmake` appends it
  unconditionally, bypassing `onnxruntime_ENABLE_LTO`, which makes the static library **LLVM bitcode
  rather than wasm objects**. `wasm-ld` must then run LTO codegen with the _consumer's_ LLVM and
  segfaults whenever that is a different major version. Passing `-fno-lto` does not help: ORT appends
  its flag last. The script patches the source and then asserts the archive members are `\0asm`.
- **Exception model.** GR4 compiles and links WASM with `-fexceptions` (JS-based); building ORT with
  `-fwasm-exceptions` leaves `__cpp_exception`, `__wasm_lpad_context` and `_Unwind_CallPersonality`
  undefined at link time. The top-level `CMakeLists.txt` scrubs the same flag out of Boost.UT.
- **ORT rewrites the emsdk it is given.** `build.py` installs and activates its own pinned SDK in
  whatever tree `cmake/external/emsdk` points at, so pointing it at the shared `$EMSDK_HOME`
  uninstalls the pinned version and every later `emsdk activate` reports _"tool is not installed"_.
  The script hands it a throw-away copy. Note also that sourcing `emsdk_env.sh` **clears
  `EMSDK_HOME`**, so nothing after that line may refer to the variable.

Once built with real wasm objects, the archive links across emsdk versions — ORT's pinned 4.0.3
objects link fine with a 5.0.2 `wasm-ld` — so no version pinning is required.

**Known limitation — models do not load under WASM yet.** `OnnxSession::load` reads through
`gr::algorithm::fileio::readAsync(...)` and blocks on `get()`. Emscripten forbids blocking the main
thread, so the wait returns immediately with no data and every load fails with an empty-model error.
The WASM targets therefore build and link, but inference needs either a synchronous read path for
embedded/MEMFS files or the load moved off the main thread.

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
