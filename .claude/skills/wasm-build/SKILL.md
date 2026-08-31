---
name: wasm-build
description: Configure and run the local Emscripten/WebAssembly build with ONNX Runtime, including the tests. Use when reproducing an emcc CI failure or changing anything under blocks/onnx that WASM consumes.
---

# WASM build

```bash
source /home/steinhagen/git/emsdk/emsdk_env.sh
emcmake cmake -S . -B build-wasm -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTING=ON \
  -DORT_STATIC_LIB_DIR=/home/steinhagen/temp/ort-nolto-eh
cmake --build build-wasm --parallel 2 --target <targets>
ctest --test-dir build-wasm -R "qa_Onnx" --output-on-failure
```

## The archive must not be LTO

`~/onnxruntime-static` holds **LLVM IR bitcode** — it predates the `-flto` strip in
`docker/build_onnxruntime.sh`. Linking it makes `wasm-ld` die with SIGSEGV and no diagnostic. Use an
`ort-nolto*` tree. To check an archive:

```bash
ar x <archive>.a && file $(ls *.o | head -1)
# want: "WebAssembly (wasm) binary"   not: "LLVM IR bitcode"
```

## Fixtures

WASM has no host filesystem. Every model a test loads must be embedded via `--embed-file` in
`blocks/onnx/test/CMakeLists.txt`, and the lists must mirror the `modelPath()` and
`deliverableModelPath()` calls in each source — including any loop over a runtime variable, which a
grep for string literals will miss.

A minimal ONNX Runtime build loads `.ort` only. `.onnx` fixtures fail with "ONNX format model is not
supported in this build"; use `deliverableModelPath()`, which selects `.ort.gz` under Emscripten.

Redirect test output through `head -c` — a looping WASM binary has filled the disk before.
