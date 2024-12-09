# ONNX Runtime integration — User-API reference

## Why ONNX?

[ONNX](https://onnx.ai) (Open Neural Network Exchange) is an open, standardised
format for ML models. Using ONNX as the inference layer in GR4 provides:

- **Framework independence** — train in TensorFlow, PyTorch, JAX, scikit-learn, or
  any framework that exports ONNX. The runtime does not link against any of them.
- **Hardware portability** — ONNX Runtime supports CPU (x86, ARM), GPU (CUDA, ROCm),
  WASM, and accelerator backends through a single API.
- **Reproducibility** — the `.onnx` file is a self-contained, versioned artefact.
  Ship it alongside the flow graph; no Python, no training environment needed at runtime.
- **Separation of concerns** — RSE/DSP engineers configure blocks and connect them to
  flow-graphs; data scientists iterate on model architecture and training independently.
- **Minimal footprint** — the `.ort` (ORT-optimised) format strips metadata and applies
  graph optimisations, producing compact models suitable for embedded and WASM targets.

This integration wraps ONNX Runtime into GR4 blocks that accept `DataSet<T>`,
`Tensor<T>`, or streaming `T` — drop-in replaceable with classical DSP blocks.

## TL;DR

`OnnxInference` is the generic building block — it runs any ONNX model and returns
the raw output tensor as a `DataSet`, `Tensor`, or streaming `T`. Specialised blocks
like `OnnxPeakDetector` -- as an introductory example -- add domain-specific
post-processing on top.

```cpp
#include <gnuradio-4.0/onnx/OnnxInference.hpp>
#include <gnuradio-4.0/onnx/OnnxPeakDetector.hpp>
#include <gnuradio-4.0/onnx/PeakDetector.hpp>

gr::Graph graph;

// generic inference — runs any ONNX model, outputs raw result tensor
auto& inference = graph.emplaceBlock<gr::blocks::onnx::OnnxInference<float>>({
    {"model_path", std::string("my_model.onnx")},
});

// practical example: ML peak detector (inference + normalisation + NMS → annotated peaks)
auto& mlDetector = graph.emplaceBlock<gr::blocks::onnx::OnnxPeakDetector>({
    {"model_path",           std::string("peak_detector_N1024.onnx")},
    {"confidence_threshold", 0.3f},
    {"min_peak_distance",    gr::Size_t(8)},
    {"max_peaks",            gr::Size_t(8)},
});

// classical peak detector — same output format, no model needed
auto& classical = graph.emplaceBlock<gr::blocks::onnx::PeakDetector>({
    {"prominence_threshold", 3.0f},
    {"min_peak_distance",    gr::Size_t(5)},
    {"max_peaks",            gr::Size_t(8)},
});

// all three accept DataSet<float> in and produce DataSet<float> out
graph.connect<"out">(fftBlock).to<"in">(mlDetector);
```

## Output format

Both `OnnxPeakDetector` and `PeakDetector` produce identical `DataSet<float>`:

```
signals:
  [0] "Spectrum"    — input spectrum (pass-through)
  [1] "Heatmap"     — ML confidence [0,1]  (or "Prominence" for classical)

timing_events[0]:   (per detected peak)
  { offset: bin_position,
    properties: { "confidence", "sigma", "amplitude", "w68", "w96", "w99", "kurtosis" } }
```

## Blocks

### `OnnxPeakDetector` — ML peak detection

Fused pipeline: resample to model dimension, normalise, infer, NMS, annotated output.

| Setting                | Type            | Default  | Description                                         |
|------------------------|-----------------|----------|-----------------------------------------------------|
| `model_path`           | `std::string`   | `""`     | path to `.onnx` or `.ort` model                     |
| `confidence_threshold` | `float`         | `0.4`    | minimum heatmap confidence                          |
| `min_peak_distance`    | `Size_t`        | `8`      | minimum bin separation                              |
| `max_peaks`            | `Size_t`        | `8`      | maximum peaks per spectrum                          |
| `normalise_mode`       | `NormaliseMode` | `LogMAD` | `LogMAD`, `MinMax`, `ZScore`, `Expression`, `None`  |
| `normalise_expr`       | `std::string`   | `""`     | custom ExprTk expression (when mode = `Expression`) |
| `resample_mode`        | `ResampleMode`  | `Linear` | `Linear` or `None`                                  |

### `OnnxInference<T, TIn, TOut>` — generic inference

Template block supporting the following port-type combinations:

| `TIn`        | `TOut`       | Mode                                        |
|--------------|--------------|---------------------------------------------|
| `DataSet<T>` | `DataSet<T>` | single-slice `[1,1,N]` or history `[1,M,N]` |
| `DataSet<T>` | `Tensor<T>`  | strip metadata from output                  |
| `Tensor<T>`  | `Tensor<T>`  | raw tensor I/O                              |
| `Tensor<T>`  | `DataSet<T>` | attach metadata to raw output               |
| `T`          | `T`          | streaming filter (chunk = model input dim)  |
| `T`          | `DataSet<T>` | streaming to spectrum (like FFT)            |

#### History-based inference (M x N)

When the model input is `[1, M, N]` with M > 1, the block accumulates M consecutive
inputs before running inference — enabling temporal tracking across a sliding window.

```cpp
auto& historyBlock = graph.emplaceBlock<gr::blocks::onnx::OnnxInference<float>>({
    {"model_path", std::string("peak_detector_history_N1024_M16.onnx")},
    // history_depth auto-detected from model metadata (M=16)
    // first 15 processOne() calls return empty (accumulating)
    // 16th triggers inference on [16, 1024]; then slides by history_stride
});
```

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `history_depth` | `Size_t` | `1` | auto-set from model; 1 = no history |
| `history_stride` | `Size_t` | `1` | slices consumed per inference (1 = sliding window) |
| `stride` | `Size_t` | `0` | streaming overlap (0 = non-overlapping) |

### `PeakDetector` — classical DSP baseline

Local-maxima detection with noise-adaptive prominence thresholding. Same output
format as `OnnxPeakDetector` — swap in a flow graph without changing downstream.

### `PeakExtractor` — standalone post-processing

Accepts raw inference output (heatmap + regression) and produces annotated peaks.

## Normalisation

All inference blocks support configurable preprocessing via `normalise_mode`:

| Mode         | Description                                             |
|--------------|---------------------------------------------------------|
| `LogMAD`     | log10 → robust z-score (median / MAD), clip to [-5, 10] |
| `MinMax`     | linear scaling to [0, 1]                                |
| `ZScore`     | (x - mean) / std, clip to [-5, 10]                      |
| `Expression` | user-defined ExprTk expression                          |
| `None`       | pass through unchanged                                  |

### ExprTk expressions

When `normalise_mode = Expression`, set `normalise_expr` to a custom ExprTk
expression. The runtime pre-computes statistics on the **raw input** before
evaluation and exposes them as scalar variables alongside the input/output vectors.
The expression reads `vecIn`, writes `vecOut`, and is evaluated once per spectrum.

Example — approximate `LogMAD` normalisation:

```
// pre-computed by the runtime (read-only):
//   vecIn[0..n-1]  — raw input vector
//   n              — number of elements
//   median, mad    — median and median absolute deviation of vecIn
//   min_val, max_val, mean_val, std_val — of vecIn
// to be written by this expression:
//   vecOut[0..n-1] — normalised output vector

// for X ~ N(μ,σ²):  MAD = σ · Φ⁻¹(¾) ≈ 0.6745·σ  →  σ ≈ 1.4826 · MAD
var scale := 1.0 / (1.4826 * mad + 1e-10);
for (var i := 0; i < n; i += 1) {
    var shifted := vecIn[i] - min_val + 1.0;
    var logged  := log10(shifted);
    vecOut[i]   := clamp(-5.0, (logged - median) * scale, 10.0);
};
```

N.B. `median` and `mad` above are computed on the raw (pre-log) input. The
built-in `LogMAD` mode computes them after the log10 step, so results differ
slightly. For an exact match, compute the post-log statistics inside the
expression itself.

## Model I/O contract

Models are standard ONNX (`.onnx`) or ORT-optimised (`.ort`) files.

### Shapes

| Mode | Input | Output |
|------|-------|--------|
| single-slice | `[batch, 1, N]` | `[batch, N + N*R]` |
| history | `[batch, M, N]` | `[batch, N + N*R]` |

- `N` — input dimension (e.g. 1024 frequency bins, or any 1D signal length)
- `M` — history depth (consecutive slices; 1 = single-shot)
- `R` — regression channels (default 8)

### Metadata keys

Models may carry custom metadata that the runtime reads on load:

| Key                     | Example     | Purpose                           |
|-------------------------|-------------|-----------------------------------|
| `input_size`            | `"1024"`    | primary input dimension N         |
| `history_depth`         | `"16"`      | M; omit or `"1"` for single-slice |
| `n_regression_channels` | `"8"`       | R (defaults to 8 if absent)       |
| `normalise_mode`        | `"LogMAD"`  | auto-configure preprocessing      |
| `architecture`          | `"unet_1d"` | informational                     |

If metadata is absent (`.ort` strips it), N is inferred from the filename
convention: `peak_detector_N1024.ort` -> N=1024.

### Bundled models

| Model                             | Shape         | Purpose                         |
|-----------------------------------|---------------|---------------------------------|
| `peak_detector_N1024`             | `[1,1,1024]`  | single-slice peak detection     |
| `peak_detector_N4096`             | `[1,1,4096]`  | single-slice, larger input      |
| `peak_detector_history_N1024_M16` | `[1,16,1024]` | history-based peak tracking     |
| `identity_N64`                    | `[1,1,64]`    | CI test model (output = input)  |
| `identity_N64_M4`                 | `[1,4,64]`    | CI test model for M x N history |

Each model ships as both `.onnx` (full format with metadata) and `.ort` (minimal).

## Exporting models to ONNX

### From TensorFlow / Keras

```python
# wrap trained model to match [batch, 1, N] input convention, then export
inp = keras.Input(shape=(1, N), dtype="float32", name="input")
x = keras.ops.transpose(inp, [0, 2, 1])  # Conv1D expects [batch, N, channels]
output = trained_model(x, training=False)
keras.Model(inputs=inp, outputs=output).export("saved_model_dir")
# then: python -m tf2onnx.convert --saved-model saved_model_dir --output model.onnx --opset 17
```

### From PyTorch

```python
model.eval()
dummy = torch.randn(1, 1, N)
torch.onnx.export(model, dummy, "model.onnx", opset_version=17,
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
```

### Converting `.onnx` to `.ort`

The `.ort` format is an optimised minimal representation required for WASM
and minimal builds:

```bash
python -m onnxruntime.tools.convert_onnx_models_to_ort model.onnx
```

### Adding metadata

```python
import onnx
model = onnx.load("model.onnx")
for key, value in {"input_size": "1024", "normalise_mode": "LogMAD"}.items():
    entry = model.metadata_props.add()
    entry.key, entry.value = key, value
onnx.save(model, "model.onnx")
```

Training scripts for the bundled models are in `src/ex0_python/` through `src/ex3_python/`.

## Build configuration

```bash
cmake -DENABLE_ONNX_INTEGRATION=opt ..   # system ORT or bundled static — recommended
cmake -DENABLE_ONNX_INTEGRATION=on  ..   # build ORT from source (cross-compile, WASM)
cmake -DENABLE_ONNX_INTEGRATION=off ..   # disable entirely
```

See `ONNX_INSTALL.md` for details.

## Examples

| Binary          | Description                                                   |
|-----------------|---------------------------------------------------------------|
| `onnx_example0` | side-by-side ML vs classical detection with dB spectrum chart |
| `onnx_example1` | detection latency benchmark (200 evolving spectra, histogram) |

```bash
cmake-build-debug-gcc15/blocks/onnx/src/onnx_example0
cmake-build-debug-gcc15/blocks/onnx/src/onnx_example1
```

## Tests

```bash
ctest --test-dir build -R "qa_Onnx|qa_Peak" --output-on-failure -j6
```

| Test | Coverage |
|------|----------|
| `qa_OnnxInstallationTest` | ORT runtime capabilities |
| `qa_OnnxUtils` | normalise, resample, NMS |
| `qa_OnnxPreprocess` | all normalisation modes, ExprTk |
| `qa_OnnxSession` | model lifecycle, M x N, history model |
| `qa_OnnxInference` | all type combinations, streaming, history accumulation |
| `qa_OnnxPeakDetector` | N1024 + N4096 parameterised (native, decimate, interpolate) |
| `qa_PeakDetector` | classical DSP peak detection |
| `qa_PeakExtractor` | standalone post-processing |
