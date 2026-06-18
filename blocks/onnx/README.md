# ONNX Runtime integration — user-API reference

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
- **Minimal footprint** — the `.ort` (ORT-optimised, flatbuffer) format applies graph
  optimisations and is the only model format a minimal/WASM build's runtime can load;
  it retains custom metadata just like `.onnx` (see `.onnx` vs `.ort` below).

This integration wraps ONNX Runtime into GR4 blocks that accept `DataSet<T>` or
`Tensor<T>` — drop-in replaceable with classical DSP blocks.

## TL;DR

`OnnxInference` is the generic building block — it runs any ONNX model and returns
the raw output as a `DataSet` or `Tensor`. `OnnxPeakDetector` — an introductory
application example — adds domain-specific peak extraction on top of the same
session/preprocess machinery.

```cpp
#include <gnuradio-4.0/onnx/OnnxInference.hpp>
#include <gnuradio-4.0/onnx/OnnxPeakDetector.hpp>
#include <gnuradio-4.0/fourier/PeakDetector.hpp>

gr::Graph graph;

// generic inference — runs any ONNX model, outputs the raw result
auto& inference = graph.emplaceBlock<gr::blocks::onnx::OnnxInference<float>>({
    {"model_path", std::string("models/affine_N64.onnx")},
});

// ML peak detector — model emits an in-graph post-processed peak set (see contract below)
auto& mlDetector = graph.emplaceBlock<gr::blocks::onnx::OnnxPeakDetector>({
    {"model_path",     std::string("models/peaks_fixture_N1024.onnx")},
    {"gate_threshold", 0.5f},
    {"max_peaks",      gr::Size_t(8)},
});

// classical peak detector — same event contract, no model needed
auto& classical = graph.emplaceBlock<gr::blocks::fourier::PeakDetector>({
    {"max_peaks", gr::Size_t(10)},
});

// all three accept DataSet<float> in and produce DataSet<float> out
if (!graph.connect(fftBlock, std::string("out"), mlDetector, std::string("in")).has_value()) {
    // handle wiring error
}
```

The four use-case binaries (`ex01_elementwise` etc., see below) wire
`OnnxInference` through `gr::Graph` and `gr::scheduler::Simple` end to end;
`ex05_ml_peak_detector`/`ex06_ml_vs_classical` drive the peak-detector blocks directly.

## Output format

`OnnxPeakDetector` produces a `DataSet<float>` with four signals; the classical
`PeakDetector` produces two:

```
OnnxPeakDetector signals:                  PeakDetector signals:
  [0] "Spectrum"        input pass-through   [0] "Spectrum"
  [1] "Heatmap"         ML confidence [0,1]  [1] "Prominence"  (in noise-sigma units)
  [2] "Reconstruction"  model peak-sum
  [3] "Residual"        spectrum − reconstruction
```

Both emit one `timing_events[0]` entry per detected peak with an identical key set,
so downstream consumers can swap the detectors without changes:
`confidence`, `centre` (fractional bin), `sigma`, `sigma_left`, `sigma_right`,
`amplitude`, `amplitude_measured`, `prominence`, `isolation`, `w68`, `w96`, `w99`,
`kurtosis`, `noise_sigma`, `noise_floor`, `position_uncertainty`,
`width_uncertainty`, `amplitude_uncertainty`.

One divergence from the drop-in claim cannot be removed: for a temporal (M > 1) model,
`OnnxPeakDetector` forwards the first M−1 input spectra unchanged with no peak events while
its frame-history buffer fills (see Lifecycle below). `PeakDetector` has no history and
never does this — every input it sees produces a real detection pass.

## Blocks

### `OnnxPeakDetector` — ML peak detection

Runs a peaks-tensor model (see model I/O contract below): the model itself performs
NMS, gating and refinement in-graph; the block resamples, normalises, infers,
thresholds and emits annotated peaks. There is deliberately no host-side NMS.

| Setting              | Type            | Default  | Description                                                 |
|----------------------|-----------------|----------|-------------------------------------------------------------|
| `model_path`         | `std::string`   | `""`     | path or URI (`file:`, `http(s):`) to `.onnx` / `.ort` model |
| `execution_provider` | `std::string`   | `"cpu"`  | `cpu`, `cuda`, `tensorrt`, `rocm` (validated at load)       |
| `resample_mode`      | `ResampleMode`  | `Linear` | `Linear` or `None`                                          |
| `normalise_mode`     | `NormaliseMode` | `None`   | `None`, `LogMAD`, `MinMax`, `ZScore`, `Expression`          |
| `clip_min`/`clip_max`| `float`         | `-5`/`10`| clip range for `LogMAD` / `ZScore`                          |
| `normalise_expr`     | `std::string`   | `""`     | ExprTk expression (when mode = `Expression`)                |
| `gate_threshold`     | `float`         | `0.5`    | accept a peak when its score reaches this                   |
| `max_peaks`          | `Size_t`        | `8`      | host-side cap: keep the highest-scoring peaks               |

Read-only (populated after model load): `model_io_shape`, `peaks_layout`,
`available_providers`. A model that declares `normalise_mode` (and for `Expression`
also `normalise_expr`) in its ONNX metadata auto-configures the preprocessing.

`OnnxPeakDetector` has no `error_policy` — unlike `OnnxInference`, it always stops on a load
or inference failure (`emitErrorMessage` then `requestStop()`). A peak detector that silently
reverted to pass-through would report "no peaks" indistinguishably from "no peaks found";
the only legitimate way to get pass-through behaviour is to load a pass-through ONNX model
(e.g. `models/identity_N64.onnx`), not to configure the block into one.

Temporal models (metadata `history_depth` M > 1) are driven from an internal frame
history: the first M−1 inputs pass through unchanged (warm-up), then every input
runs one inference over the newest M frames (sliding window, stride 1). M comes
from the model, not from a user setting.

### `OnnxInference<T, TIn, TOut>` — generic inference

Template block supporting the following port-type combinations
(all four are registered for `T = float`):

| `TIn`        | `TOut`       | Mode                                        |
|--------------|--------------|---------------------------------------------|
| `DataSet<T>` | `DataSet<T>` | single-slice `[1,1,N]` or history `[1,M,N]` |
| `DataSet<T>` | `Tensor<T>`  | strip metadata from output                  |
| `Tensor<T>`  | `Tensor<T>`  | raw tensor I/O                              |
| `Tensor<T>`  | `DataSet<T>` | attach metadata to raw output               |

| Setting           | Type           | Default  | Description                                            |
|-------------------|----------------|----------|--------------------------------------------------------|
| `model_path`      | `std::string`  | `""`     | path or URI to `.onnx` / `.ort` model                  |
| `error_policy`    | `ErrorPolicy`  | `Stop`   | `Stop` or `Passthrough` on load/inference failure      |
| `execution_provider` | `std::string` | `"cpu"` | `cpu`, `cuda`, `tensorrt`, `rocm` (validated at load)  |
| `resample_mode`   | `ResampleMode` | `Linear` | `Linear` or `None`                                     |
| `normalise_mode`  | `NormaliseMode`| `LogMAD` | `None`, `LogMAD`, `MinMax`, `ZScore`, `Expression`     |
| `normalise_expr`  | `std::string`  | `""`     | ExprTk expression (when mode = `Expression`)           |
| `history_depth`   | `Size_t`       | `1`      | M; auto-set from model metadata, 1 = no history        |
| `history_stride`  | `Size_t`       | `1`      | 1 = sliding window (1:1); S > 1 = genuine S:1 decimation |
| `model_overrides` | `property_map` | `{}`     | overridable-initializer values by name                 |

Read-only: `model_io_shape`, `meta_output_names`, `config_input_names`,
`available_providers`.

**History mode** (`DataSet` → `DataSet`, M > 1, `history_stride` = 1): the inference window
always covers the newest M frames. While it first fills, each input passes through unchanged
(1:1 output count) — the same warm-up contract as `OnnxPeakDetector`.

With `history_stride` S > 1 the block genuinely decimates S:1 through the framework's
resampling machinery: once the window is full, one output emits per S inputs. Warm-up is
**not** symmetric with the stride-1 case above: input chunks that arrive before the window
first fills are consumed and produce no output at all — not even a marked pass-through frame
— only once M frames have accumulated does the block start emitting.

**Multi-output models**: `out` carries model output 0; outputs 1..n−1 appear on
the optional `meta_out` ports (one `Tensor<T>` per inference). The full output set
is only requested from ORT when a `meta_out` port is connected, so single-output
use keeps the pruned fast path.

**Model configuration**: each overridable initializer of the model is exposed as an
optional scalar `config_in` port and can also be set by name via `model_overrides`.
Precedence: fed `config_in` port > `model_overrides` entry > model-baked default.
An entry that cannot be applied (unknown name, size or type mismatch) is reported
through the error policy instead of being silently ignored.

### `PeakDetector` — classical DSP baseline

Iterative detect–fit–subtract peak stripping with noise-adaptive prominence
thresholding. Settings: `noise_rejection_threshold` (5.0), `min_amplitude` (0),
`min_isolation` (0.5), `max_iterations` (10), `min_prominence` (5.0), `max_peaks`
(10), `subtraction_shape` (0 = Auto, 1 = Gaussian, 2 = Lorentzian, 3 = Voigt).
`noise_rejection_threshold`/`min_prominence` default to 5 (not the 2 order-statistics
naively suggest) because a 1024-bin spectrum routinely has noise excursions up to ~4.5σ,
particularly near band edges — see the why-comment on those settings in `PeakDetector.hpp`.
Same output/event contract as `OnnxPeakDetector` — swap in a flow graph without
changing downstream.

## Lifecycle

`start()` loads the model and configures preprocessing; an empty `model_path` is reported as
an error without attempting a load — through `OnnxInference`'s configurable `error_policy`,
or unconditionally stopping `OnnxPeakDetector` (see above). `stop()` tears the session down
and clears any frame history, so a later `start()` reloads the model from scratch — for
`http(s):` model paths this means a real network re-fetch, not a cached reopen.

Restarting a temporal (M > 1) block re-enters warm-up. `OnnxPeakDetector` and `OnnxInference` in
sliding-window mode (`history_stride` = 1) pass the first M-1 inputs through unchanged, keeping
a 1:1 output count. `OnnxInference` in decimating mode (`history_stride` > 1) does not: it
consumes input chunks silently until the window first fills, emitting nothing at all in the
meantime — see history mode above.

## Threading and multiple instances

Each block owns its own `OnnxSession`, including a private `Ort::Env` — there is no runtime
state shared between block instances. Intra-op threading is hardcoded to a single thread and no
inter-op thread pool is configured: GR4's scheduler already parallelises across blocks, so
letting ORT also spawn worker threads per session would oversubscribe the machine.

- a single session is **not** thread-safe: its input/output buffers are bound once at load and
  reused in place on every inference call, so concurrent calls into one block instance would
  race on those buffers
- one session per block instance is the supported model — several `OnnxPeakDetector` /
  `OnnxInference` blocks in the same flow graph are independent, each with its own model,
  buffers and (optionally) execution provider

## Execution providers

Both blocks accept `execution_provider` = `cpu` (default), `cuda`, `tensorrt` or
`rocm`. The request is validated against `Ort::GetAvailableProviders()` of the
linked ONNX Runtime at model load: an unavailable or unknown provider fails the
load through the block's error handling, naming the request and listing the
available providers — never a silent CPU fallback. The read-only
`available_providers` setting mirrors what the linked runtime offers. The data
plane stays on CPU tensors; ORT performs device transfers internally.

N.B. the system ONNX Runtime this tree currently links is CPU-only, so the GPU
provider paths are validated (error handling) but not exercised here.

## Normalisation

All inference blocks support configurable preprocessing via `normalise_mode`
(defaults differ: `OnnxInference` = `LogMAD`, `OnnxPeakDetector` = `None`; a model
declaring `normalise_mode` in its metadata overrides the setting at load):

| Mode         | Description                                             |
|--------------|---------------------------------------------------------|
| `None`       | pass through unchanged                                  |
| `LogMAD`     | log10 → robust z-score (median / MAD), clip to [-5, 10] |
| `MinMax`     | linear scaling to [0, 1]                                |
| `ZScore`     | (x - mean) / std, clip to [-5, 10]                      |
| `Expression` | user-defined ExprTk expression                          |

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

### `.onnx` vs `.ort`

| | `.onnx` | `.ort` |
|---|---|---|
| format | protobuf; human-inspectable, editable with the `onnx` Python package | flatbuffer (`ORTM` magic); pre-optimised, not hand-editable |
| loadable by | `opt`-mode (system ORT) builds only — `validateModelPath` (`OnnxHelper.hpp`) rejects `.onnx` under `GR_ONNX_MINIMAL_BUILD` | `opt`-mode **and** minimal (`on`-mode / WASM) builds — `validateModelPath` accepts `.ort` unconditionally |
| use it for | portable interchange, inspection, editing metadata | deploying to minimal/size-reduced native builds and Emscripten/WASM (`ENABLE_ONNX_INTEGRATION=on` — the minimal ORT runtime linked there cannot parse `.onnx` protobuf at all, see `ONNX_INSTALL.md`) |
| custom metadata | present | present — carried through unchanged by both conversion routes this repo uses (verified: ORT's `optimized_model_filepath` session option, used by this module's generators, and the `onnxruntime.tools.convert_onnx_models_to_ort` CLI both retain every custom metadata key) |

Ship `.onnx` as the portable, inspectable artefact; ship `.ort` alongside it whenever the
model might run in a minimal or WASM build. The generic-fixture and `models/` generators
write both files together from one source model; the peaks fixtures need the separate
conversion step instead (see Regenerating the tracked models below) — either way there
is no "strip metadata for deployment" step.

Do not confuse file format with the `GR_ONNX_MINIMAL_BUILD` compile-time flag (see
Metadata keys below): that flag deletes the `readMetadata()` code path itself, so a
minimal build never reads metadata regardless of which format it loads — while a
full/`opt` build reads metadata from either format.

`OnnxInference` accepts any model with one float input `[1, M, N]` (M = 1 for
single-slice) and at least one output — output 0 goes to `out`, the rest to
`meta_out`.

`OnnxPeakDetector` additionally requires **named outputs**: the model must emit a
fully post-processed peak set (NMS, gating and refinement in-graph):

| Output           | Shape       | Required | Purpose                                        |
|------------------|-------------|----------|------------------------------------------------|
| `peaks`          | `[1, K, P]` | yes      | K peak candidates × P properties               |
| `peak_rescore`   | `[1, K]`    | no       | learned gate re-scorer; supersedes column 0    |
| `heatmap`        | `[1, N]`    | no       | confidence curve for signal 1                  |
| `reconstruction` | `[1, N]`    | no       | peak-sum for signal 2                          |
| `residual`       | `[1, N]`    | no       | spectrum − reconstruction for signal 3         |

The peaks column layout is compiled in (`kPeaksLayoutPrefix`), P ≥ 10:

```
0 peak_present, 1 centre, 2 amplitude, 3 sigma_left, 4 sigma_right,
5 eta, 6 sigma_avg, 7 score, 8 type_tag, 9 is_top1, 10-12 reserved
```

Only columns 0-6 are validated: a loaded model whose `property_layout` metadata does not
start with `peak_present,centre,amplitude,sigma_left,sigma_right,eta,sigma_avg` is rejected
at load time instead of being silently misparsed (`matchesPeaksLayout` is a prefix check over
those seven names). Columns 7-9 are the documented convention but are not name-checked, and
`extractPeaksRegressor` does not read them directly — the acceptance score instead comes from
the optional `peak_rescore` output, falling back to column 0 when absent. A model may append
further columns beyond the validated prefix. Legacy heatmap+regression models
(`[batch, N + N*R]`, no `peaks` output) are likewise rejected; a qa test pins this.

### Metadata keys

Models self-describe via ONNX custom metadata, read at load:

| Key                 | Example                    | Purpose                                                    |
|---------------------|----------------------------|------------------------------------------------------------|
| `input_size`        | `"1024"`                   | primary input dimension N (`fft_size` accepted as fallback)|
| `history_depth`     | `"16"`                     | M; omit or `"1"` for single-slice; shape fallback `[1,M,N]`|
| `n_max_peaks`       | `"8"`                      | K; falls back to the `peaks` output shape                  |
| `n_peak_properties` | `"13"`                     | P; falls back to the `peaks` output shape                  |
| `property_layout`   | `"peak_present,centre,…"`  | peaks column names; only the first 7 are validated against the compiled-in prefix |
| `score_output`      | `"peak_rescore"`           | output carrying the acceptance score (default `peak_rescore`) |
| `normalise_mode`    | `"LogMAD"`                 | auto-configure preprocessing                               |
| `normalise_expr`    | `"vecOut := vecIn"`        | ExprTk expression (when `normalise_mode` = `Expression`)   |
| `architecture`      | `"analytic-fixture"`       | informational                                              |

Minimal builds (`GR_ONNX_MINIMAL_BUILD`) cannot read metadata at all — `readMetadata()`
returns immediately. N then falls back to the trailing dimension of the model's own input
tensor shape, queried from ONNX Runtime rather than parsed from the file name; history depth
M falls back the same way when the shape's middle dimension exceeds one. The compiled-in
peaks layout is assumed in both cases.

Overridable initializers in the model graph (runtime-tunable operating points with
a baked default) surface as `config_in` ports / `model_overrides` entries in
`OnnxInference`.

### Bundled models

Every tracked model ships as both `.onnx` and `.ort` (see `.onnx` vs `.ort` above) —
with one current exception, `peaks_fixture_expr_N1024.onnx`, which has no `.ort`
sibling yet.

| Model                     | Shape          | Purpose                                                        |
|---------------------------|----------------|----------------------------------------------------------------|
| `peaks_fixture_N1024`     | `[1,1,1024]`   | analytic peaks-tensor fixture — qa suite and peak examples     |
| `peaks_fixture_M16_N1024` | `[1,16,1024]`  | temporal fixture (detects on the newest frame only)            |
| `affine_N64`              | `[1,1,64]`     | 1:1 elementwise use-case: y = 2x + 1                            |
| `frame_mean_N64_M4`       | `[1,4,64]`     | N:1 decimating use-case: mean over 4 frames                     |
| `frame_delta_N64_M8`      | `[1,8,64]`     | sliding-window history use-case: newest − oldest frame          |
| `mean_rms_N64`            | `[1,1,64]`     | multi-output use-case: mean + rms meta output                   |
| `identity_N64[_M4]`       | `[1,{1,4},64]` | CI models (output = input); the M4 variant drives the M x N session/inference tests |
| `peak_detector_N1024`, `_N4096`, `_history_N1024_M16` | `[1,{1,16},N]` | legacy heatmap+regression models — generic-inference test inputs only; **rejected by `OnnxPeakDetector`** |

### Regenerating the tracked models

The generic-fixture generators (`src/ex4_python`..`ex8_python`) live in a separate
repository and are present on the development machine but untracked here. A few
framework-agnostic generators do live in `models/` and are tracked: `fixtures_gen.py`
(peaks-tensor fixtures), `ex01_elementwise_gen.py`/`ex01_elementwise_inference.py` (minimal hand-built
round-trip, y = a·x + b), and `ex00_identity_gen.py` (identity model built and exported
from TensorFlow/Keras via tf2onnx — the export path an ML user actually trains through;
writes untracked `identity_N64_tf.onnx`/`.ort` alongside the hand-built `identity_N64.onnx`
reference). All of the above require TensorFlow, `tf2onnx` and/or `onnx`/`onnxruntime`
installed locally — not a build dependency, not installed by these scripts, and
**not run by CI**. The fixtures record their provenance where possible
(`generator_cmd`, `git_rev`, `created` in their ONNX metadata):

```bash
# peak fixtures (writes .onnx for both M=1 and M=16 and self-tests them)
python models/fixtures_gen.py models/

# minimal hand-built round-trip example + inference check
python models/ex01_elementwise_gen.py
python models/ex01_elementwise_inference.py

# TensorFlow/Keras identity example (writes identity_N64_tf.onnx/.ort)
python models/ex00_identity_gen.py

# generic use-case fixtures (each writes .onnx + .ort and self-tests)
python src/ex5_python/ex5_affine_model.py
python src/ex6_python/ex6_frame_mean_model.py
python src/ex7_python/ex7_frame_delta_model.py
python src/ex8_python/ex8_mean_rms_model.py

# litmus grid consumed by onnx_litmus (untracked, regenerable)
python src/ex4_python/litmus.py --dump-bin models/litmus_cases.bin

# .onnx → .ort conversion (needed for the peak fixtures)
python -m onnxruntime.tools.convert_onnx_models_to_ort models/peaks_fixture_N1024.onnx
```

## Exporting models to ONNX

### From TensorFlow / Keras

```python
# wrap trained model to match the [batch, M, N] input convention, then export
inp = keras.Input(shape=(1, N), dtype="float32", name="input")
x = keras.ops.transpose(inp, [0, 2, 1])  # Conv1D expects [batch, N, channels]
output = trained_model(x, training=False)
keras.Model(inputs=inp, outputs=output).export("saved_model_dir")
# then: python -m tf2onnx.convert --saved-model saved_model_dir --output model.onnx --opset 17
```

`models/ex00_identity_gen.py` is a complete, runnable version of this pattern (Keras
model → `tf2onnx.convert.from_keras` → stamp metadata → `.onnx` + `.ort`), for the
identity case — see Regenerating the tracked models above.

### From PyTorch

```python
model.eval()
dummy = torch.randn(1, 1, N)
torch.onnx.export(model, dummy, "model.onnx", opset_version=17,
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
```

### Converting `.onnx` to `.ort`

Required for minimal (`on`-mode) and WASM builds — see `.onnx` vs `.ort` above; the
conversion retains custom metadata:

```bash
python -m onnxruntime.tools.convert_onnx_models_to_ort model.onnx
```

This module's own generators use the equivalent `ort.SessionOptions.optimized_model_filepath`
API instead, so both `.onnx` and `.ort` fall out of one generator run (see
`ex01_elementwise_gen.py` / `ex00_identity_gen.py` for the pattern).

### Adding metadata

```python
import onnx
model = onnx.load("model.onnx")
for key, value in {"input_size": "1024", "normalise_mode": "LogMAD"}.items():
    entry = model.metadata_props.add()
    entry.key, entry.value = key, value
onnx.save(model, "model.onnx")
```

## Generic `OnnxInference` use-cases

Four documented rate configurations, each backed by a small analytic fixture
(hand-built opset-17 graph, deterministic, self-tested by its generator), a qa
suite (`test/qa_OnnxInferenceUseCases.cpp`) and a graph-wired example binary.
Each fixture carries its own provenance in the ONNX custom metadata
(`generator_cmd`, `git_rev`, `created`) next to the self-configuration keys
(`input_size`, `history_depth`, `normalise_mode`).

| Use-case                   | Fixture (computes)                            | Example binary                | Regeneration                                    |
|----------------------------|-----------------------------------------------|-------------------------------|-------------------------------------------------|
| 1:1 elementwise            | `affine_N64` (y = 2x + 1)                     | `ex01_elementwise`            | `python src/ex5_python/ex5_affine_model.py`     |
| N:1 decimating             | `frame_mean_N64_M4` (mean over M = 4 frames)  | `ex02_decimating`             | `python src/ex6_python/ex6_frame_mean_model.py` |
| sliding-window history     | `frame_delta_N64_M8` (newest − oldest, M = 8) | `ex03_sliding_window`         | `python src/ex7_python/ex7_frame_delta_model.py`|
| multi-output                | `mean_rms_N64` (out = mean, meta_out#0 = rms) | `ex04_multi_output`           | `python src/ex8_python/ex8_mean_rms_model.py`   |

Each generator writes both `.onnx` and `.ort` into `models/` and fails loudly if
the graph does not compute what its name says; a matching
`src/ex<N>_python/ex<N>_evaluate.py` re-checks the shipped artefact against the
closed-form reference from Python.

## Building your own block on this module

The module separates reusable machinery from the peak-detection application; a new
ONNX-backed block composes the former and replaces the latter:

- **`OnnxSession`** (`OnnxSession.hpp`) — model loading (`load` for path/URI,
  `loadFromMemory`), execution-provider selection, metadata/shape introspection,
  and inference: `run()` returns output 0, `runAll()` returns all outputs as
  `NamedTensor`s (name, shape, values). Pick tensors by name from the `runAll`
  result, as `OnnxPeakDetector::findOutput` does.
- **`OnnxPreprocess<T>`** (`OnnxPreprocess.hpp`) — `configure(resample, normalise,
  expr, clipMin, clipMax)` once, then `normalise(raw, out)` per frame; static
  `resample()` for grid changes.
- **Error handling** — a `handleError()` helper that calls `emitErrorMessage` then
  `requestStop()`; `std::expected` carries errors up to it. `OnnxInference` additionally
  exposes an `error_policy` setting so a caller can instead latch pass-through
  (`ErrorPolicy::Passthrough`) — a choice deliberately not offered on `OnnxPeakDetector`,
  where silently reverting to pass-through would be indistinguishable from "no peaks found".
- **Load state** — expose a public `isModelLoaded()` predicate (as `OnnxInference`
  does) rather than letting callers poke the session member.

In `OnnxPeakDetector` the reusable part is `start`/`stop`/`loadModel`/
`configurePreprocess`, the resample-normalise-infer sequence in `processOne` and
the history handling. Peak-specific — replace it with your own post-processing —
is everything downstream of `runAll`: `kPeaksLayoutPrefix`/`matchesPeaksLayout`,
`extractPeaksRegressor`, the spectrum statistics from `PeakResult.hpp` and
`buildOutput`. Copy the block, strip the peak-specific half, and keep the
settings/reflection layout.

## Build configuration

```bash
cmake -DENABLE_ONNX_INTEGRATION=opt ..   # system ORT or bundled static — recommended
cmake -DENABLE_ONNX_INTEGRATION=on  ..   # build ORT from source (cross-compile, WASM)
cmake -DENABLE_ONNX_INTEGRATION=off ..   # disable entirely
```

See `ONNX_INSTALL.md` for details — including two local environment gotchas
(shadowed ONNX Runtime needing `LD_LIBRARY_PATH`, and the flags a fresh build
directory needs) that look like build breaks but are not.

## Examples

| Binary                        | Description                                                                     |
|-------------------------------|---------------------------------------------------------------------------------|
| `ex05_ml_peak_detector`       | `OnnxPeakDetector` single-stage vs cascaded (detect-subtract-repeat) modes      |
| `ex06_ml_vs_classical`        | detection latency benchmark, ML vs classical (200 evolving spectra, histogram)  |
| `ex01_elementwise`            | 1:1 elementwise use-case: graph-wired inference on `affine_N64`                 |
| `ex02_decimating`             | N:1 decimating use-case: genuine decimation on `frame_mean_N64_M4`              |
| `ex03_sliding_window`         | sliding-window history use-case on `frame_delta_N64_M8`                        |
| `ex04_multi_output`           | multi-output use-case: `meta_out` wiring on `mean_rms_N64`                     |
| `onnx_litmus`                 | estimator-quality regression table (width × S/N grid), cross-checks the Python reference |

```bash
build/blocks/onnx/src/ex05_ml_peak_detector    # defaults to the bundled analytic fixture
build/blocks/onnx/src/onnx_litmus <model> models/litmus_cases.bin
```

`onnx_litmus` needs `models/litmus_cases.bin` (untracked — regenerate with
`python src/ex4_python/litmus.py --dump-bin models/litmus_cases.bin`).

## Tests

```bash
ctest --test-dir build -R "qa_Onnx|qa_Peak" --output-on-failure -j6
```

(Prefix with `LD_LIBRARY_PATH=/usr/lib64` if another ONNX Runtime shadows the
linked one — see `ONNX_INSTALL.md` troubleshooting.)

| Test                       | Coverage                                                                        |
|----------------------------|---------------------------------------------------------------------------------|
| `qa_OnnxInstallationTest`  | ORT runtime capabilities                                                        |
| `qa_OnnxPreprocess`        | all normalisation modes, ExprTk                                                 |
| `qa_OnnxSession`           | model lifecycle, execution providers, M x N, history model                      |
| `qa_OnnxInference`         | all four port-type combinations, history/decimation, `meta_out`, `model_overrides`, `config_in` |
| `qa_OnnxInferenceUseCases` | the four documented rate configurations on the analytic fixtures                |
| `qa_OnnxPeakDetector`      | peaks-tensor contract, temporal M=16, layout validation, legacy-model rejection, always-stop-on-error |
| `qa_PeakDetector`          | classical DSP peak detection                                                    |
