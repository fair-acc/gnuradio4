# ONNX Runtime integration

## TL;DR

`OnnxInference<T, TIn, TOut>` runs any ONNX model over `DataSet<T>`/`Tensor<T>`
in four rate configurations: 1:1 elementwise, N:1 decimating, sliding-window
history, and multi-output. Input rank 1, 2 and 3 are accepted, the batch
dimension may be symbolic, and `T` may be `float` or `double`.

`OnnxPeakDetector` is one worked consumer of that machinery — a drop-in
alternative to the classical `gr::blocks::fourier::PeakDetector` sharing its
output event contract — and is documented at the end.

The governing idea: **the algorithm lives in the `.onnx` file, not in C++.**
NMS, gating, refinement and (for temporal models) frame fusion all run inside
the model graph; the block is plumbing — session lifecycle, resampling,
normalisation, tensor↔`DataSet` marshalling. Swapping the model swaps the topology.

## Minimal example

```cpp
auto& inference = graph.emplaceBlock<gr::blocks::onnx::OnnxInference<float>>({
    {"model_path",     std::string("models/frame_delta_N64_M8.ort.gz")},
    {"history_depth",  gr::Size_t(8)},  // rank-3 [batch, frames, N] model
    {"normalise_mode", "LogMAD"},
});
```

The peak detector is configured the same way, with its own settings:

```cpp
auto& detector = graph.emplaceBlock<gr::blocks::onnx::OnnxPeakDetector>({
    {"model_path",     std::string("models/ex05_peak_detector_cascaded.ort.gz")},
    {"gate_threshold", 0.45f},
    {"max_peaks",      gr::Size_t(8)},
});
```

## Settings

Shared by both blocks:

| Setting               | Type            | Default   | Meaning                                                                                  |
| --------------------- | --------------- | --------- | ---------------------------------------------------------------------------------------- |
| `model_path`          | `std::string`   | `""`      | path or URI (`file:`, `http(s):`) to `.onnx`/`.ort`                                      |
| `execution_provider`  | `std::string`   | `"cpu"`   | `cpu`, `cuda`, `tensorrt`, `rocm` (validated at load)                                    |
| `resample_mode`       | `ResampleMode`  | `Linear`  | `Linear` or `None`                                                                       |
| `normalise_mode`      | `NormaliseMode` | `None`    | `None`, `LogMAD`, `MinMax`, `ZScore`, `Expression`                                       |
| `clip_min`/`clip_max` | `T` or `float`  | `-5`/`10` | clip range for `LogMAD`/`ZScore` (`T` in `OnnxInference`, `float` in `OnnxPeakDetector`) |
| `normalise_expr`      | `std::string`   | `""`      | ExprTk expression (mode = `Expression`)                                                  |

Read-only on both: `model_input_shape`, `model_output_shape`, `available_providers`.

`OnnxPeakDetector`-only: `gate_threshold` (`float`, `0.5`), `max_peaks`
(`Size_t`, `8`, host-side cap). Read-only: `peaks_layout`. No `error_policy`
— it always stops on a load/inference failure rather than risk a silent
pass-through reading as "no peaks found". No host-side NMS: the model has
already decided which peaks to keep. Temporal models (metadata
`history_depth` M > 1) pass the first M−1 inputs through unchanged while
history fills, then infer once per input over the newest M frames.

`OnnxInference<T, TIn, TOut>`-only: `error_policy` (`ErrorPolicy`, `Stop` or
`Passthrough`), `history_depth` (`Size_t`, `1`, M, auto-set from metadata),
`history_stride` (`Size_t`, `1`), `model_overrides` (`property_map`, `{}`,
overridable-initializer values by name). Read-only: `meta_output_names`,
`config_input_names`. Supports `DataSet<T>`↔`DataSet<T>` (single-slice
`[1,1,N]` or history `[1,M,N]`), `DataSet<T>`→`Tensor<T>`,
`Tensor<T>`→`Tensor<T>` and `Tensor<T>`→`DataSet<T>`; all four registered for
`T = float`. With `history_stride` = 1: sliding window, 1:1 output, warm-up
passes inputs through unchanged. With `history_stride` S > 1: genuine S:1
decimation; warm-up is asymmetric here — chunks arriving before the window
first fills produce no output at all, not even a pass-through frame. For a
multi-output model, outputs 1..n−1 land on the optional `meta_out` ports
(only requested from ORT when connected). Overridable model initializers are
exposed both as an optional `config_in` port and a `model_overrides` entry;
precedence is `config_in` > `model_overrides` > model-baked default, and an
entry that cannot be applied (unknown name, size or type mismatch) goes
through the error policy rather than being silently ignored.

## Model contract (for model authors)

| Metadata key        | Example                   | Purpose                                                                 |
| ------------------- | ------------------------- | ----------------------------------------------------------------------- |
| `input_size`        | `"1024"`                  | primary input dimension N (`fft_size` accepted as fallback)             |
| `history_depth`     | `"16"`                    | M; omit or `"1"` for single-slice                                       |
| `n_max_peaks`       | `"8"`                     | K; falls back to the `peaks` output shape                               |
| `n_peak_properties` | `"16"`                    | P; falls back to the `peaks` output shape                               |
| `property_layout`   | `"peak_present,centre,…"` | column names; only the first 10 are validated                           |
| `score_output`      | `"peak_rescore"`          | acceptance-score output (default `peak_rescore`)                        |
| `normalise_mode`    | `"LogMAD"`                | auto-configure preprocessing; `"InGraph"` = model normalises internally |
| `normalise_expr`    | `"vecOut := vecIn"`       | ExprTk expression (mode = `Expression`)                                 |

Metadata is adopted only when the block's own setting is still at its default
(`normalise_mode = None`); an explicit setting always wins. A model that
normalises in-graph (`"InGraph"`) restores input units before emitting —
`amplitude`, `reconstruction`, `residual` come back on the caller's scale;
centre, sigma, eta, score and the heatmap are scale-free.

Normalisation modes: `None` (default, pass through), `LogMAD` (log10 →
robust z-score via median/MAD, clipped), `MinMax` (linear scale to `[0, 1]`),
`ZScore` (`(x - mean) / std`, clipped), `Expression` (`normalise_expr` ExprTk,
reads `vecIn` and pre-computed statistics, writes `vecOut`).

## Formats, build, tests

`.onnx` (protobuf) is portable/inspectable; `.ort` (flatbuffer) is
pre-optimised and the only format a minimal/WASM ORT build can load. Either
may be gzip-compressed (`.ort.gz`, `.onnx.gz`); `load()` inflates on read.
Most tracked fixtures under `models/` ship `.ort.gz` only (the WASM-required
form); the trained `ex05_*` deliverables ship both. Regenerate the analytic qa fixtures with
`python src/ex00_ex04_fixtures_gen.py [out_dir] [--only peaks heatmap_only ...]`
(requires TensorFlow/`tf2onnx`/`onnx`/`onnxruntime` locally; not a build
dependency, not run by CI).

```bash
cmake -DENABLE_ONNX_INTEGRATION=opt ..   # system ORT or bundled static — recommended
cmake -DENABLE_ONNX_INTEGRATION=on  ..   # build ORT from source (cross-compile, WASM)
cmake -DENABLE_ONNX_INTEGRATION=off ..   # disable entirely
ctest --test-dir build -R "qa_Onnx|qa_Peak" --output-on-failure -j6
```

## Peak detection

`OnnxPeakDetector` is the non-trivial consumer of the above: it requires named
outputs and a declared column layout, and performs NMS, gating and refinement
in-graph before anything leaves the model.

`OnnxPeakDetector` requires named outputs — the model performs NMS, gating
and refinement in-graph before the output leaves the model:

| Output           | Shape       | Required | Purpose                                                                     |
| ---------------- | ----------- | -------- | --------------------------------------------------------------------------- |
| `peaks`          | `[1, K, P]` | yes      | K peak candidates × P properties                                            |
| `peak_rescore`   | `[1, K]`    | no       | learned gate re-scorer; falls back to column 0 (`peak_present`) when absent |
| `heatmap`        | `[1, N]`    | no       | confidence curve                                                            |
| `reconstruction` | `[1, N]`    | no       | peak-sum spectrum                                                           |
| `residual`       | `[1, N]`    | no       | spectrum − reconstruction                                                   |

Peaks column layout, currently 16 columns (compiled in as `kPeaksLayoutPrefix`):

```
0 peak_present, 1 centre, 2 amplitude, 3 sigma_left, 4 sigma_right, 5 eta,
6 sigma_avg, 7 score, 8 type_tag, 9 is_top1, 10 local_snr, 11 prominence,
12 noise_sigma, 13 w68, 14 w96, 15 w99
```

The block requires P ≥ 10; only columns 0-9 are validated, and a
`property_layout` not starting with those ten names is rejected at load.
Further columns beyond the prefix are allowed. Columns 13-15 are
energy-containment widths of the model's own fitted asymmetric pseudo-Voigt
profile, computed in-graph, so the width definition ships with the model
rather than being a host-side constant; a model exported without them still
loads, falling back to Gaussian-equivalent multiples of `sigma_avg`.

Both detectors emit one `timing_events[0]` entry per peak with the same key
set (`confidence`, `centre`, `fwhm`, `hwhm_l`, `hwhm_r`, `amplitude`,
`prominence`, `noise_sigma`, `w68`, `w96`, `w99`, …) so a flow graph can swap
detectors without touching downstream code — except `confidence` itself:
`[0, 1]` score for `OnnxPeakDetector`, prominence in noise-sigma units
(routinely > 1) for `PeakDetector`. Retune any downstream threshold on swap.

## Examples

| Binary                  | Description                                                               |
| ----------------------- | ------------------------------------------------------------------------- |
| `ex01_elementwise`      | 1:1 use-case on `affine_N64` (y = 2x + 1)                                 |
| `ex02_decimating`       | N:1 use-case on `frame_mean_N64_M4`                                       |
| `ex03_sliding_window`   | history use-case on `frame_delta_N64_M8`                                  |
| `ex04_multi_output`     | `meta_out` wiring on `mean_rms_N64`                                       |
| `ex05_ml_peak_detector` | `OnnxPeakDetector`, single-stage vs 10-stage cascade, chart below         |
| `ex06_ml_vs_classical`  | detection-quality comparison, ML vs classical; `spectra=N` (default 2000) |

`ex06` is evaluated on `SyntheticPeakSpectrum`, which matches the ML model's
training distribution — its numbers are not an out-of-distribution measurement.

Captured run of `ex05_ml_peak_detector` on the cascade model. Braille dots are
the spectrum, `O` a designed peak, `★` a detection; the terminal renders the
three in colour, which is lost here. The `stage` column is the promoting stage,
read from `type_tag`:

```
   +5.5865 ┤cascaded (10 stages, in-graph) [a.u.]
           │
           │                O
           │                ★
           │                ⠠
           │
           │                ⠂
           │                ⠐
           │
  +3.57845 ┤                ⠂
           │                ⠠
           │                                               O
           │               ⢀                               ★⡂
           │                                              ⢀⠂⡁
           │                ⠐                             ⠐ ⢐
           │                                              ⢈ ⠐
           │               ⠈                              ⠄ ⢈
           │                 ⠂                            ⡁  ⠄
  +1.57039 ┤                                              ⠄  ⡂                             ⣀★
           │               ⠈                             ⢐   ⠄                            ⡰⠉⠈⢦
           │                 ⠁                           ⠐   ⢐                           ⢰⠃  ⠘⡆
           │               ⠐                             ⠨   ⢐                          ⢀⠇    ⠸⡄              ★
           │                 ⠁                           ⠅    ⡂                         ⡜      ⢣             ⢀⠂⠆
           │               ⠁ ⠄                          ⢀⠃    ⠆                        ⡼⠁       ⢧            ⠨ ⠆
           │               ⡁ ⠠                          ⡸     ⢱                      ⢀⡼⠁        ⠈⢧           ⠌ ⢘
           ├──────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────┐
           │                                                         512                                                      1024
  -0.43766 ┘ █: spectrum █: reference █: detected
  cascaded (10 stages, in-graph) — expected vs measured:
       pos     pos'    Δpos |     amp    amp'   Δamp% |   fwhm   hwhmL'/R'  Δfwhm% | stage
     150.3   151.18   +0.88 |    5.00    5.02   +0.3% |    9.4   5.8/4.0     +3.8% |     1
     420.7   421.67   +0.97 |    3.00    3.01   +0.5% |   28.3  15.5/13.5    +2.6% |     2
     700.0   701.19   +1.19 |    1.50    1.58   +5.1% |   58.9  32.6/29.8    +6.0% |     4
     860.5   861.10   +0.60 |    1.00    1.03   +2.8% |   14.1   8.8/7.1    +12.0% |     3
  summary: 4/4 reference peaks detected, 0 spurious
```
