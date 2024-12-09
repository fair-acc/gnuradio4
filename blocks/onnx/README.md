# ONNX integration — ML inference blocks for GNU Radio 4

Optional ONNX Runtime integration providing ML-based signal analysis blocks.
The primary use case is multi-peak detection in FFT spectra using a trained 1D U-Net model,
with a classical peak detector as a drop-in baseline for comparison.

Both detectors produce identical output formats (`DataSet<float>` with spectrum, heatmap/prominence,
and peak positions as `timing_events`), so they are interchangeable in any flow graph.

## TL;DR

```cpp
#include <gnuradio-4.0/onnx/OnnxPeakDetector.hpp>
#include <gnuradio-4.0/onnx/PeakDetector.hpp>

// ML peak detector (requires a trained .onnx or .ort model)
gr::blocks::onnx::OnnxPeakDetector mlDetector;
mlDetector.model_path           = "peak_detector_N1024.onnx";
mlDetector.confidence_threshold = 0.3f;
mlDetector.min_peak_distance    = 8;
mlDetector.max_peaks            = 8;
mlDetector.start();

gr::DataSet<float> spectrum = /* from FFT block */;
auto result = mlDetector.processOne(std::move(spectrum));
// result.signal_names    = {"Spectrum", "Heatmap"}
// result.timing_events[0] = detected peaks with confidence, sigma, amplitude, ...

// classical peak detector (no model needed)
gr::blocks::onnx::PeakDetector classical;
classical.prominence_threshold = 3.0f;  // in noise sigma
classical.min_peak_distance    = 5;
classical.max_peaks            = 8;

auto result2 = classical.processOne(std::move(spectrum));
// same output format: timing_events[0] = peaks with confidence, sigma, amplitude, ...
```

## Components

### Blocks

| Block              | Header                 | Purpose                                                                           |
|--------------------|------------------------|-----------------------------------------------------------------------------------|
| `OnnxInference`    | `OnnxInference.hpp`    | generic ONNX inference on `DataSet<float>` — runs any model, outputs raw results  |
| `OnnxPeakDetector` | `OnnxPeakDetector.hpp` | ML peak detection: resample → normalise → infer → NMS → annotated output          |
| `PeakDetector`     | `PeakDetector.hpp`     | classical peak detection: local maxima + prominence threshold, same output format |

### Utilities (not blocks)

| Component     | Header            | Purpose                                                                        |
|---------------|-------------------|--------------------------------------------------------------------------------|
| `OnnxSession` | `OnnxSession.hpp` | ONNX Runtime session lifecycle, model loading, metadata extraction             |
| `OnnxUtils`   | `OnnxUtils.hpp`   | `normalise()`, `resample()`, `extractPeaks()` — header-only, no ORT dependency |

### Test sources (in `blocks/testing/`)

| Block                      | Header                      | Purpose                                                                    |
|----------------------------|-----------------------------|----------------------------------------------------------------------------|
| `SyntheticPeakSpectrum<T>` | `SyntheticPeakSpectrum.hpp` | static spectra with random peaks (7 shapes), matching Python training data |
| `EvolvingPeakSpectrum<T>`  | `EvolvingPeakSpectrum.hpp`  | time-varying spectra with peaks that fade in/out and drift in frequency    |
| `TestSpectrumGenerator<T>` | `TestSpectrumGenerator.hpp` | beam-like spectra with Schottky peak, sweep, and morse-keyed interference  |

## Output format

Both `OnnxPeakDetector` and `PeakDetector` produce the same `DataSet<float>`:

```
signals:
  [0] "Spectrum"   — input spectrum (pass-through)
  [1] "Heatmap"    — ML confidence heatmap [0, 1]  (or "Prominence" for classical)

timing_events[0]:  (one entry per detected peak, sorted by confidence)
  { offset: bin_position,
    properties: {
      "confidence": float,   // detection confidence
      "sigma":      float,   // width in bins
      "amplitude":  float,   // relative amplitude
      "w68":        float,   // 68% energy containment width
      "w96":        float,   // 96% energy containment width
    }
  }
```

## Model I/O contract

The bundled 1D U-Net models expect:

- **Input:** `[batch, 1, N]` float32 — log-MAD normalised spectrum
- **Output:** `[batch, N + N*R]` float32 — first N = heatmap, remaining N*R = regression channels (R=8)
- **Metadata keys:** `fft_size`, `n_regression_channels`, `architecture`
- **Bundled models:** `peak_detector_N1024.{onnx,ort}` and `peak_detector_N4096.{onnx,ort}`

The block handles input size mismatches by resampling to the model's trained dimension.

## Build configuration

```bash
cmake -DENABLE_ONNX_INTEGRATION=opt ..   # system ORT (.so) or bundled static (.a)
cmake -DENABLE_ONNX_INTEGRATION=on  ..   # build ORT from source (for AdaptiveCpp/cross-compile)
cmake -DENABLE_ONNX_INTEGRATION=off ..   # disable entirely
```

See `ONNX_INSTALL.md` for detailed installation instructions.

## Examples

```bash
# side-by-side ML vs classical with dB spectrum chart and peak matching
cmake-build-debug-gcc-15/blocks/onnx/src/onnx_example0

# detection latency benchmark (200 evolving spectra, histogram)
cmake-build-debug-gcc-15/blocks/onnx/src/onnx_example1
```

## Training

The Python training script (`src/ex1_python/ex1_training.py`) trains the 1D U-Net from scratch:

```bash
# activate the venv with TF 2.20, Keras 3, onnxruntime, tf2onnx
source ~/venvs/ort-gpu/bin/activate

# train N=1024 model
cd blocks/onnx/src/ex1_python
PYTHONUNBUFFERED=1 python ex1_training.py --epochs 100 --fft-size 1024 --no-show --output-dir ../../models

# train N=4096 model (needs --train-samples 15000 to avoid OOM)
PYTHONUNBUFFERED=1 python ex1_training.py --epochs 100 --fft-size 4096 --batch-size 8 --train-samples 15000 --no-show --output-dir ../../models
```

## Tests

```bash
ctest --test-dir build -R "qa_Onnx|qa_PeakDetector" --output-on-failure -j6
```

| Test                      | Coverage                                                 |
|---------------------------|----------------------------------------------------------|
| `qa_OnnxInstallationTest` | ORT runtime capability verification                      |
| `qa_OnnxUtils`            | normalise, resample, extractPeaks with synthetic spectra |
| `qa_OnnxSession`          | model load, inference, error handling                    |
| `qa_OnnxInference`        | generic inference block (4 tests)                        |
| `qa_OnnxPeakDetector`     | ML peak detection block (6 tests)                        |
| `qa_PeakDetector`         | classical peak detection (7 tests)                       |

## Current model performance

Both models have sub-bin position accuracy but low recall (F1 well below the 0.80 target).
Position accuracy is the primary strength; detection rate needs more training data.

| Metric             | N=1024                                    | N=4096         |
|--------------------|-------------------------------------------|----------------|
| Epochs             | 58 (early stop)                           | 100 (full run) |
| F1 (threshold=0.4) | 0.540                                     | 0.446          |
| Precision          | 0.609                                     | 0.538          |
| Recall             | 0.485                                     | 0.380          |
| Position median    | 1.03 bins                                 | 0.98 bins      |
| Position 90th pct  | 6.83 bins                                 | 23.78 bins     |
| Model size         | 5.0 MB                                    | 5.1 MB         |
| Architecture       | 1D U-Net, 1.3M params, 3 skip connections | same           |

Detection latency benchmark (200 evolving spectra, seed=42, 29 ground-truth peaks):

| Metric              | Classical  | ML (ONNX)  |
|---------------------|------------|------------|
| Median latency      | 2 spectra  | 6 spectra  |
| 90th pct latency    | 16 spectra | 17 spectra |
| False positive rate | 60%        | 17.5%      |
| False negative rate | 7%         | 3.4%       |
| Position MAE        | 36 bins    | 27 bins    |

## Known issues and gotchas

1. **ORT ≤ 1.19 shape introspection crash:** `GetInputTypeInfo().GetShape()` crashes with
   `std::bad_alloc` on tf2onnx models. Worked around by reading FFT size from ONNX model
   metadata or filename convention (`peak_detector_N1024.onnx` → N=1024).
2. **N=4096 training OOM:** 50k samples × 4096 bins needs ~9 GB RAM — use
   `--train-samples 15000` to stay within 32 GB.
3. **Python stdout buffering:** always use `PYTHONUNBUFFERED=1` when running training in
   background, or output is invisible.
4. **tf2onnx numpy 2.x warning:** `np.cast` deprecation causes a non-fatal `ERROR:tf2onnx`
   log message during export. The ONNX file is still valid.
5. **SavedModel directories:** training creates `peak_detector_N{size}_saved/` directories —
   intermediate artefacts, safe to delete after export.
6. **Graph optimisation disabled:** `ORT_DISABLE_ALL` is set for maximum compatibility across
   ORT versions. Re-enable for production if targeting a specific ORT version.

## To do

### Model quality (high priority)

- [ ] improve F1 score (currently 0.54 for N=1024, 0.45 for N=4096 — target 0.80)
    - increase training data to 200k+ samples (currently 50k / 15k)
    - add data augmentation (spectral shift, noise level perturbation, SNR jitter)
    - tune focal loss alpha (currently 0.25, may under-weight positive examples)
    - N=4096 particularly data-starved (OOM forced 15k sample limit)
    - consider PyTorch rewrite for more flexible export (tf2onnx has fragility)

### Visual review (high priority)

- [ ] review ImChart output of `onnx_example0` and `onnx_example1` — verify detected peak
  markers (tags) are correctly drawn on top of the noisy spectrum in dB
- [ ] review ImChart output in `qa_OnnxPeakDetector` and `qa_PeakDetector` — verify tag overlay
  positions match ground-truth peaks visually
- [ ] consider adding chart output to `qa_OnnxSession` (inference result heatmap visualisation)

### Code quality (medium priority)

- [ ] add public accessor methods to `OnnxSession` — `isLoaded()`, `modelN()`,
  `regressionChannels()` are currently accessed via `_session` (private member) in
  `OnnxInference.hpp`, `OnnxPeakDetector.hpp`, `onnx_example0.cpp`, and tests
- [ ] extract `normalise()` clip bounds `[-5, 10]` to named constants (`kNormClipMin`,
  `kNormClipMax`)
- [ ] extract default regression channels (8) to a named constant in `OnnxSession.hpp`
- [ ] document `const_cast<float*>` safety in `OnnxSession::run()` (ORT C API requires
  non-const pointer but does not modify the data)
- [ ] remove or complete the empty `Onnx.hpp` stub block and `qa_Onnx.cpp` (dead code)
- [ ] fix AdaptiveCpp detection regex in CMakeLists.txt (`"acpp"` → `"acpp\\+\\+"`)
- [ ] replace hardcoded `re2` library search paths in CMake `on` mode with `find_library()`
- [ ] add bounds checks / clamp to `extractPeaks()` regression channel scaling

### Cross-platform verification (medium priority)

- [ ] verify Phase 2 additions (`EvolvingPeakSpectrum`, examples) compile on Clang 20 and
  Emscripten
- [ ] run full 16-configuration build matrix for Phase 2 changes
- [ ] verify WASM embedding works for new examples (model + test data in virtual filesystem)

### Detection latency (low priority)

- [ ] add `qa_DetectionLatency.cpp` unit test with deterministic assertions on max latency
- [ ] add `"peak_end"` tag emission when peaks finish decay (currently only `"peak_start"`)
- [ ] add exponential onset/decay ramp option (currently linear only)
- [ ] add `SyntheticPeakSpectrum` / `EvolvingPeakSpectrum` unit tests in `blocks/testing/test/`

### Future features (out of scope for this iteration)

- [ ] GPU execution providers (ROCm, CUDA) — CMake infrastructure exists, blocks use CPU only
- [ ] streaming/stateful models (RNNs)
- [ ] complex IQ input (multi-channel)
- [ ] quantised/pruned models for smaller footprint
- [ ] model download registry / auto-update
- [ ] dynamic batch inference (multiple spectra per call)
