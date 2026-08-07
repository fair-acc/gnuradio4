#ifndef GR_ONNX_PEAK_DETECTOR_ONNX_HPP
#define GR_ONNX_PEAK_DETECTOR_ONNX_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/fourier/PeakResult.hpp>
#include <gnuradio-4.0/onnx/OnnxPreprocess.hpp>
#include <gnuradio-4.0/onnx/OnnxSession.hpp>

#include <deque>
#include <format>
#include <limits>
#include <source_location>
#include <string_view>

namespace gr::blocks::onnx {

// PeakResult and its estimator free functions moved to gr::blocks::fft (blocks/fourier) as
// classical spectrum-domain utilities shared by the classical and ML peak detectors; pulled in
// unqualified here rather than at every use site below
using gr::blocks::fourier::estimateIsolation;
using gr::blocks::fourier::estimateNoise;
using gr::blocks::fourier::estimateProminence;
using gr::blocks::fourier::estimateUncertainty;
using gr::blocks::fourier::NoiseEstimate;
using gr::blocks::fourier::PeakResult;

// compiled-in column order of the leading peaks-tensor columns consumed below; a model may
// append further columns (score, type_tag, ...) but must not rename or reorder this prefix
inline constexpr std::string_view kPeaksLayoutPrefix = "peak_present,centre,amplitude,sigma_left,sigma_right,eta,sigma_avg";

// empty = metadata absent (e.g. minimal builds cannot read it) — the compiled-in layout is assumed
[[nodiscard]] inline bool matchesPeaksLayout(std::string_view declaredLayout) { return declaredLayout.empty() || declaredLayout.starts_with(kPeaksLayoutPrefix); }

// scale-free asymmetry delta = 1/2 log(sigma_R/sigma_L); 0 when either flank is non-positive
[[nodiscard]] inline float asymmetryOf(const PeakResult& peak) noexcept {
    if (peak.sigmaLeft <= 0.f || peak.sigmaRight <= 0.f) {
        return 0.f;
    }
    return 0.5f * std::log(peak.sigmaRight / peak.sigmaLeft);
}

// Peaks-tensor extraction: the model emits a fully post-processed peak set (NMS, σ-dedup,
// gate and Gauss-Newton refinement are all in-graph), so this only reshapes
// [nMaxPeaks × nProperties], thresholds, and derives the post-hoc spectrum statistics.
// No host-side NMS — applying one here would re-suppress peaks the graph decided to keep.
//
// The column layout is compiled in (kPeaksLayoutPrefix); a loaded model's declared
// `property_layout` metadata is validated against it at load time:
//   0 peak_present, 1 centre, 2 amplitude, 3 sigma_left, 4 sigma_right,
//   5 eta, 6 sigma_avg, 7 score, 8 type_tag, 9 is_top1, 10-12 reserved
//
// `rescore` is the optional `peak_rescore` output (the learned gate re-scorer); when
// present it supersedes column 0 as the acceptance score. `maxPeaks` caps the result to
// the highest-scoring peaks.
[[nodiscard]] inline std::vector<PeakResult> extractPeaksRegressor(std::span<const float> peakTensor, std::size_t nMaxPeaks, std::size_t nProperties, float gateThreshold = 0.5f, std::span<const float> rescore = {}, std::span<const float> spectrum = {}, NoiseEstimate noise = {0.f, 1.f}, std::size_t maxPeaks = std::numeric_limits<std::size_t>::max()) {

    if (peakTensor.size() < nMaxPeaks * nProperties || nProperties < 10) {
        return {};
    }

    std::vector<PeakResult> peaks;
    peaks.reserve(nMaxPeaks);

    for (std::size_t i = 0; i < nMaxPeaks; ++i) {
        const float* p     = peakTensor.data() + i * nProperties;
        const float  score = (rescore.size() > i) ? rescore[i] : p[0];
        if (score < gateThreshold) {
            continue;
        }

        const float sigmaAvg = p[6];
        PeakResult  peak{
             .confidence = score,
             .centre     = p[1],
             .amplitude  = p[2],
             .sigmaLeft  = p[3],
             .sigmaRight = p[4],
            // energy-containment widths follow the same convention as detectPeaksIterative
             .w68 = sigmaAvg * 2.f,
             .w96 = sigmaAvg * 4.f,
             .w99 = sigmaAvg * 6.f,
            // eta interpolates Gaussian↔Lorentzian; the model's own kurt = 5·eta mapping
            // lets subtractPeak's Auto shape selection reproduce the fitted shape
             .kurtosis = 5.f * p[5],
        };

        // post-hoc spectrum statistics matching the classical PeakDetector key set
        if (!spectrum.empty()) {
            auto n                          = spectrum.size();
            auto rawIdx                     = static_cast<std::size_t>(std::clamp(std::lround(peak.centre), 0L, static_cast<long>(n - 1)));
            peak.amplitudeMeasured          = spectrum[rawIdx] - noise.median;
            peak.isolation                  = estimateIsolation(spectrum, rawIdx);
            peak.prominence                 = estimateProminence(spectrum, rawIdx) / (noise.sigma + 1e-10f);
            auto [posUnc, widthUnc, ampUnc] = estimateUncertainty(noise.sigma, std::max(peak.amplitudeMeasured, 0.f), peak.sigma());
            peak.positionUncertainty        = posUnc;
            peak.widthUncertainty           = widthUnc;
            peak.amplitudeUncertainty       = ampUnc;
        }

        peaks.push_back(peak);
    }

    if (peaks.size() > maxPeaks) {
        std::ranges::nth_element(peaks, peaks.begin() + static_cast<std::ptrdiff_t>(maxPeaks), [](const PeakResult& a, const PeakResult& b) { return a.confidence > b.confidence; });
        peaks.resize(maxPeaks);
    }

    std::ranges::sort(peaks, [](const PeakResult& a, const PeakResult& b) { return a.centre < b.centre; });

    return peaks;
}

GR_REGISTER_BLOCK(gr::blocks::onnx::OnnxPeakDetector)

/**
 * @brief ML peak detection for models that emit a fully post-processed peak set.
 *
 * Drop-in replacement for the classical PeakDetector: same in/out DataSet<float> ports,
 * signal 0 is the pass-through "Spectrum", and timing_events[0] carries the same
 * per-peak property keys. A peaks-tensor model (UnifiedPeakRegressor) performs NMS,
 * sigma-dedup, gating and Gauss-Newton refinement inside the graph and emits:
 *
 *   spectra[1, M, N] -> heatmap[1, N], peaks[1, K, P], peak_rescore[1, K] (optional),
 *                       reconstruction[1, N], residual[1, N]
 *
 * The model is largely self-describing: K, P, which output carries the acceptance score,
 * and the required input normalisation (`normalise_mode`, for Expression also
 * `normalise_expr`) come from its ONNX custom metadata. The peaks column layout itself is
 * compiled in (kPeaksLayoutPrefix); a model declaring a different `property_layout` is
 * rejected at load time instead of being silently misparsed. There is deliberately no
 * host-side NMS here — suppressing peaks the graph already decided to keep would
 * double-suppress.
 *
 * Temporal models (model metadata `history_depth` M > 1, shape fallback) are driven from
 * an internal frame history: while the buffer fills, the first M-1 input spectra pass
 * through unchanged (one signal, no peak events — same as the no-model pass-through).
 * This warm-up pass-through is the module-wide contract, shared with OnnxInference's
 * history mode. From the M-th sample on every input runs one inference over the newest
 * M frames (sliding window, stride 1, newest frame in row M-1), keeping the 1:1 in/out
 * contract. M comes from the model, not from a user setting.
 *
 * Output DataSet carries four signals (Spectrum, Heatmap, Reconstruction, Residual) and
 * one timing_events entry per accepted peak: confidence, centre, sigma/sigma_left/sigma_right,
 * asymmetry, amplitude and the post-hoc spectrum statistics.
 *
 * Uncertainties cover THREE of the five estimands -- position, width and amplitude. Confidence and
 * asymmetry carry none, deliberately: no calibrated estimate exists for them, and an invented column
 * would be indistinguishable from a measured one. The three that are published come from the
 * classical closed form estimateUncertainty(), NOT from the model, so they describe the achievable
 * bound rather than this estimator's realised scatter -- treat them as a floor, not a calibration.
 *
 * Frames forwarded unchanged (no model loaded,
 * warm-up) carry meta_information[0]["onnx_passthrough"] = true. Any model or inference
 * error stops the block outright — a peak detector that silently reverted to pass-through
 * would report "no peaks" indistinguishably from "no peaks found".
 */
struct OnnxPeakDetector : gr::Block<OnnxPeakDetector> {
    using Description = Doc<"Detects peaks using an ONNX model that emits an in-graph post-processed peak set.">;

    gr::PortIn<gr::DataSet<float>>  in;
    gr::PortOut<gr::DataSet<float>> out;

    Annotated<std::string, "model path">                                                                                                                                 model_path         = "";
    Annotated<std::string, "execution provider", Doc<"cpu | cuda | tensorrt | rocm — validated against the linked ONNX Runtime at model load, never a silent fallback">> execution_provider = "cpu";

    Annotated<ResampleMode, "resample mode">       resample_mode  = ResampleMode::Linear;
    Annotated<NormaliseMode, "normalise mode">     normalise_mode = NormaliseMode::None;
    Annotated<float, "clip min">                   clip_min       = -5.f;
    Annotated<float, "clip max">                   clip_max       = 10.f;
    Annotated<std::string, "normalise expression"> normalise_expr = "";

    Annotated<float, "gate threshold", Doc<"accept a peak when its score reaches this; the model bakes no threshold">> gate_threshold = 0.5f;
    Annotated<gr::Size_t, "max peaks", Doc<"host-side cap: keep the highest-scoring peaks">>                           max_peaks      = 8U;

    Annotated<gr::property_map, "model overrides", Doc<"overridable-initializer values by name (e.g. nms_factor)">> model_overrides;

    Annotated<std::vector<gr::Size_t>, "model I/O shape", Doc<"read-only: populated after model load">>                                  model_io_shape;
    Annotated<std::string, "peaks layout", Doc<"read-only: peaks-tensor column names from model metadata">>                              peaks_layout;
    Annotated<std::vector<std::string>, "available providers", Doc<"read-only: execution providers offered by the linked ONNX Runtime">> available_providers;

    GR_MAKE_REFLECTABLE(OnnxPeakDetector, in, out, model_path, execution_provider, resample_mode, normalise_mode, clip_min, clip_max, normalise_expr, gate_threshold, max_peaks, model_overrides, model_io_shape, peaks_layout, available_providers);

    OnnxSession                    _session;
    OnnxPreprocess<float>          _preprocess;
    std::deque<std::vector<float>> _history; // resampled raw frames for temporal (M > 1) models

    void start() {
        if (model_path.value.empty()) {
            handleError("start()", "model_path is empty");
            return;
        }
        loadModel();
        configurePreprocess();
    }

    void stop() {
        _session.reset();
        _history.clear();
    }

    // not noexcept: the pipeline allocates and bad_alloc must propagate to the framework
    [[nodiscard]] gr::DataSet<float> processOne(gr::DataSet<float> inData) {
        if (!_session.isLoaded() || inData.signal_values.empty()) {
            markPassthrough(inData);
            return inData;
        }

        const std::size_t modelN    = _session.modelN();
        const std::size_t nSignals  = std::max(1UZ, inData.signal_names.size());
        const std::size_t inputSize = inData.signal_values.size() / nSignals;

        std::vector<float> modelInput(modelN);
        if (inputSize == modelN || resample_mode == ResampleMode::None) {
            std::copy_n(inData.signal_values.begin(), std::min(inputSize, modelN), modelInput.begin());
        } else {
            std::span<const float> firstSignal(inData.signal_values.data(), inputSize);
            OnnxPreprocess<float>::resample(firstSignal, modelInput);
        }

        const std::size_t historyDepth = _session.historyDepth();
        if (historyDepth > 1UZ) {
            _history.push_back(modelInput);
            if (_history.size() < historyDepth) {
                markPassthrough(inData);
                return inData; // buffer still filling — spectrum passes through without peak annotation
            }
        }

        std::vector<float> normalised(historyDepth * modelN);
        if (historyDepth > 1UZ) {
            for (std::size_t row = 0UZ; row < historyDepth; ++row) {
                _preprocess.normalise(_history[row], std::span<float>(normalised.data() + row * modelN, modelN));
            }
            _history.pop_front(); // sliding window, stride 1: inference fires on every sample once warm
        } else {
            _preprocess.normalise(modelInput, normalised);
        }

        auto                           noHigherPrecedence = [](std::size_t, std::size_t) -> std::optional<std::vector<float>> { return std::nullopt; };        // no config_in ports here: the map is the only source
        const std::vector<NamedTensor> extras             = buildOverrideInputs(model_overrides.value, _session.overridableInitializers(), noHigherPrecedence, //
                        [this](std::string message) { handleError("processOne()", message); });

        auto result = _session.runAll(normalised, extras);
        if (!result) {
            handleError("processOne()", result.error().message);
            markPassthrough(inData);
            return inData;
        }

        const auto* peakTensor = findOutput(*result, "peaks");
        if (peakTensor == nullptr) {
            handleError("processOne()", "model has no 'peaks' output — OnnxPeakDetector requires the peaks-tensor contract");
            markPassthrough(inData);
            return inData;
        }

        const auto& meta        = _session.metadata();
        const auto  nMaxPeaks   = meta.nMaxPeaks != 0 ? meta.nMaxPeaks : (peakTensor->shape.size() >= 2 ? peakTensor->shape[peakTensor->shape.size() - 2] : 0UZ);
        const auto  nProps      = meta.nPeakProperties != 0 ? meta.nPeakProperties : (peakTensor->shape.empty() ? 0UZ : peakTensor->shape.back());
        const auto* scoreTensor = findOutput(*result, meta.scoreOutput.empty() ? "peak_rescore" : meta.scoreOutput);
        const auto* heatmap     = findOutput(*result, "heatmap");

        std::span<const float> rescore;
        if (scoreTensor != nullptr && scoreTensor->values.size() >= nMaxPeaks) {
            rescore = std::span<const float>(scoreTensor->values);
        }
        std::span<const float> rawSpectrum(modelInput);
        const NoiseEstimate    noise = estimateNoise(rawSpectrum);
        auto                   peaks = extractPeaksRegressor(peakTensor->values, nMaxPeaks, nProps, gate_threshold, rescore, rawSpectrum, noise, max_peaks);

        return buildOutput(inData, *result, heatmap, peaks, noise, inputSize, modelN);
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& newSettings) {
        if (newSettings.contains("model_path") || newSettings.contains("execution_provider")) {
            loadModel();
        }
        if (newSettings.contains("normalise_mode") || newSettings.contains("normalise_expr") || newSettings.contains("clip_min") || newSettings.contains("clip_max") || newSettings.contains("resample_mode")) {
            configurePreprocess();
        }
    }

    [[nodiscard]] bool isModelLoaded() const noexcept { return _session.isLoaded(); }

private:
    [[nodiscard]] static const NamedTensor* findOutput(const std::vector<NamedTensor>& outputs, std::string_view name) {
        auto it = std::ranges::find(outputs, name, &NamedTensor::name);
        return it != outputs.end() ? std::to_address(it) : nullptr;
    }

    // model-domain signal resampled back to the input grid, or zeros when absent
    [[nodiscard]] static std::vector<float> toInputGrid(const NamedTensor* tensor, std::size_t inputSize, std::size_t modelN, ResampleMode mode) {
        std::vector<float> scaled(inputSize, 0.f);
        if (tensor == nullptr || tensor->values.size() < modelN) {
            return scaled;
        }
        std::span<const float> source(tensor->values.data(), modelN);
        if (inputSize != modelN && mode != ResampleMode::None) {
            OnnxPreprocess<float>::resample(source, scaled);
        } else {
            std::copy_n(source.begin(), std::min(inputSize, modelN), scaled.begin());
        }
        return scaled;
    }

    [[nodiscard]] gr::DataSet<float> buildOutput(const gr::DataSet<float>& inData, const std::vector<NamedTensor>& outputs, const NamedTensor* heatmap, const std::vector<PeakResult>& peaks, NoiseEstimate noise, std::size_t inputSize, std::size_t modelN) const {
        const auto heatmapOut = toInputGrid(heatmap, inputSize, modelN, resample_mode);
        const auto reconOut   = toInputGrid(findOutput(outputs, "reconstruction"), inputSize, modelN, resample_mode);
        const auto residOut   = toInputGrid(findOutput(outputs, "residual"), inputSize, modelN, resample_mode);

        gr::DataSet<float> output;
        output.timestamp   = inData.timestamp;
        output.axis_names  = inData.axis_names;
        output.axis_units  = inData.axis_units;
        output.axis_values = inData.axis_values;

        const std::string inputUnit = inData.signal_units.empty() ? "" : inData.signal_units[0];
        output.signal_names         = {"Spectrum", "Heatmap", "Reconstruction", "Residual"};
        output.signal_quantities    = {"", "", "", ""};
        output.signal_units         = {inputUnit, "", inputUnit, inputUnit};
        output.signal_ranges        = {gr::Range<float>{0.f, 0.f}, gr::Range<float>{0.f, 1.f}, gr::Range<float>{0.f, 0.f}, gr::Range<float>{0.f, 0.f}};
        output.extents              = {static_cast<std::int32_t>(inputSize)};

        output.signal_values.resize(4 * inputSize);
        auto dest = output.signal_values.begin();
        std::copy_n(inData.signal_values.begin(), inputSize, dest);
        dest = std::copy(heatmapOut.begin(), heatmapOut.end(), dest + static_cast<std::ptrdiff_t>(inputSize));
        dest = std::copy(reconOut.begin(), reconOut.end(), dest);
        std::copy(residOut.begin(), residOut.end(), dest);

        output.meta_information = {{}, {}, {}, {}};

        const float posScale = (inputSize != modelN) ? static_cast<float>(inputSize) / static_cast<float>(modelN) : 1.f;

        std::vector<gr::DataSet<float>::idx_pmt_map> peakEvents;
        peakEvents.reserve(peaks.size());
        for (const auto& p : peaks) {
            gr::property_map props{
                {std::pmr::string("confidence"), gr::pmt::Value(p.confidence)},
                // the timing_events index is an integer bin, so the fractional centre — the
                // whole point of the sub-bin position head — must be carried separately or
                // position accuracy is silently truncated to +-0.5 bin
                {std::pmr::string("centre"), gr::pmt::Value(p.centre * posScale)},
                {std::pmr::string("sigma"), gr::pmt::Value(p.sigma() * posScale)},
                {std::pmr::string("sigma_left"), gr::pmt::Value(p.sigmaLeft * posScale)},
                {std::pmr::string("sigma_right"), gr::pmt::Value(p.sigmaRight * posScale)},
                // delta = 1/2 log(sigma_R/sigma_L): the scale-free asymmetry the estimator is trained
                // against, published so consumers need not re-derive it. 0 when either flank is degenerate.
                {std::pmr::string("asymmetry"), gr::pmt::Value(asymmetryOf(p))},
                {std::pmr::string("amplitude"), gr::pmt::Value(p.amplitude)},
                {std::pmr::string("amplitude_measured"), gr::pmt::Value(p.amplitudeMeasured)},
                {std::pmr::string("prominence"), gr::pmt::Value(p.prominence)},
                {std::pmr::string("isolation"), gr::pmt::Value(p.isolation * posScale)},
                {std::pmr::string("w68"), gr::pmt::Value(p.w68 * posScale)},
                {std::pmr::string("w96"), gr::pmt::Value(p.w96 * posScale)},
                {std::pmr::string("w99"), gr::pmt::Value(p.w99 * posScale)},
                {std::pmr::string("kurtosis"), gr::pmt::Value(p.kurtosis)},
                {std::pmr::string("noise_sigma"), gr::pmt::Value(noise.sigma)},
                {std::pmr::string("noise_floor"), gr::pmt::Value(noise.median)},
                {std::pmr::string("position_uncertainty"), gr::pmt::Value(p.positionUncertainty * posScale)},
                {std::pmr::string("width_uncertainty"), gr::pmt::Value(p.widthUncertainty)},
                {std::pmr::string("amplitude_uncertainty"), gr::pmt::Value(p.amplitudeUncertainty)},
            };
            peakEvents.emplace_back(std::lround(p.centre * posScale), std::move(props));
        }
        output.timing_events = {std::move(peakEvents), {}, {}, {}};

        return output;
    }

    void loadModel() {
        _history.clear();
        available_providers.value = OnnxSession::availableProviders(); // before the empty-path return: "which providers do I have?" is asked before a model is chosen
        if (model_path.value.empty()) {
            return;
        }
        _session.setExecutionProvider(execution_provider);
        if (auto result = _session.load(model_path); !result) {
            handleError("loadModel()", result.error().message);
            return;
        }
        model_io_shape.value = _session.modelIoShape<gr::Size_t>();
        peaks_layout.value   = _session.metadata().propertyLayout;
        if (!matchesPeaksLayout(peaks_layout.value)) {
            handleError("loadModel()", std::format("model property_layout '{}' does not start with the compiled-in layout '{}'", peaks_layout.value, kPeaksLayoutPrefix));
            return;
        }
        if (auto mode = normaliseModeFromString(_session.metadata().normaliseMode)) {
            normalise_mode = *mode;
            if (*mode == NormaliseMode::Expression && !_session.metadata().normaliseExpr.empty()) {
                normalise_expr = _session.metadata().normaliseExpr;
            }
        }
        configurePreprocess();
    }

    void configurePreprocess() {
        if (auto r = _preprocess.configure(resample_mode, normalise_mode, normalise_expr, clip_min, clip_max); !r) {
            handleError("configurePreprocess()", r.error().message);
        }
    }

    void handleError(std::string_view context, std::string_view message, std::source_location location = std::source_location::current()) {
        this->emitErrorMessage(context, gr::Error{message, location});
        this->requestStop();
    }
};

} // namespace gr::blocks::onnx

#endif // GR_ONNX_PEAK_DETECTOR_ONNX_HPP
