#ifndef GR_ONNX_PEAK_DETECTOR_ONNX_HPP
#define GR_ONNX_PEAK_DETECTOR_ONNX_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/fourier/PeakResult.hpp>
#include <gnuradio-4.0/onnx/OnnxPreprocess.hpp>
#include <gnuradio-4.0/onnx/OnnxSession.hpp>

#include <format>
#include <limits>
#include <source_location>
#include <string_view>

namespace gr::blocks::onnx {

using gr::blocks::fourier::estimateIsolation;
using gr::blocks::fourier::estimateMagnitudeNoise;
using gr::blocks::fourier::estimateProminence;
using gr::blocks::fourier::estimateUncertainty;
using gr::blocks::fourier::NoiseEstimate;
using gr::blocks::fourier::PeakResult;

// a model may append columns but must not rename or reorder this prefix
inline constexpr std::string_view kPeaksLayoutPrefix = "peak_present,centre,amplitude,sigma_left,sigma_right,eta,sigma_avg,score,type_tag,is_top1";

// an empty declaration means the model carries no layout metadata; the compiled-in one is assumed
[[nodiscard]] inline bool matchesPeaksLayout(std::string_view declaredLayout) {
    if (declaredLayout.empty()) {
        return true; // model declares no layout: the compiled-in one is assumed
    }
    if (!declaredLayout.starts_with(kPeaksLayoutPrefix)) {
        return false;
    }
    const auto tail = declaredLayout.substr(kPeaksLayoutPrefix.size());
    return tail.empty() || tail.starts_with(','); // a column boundary, not a longer name
}

// Peaks-tensor columns:
//   0 peak_present, 1 centre, 2 amplitude, 3 sigma_left, 4 sigma_right,
//   5 eta, 6 sigma_avg, 7 score, 8 type_tag, 9 is_top1, 10 local_snr, 11 prominence,
//   12 noise_sigma, 13 w68, 14 w96, 15 w99
// The containment widths are computed in-graph from the model's own fitted profile, so the width
// definition ships with the model. Models exported before that column existed fall back to the
// Gaussian-equivalent multiples below, which are only correct for a Gaussian peak.
// No host-side NMS: the graph has already decided which peaks to keep, and a second pass here
// would re-suppress them.
[[nodiscard]] inline std::vector<PeakResult> extractPeaksRegressor(std::span<const float> peakTensor, std::size_t nMaxPeaks, std::size_t nProperties, float gateThreshold = 0.5f, std::span<const float> rescore = {}, std::span<const float> spectrum = {}, NoiseEstimate noise = {0.f, 1.f}, std::size_t maxPeaks = std::numeric_limits<std::size_t>::max()) {
    // nMaxPeaks and nProperties both come from model metadata and are unbounded, so their product
    // can wrap and let a short tensor through; dividing cannot overflow
    if (nProperties < 10 || nMaxPeaks > peakTensor.size() / nProperties) {
        return {};
    }

    std::vector<PeakResult> peaks;
    peaks.reserve(nMaxPeaks);

    for (std::size_t i = 0; i < nMaxPeaks; ++i) {
        const float* p     = peakTensor.data() + i * nProperties;
        const float  score = (rescore.size() > i) ? rescore[i] : p[0];
        if (!(score >= gateThreshold)) { // NaN must not pass: it compares false either way
            continue;
        }
        // a model exported before the containment columns still reports its fitted flanks and eta,
        // so the host measures the same quantity rather than assuming the peak is Gaussian
        const gr::blocks::fourier::ContainmentWidths widths = nProperties >= 16                                                 //
                                                                  ? gr::blocks::fourier::ContainmentWidths{p[13], p[14], p[15]} //
                                                                  : gr::blocks::fourier::containmentWidths(p[3], p[4], p[5]);
        PeakResult                                   peak{
                                              .confidence = score,
                                              .centre     = p[1],
                                              .amplitude  = p[2],
                                              .hwhmLeft   = p[3] / gr::blocks::fourier::kGaussianHalfMaxToSigma,
                                              .hwhmRight  = p[4] / gr::blocks::fourier::kGaussianHalfMaxToSigma,
                                              .w68        = widths.w68,
                                              .w96        = widths.w96,
                                              .w99        = widths.w99,
                                              .kurtosis   = 5.f * p[5],
                                              .typeTag    = p[8],
        };

        if (!spectrum.empty()) {
            auto n                          = spectrum.size();
            auto rawIdx                     = static_cast<std::size_t>(std::clamp(std::lround(peak.centre), 0L, static_cast<long>(n - 1)));
            peak.amplitudeMeasured          = spectrum[rawIdx] - noise.median;
            peak.isolation                  = estimateIsolation(spectrum, rawIdx);
            peak.prominence                 = estimateProminence(spectrum, rawIdx) / (noise.sigma + 1e-10f);
            auto [posUnc, widthUnc, ampUnc] = estimateUncertainty(noise.sigma, std::max(peak.amplitudeMeasured, 0.f), peak.gaussianEquivalentSigma());
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

struct OnnxPeakDetector : gr::Block<OnnxPeakDetector> {
    using Description = Doc<R""(ML peak detection for models that emit a fully post-processed peak set.

Drop-in replacement for the classical PeakDetector: same DataSet<float> in/out ports, signal 0
is the pass-through "Spectrum", and timing_events[0] carries the same per-peak property keys.
The model performs NMS, sigma-dedup, gating and Gauss-Newton refinement in-graph; a "peaks"
output is required, "heatmap"/"reconstruction"/"residual" are consumed when present. K, P, the
acceptance-score output and the required normalisation come from the model's ONNX metadata;
only the peaks column layout is compiled in, and an incompatible one is rejected at load.

Temporal models (metadata history_depth M > 1) run from an internal frame history: the first
M-1 spectra pass through unchanged, then every input triggers one inference over the newest M
frames, keeping the 1:1 in/out contract. M comes from the model, not a user setting.

Pass-through frames are marked meta_information[0]["onnx_passthrough"]. Any model or inference
error stops the block rather than silently reverting to pass-through, since "no peaks" would
otherwise be indistinguishable from "no peaks found".

Published uncertainties (position, width, amplitude only) come from the classical closed-form
estimator, not the model — an achievable floor, not this model's realised scatter. Confidence
and asymmetry carry none, since no calibrated estimate exists.)"">;

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

    Annotated<gr::Size_t, "intra-op threads", Doc<"ONNX Runtime threads per operator; 1 is deterministic, 4 is ~2.5x faster">> intra_op_threads = 4U;

    Annotated<gr::property_map, "model overrides", Doc<"overridable-initializer values by name (e.g. nms_factor)">> model_overrides;

    Annotated<std::vector<gr::Size_t>, "model input shape", Doc<"read-only: populated after model load">>                                model_input_shape;
    Annotated<std::vector<gr::Size_t>, "model output shape", Doc<"read-only: populated after model load">>                               model_output_shape;
    Annotated<std::string, "peaks layout", Doc<"read-only: peaks-tensor column names from model metadata">>                              peaks_layout;
    Annotated<std::vector<std::string>, "available providers", Doc<"read-only: execution providers offered by the linked ONNX Runtime">> available_providers;

    GR_MAKE_REFLECTABLE(OnnxPeakDetector, in, out, model_path, execution_provider, resample_mode, normalise_mode, clip_min, clip_max, normalise_expr, gate_threshold, max_peaks, intra_op_threads, model_overrides, model_input_shape, model_output_shape, peaks_layout, available_providers);

    OnnxSession           _session;
    OnnxPreprocess<float> _preprocess;

    void start() {
        if (model_path.value.empty()) {
            handleError("start()", "model_path is empty");
            return;
        }
        loadModel();
        configurePreprocess();
    }

    void stop() { _session.reset(); }

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

        std::vector<float> normalised(modelN);
        _preprocess.normalise(modelInput, normalised);

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
        const NoiseEstimate    noise = estimateMagnitudeNoise(rawSpectrum);
        auto                   peaks = extractPeaksRegressor(peakTensor->values, nMaxPeaks, nProps, gate_threshold, rescore, rawSpectrum, noise, max_peaks);

        return buildOutput(inData, *result, heatmap, peaks, noise, inputSize, modelN);
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& newSettings) {
        if (newSettings.contains("model_path") || newSettings.contains("execution_provider") || newSettings.contains("intra_op_threads")) {
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

        // OnnxPreprocess::resample maps model bin i to input bin i*(inputSize-1)/(modelN-1), so the inverse must
        // preserve the same endpoints; inputSize/modelN would bias every position towards zero, growing with the bin index
        const bool  isIdentityMapping = inputSize == modelN || inputSize == 0UZ || modelN <= 1UZ;
        const float posScale          = isIdentityMapping ? 1.f : static_cast<float>(inputSize - 1UZ) / static_cast<float>(modelN - 1UZ);

        std::vector<gr::DataSet<float>::idx_pmt_map> peakEvents;
        peakEvents.reserve(peaks.size());
        for (const auto& p : peaks) {
            gr::property_map props = gr::blocks::fourier::peakEventProps(p, noise, posScale);
            // inputSize == 0 would make the clamp's upper bound precede its lower bound, which is UB
            const long lastBin = inputSize == 0UZ ? 0L : static_cast<long>(inputSize) - 1L;
            peakEvents.emplace_back(std::clamp<long>(std::lround(p.centre * posScale), 0L, lastBin), std::move(props));
        }
        output.timing_events = {std::move(peakEvents), {}, {}, {}};

        return output;
    }

    void loadModel() {
        available_providers.value = OnnxSession::availableProviders(); // before the empty-path return: "which providers do I have?" is asked before a model is chosen
        if (model_path.value.empty()) {
            return;
        }
        _session.setExecutionProvider(execution_provider);
        _session.setIntraOpThreads(static_cast<std::size_t>(intra_op_threads));
        if (auto result = _session.load(model_path); !result) {
            handleError("loadModel()", result.error().message);
            return;
        }
        model_input_shape.value  = _session.modelInputShape<gr::Size_t>();
        model_output_shape.value = _session.modelOutputShape<gr::Size_t>();
        peaks_layout.value       = _session.metadata().propertyLayout;
        if (const auto& meta = _session.metadata(); meta.nPeakProperties > 0UZ && meta.nPeakProperties < 10UZ) {
            handleError("loadModel()", std::format("model declares {} peak properties; the extractor reads columns 0-9", meta.nPeakProperties));
            _session.reset();
            return;
        }
        if (!matchesPeaksLayout(peaks_layout.value)) {
            handleError("loadModel()", std::format("model property_layout '{}' does not start with the compiled-in layout '{}'", peaks_layout.value, kPeaksLayoutPrefix));
            _session.reset();
            return;
        }
        // metadata fills in what the caller left at its default; an explicit setting always wins,
        // otherwise loading a re-exported model would silently retarget the caller's normalisation
        adoptDeclaredNormalisation(_session.metadata(), normalise_mode, normalise_expr);
        configurePreprocess();
    }

    void configurePreprocess() {
        if (auto r = _preprocess.configure(normalise_mode, normalise_expr, clip_min, clip_max); !r) {
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
