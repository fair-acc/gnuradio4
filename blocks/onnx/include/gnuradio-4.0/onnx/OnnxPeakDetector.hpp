#ifndef GR_ONNX_PEAK_DETECTOR_ONNX_HPP
#define GR_ONNX_PEAK_DETECTOR_ONNX_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/onnx/OnnxPreprocess.hpp>
#include <gnuradio-4.0/onnx/OnnxSession.hpp>
#include <gnuradio-4.0/onnx/PeakExtraction.hpp>

namespace gr::blocks::onnx {

struct OnnxPeakDetector : gr::Block<OnnxPeakDetector> {
    using Description = Doc<"Detects peaks in spectra using a trained ONNX model with heatmap NMS.">;

    gr::PortIn<gr::DataSet<float>>  in;
    gr::PortOut<gr::DataSet<float>> out;

    // --- model settings ---
    Annotated<std::string, "model path">   model_path   = "";
    Annotated<ErrorPolicy, "error policy"> error_policy = ErrorPolicy::Stop;

    // --- preprocessing ---
    Annotated<ResampleMode, "resample mode">       resample_mode  = ResampleMode::Linear;
    Annotated<NormaliseMode, "normalise mode">     normalise_mode = NormaliseMode::LogMAD;
    Annotated<float, "clip min">                   clip_min       = -5.f;
    Annotated<float, "clip max">                   clip_max       = 10.f;
    Annotated<std::string, "normalise expression"> normalise_expr = "";

    // --- peak extraction ---
    Annotated<float, "confidence threshold">   confidence_threshold = 0.4f;
    Annotated<gr::Size_t, "min peak distance"> min_peak_distance    = 8U;
    Annotated<gr::Size_t, "max peaks">         max_peaks            = 8U;

    // --- read-only info ---
    Annotated<std::vector<gr::Size_t>, "model I/O shape", Doc<"read-only: populated after model load">> model_io_shape;

    GR_MAKE_REFLECTABLE(OnnxPeakDetector, in, out, model_path, error_policy, resample_mode, normalise_mode, clip_min, clip_max, normalise_expr, confidence_threshold, min_peak_distance, max_peaks, model_io_shape);

    OnnxSession           _session;
    OnnxPreprocess<float> _preprocess;
    bool                  _passthrough = false;

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
        _passthrough = false;
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& newSettings) {
        if (newSettings.contains("model_path")) {
            loadModel();
        }
        if (newSettings.contains("normalise_mode") || newSettings.contains("normalise_expr") || newSettings.contains("clip_min") || newSettings.contains("clip_max") || newSettings.contains("resample_mode")) {
            configurePreprocess();
        }
    }

    [[nodiscard]] gr::DataSet<float> processOne(gr::DataSet<float> inData) noexcept {
        if (_passthrough || !_session.isLoaded() || inData.signal_values.empty()) {
            return inData;
        }

        const std::size_t modelN    = _session.modelN();
        const std::size_t R         = _session.regressionChannels();
        const std::size_t nSignals  = std::max(1UZ, inData.signal_names.size());
        const std::size_t inputSize = inData.signal_values.size() / nSignals;

        // resample to model dimension if needed
        std::vector<float> modelInput(modelN);
        if (inputSize == modelN || resample_mode == ResampleMode::None) {
            const auto copyN = std::min(inputSize, modelN);
            std::copy_n(inData.signal_values.begin(), copyN, modelInput.begin());
        } else {
            std::span<const float> firstSignal(inData.signal_values.data(), inputSize);
            OnnxPreprocess<float>::resample(firstSignal, modelInput);
        }

        // normalise
        std::vector<float> normalised(modelN);
        _preprocess.normalise(modelInput, normalised);

        // run inference
        auto result = _session.run(normalised);
        if (!result) {
            return inData;
        }

        // split output: first N = heatmap, remaining N*R = regression channels
        const auto&            raw = *result;
        std::span<const float> heatmap(raw.data(), modelN);
        std::span<const float> regression(raw.data() + modelN, modelN * R);

        // extract peaks via NMS
        auto peaks = extractPeaks(heatmap, regression, R, confidence_threshold, min_peak_distance, max_peaks);

        // resample heatmap back to input dimension if sizes differ
        std::vector<float> heatmapOut;
        if (inputSize != modelN && resample_mode != ResampleMode::None) {
            heatmapOut.resize(inputSize);
            OnnxPreprocess<float>::resample(heatmap, heatmapOut);
        } else {
            heatmapOut.assign(heatmap.begin(), heatmap.end());
        }

        // build output DataSet
        gr::DataSet<float> output;
        output.timestamp   = inData.timestamp;
        output.axis_names  = inData.axis_names;
        output.axis_units  = inData.axis_units;
        output.axis_values = inData.axis_values;

        output.signal_names      = {"Spectrum", "Heatmap"};
        output.signal_quantities = {"", ""};
        output.signal_units      = {inData.signal_units.empty() ? "" : inData.signal_units[0], ""};
        output.signal_ranges     = {gr::Range<float>{0.f, 0.f}, gr::Range<float>{0.f, 1.f}};

        output.extents = {static_cast<std::int32_t>(inputSize)};

        output.signal_values.resize(2 * inputSize);
        std::copy_n(inData.signal_values.begin(), inputSize, output.signal_values.begin());
        std::copy(heatmapOut.begin(), heatmapOut.end(), output.signal_values.begin() + static_cast<std::ptrdiff_t>(inputSize));

        output.meta_information = {{}, {}};

        // scale peak positions from model domain back to input domain
        float posScale = (inputSize != modelN) ? static_cast<float>(inputSize) / static_cast<float>(modelN) : 1.f;

        std::vector<gr::DataSet<float>::idx_pmt_map> peakEvents;
        for (const auto& p : peaks) {
            float            scaledPos = p.position * posScale;
            gr::property_map props{
                {std::pmr::string("confidence"), gr::pmt::Value(p.confidence)},
                {std::pmr::string("sigma"), gr::pmt::Value(p.sigma * posScale)},
                {std::pmr::string("amplitude"), gr::pmt::Value(p.amplitude)},
                {std::pmr::string("w68"), gr::pmt::Value(p.w68 * posScale)},
                {std::pmr::string("w96"), gr::pmt::Value(p.w96 * posScale)},
                {std::pmr::string("w99"), gr::pmt::Value(p.w99 * posScale)},
                {std::pmr::string("kurtosis"), gr::pmt::Value(p.kurtosis)},
            };
            peakEvents.emplace_back(static_cast<std::ptrdiff_t>(scaledPos), std::move(props));
        }
        output.timing_events = {std::move(peakEvents), {}};

        return output;
    }

private:
    void loadModel() {
        _passthrough = false;
        if (model_path.value.empty()) {
            return;
        }
        auto result = _session.load(model_path);
        if (!result) {
            handleError("loadModel()", result.error().message);
            return;
        }
        model_io_shape.value = _session.modelIoShape<gr::Size_t>();
        if (auto mode = normaliseModeFromString(_session.metadata().normaliseMode)) {
            normalise_mode = *mode;
        }
        configurePreprocess();
    }

    void configurePreprocess() {
        if (auto r = _preprocess.configure(resample_mode, normalise_mode, normalise_expr, clip_min, clip_max); !r) {
            handleError("configurePreprocess()", r.error().message);
        }
    }

    void handleError(std::string_view context, std::string_view message, std::source_location location = std::source_location::current()) {
        this->emitErrorMessage(std::string(context), gr::Error{message, location});
        if (error_policy == ErrorPolicy::Stop) {
            this->requestStop();
        } else {
            _passthrough = true;
        }
    }
};

} // namespace gr::blocks::onnx

inline const auto registerOnnxPeakDetector = gr::registerBlock<gr::blocks::onnx::OnnxPeakDetector>(gr::globalBlockRegistry());

#endif // GR_ONNX_PEAK_DETECTOR_ONNX_HPP
