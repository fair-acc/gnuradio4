#ifndef GR_ONNX_INFERENCE_HPP
#define GR_ONNX_INFERENCE_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/onnx/OnnxPreprocess.hpp>
#include <gnuradio-4.0/onnx/OnnxSession.hpp>

#include <deque>

namespace gr::blocks::onnx {

/**
 * @brief Generic ONNX Runtime inference block.
 *
 * Template parameters:
 *   T    - value type (float, double)
 *   TIn  - input port type: T (streaming), DataSet<T>, Tensor<T>
 *   TOut - output port type: T (streaming), DataSet<T>, Tensor<T>
 *
 * Processing modes (determined by TIn/TOut combination):
 *   DataSet<T> -> DataSet<T>:  [1xN] single-shot or [MxN] history-based inference
 *   DataSet<T> -> Tensor<T>:   spectrum to raw tensor (strip metadata)
 *   Tensor<T>  -> Tensor<T>:   raw tensor I/O (no metadata either side)
 *   Tensor<T>  -> DataSet<T>:  raw tensor to annotated output (attach metadata)
 *   T -> T:                    Streaming filter (M->M) or sliding-window prediction
 *   T -> DataSet<T>:           Streaming to spectrum (M->1 DataSet, like FFT)
 */
template<typename T, typename TIn = gr::DataSet<T>, typename TOut = gr::DataSet<T>>
requires std::floating_point<T>
struct OnnxInference : gr::Block<OnnxInference<T, TIn, TOut>> {
    using Description = Doc<R""(@brief Generic ONNX Runtime inference block.

Supports DataSet<T> and streaming T input/output with configurable
preprocessing (resample, normalise) and history-based batched inference.)"">;

    gr::PortIn<TIn>   in;
    gr::PortOut<TOut> out;

    // --- model settings ---
    Annotated<std::string, "model path">   model_path   = "";
    Annotated<ErrorPolicy, "error policy"> error_policy = ErrorPolicy::Stop;

    // --- preprocessing ---
    Annotated<ResampleMode, "resample mode">       resample_mode  = ResampleMode::Linear;
    Annotated<NormaliseMode, "normalise mode">     normalise_mode = NormaliseMode::LogMAD;
    Annotated<T, "clip min">                       clip_min       = T(-5);
    Annotated<T, "clip max">                       clip_max       = T(10);
    Annotated<std::string, "normalise expression"> normalise_expr = "";

    // --- history / batching (for DataSet input) ---
    Annotated<gr::Size_t, "history depth">  history_depth  = 1U; // M; 1 = no history
    Annotated<gr::Size_t, "history stride"> history_stride = 1U; // advance per inference

    // --- streaming settings (for T input) ---
    Annotated<gr::Size_t, "stride"> stride = 0U; // 0 = non-overlapping (stride = model input dim)

    // --- read-only info ---
    Annotated<std::vector<gr::Size_t>, "model I/O shape", Doc<"read-only: populated after model load">> model_io_shape;

    GR_MAKE_REFLECTABLE(OnnxInference, in, out, model_path, error_policy, resample_mode, normalise_mode, clip_min, clip_max, normalise_expr, history_depth, history_stride, stride, model_io_shape);

    OnnxSession       _session;
    OnnxPreprocess<T> _preprocess;
    bool              _passthrough = false;

    // history buffer for [MxN] DataSet mode
    std::deque<gr::DataSet<T>> _historyBuffer;

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
        _historyBuffer.clear();
        _passthrough = false;
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& newSettings) {
        if (newSettings.contains("model_path")) {
            loadModel();
        }
        if (newSettings.contains("normalise_mode") || newSettings.contains("normalise_expr") || newSettings.contains("clip_min") || newSettings.contains("clip_max") || newSettings.contains("resample_mode")) {
            configurePreprocess();
        }
        if (newSettings.contains("history_depth") || newSettings.contains("history_stride")) {
            updatePortConstraints();
        }
    }

    // === DataSet<T> -> DataSet<T> mode (processOne, 1xN or MxN) ===
    [[nodiscard]] TOut processOne(TIn inData) noexcept
    requires std::same_as<TIn, gr::DataSet<T>> && std::same_as<TOut, gr::DataSet<T>>
    {
        if (_passthrough || !_session.isLoaded()) {
            return inData;
        }
        if (history_depth > 1U) {
            _historyBuffer.push_back(std::move(inData));
            if (_historyBuffer.size() < history_depth) {
                return {}; // accumulating — return empty DataSet until buffer is full
            }
            auto result = processHistoryBatch();
            // consume history_stride items, keep the rest for the next window
            const auto toConsume = std::min(static_cast<std::size_t>(history_stride.value), _historyBuffer.size());
            for (std::size_t i = 0; i < toConsume; ++i) {
                _historyBuffer.pop_front();
            }
            return result;
        }
        return processDataSet(inData);
    }

    // === DataSet<T> -> Tensor<T> mode (strip metadata from output) ===
    [[nodiscard]] TOut processOne(TIn inData) noexcept
    requires std::same_as<TIn, gr::DataSet<T>> && std::same_as<TOut, gr::Tensor<T>>
    {
        if (_passthrough || !_session.isLoaded() || inData.signal_values.empty()) {
            return gr::Tensor<T>(gr::data_from, inData.signal_values);
        }
        const std::size_t nSignals  = std::max(1UZ, inData.signal_names.size());
        const std::size_t inputSize = inData.signal_values.size() / nSignals;
        auto              result    = runInference({inData.signal_values.data(), inputSize});
        if (!result) {
            return gr::Tensor<T>(gr::data_from, inData.signal_values);
        }
        return gr::Tensor<T>(gr::data_from, *result);
    }

    // === Tensor<T> -> Tensor<T> mode (raw tensor I/O) ===
    [[nodiscard]] TOut processOne(TIn inData) noexcept
    requires std::same_as<TIn, gr::Tensor<T>> && std::same_as<TOut, gr::Tensor<T>>
    {
        if (_passthrough || !_session.isLoaded() || inData.empty()) {
            return inData;
        }
        auto result = runInference({inData.data(), inData.size()});
        if (!result) {
            return inData;
        }
        return gr::Tensor<T>(gr::data_from, *result);
    }

    // === Tensor<T> -> DataSet<T> mode (attach metadata to raw output) ===
    [[nodiscard]] TOut processOne(TIn inData) noexcept
    requires std::same_as<TIn, gr::Tensor<T>> && std::same_as<TOut, gr::DataSet<T>>
    {
        if (_passthrough || !_session.isLoaded() || inData.empty()) {
            gr::DataSet<T> ds;
            ds.signal_names = {"pass-through"};
            ds.signal_values.assign(inData.begin(), inData.end());
            ds.extents          = {static_cast<std::int32_t>(inData.size())};
            ds.meta_information = {{}};
            ds.timing_events    = {{}};
            return ds;
        }
        auto result = runInference({inData.data(), inData.size()});
        if (!result) {
            gr::DataSet<T> ds;
            ds.signal_names = {"pass-through"};
            ds.signal_values.assign(inData.begin(), inData.end());
            ds.extents          = {static_cast<std::int32_t>(inData.size())};
            ds.meta_information = {{}};
            ds.timing_events    = {{}};
            return ds;
        }
        const auto     outSize = result->size();
        gr::DataSet<T> output;
        output.signal_names      = {"inference_output"};
        output.signal_quantities = {""};
        output.signal_units      = {""};
        output.signal_values     = std::move(*result);
        output.signal_ranges     = {gr::Range<T>{T(0), T(0)}};
        output.extents           = {static_cast<std::int32_t>(outSize)};
        output.meta_information  = {{}};
        output.timing_events     = {{}};
        return output;
    }

    // === T -> T mode (streaming filter) ===
    [[nodiscard]] gr::work::Status processBulk(auto& inPort, auto& outPort)
    requires std::same_as<TIn, T> && std::same_as<TOut, T>
    {
        if (_passthrough || !_session.isLoaded()) {
            return consumeAndDiscard(inPort);
        }

        const std::size_t modelN     = _session.modelN();
        const std::size_t chunkSize  = modelN;
        const auto        nAvailable = inPort.size();

        if (nAvailable < chunkSize) {
            return gr::work::Status::INSUFFICIENT_INPUT_ITEMS;
        }

        const std::size_t nChunks = std::min(nAvailable / chunkSize, outPort.size() / chunkSize);
        if (nChunks == 0) {
            return gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS;
        }

        for (std::size_t c = 0; c < nChunks; ++c) {
            std::span<const T> inputChunk(&inPort[c * chunkSize], chunkSize);

            // normalise
            std::vector<T> normalised(modelN);
            _preprocess.normalise(inputChunk, normalised);

            auto result = _session.run(normalised);
            if (!result) {
                // fill output with zeros on error
                std::fill_n(&outPort[c * chunkSize], chunkSize, T(0));
                continue;
            }

            const auto& raw      = *result;
            const auto  outCount = std::min(raw.size(), chunkSize);
            std::copy_n(raw.begin(), outCount, &outPort[c * chunkSize]);
            if (outCount < chunkSize) {
                std::fill_n(&outPort[c * chunkSize + outCount], chunkSize - outCount, T(0));
            }
        }

        std::ignore = inPort.consume(nChunks * chunkSize);
        outPort.publish(nChunks * chunkSize);
        return gr::work::Status::OK;
    }

    // === T -> DataSet<T> mode (streaming to spectrum, like FFT) ===
    [[nodiscard]] gr::work::Status processBulk(auto& inPort, auto& outPort)
    requires std::same_as<TIn, T> && std::same_as<TOut, gr::DataSet<T>>
    {
        if (_passthrough || !_session.isLoaded()) {
            return consumeAndDiscard(inPort);
        }

        const std::size_t modelN     = _session.modelN();
        const auto        nAvailable = inPort.size();

        if (nAvailable < modelN) {
            return gr::work::Status::INSUFFICIENT_INPUT_ITEMS;
        }

        const std::size_t effectiveStride = (stride > 0U) ? static_cast<std::size_t>(stride.value) : modelN;
        const std::size_t nOutputs        = std::min((nAvailable - modelN) / effectiveStride + 1, outPort.size());
        if (nOutputs == 0) {
            return gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS;
        }

        for (std::size_t i = 0; i < nOutputs; ++i) {
            std::span<const T> inputWindow(&inPort[i * effectiveStride], modelN);

            std::vector<T> normalised(modelN);
            _preprocess.normalise(inputWindow, normalised);

            auto result = _session.run(normalised);
            if (!result) {
                outPort[i] = gr::DataSet<T>{};
                continue;
            }

            gr::DataSet<T> ds;
            ds.signal_names      = {"inference_output"};
            ds.signal_values     = std::move(*result);
            ds.extents           = {static_cast<std::int32_t>(ds.signal_values.size())};
            ds.signal_ranges     = {gr::Range<T>{T(0), T(0)}};
            ds.signal_quantities = {""};
            ds.signal_units      = {""};
            ds.meta_information  = {{}};
            ds.timing_events     = {{}};
            outPort[i]           = std::move(ds);
        }

        std::ignore = inPort.consume(nOutputs * effectiveStride);
        outPort.publish(nOutputs);
        return gr::work::Status::OK;
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
        // auto-set history_depth from model metadata if the model carries it
        if (_session.historyDepth() > 1 && history_depth <= 1U) {
            history_depth = static_cast<gr::Size_t>(_session.historyDepth());
        }
        _historyBuffer.clear();
        updatePortConstraints();
        configurePreprocess();
    }

    void configurePreprocess() {
        if (auto r = _preprocess.configure(resample_mode, normalise_mode, normalise_expr, clip_min, clip_max); !r) {
            handleError("configurePreprocess()", r.error().message);
        }
    }

    void updatePortConstraints() {
        if constexpr (std::same_as<TIn, gr::DataSet<T>>) {
            if (history_depth > 1U) {
                in.min_samples = history_depth;
            } else {
                in.min_samples = 1U;
            }
        } else if constexpr (std::same_as<TIn, gr::Tensor<T>>) {
            in.min_samples = 1U;
        } else if constexpr (std::same_as<TIn, T>) {
            if (_session.isLoaded()) {
                in.min_samples = _session.modelN();
            }
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

    gr::work::Status consumeAndDiscard(auto& inPort) {
        const auto n = inPort.size();
        if (n > 0) {
            std::ignore = inPort.consume(n);
        }
        return gr::work::Status::OK;
    }

    // shared: resample → normalise → inference pipeline
    [[nodiscard]] auto runInference(std::span<const T> input) {
        const std::size_t modelN    = _session.modelN();
        const std::size_t inputSize = input.size();

        std::vector<T> modelInput(modelN);
        if (inputSize == modelN || resample_mode == ResampleMode::None) {
            const auto copyN = std::min(inputSize, modelN);
            std::copy_n(input.begin(), copyN, modelInput.begin());
        } else {
            OnnxPreprocess<T>::resample(input, modelInput);
        }

        std::vector<T> normalised(modelN);
        _preprocess.normalise(modelInput, normalised);

        return _session.run(normalised);
    }

    // [1xN] DataSet processing
    [[nodiscard]] gr::DataSet<T> processDataSet(const gr::DataSet<T>& inData) {
        if (inData.signal_values.empty()) {
            return inData;
        }

        const std::size_t nSignals  = std::max(1UZ, inData.signal_names.size());
        const std::size_t inputSize = inData.signal_values.size() / nSignals;

        auto result = runInference({inData.signal_values.data(), inputSize});
        if (!result) {
            return inData;
        }

        gr::DataSet<T> output;
        output.timestamp   = inData.timestamp;
        output.axis_names  = inData.axis_names;
        output.axis_units  = inData.axis_units;
        output.axis_values = inData.axis_values;

        const auto outSize = result->size();

        output.signal_names      = {"inference_output"};
        output.signal_quantities = {""};
        output.signal_units      = {""};
        output.signal_values     = std::move(*result);
        output.signal_ranges     = {gr::Range<T>{T(0), T(0)}};
        output.extents           = {static_cast<std::int32_t>(outSize)};
        output.meta_information  = {{}};
        output.timing_events     = {{}};

        return output;
    }

    // [MxN] history batch processing
    [[nodiscard]] gr::DataSet<T> processHistoryBatch() {
        const std::size_t M      = history_depth;
        const std::size_t modelN = _session.modelN();

        // stack M DataSets into [M, N] tensor
        std::vector<T> batchInput(M * modelN, T(0));
        for (std::size_t row = 0; row < M && row < _historyBuffer.size(); ++row) {
            const auto& ds      = _historyBuffer[row];
            const auto  nSigs   = std::max(1UZ, ds.signal_names.size());
            const auto  sigSize = ds.signal_values.size() / nSigs;

            if (sigSize == modelN) {
                std::copy_n(ds.signal_values.begin(), modelN, batchInput.begin() + static_cast<std::ptrdiff_t>(row * modelN));
            } else {
                std::span<const T> firstSig(ds.signal_values.data(), sigSize);
                std::span<T>       target(batchInput.data() + row * modelN, modelN);
                OnnxPreprocess<T>::resample(firstSig, target);
            }
        }

        // normalise each row
        std::vector<T> normalised(M * modelN);
        for (std::size_t row = 0; row < M; ++row) {
            std::span<const T> rowIn(batchInput.data() + row * modelN, modelN);
            std::span<T>       rowOut(normalised.data() + row * modelN, modelN);
            _preprocess.normalise(rowIn, rowOut);
        }

        // run full [M×N] batch inference
        auto result = _session.run(normalised);

        if (!result) {
            // return pass-through of latest input
            return _historyBuffer.back();
        }

        // format output using latest DataSet's metadata
        const auto&    latest = _historyBuffer.back();
        gr::DataSet<T> output;
        output.timestamp   = latest.timestamp;
        output.axis_names  = latest.axis_names;
        output.axis_units  = latest.axis_units;
        output.axis_values = latest.axis_values;

        output.signal_names      = {"inference_output"};
        output.signal_quantities = {""};
        output.signal_units      = {""};
        output.signal_values     = std::move(*result);
        output.signal_ranges     = {gr::Range<T>{T(0), T(0)}};
        output.extents           = {static_cast<std::int32_t>(output.signal_values.size())};
        output.meta_information  = {{}};
        output.timing_events     = {{}};

        return output;
    }
};

} // namespace gr::blocks::onnx

inline const auto registerOnnxInference     = gr::registerBlock<gr::blocks::onnx::OnnxInference<float>>(gr::globalBlockRegistry());
inline const auto registerOnnxInferenceDS2T = gr::registerBlock<gr::blocks::onnx::OnnxInference<float, gr::DataSet<float>, gr::Tensor<float>>>(gr::globalBlockRegistry());
inline const auto registerOnnxInferenceT2T  = gr::registerBlock<gr::blocks::onnx::OnnxInference<float, gr::Tensor<float>, gr::Tensor<float>>>(gr::globalBlockRegistry());
inline const auto registerOnnxInferenceT2DS = gr::registerBlock<gr::blocks::onnx::OnnxInference<float, gr::Tensor<float>, gr::DataSet<float>>>(gr::globalBlockRegistry());

#endif // GR_ONNX_INFERENCE_HPP
