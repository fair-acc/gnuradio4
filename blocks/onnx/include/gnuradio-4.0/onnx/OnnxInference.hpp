#ifndef GR_ONNX_INFERENCE_HPP
#define GR_ONNX_INFERENCE_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/onnx/OnnxPreprocess.hpp>
#include <gnuradio-4.0/onnx/OnnxSession.hpp>

#include <deque>
#include <format>
#include <numeric>
#include <optional>
#include <print>
#include <source_location>
#include <string_view>

namespace gr::blocks::onnx {

// one registration per value type: a combined list would also admit mixes such as
// OnnxInference<float, DataSet<double>, ...>, which the block's own constraints reject
GR_REGISTER_BLOCK(gr::blocks::onnx::OnnxInference, ([T], [U], [A]), [float], [ gr::DataSet<float>, gr::Tensor<float> ], [ gr::DataSet<float>, gr::Tensor<float> ])
GR_REGISTER_BLOCK(gr::blocks::onnx::OnnxInference, ([T], [U], [A]), [double], [ gr::DataSet<double>, gr::Tensor<double> ], [ gr::DataSet<double>, gr::Tensor<double> ])

template<typename T, typename TIn = gr::DataSet<T>, typename TOut = gr::DataSet<T>>
requires std::floating_point<T>
struct OnnxInference : gr::Block<OnnxInference<T, TIn, TOut>, gr::Resampling<1UL, 1UL, false>> {
    using Description = Doc<R""(Generic ONNX Runtime inference block.

Processing mode follows the TIn/TOut port types: DataSet<T> -> DataSet<T> supports
single-shot and history-based (M-frame) inference; DataSet<T> -> Tensor<T> strips
metadata; Tensor<T> -> Tensor<T> is raw I/O; Tensor<T> -> DataSet<T> attaches metadata.

History mode (history_depth M > 1): the window always covers the newest M frames.
history_stride 1 slides 1:1, with a warm-up pass-through until the window fills;
history_stride S > 1 decimates S:1 (one output per S inputs) via the framework's
Resampling machinery.

meta_out exposes model outputs 1..n-1, one Tensor<T> per inference. config_in ports
and the model_overrides map both set overridable model initializers; precedence is
fed config_in port > model_overrides entry > model-baked default, and a rejected
model_overrides entry (unknown name, size mismatch, unsupported type) is reported
through the block's error policy rather than silently dropped.

normalise_mode selects None/LogMAD/MinMax/ZScore/Expression; Expression evaluates
the ExprTk normalise_expr over the resampled input, with vecIn -> vecOut and
n/median/mad/min_val/max_val/mean_val/std_val pre-computed. A model may request its
own normalisation via ONNX metadata.

Frames forwarded unchanged (no model, history warm-up, ErrorPolicy::Passthrough)
carry meta_information[0]["onnx_passthrough"] = true on DataSet outputs; bare
Tensor<T> outputs have no metadata and stay unmarked.)"">;

    gr::PortIn<TIn>                                     in;
    std::vector<gr::PortIn<T, gr::Async, gr::Optional>> config_in; // one per overridable model initializer

    gr::PortOut<TOut>                                                out;
    std::vector<gr::PortOut<gr::Tensor<T>, gr::Async, gr::Optional>> meta_out; // model outputs 1..n-1

    Annotated<std::string, "model path">                                                                                                                                 model_path         = "";
    Annotated<ErrorPolicy, "error policy">                                                                                                                               error_policy       = ErrorPolicy::Stop;
    Annotated<std::string, "execution provider", Doc<"cpu | cuda | tensorrt | rocm — validated against the linked ONNX Runtime at model load, never a silent fallback">> execution_provider = "cpu";

    Annotated<ResampleMode, "resample mode"> resample_mode = ResampleMode::Linear;
    // identity by default: a model that wants normalisation declares it in its own metadata, which
    // loadModel() adopts. Defaulting to a transform would silently alter a user's own model's input.
    Annotated<NormaliseMode, "normalise mode">     normalise_mode = NormaliseMode::None;
    Annotated<T, "clip min">                       clip_min       = T(-5);
    Annotated<T, "clip max">                       clip_max       = T(10);
    Annotated<std::string, "normalise expression"> normalise_expr = "";

    Annotated<gr::Size_t, "history depth">  history_depth  = 1U; // M; 1 = no history
    Annotated<gr::Size_t, "history stride"> history_stride = 1U; // advance per inference

    Annotated<gr::Size_t, "intra-op threads", Doc<"ONNX Runtime threads per operator; 1 is deterministic, 4 is ~2.5x faster">> intra_op_threads = 4U;

    Annotated<gr::property_map, "model overrides", Doc<"overridable-initializer values by name">> model_overrides;

    Annotated<std::vector<gr::Size_t>, "model input shape", Doc<"read-only: populated after model load">>                                model_input_shape;
    Annotated<std::vector<gr::Size_t>, "model output shape", Doc<"read-only: populated after model load">>                               model_output_shape;
    Annotated<std::vector<std::string>, "meta output names", Doc<"read-only: model outputs 1..n-1 feeding meta_out">>                    meta_output_names;
    Annotated<std::vector<std::string>, "config input names", Doc<"read-only: overridable initializers feeding config_in">>              config_input_names;
    Annotated<std::vector<std::string>, "available providers", Doc<"read-only: execution providers offered by the linked ONNX Runtime">> available_providers;
    Annotated<std::string, "passthrough reason", Doc<"read-only: empty unless an error latched the block into pass-through">>            passthrough_reason;

    GR_MAKE_REFLECTABLE(OnnxInference, in, config_in, out, meta_out, model_path, error_policy, execution_provider, resample_mode, normalise_mode, clip_min, clip_max, normalise_expr, history_depth, history_stride, intra_op_threads, model_overrides, model_input_shape, model_output_shape, meta_output_names, config_input_names, available_providers, passthrough_reason);

    OnnxSession       _session;
    OnnxPreprocess<T> _preprocess;
    bool              _passthrough = false;

    std::deque<gr::DataSet<T>> _historyBuffer;
    std::vector<float>         _sessionInput; // narrowed copy when T is not float

    std::vector<std::optional<T>> _latchedConfig; // index-aligned with the overridable initializers

    void start() {
        if (model_path.value.empty()) {
            handleError("start()", "model_path is empty");
            return;
        }
        loadModel();
        configurePreprocess();
    }

    void reset() { _historyBuffer.clear(); }

    void stop() {
        _session.reset();
        _passthrough = false;
        passthrough_reason.value.clear();
        _historyBuffer.clear();
    }

    // one output (+ optional meta tensors) per input; history_stride > 1 decimates instead
    template<typename TCfgSpan, typename TMetaSpan>
    [[nodiscard]] gr::work::Status processBulk(auto& inSpan, std::span<TCfgSpan>& configIns, auto& outSpan, std::span<TMetaSpan>& metaOuts) {
        latchConfig(configIns);
        const std::vector<NamedTensor> extras   = buildExtraInputs();
        const bool                     wantMeta = anyMetaConnected();
        std::vector<std::size_t>       metaCounts(metaOuts.size(), 0UZ);

        if constexpr (std::same_as<TIn, gr::DataSet<T>> && std::same_as<TOut, gr::DataSet<T>>) {
            if (history_depth > 1U && history_stride > 1U) {
                return processDecimating(inSpan, outSpan, extras, wantMeta, metaOuts, metaCounts);
            }
        }
        const std::size_t n = std::min(inSpan.size(), outSpan.size());
        for (std::size_t i = 0; i < n; ++i) {
            outSpan[i] = processSample(inSpan[i], extras, wantMeta, metaOuts, metaCounts);
        }
        publishMeta(metaOuts, metaCounts);
        return gr::work::Status::OK;
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& newSettings) {
        if (newSettings.contains("model_path") || newSettings.contains("execution_provider") || newSettings.contains("intra_op_threads")) {
            loadModel();
        }
        if (newSettings.contains("normalise_mode") || newSettings.contains("normalise_expr") || newSettings.contains("clip_min") || newSettings.contains("clip_max") || newSettings.contains("resample_mode")) {
            configurePreprocess();
        }
        if (newSettings.contains("history_depth") || newSettings.contains("history_stride")) {
            updatePortConstraints();
            while (_historyBuffer.size() > std::max<std::size_t>(1UZ, static_cast<std::size_t>(history_depth.value))) {
                _historyBuffer.pop_front(); // a shrunk window must drop the oldest frames, not keep serving them
            }
        }
    }

    [[nodiscard]] bool isModelLoaded() const noexcept { return _session.isLoaded(); }

private:
    [[nodiscard]] gr::DataSet<T> processSample(const gr::DataSet<T>& inData, std::span<const NamedTensor> extras, bool wantMeta, auto& metaOuts, std::vector<std::size_t>& metaCounts)
    requires std::same_as<TIn, gr::DataSet<T>> && std::same_as<TOut, gr::DataSet<T>>
    {
        if (_passthrough || !_session.isLoaded()) {
            return markedCopy(inData);
        }
        if (history_depth > 1U) {
            _historyBuffer.push_back(inData);
            if (_historyBuffer.size() < history_depth) {
                return markedCopy(inData); // warm-up: input passes through unchanged until the window is full
            }
            auto result = processHistoryBatch(extras, wantMeta, metaOuts, metaCounts);
            if (!_historyBuffer.empty()) {  // a failed inference under ErrorPolicy::Stop clears the buffer
                _historyBuffer.pop_front(); // sliding window, stride 1
            }
            return result;
        }
        return processDataSet(inData, extras, wantMeta, metaOuts, metaCounts);
    }

    // genuine N:1 decimation (history_stride S > 1): S inputs per output over the newest
    // history_depth frames; outputs start once the window is full — no placeholders
    [[nodiscard]] gr::work::Status processDecimating(auto& inSpan, auto& outSpan, std::span<const NamedTensor> extras, bool wantMeta, auto& metaOuts, std::vector<std::size_t>& metaCounts)
    requires std::same_as<TIn, gr::DataSet<T>> && std::same_as<TOut, gr::DataSet<T>>
    {
        const std::size_t chunk    = static_cast<std::size_t>(history_stride.value);
        const std::size_t nChunks  = std::min(inSpan.size() / chunk, outSpan.size());
        std::size_t       produced = 0UZ;
        for (std::size_t c = 0; c < nChunks; ++c) {
            if (_passthrough || !_session.isLoaded()) {
                outSpan[produced++] = markedCopy(inSpan[(c + 1UZ) * chunk - 1UZ]); // keep the N:1 contract: pass the newest input of the chunk through
                continue;
            }
            for (std::size_t i = 0; i < chunk; ++i) {
                _historyBuffer.push_back(inSpan[c * chunk + i]);
                if (_historyBuffer.size() > history_depth) {
                    _historyBuffer.pop_front();
                }
            }
            if (_historyBuffer.size() == history_depth) {
                outSpan[produced++] = processHistoryBatch(extras, wantMeta, metaOuts, metaCounts);
            }
        }
        publishMeta(metaOuts, metaCounts);
        std::ignore = inSpan.consume(nChunks * chunk);
        outSpan.publish(produced);
        return gr::work::Status::OK;
    }

    [[nodiscard]] gr::Tensor<T> processSample(const gr::DataSet<T>& inData, std::span<const NamedTensor> extras, bool wantMeta, auto& metaOuts, std::vector<std::size_t>& metaCounts)
    requires std::same_as<TIn, gr::DataSet<T>> && std::same_as<TOut, gr::Tensor<T>>
    {
        if (_passthrough || !_session.isLoaded() || inData.signal_values.empty()) {
            return gr::Tensor<T>(gr::data_from, inData.signal_values);
        }
        const std::size_t nSignals  = std::max(1UZ, inData.signal_names.size());
        const std::size_t inputSize = inData.signal_values.size() / nSignals;
        auto              result    = runInference({inData.signal_values.data(), inputSize}, extras, wantMeta, metaOuts, metaCounts);
        if (!result) {
            return gr::Tensor<T>(gr::data_from, inData.signal_values);
        }
        return gr::Tensor<T>(gr::data_from, *result);
    }

    [[nodiscard]] gr::Tensor<T> processSample(const gr::Tensor<T>& inData, std::span<const NamedTensor> extras, bool wantMeta, auto& metaOuts, std::vector<std::size_t>& metaCounts)
    requires std::same_as<TIn, gr::Tensor<T>> && std::same_as<TOut, gr::Tensor<T>>
    {
        if (_passthrough || !_session.isLoaded() || inData.empty()) {
            return inData;
        }
        auto result = runInference({inData.data(), inData.size()}, extras, wantMeta, metaOuts, metaCounts);
        if (!result) {
            return inData;
        }
        return gr::Tensor<T>(gr::data_from, *result);
    }

    [[nodiscard]] gr::DataSet<T> processSample(const gr::Tensor<T>& inData, std::span<const NamedTensor> extras, bool wantMeta, auto& metaOuts, std::vector<std::size_t>& metaCounts)
    requires std::same_as<TIn, gr::Tensor<T>> && std::same_as<TOut, gr::DataSet<T>>
    {
        auto passThrough = [&inData]() {
            gr::DataSet<T> ds;
            ds.signal_names = {"pass-through"};
            ds.signal_values.assign(inData.begin(), inData.end());
            ds.extents          = {static_cast<std::int32_t>(inData.size())};
            ds.meta_information = {{}};
            ds.timing_events    = {{}};
            markPassthrough(ds);
            return ds;
        };
        if (_passthrough || !_session.isLoaded() || inData.empty()) {
            return passThrough();
        }
        auto result = runInference({inData.data(), inData.size()}, extras, wantMeta, metaOuts, metaCounts);
        if (!result) {
            return passThrough();
        }
        const auto     outSize = result->size();
        gr::DataSet<T> output;
        output.signal_names      = {"inference_output"};
        output.signal_quantities = {""};
        output.signal_units      = {""};
        output.signal_values     = toValueType(std::move(*result));
        output.signal_ranges     = {gr::Range<T>{T(0), T(0)}};
        output.extents           = {static_cast<std::int32_t>(outSize)};
        output.meta_information  = {{}};
        output.timing_events     = {{}};
        return output;
    }

    template<typename TCfgSpans>
    void latchConfig(TCfgSpans& configIns) {
        if (_latchedConfig.size() < configIns.size()) {
            _latchedConfig.resize(configIns.size());
        }
        for (std::size_t i = 0; i < configIns.size(); ++i) {
            auto& cfgSpan = configIns[i];
            if (cfgSpan.size() == 0UZ) {
                std::ignore = cfgSpan.consume(0UZ);
                continue;
            }
            _latchedConfig[i] = static_cast<T>(cfgSpan[cfgSpan.size() - 1UZ]);
            std::ignore       = cfgSpan.consume(cfgSpan.size());
        }
    }

    [[nodiscard]] std::vector<NamedTensor> buildExtraInputs() {
        if (_passthrough || !_session.isLoaded()) {
            return {};
        }
        auto latched = [this](std::size_t i, std::size_t elemCount) -> std::optional<std::vector<float>> {
            if (i < _latchedConfig.size() && _latchedConfig[i].has_value()) {
                return std::vector<float>(elemCount, static_cast<float>(*_latchedConfig[i]));
            }
            return std::nullopt;
        };
        return buildOverrideInputs(model_overrides.value, _session.overridableInitializers(), latched, //
            [this](std::string message) { handleError("buildExtraInputs()", message); });
    }

    [[nodiscard]] bool anyMetaConnected() const {
        return std::ranges::any_of(meta_out, [](const auto& port) { return port.isConnected(); });
    }

    [[nodiscard]] static gr::DataSet<T> markedCopy(const gr::DataSet<T>& inData) {
        gr::DataSet<T> copy = inData;
        markPassthrough(copy);
        return copy;
    }

    [[nodiscard]] static gr::Tensor<T> toTensor(NamedTensor&& source) {
        if (source.shape.empty() || std::ranges::any_of(source.shape, [](std::size_t d) { return d == 0UZ; })) {
            std::vector<T> converted(source.values.begin(), source.values.end());
            return gr::Tensor<T>(gr::data_from, converted);
        }
        gr::Tensor<T> tensor(gr::extents_from, std::span<const std::size_t>(source.shape));
        std::copy_n(source.values.begin(), std::min(source.values.size(), tensor.size()), tensor.begin());
        return tensor;
    }

    // runAll only when a meta port is connected, so ORT can prune the unused outputs otherwise
    // ONNX Runtime tensors are float32; a double-valued graph is narrowed here and widened again on
    // the way out, so the block accepts double without pretending the arithmetic is done in it
    // ORT hands back float32; widen once here so every caller works in the block's value type
    template<typename TValue>
    [[nodiscard]] static std::vector<T> toValueType(std::vector<TValue>&& values) {
        if constexpr (std::same_as<TValue, T>) {
            return std::move(values);
        } else {
            return std::vector<T>(values.begin(), values.end());
        }
    }

    [[nodiscard]] std::expected<std::vector<float>, gr::Error> runSession(std::span<const T> values, std::span<const NamedTensor> extras, bool wantMeta, auto& metaOuts, std::vector<std::size_t>& metaCounts) {
        std::span<const float> normalised;
        if constexpr (std::same_as<T, float>) {
            normalised = values;
        } else {
            _sessionInput.assign(values.begin(), values.end());
            normalised = _sessionInput;
        }
        if (!wantMeta) {
            return _session.run(normalised, extras);
        }
        auto all = _session.runAll(normalised, extras);
        if (!all) {
            return std::unexpected(all.error());
        }
        if (all->empty()) {
            return std::unexpected(gr::Error{"model produced no outputs"});
        }
        for (std::size_t k = 1; k < all->size(); ++k) {
            const std::size_t portIdx = k - 1;
            if (portIdx >= metaOuts.size() || metaCounts[portIdx] >= metaOuts[portIdx].size()) {
                continue;
            }
            metaOuts[portIdx][metaCounts[portIdx]++] = toTensor(std::move((*all)[k]));
        }
        return std::move(all->front().values);
    }

    static void publishMeta(auto& metaOuts, const std::vector<std::size_t>& metaCounts) {
        for (std::size_t k = 0; k < metaOuts.size(); ++k) {
            metaOuts[k].publish(metaCounts[k]);
        }
    }

    void loadModel() {
        _passthrough = false;
        passthrough_reason.value.clear();
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
        model_input_shape.value  = _session.template modelInputShape<gr::Size_t>();
        model_output_shape.value = _session.template modelOutputShape<gr::Size_t>();
        // metadata fills in what the caller left at its default, matching how history_depth below is
        // adopted; an explicit setting always wins so a re-exported model cannot silently retarget it
        adoptDeclaredNormalisation(_session.metadata(), normalise_mode, normalise_expr);
        if constexpr (!(std::same_as<TIn, gr::DataSet<T>> && std::same_as<TOut, gr::DataSet<T>>)) {
            if (history_depth > 1U) {
                handleError("loadModel()", std::format("history_depth {} needs DataSet in and out; this port combination has no frame history", history_depth.value));
            }
        }
        if (_session.historyDepth() > 1 && history_depth <= 1U) {
            history_depth = static_cast<gr::Size_t>(_session.historyDepth());
        }
        meta_output_names.value.assign(_session.outputNames().begin() + (_session.outputNames().empty() ? 0 : 1), _session.outputNames().end());
        meta_out.resize(meta_output_names.value.size());

        config_input_names.value.clear();
        for (const NamedTensor& init : _session.overridableInitializers()) {
            config_input_names.value.push_back(init.name);
        }
        config_in.resize(config_input_names.value.size());
        _latchedConfig.assign(config_in.size(), std::nullopt);

        _historyBuffer.clear();
        updatePortConstraints();
        configurePreprocess();
    }

    void configurePreprocess() {
        if (auto r = _preprocess.configure(normalise_mode, normalise_expr, clip_min, clip_max); !r) {
            handleError("configurePreprocess()", r.error().message);
        }
    }

    void updatePortConstraints() {
        if constexpr (std::same_as<TIn, gr::DataSet<T>> && std::same_as<TOut, gr::DataSet<T>>) {
            const bool decimating   = history_depth > 1U && history_stride > 1U;
            this->input_chunk_size  = decimating ? history_stride.value : 1U;
            this->output_chunk_size = 1U;
        }
    }

    void handleError(std::string_view context, std::string_view message, std::source_location location = std::source_location::current()) {
        if (error_policy == ErrorPolicy::Stop) {
            this->emitErrorMessage(context, gr::Error{message, location});
            this->requestStop();
            return;
        }
        // Passthrough must NOT emitErrorMessage: with no listener on msgOut the scheduler converts an
        // unconsumed error message into a thrown exception, aborting the very graph this policy exists
        // to keep running. Report on the block's own channel and latch the reason so it stays visible.
        if (!_passthrough) {
            std::print("{}: degraded to pass-through — {}: {}\n", this->name, context, message);
        }
        _passthrough       = true;
        passthrough_reason = std::format("{}: {}", context, message);
    }

    // nullopt on a failed inference, already reported through handleError, so callers pass through
    [[nodiscard]] std::optional<std::vector<T>> runInference(std::span<const T> input, std::span<const NamedTensor> extras, bool wantMeta, auto& metaOuts, std::vector<std::size_t>& metaCounts) {
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

        auto result = runSession(normalised, extras, wantMeta, metaOuts, metaCounts);
        if (!result) {
            handleError("runInference()", result.error().message);
            return std::nullopt;
        }
        return toValueType(std::move(*result));
    }

    [[nodiscard]] gr::DataSet<T> processDataSet(const gr::DataSet<T>& inData, std::span<const NamedTensor> extras, bool wantMeta, auto& metaOuts, std::vector<std::size_t>& metaCounts) {
        if (inData.signal_values.empty()) {
            return markedCopy(inData);
        }

        const std::size_t nSignals  = std::max(1UZ, inData.signal_names.size());
        const std::size_t inputSize = inData.signal_values.size() / nSignals;

        auto result = runInference({inData.signal_values.data(), inputSize}, extras, wantMeta, metaOuts, metaCounts);
        if (!result) {
            return markedCopy(inData);
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
        output.signal_values     = toValueType(std::move(*result));
        output.signal_ranges     = {gr::Range<T>{T(0), T(0)}};
        output.extents           = {static_cast<std::int32_t>(outSize)};
        output.meta_information  = {{}};
        output.timing_events     = {{}};

        return output;
    }

    [[nodiscard]] gr::DataSet<T> processHistoryBatch(std::span<const NamedTensor> extras, bool wantMeta, auto& metaOuts, std::vector<std::size_t>& metaCounts) {
        const std::size_t M      = history_depth;
        const std::size_t modelN = _session.modelN();

        std::vector<T> batchInput(M * modelN, T(0));
        for (std::size_t row = 0; row < M && row < _historyBuffer.size(); ++row) {
            const auto& ds      = _historyBuffer[row];
            const auto  nSigs   = std::max(1UZ, ds.signal_names.size());
            const auto  sigSize = ds.signal_values.size() / nSigs;

            if (sigSize == modelN || resample_mode == ResampleMode::None) {
                std::copy_n(ds.signal_values.begin(), std::min(sigSize, modelN), batchInput.begin() + static_cast<std::ptrdiff_t>(row * modelN));
            } else {
                std::span<const T> firstSig(ds.signal_values.data(), sigSize);
                std::span<T>       target(batchInput.data() + row * modelN, modelN);
                OnnxPreprocess<T>::resample(firstSig, target);
            }
        }

        std::vector<T> normalised(M * modelN);
        for (std::size_t row = 0; row < M; ++row) {
            std::span<const T> rowIn(batchInput.data() + row * modelN, modelN);
            std::span<T>       rowOut(normalised.data() + row * modelN, modelN);
            _preprocess.normalise(rowIn, rowOut);
        }

        auto result = runSession(normalised, extras, wantMeta, metaOuts, metaCounts);

        if (!result) {
            // copy before handleError: ErrorPolicy::Stop triggers stop(), which clears the buffer
            gr::DataSet<T> passThrough = _historyBuffer.back();
            handleError("processHistoryBatch()", result.error().message);
            markPassthrough(passThrough);
            return passThrough;
        }

        const auto&    latest = _historyBuffer.back();
        gr::DataSet<T> output;
        output.timestamp   = latest.timestamp;
        output.axis_names  = latest.axis_names;
        output.axis_units  = latest.axis_units;
        output.axis_values = latest.axis_values;

        output.signal_names      = {"inference_output"};
        output.signal_quantities = {""};
        output.signal_units      = {""};
        output.signal_values     = toValueType(std::move(*result));
        output.signal_ranges     = {gr::Range<T>{T(0), T(0)}};
        output.extents           = {static_cast<std::int32_t>(output.signal_values.size())};
        output.meta_information  = {{}};
        output.timing_events     = {{}};

        return output;
    }
};

} // namespace gr::blocks::onnx

#endif // GR_ONNX_INFERENCE_HPP
