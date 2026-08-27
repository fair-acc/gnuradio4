#ifndef GR_ONNX_SESSION_HPP
#define GR_ONNX_SESSION_HPP

#include <gnuradio-4.0/Compression.hpp>
#include <gnuradio-4.0/Message.hpp>
#include <gnuradio-4.0/algorithm/fileio/FileIo.hpp>

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gr::blocks::onnx {

struct NamedTensor {
    std::string              name;
    std::vector<std::size_t> shape;
    std::vector<float>       values;
};

// marks frames forwarded unchanged, so downstream can tell forwarded input from inference output;
// bare Tensor<T> outputs have no metadata and stay unmarked by design
inline constexpr const char* kPassthroughKey = "onnx_passthrough";

template<typename TDataSet>
void markPassthrough(TDataSet& dataSet) {
    if (dataSet.meta_information.empty()) {
        dataSet.meta_information.resize(1UZ);
    }
    dataSet.meta_information[0][kPassthroughKey] = true;
}

template<typename TDataSet>
[[nodiscard]] bool isMarkedPassthrough(const TDataSet& dataSet) {
    if (dataSet.meta_information.empty()) {
        return false;
    }
    const auto& meta = dataSet.meta_information[0];
    const auto  it   = meta.find(std::pmr::string(kPassthroughKey));
    if (it == meta.end()) {
        return false;
    }
    const bool* flag = it->second.template get_if<bool>();
    return flag != nullptr && *flag;
}

struct ModelMetadata {
    std::size_t              inputSize       = 0; // primary input dimension (N)
    std::size_t              historyDepth    = 1; // M; 1 = no history, >1 = [1, M, N] input
    std::size_t              nMaxPeaks       = 0; // peaks-tensor models: K
    std::size_t              nPeakProperties = 0; // peaks-tensor models: properties per peak
    std::string              architecture;
    std::string              normaliseMode;  // auto-detect hint from model metadata
    std::string              normaliseExpr;  // ExprTk expression for NormaliseMode::Expression
    std::string              propertyLayout; // comma-separated peaks-tensor column names
    std::string              scoreOutput;    // output carrying the acceptance score
    std::vector<std::size_t> inputShape;     // full input shape from model
    std::vector<std::size_t> outputShape;    // full output shape from model
};

// Sizes parsed from model metadata are otherwise unbounded and drive allocations and index
// arithmetic; anything beyond this is a malformed or hostile file rather than a large model.
inline constexpr std::size_t kMaxMetadataExtent = 1UZ << 26; // 67 M elements, 256 MiB as float

/**
 * @brief RAII owner of one ONNX Runtime session and its cached model introspection data.
 *
 * class, not struct: the Ort::Session must never outlive the model buffer it was created
 * from, and the cached I/O names, shapes and metadata must stay consistent with the loaded
 * session — state changes only through load()/loadFromMemory()/reset().
 */
class OnnxSession {
    // ERROR, not WARNING: ORT warns on every load about overridable initializers appearing in
    // graph inputs, which is the mechanism working as designed and therefore pure noise.
#if defined(__EMSCRIPTEN__)
    // --enable_wasm_threads makes every session take its thread pools from the environment, so an
    // env built without them fails every load
    Ort::ThreadingOptions _threadingOptions{};
    Ort::Env              _env{_threadingOptions, ORT_LOGGING_LEVEL_ERROR, "gr-onnx"};
#else
    Ort::Env _env{ORT_LOGGING_LEVEL_ERROR, "gr-onnx"};
#endif
    std::optional<Ort::Session> _session;
    Ort::SessionOptions         _sessionOptions;
    Ort::RunOptions             _runOptions;
    std::string                 _executionProvider = "cpu";

    std::vector<int64_t>     _inputShape; // e.g. [1, 1, N]
    std::string              _inputName;
    std::string              _outputName;
    std::vector<std::string> _outputNames;             // all outputs, in graph order
    std::vector<NamedTensor> _overridableInitializers; // name + shape only, values stay empty
    ModelMetadata            _metadata;
    std::vector<uint8_t>     _modelData; // retained for ORT session lifetime

    // I/O bound once per load: shapes are fixed, so Run reads/writes these buffers in place
    // (buffers before the Ort::Values wrapping them — reverse destruction order)
    std::size_t                           _intraOpThreads = 4UZ; // 4: measured 2.5x over one thread, 6 adds only ~20 % more
    std::vector<float>                    _inputBuffer;
    std::vector<std::vector<float>>       _outputBuffers;
    std::vector<std::vector<std::size_t>> _outputShapes; // concrete, from shape info or the load-time probe
    Ort::Value                            _boundInput{nullptr};
    std::vector<Ort::Value>               _boundOutputs;

public:
    // applied on the next load()/loadFromMemory(); validated there against availableProviders()
    void setExecutionProvider(std::string_view provider) { _executionProvider = provider; }
    void setIntraOpThreads(std::size_t threads) { _intraOpThreads = std::max<std::size_t>(1UZ, threads); }

    [[nodiscard]] static std::vector<std::string> availableProviders() {
        try {
            return Ort::GetAvailableProviders();
        } catch (...) {
            return {"CPUExecutionProvider"}; // the CPU provider exists in every ORT build
        }
    }

    [[nodiscard]] std::expected<void, gr::Error> load(std::string_view modelUri) {
        // FileIo accepts file:, http:// and https:// URIs; a bare path is rewritten to file: below
        std::string uri(modelUri);
        if (!uri.starts_with("file:") && !uri.starts_with("http://") && !uri.starts_with("https://")) {
            auto absPath = std::filesystem::absolute(uri).string();
            uri          = "file:" + absPath;
        }

#if defined(__EMSCRIPTEN__)
        // fileio cannot make progress on the main WASM thread, and an embedded model is already resident
        if (!uri.starts_with("http://") && !uri.starts_with("https://")) {
            const std::string localPath = uri.starts_with("file:") ? uri.substr(5) : uri;
            std::ifstream     in(localPath, std::ios::binary);
            if (!in) {
                return std::unexpected(gr::Error{std::format("failed to open model file '{}'", localPath)});
            }
            std::vector<std::uint8_t> raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            if (raw.empty()) {
                return std::unexpected(gr::Error{std::format("model data is empty: '{}'", modelUri)});
            }
            if (raw.size() > 2UZ && raw[0] == 0x1FU && raw[1] == 0x8BU) { // gzip magic
                const auto inflated = gr::compression::decompress(std::as_bytes(std::span(raw)), gr::compression::Format::gzip);
                if (!inflated) {
                    return std::unexpected(gr::Error{std::format("failed to inflate model '{}'", modelUri)});
                }
                std::vector<std::uint8_t> bytes(inflated->size());
                std::ranges::transform(*inflated, bytes.begin(), [](std::byte b) { return static_cast<std::uint8_t>(b); });
                return loadFromMemory(std::move(bytes), modelUri);
            }
            return loadFromMemory(std::move(raw), modelUri);
        }
#endif

        auto reader = gr::algorithm::fileio::readAsync(uri);
        if (!reader) {
            return std::unexpected(gr::Error{std::format("failed to open model URI '{}': {}", modelUri, reader.error().message)});
        }

        auto data = reader->get();
        if (!data) {
            return std::unexpected(gr::Error{std::format("failed to read model data from '{}': {}", modelUri, data.error().message)});
        }

        if (data->empty()) {
            return std::unexpected(gr::Error{std::format("model data is empty: '{}'", modelUri)});
        }

        return loadFromMemory(std::move(*data), modelUri);
    }

    [[nodiscard]] std::expected<void, gr::Error> loadFromMemory(std::vector<uint8_t> data, std::string_view sourceUri = "<memory>") {
        _modelData = std::move(data);

        // reset(), not fall back: keeping a previous CPU session alive would make a failed
        // GPU provider request look like a successful GPU deployment
        if (auto configured = configureSessionOptions(); !configured) {
            reset();
            return std::unexpected(configured.error());
        }

        try {
            _session.emplace(_env, _modelData.data(), _modelData.size(), _sessionOptions);
        } catch (const std::exception& e) {
            reset();
            return std::unexpected(gr::Error{std::format("failed to load model '{}': {}", sourceUri, e.what())});
        }

        auto& session = *_session;

        try {
            const std::size_t nInputs  = session.GetInputCount();
            const std::size_t nOutputs = session.GetOutputCount();
            if (nInputs < 1 || nOutputs < 1) {
                reset();
                return std::unexpected(gr::Error{std::format("model '{}' must have at least one input and one output (has {} inputs, {} outputs)", sourceUri, nInputs, nOutputs)});
            }

            Ort::AllocatorWithDefaultOptions alloc;
            _inputName  = session.GetInputNameAllocated(0, alloc).get();
            _outputName = session.GetOutputNameAllocated(0, alloc).get();
            _outputNames.clear();
            for (std::size_t i = 0; i < nOutputs; ++i) {
                _outputNames.emplace_back(session.GetOutputNameAllocated(i, alloc).get());
            }
        } catch (const std::exception& e) {
            reset();
            return std::unexpected(gr::Error{std::format("failed to load model '{}': {}", sourceUri, e.what())});
        }

        _overridableInitializers.clear();
        try {
            Ort::AllocatorWithDefaultOptions alloc;
            for (std::size_t i = 0; i < session.GetOverridableInitializerCount(); ++i) {
                NamedTensor initializer{.name = session.GetOverridableInitializerNameAllocated(i, alloc).get(), .shape = {}, .values = {}};
                for (auto dim : session.GetOverridableInitializerTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape()) {
                    initializer.shape.push_back(dim > 0 ? static_cast<std::size_t>(dim) : 1UZ);
                }
                _overridableInitializers.push_back(std::move(initializer));
            }
        } catch (...) {
            _overridableInitializers.clear(); // initializer introspection is best-effort
        }

        _metadata = {};
        readMetadata();
        readShapes();

        // metadata absent: N is the primary input's trailing dimension (0 = symbolic)
        if (_metadata.inputSize == 0 && !_metadata.inputShape.empty()) {
            _metadata.inputSize = _metadata.inputShape.back();
        }
        if (_metadata.inputSize == 0) {
            reset();
            return std::unexpected(gr::Error{std::format("could not determine the input size of '{}': its declared input shape is {} and the trailing "
                                                         "dimension is symbolic, so add an 'input_size' metadata key to the model or re-export it with a "
                                                         "fixed trailing dimension",
                sourceUri, formatShape(_metadata.inputShape))});
        }

        auto shape = inferenceInputShape(_metadata.inputShape, _metadata.historyDepth, _metadata.inputSize, sourceUri);
        if (!shape) {
            reset();
            return std::unexpected(shape.error());
        }
        _inputShape = std::move(*shape);

        if (auto bound = bindIo(sourceUri); !bound) {
            reset();
            return std::unexpected(bound.error());
        }
        _metadata.outputShape = _outputShapes.front();

        return {};
    }

    [[nodiscard]] std::expected<std::vector<float>, gr::Error> run(std::span<const float> input, std::span<const NamedTensor> extraInputs = {}) {
        if (_boundOutputs.empty()) {
            return std::unexpected(gr::Error{"no model loaded"});
        }
        const char* outNames[] = {_outputName.c_str()};
        if (auto ok = runBound(input, extraInputs, outNames, {_boundOutputs.data(), 1UZ}); !ok) {
            return std::unexpected(ok.error());
        }
        return _outputBuffers.front();
    }

    // run() above returns only output 0, which lets ORT prune the branches feeding the others
    [[nodiscard]] std::expected<std::vector<NamedTensor>, gr::Error> runAll(std::span<const float> input, std::span<const NamedTensor> extraInputs = {}) {
        std::vector<const char*> outNames;
        outNames.reserve(_outputNames.size());
        for (const auto& name : _outputNames) {
            outNames.push_back(name.c_str());
        }

        if (auto ok = runBound(input, extraInputs, outNames, _boundOutputs); !ok) {
            return std::unexpected(ok.error());
        }

        std::vector<NamedTensor> result;
        result.reserve(_outputNames.size());
        for (std::size_t i = 0; i < _outputNames.size(); ++i) {
            result.push_back(NamedTensor{.name = _outputNames[i], .shape = _outputShapes[i], .values = _outputBuffers[i]});
        }
        return result;
    }

    void reset() {
        _session.reset();
        _modelData.clear();
        _inputShape.clear();
        _inputName.clear();
        _outputName.clear();
        _outputNames.clear();
        _overridableInitializers.clear();
        _metadata   = {};
        _boundInput = Ort::Value{nullptr};
        _boundOutputs.clear();
        _inputBuffer.clear();
        _outputBuffers.clear();
        _outputShapes.clear();
    }

    [[nodiscard]] std::size_t                     modelN() const noexcept { return _metadata.inputSize; }
    [[nodiscard]] std::size_t                     historyDepth() const noexcept { return _metadata.historyDepth; }
    [[nodiscard]] const ModelMetadata&            metadata() const noexcept { return _metadata; }
    [[nodiscard]] bool                            isLoaded() const noexcept { return _session.has_value(); }
    [[nodiscard]] const std::vector<std::string>& outputNames() const noexcept { return _outputNames; }
    [[nodiscard]] const std::vector<NamedTensor>& overridableInitializers() const noexcept { return _overridableInitializers; }

    template<typename SizeType = std::size_t>
    [[nodiscard]] std::vector<SizeType> modelInputShape() const {
        return {_metadata.inputShape.begin(), _metadata.inputShape.end()};
    }

    template<typename SizeType = std::size_t>
    [[nodiscard]] std::vector<SizeType> modelOutputShape() const {
        return {_metadata.outputShape.begin(), _metadata.outputShape.end()};
    }

private:
    // fresh SessionOptions per load (appended providers accumulate, so never reuse). The data
    // plane deliberately stays on CPU tensors even under CUDA/TensorRT/ROCm — ORT performs the
    // device transfers internally, so no device-memory handling is needed here.
    [[nodiscard]] std::expected<void, gr::Error> configureSessionOptions() {
        try {
            _sessionOptions = Ort::SessionOptions{};
            _sessionOptions.SetIntraOpNumThreads(static_cast<int>(_intraOpThreads));
            // DISABLE_ALL, deliberately: measured vs ENABLE_EXTENDED (2026-08, fixtures + v31 peak
            // regressor) the latency difference is within run-to-run noise (≤2 %), while DISABLE_ALL
            // is the one level guaranteed valid for pre-optimised .ort models in minimal builds
            _sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);

            if (_executionProvider == "cpu") {
                return {}; // ORT's built-in default provider
            }

            const std::vector<std::string> available       = availableProviders();
            auto                           requireProvider = [&](std::string_view ortName) -> std::expected<void, gr::Error> {
                if (std::ranges::contains(available, ortName)) {
                    return {};
                }
                return std::unexpected(gr::Error{std::format("execution provider '{}' ({}) is not available in the linked ONNX Runtime; available: {}", _executionProvider, ortName, available)});
            };

            if (_executionProvider == "cuda") {
                if (auto ok = requireProvider("CUDAExecutionProvider"); !ok) {
                    return ok;
                }
                OrtCUDAProviderOptions cudaOptions{};
                _sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
                return {};
            }
            if (_executionProvider == "tensorrt") {
                if (auto ok = requireProvider("TensorrtExecutionProvider"); !ok) {
                    return ok;
                }
                OrtTensorRTProviderOptions trtOptions{};
                _sessionOptions.AppendExecutionProvider_TensorRT(trtOptions);
                if (std::ranges::contains(available, "CUDAExecutionProvider")) { // CUDA handles the ops TensorRT rejects
                    OrtCUDAProviderOptions cudaOptions{};
                    _sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
                }
                return {};
            }
            if (_executionProvider == "rocm") {
                if (auto ok = requireProvider("ROCMExecutionProvider"); !ok) {
                    return ok;
                }
                OrtROCMProviderOptions rocmOptions{};
                _sessionOptions.AppendExecutionProvider_ROCM(rocmOptions);
                return {};
            }
            return std::unexpected(gr::Error{std::format("unknown execution provider '{}': supported are cpu, cuda, tensorrt, rocm; the linked ONNX Runtime provides: {}", _executionProvider, available)});
        } catch (const std::exception& e) {
            return std::unexpected(gr::Error{std::format("failed to configure execution provider '{}': {}", _executionProvider, e.what())});
        }
    }

    // runs into the I/O bound at load, so the extras-free fast path allocates nothing per call;
    // extraInputs override overridable initializers by name and wrap the caller's storage
    [[nodiscard]] std::expected<void, gr::Error> runBound(std::span<const float> input, std::span<const NamedTensor> extraInputs, std::span<const char* const> outNames, std::span<Ort::Value> outValues) {
        if (!_session) {
            return std::unexpected(gr::Error{"no model loaded"});
        }
        if (input.size() != _inputBuffer.size()) {
            return std::unexpected(gr::Error{std::format("input size mismatch: expected {} got {}", _inputBuffer.size(), input.size())});
        }
        std::ranges::copy(input, _inputBuffer.begin());

        try {
            if (extraInputs.empty()) {
                const char* inName = _inputName.c_str();
                _session->Run(_runOptions, &inName, &_boundInput, 1UZ, outNames.data(), outValues.data(), outValues.size());
                return {};
            }

            auto                     memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            std::vector<Ort::Value>  inputTensors;
            std::vector<const char*> inNames;
            inputTensors.reserve(1UZ + extraInputs.size());
            inNames.reserve(1UZ + extraInputs.size());
            inputTensors.push_back(Ort::Value::CreateTensor<float>(memInfo, _inputBuffer.data(), _inputBuffer.size(), _inputShape.data(), _inputShape.size()));
            inNames.push_back(_inputName.c_str());
            std::vector<std::vector<int64_t>> extraShapes(extraInputs.size());
            for (std::size_t i = 0; i < extraInputs.size(); ++i) {
                const NamedTensor& extra = extraInputs[i];
                if (extra.shape.empty()) {
                    extraShapes[i] = {static_cast<int64_t>(extra.values.size())};
                } else {
                    for (auto dim : extra.shape) {
                        extraShapes[i].push_back(static_cast<int64_t>(dim));
                    }
                }
                inputTensors.push_back(Ort::Value::CreateTensor<float>(memInfo, const_cast<float*>(extra.values.data()), extra.values.size(), extraShapes[i].data(), extraShapes[i].size()));
                inNames.push_back(extra.name.c_str());
            }
            _session->Run(_runOptions, inNames.data(), inputTensors.data(), inputTensors.size(), outNames.data(), outValues.data(), outValues.size());
            return {};
        } catch (const std::exception& e) {
            return std::unexpected(gr::Error{std::format("inference failed: {}", e.what())});
        }
    }

    // symbolic output dims are pinned by one zero-filled probe inference at load
    [[nodiscard]] std::expected<void, gr::Error> bindIo(std::string_view sourceUri) {
        const std::size_t nOutputs = _outputNames.size();
        _outputShapes.assign(nOutputs, {});
        if (_metadata.inputSize != 0UZ && _metadata.historyDepth > kMaxMetadataExtent / _metadata.inputSize) {
            return std::unexpected(gr::Error{std::format("model '{}' declares an input of {} x {} elements, beyond the {} element limit", //
                sourceUri, _metadata.historyDepth, _metadata.inputSize, kMaxMetadataExtent)});
        }
        _inputBuffer.assign(_metadata.historyDepth * _metadata.inputSize, 0.f);

        bool allConcrete = true;
        try {
            for (std::size_t i = 0; i < nOutputs; ++i) {
                for (auto dim : _session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape()) {
                    allConcrete = allConcrete && dim > 0;
                    _outputShapes[i].push_back(dim > 0 ? static_cast<std::size_t>(dim) : 0UZ);
                }
            }
        } catch (...) {
            allConcrete = false;
        }

        try {
            auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            if (!allConcrete) {
                Ort::Value               probeInput = Ort::Value::CreateTensor<float>(memInfo, _inputBuffer.data(), _inputBuffer.size(), _inputShape.data(), _inputShape.size());
                const char*              inName     = _inputName.c_str();
                std::vector<const char*> outNames;
                outNames.reserve(nOutputs);
                for (const auto& name : _outputNames) {
                    outNames.push_back(name.c_str());
                }
                std::vector<Ort::Value> probed = _session->Run(_runOptions, &inName, &probeInput, 1UZ, outNames.data(), outNames.size());
                for (std::size_t i = 0; i < probed.size(); ++i) {
                    _outputShapes[i].clear();
                    for (auto dim : probed[i].GetTensorTypeAndShapeInfo().GetShape()) {
                        _outputShapes[i].push_back(static_cast<std::size_t>(std::max<int64_t>(dim, 0)));
                    }
                }
            }

            _boundInput = Ort::Value::CreateTensor<float>(memInfo, _inputBuffer.data(), _inputBuffer.size(), _inputShape.data(), _inputShape.size());
            _outputBuffers.assign(nOutputs, {});
            _boundOutputs.clear();
            _boundOutputs.reserve(nOutputs);
            for (std::size_t i = 0; i < nOutputs; ++i) {
                const std::size_t count = std::accumulate(_outputShapes[i].begin(), _outputShapes[i].end(), 1UZ, std::multiplies<>{});
                _outputBuffers[i].assign(count, 0.f);
                std::vector<int64_t> shape(_outputShapes[i].begin(), _outputShapes[i].end());
                _boundOutputs.push_back(Ort::Value::CreateTensor<float>(memInfo, _outputBuffers[i].data(), count, shape.data(), shape.size()));
            }
            return {};
        } catch (const std::exception& e) {
            return std::unexpected(gr::Error{std::format("failed to bind model I/O for model '{}': {}", sourceUri, e.what())});
        }
    }

    void readMetadata() {
        // attempted on every build: GetModelMetadata() carries no ORT_MINIMAL_BUILD guard, and the
        // catch below already falls back to shape-derived values where it is not served
        if (!_session) {
            return;
        }
        try {
            auto                             meta = _session->GetModelMetadata();
            Ort::AllocatorWithDefaultOptions alloc;

            auto tryRead = [&](const char* key) -> std::optional<std::string> {
                auto val = meta.LookupCustomMetadataMapAllocated(key, alloc);
                if (val) {
                    return std::string(val.get());
                }
                return std::nullopt;
            };
            auto tryReadSize = [&](const char* key) -> std::optional<std::size_t> {
                auto v = tryRead(key);
                if (!v) {
                    return std::nullopt;
                }
                std::size_t parsed = 0;
                auto [ptr, ec]     = std::from_chars(v->data(), v->data() + v->size(), parsed);
                if (ec != std::errc() || parsed > kMaxMetadataExtent) {
                    return std::nullopt; // a size this large is a malformed or hostile model, not a big one
                }
                return std::optional{parsed};
            };

            if (auto v = tryReadSize("input_size").or_else([&]() { return tryReadSize("fft_size"); })) {
                _metadata.inputSize = *v;
            }
            if (auto v = tryReadSize("history_depth"); v && *v > 0) {
                _metadata.historyDepth = *v;
            }
            if (auto v = tryReadSize("n_max_peaks")) {
                _metadata.nMaxPeaks = *v;
            }
            if (auto v = tryReadSize("n_peak_properties")) {
                _metadata.nPeakProperties = *v;
            }
            if (auto v = tryRead("property_layout")) {
                _metadata.propertyLayout = *v;
            }
            if (auto v = tryRead("score_output")) {
                _metadata.scoreOutput = *v;
            }
            if (auto v = tryRead("architecture")) {
                _metadata.architecture = *v;
            }
            if (auto v = tryRead("normalise_mode")) {
                _metadata.normaliseMode = *v;
            }
            if (auto v = tryRead("normalise_expr")) {
                _metadata.normaliseExpr = *v;
            }
        } catch (...) {
            // metadata is best-effort — shape-derived values fill the gaps
        }
    }

    // A symbolic dimension reads back as 0. The batch axis is always pinned to 1 -- the block runs
    // one inference per frame -- so only a symbolic trailing dimension is fatal.
    [[nodiscard]] static std::string formatShape(std::span<const std::size_t> shape) {
        if (shape.empty()) {
            return "[unknown]";
        }
        std::string out = "[";
        for (std::size_t i = 0UZ; i < shape.size(); ++i) {
            out += (shape[i] == 0UZ) ? "symbolic" : std::to_string(shape[i]);
            if (i + 1UZ < shape.size()) {
                out += ", ";
            }
        }
        return out + "]";
    }

    // Feed the rank the model declares rather than imposing one: [features] and [batch, features]
    // are what most exported classifiers and regressors carry, and only history models need
    // [batch, frames, features].
    [[nodiscard]] static std::expected<std::vector<int64_t>, gr::Error> inferenceInputShape(std::span<const std::size_t> declared, std::size_t historyDepth, std::size_t inputSize, std::string_view sourceUri) {
        const std::size_t rank = declared.empty() ? 3UZ : declared.size();
        const auto        n    = static_cast<int64_t>(inputSize);
        if (historyDepth > 1UZ && rank != 3UZ) {
            return std::unexpected(gr::Error{std::format("history_depth {} needs a rank-3 [batch, frames, {}] input, but '{}' declares rank {} {}", historyDepth, inputSize, sourceUri, rank, formatShape(declared))});
        }
        switch (rank) {
        case 1UZ: return std::vector<int64_t>{n};
        case 2UZ: return std::vector<int64_t>{1, n};
        case 3UZ: return std::vector<int64_t>{1, static_cast<int64_t>(historyDepth), n};
        default: return std::unexpected(gr::Error{std::format("'{}' declares a rank-{} input {}: this block feeds rank 1 [{}], rank 2 [batch, {}] or rank 3 [batch, frames, {}]", sourceUri, rank, formatShape(declared), inputSize, inputSize, inputSize)});
        }
    }

    void readShapes() {
        if (!_session) {
            return;
        }
        try {
            auto inputInfo  = _session->GetInputTypeInfo(0);
            auto inputShape = inputInfo.GetTensorTypeAndShapeInfo().GetShape();
            _metadata.inputShape.clear();
            for (auto dim : inputShape) {
                _metadata.inputShape.push_back(dim > 0 ? static_cast<std::size_t>(dim) : 0);
            }
            if (_metadata.historyDepth == 1 && _metadata.inputShape.size() == 3 && _metadata.inputShape[1] > 1) {
                _metadata.historyDepth = _metadata.inputShape[1];
            }
            auto outputInfo  = _session->GetOutputTypeInfo(0);
            auto outputShape = outputInfo.GetTensorTypeAndShapeInfo().GetShape();
            _metadata.outputShape.clear();
            for (auto dim : outputShape) {
                _metadata.outputShape.push_back(dim > 0 ? static_cast<std::size_t>(dim) : 0);
            }
        } catch (...) {
            // shape introspection is best-effort
        }
    }
};

// std::vector<float>/<double> overrides also land here: pmt::Value stores them as Tensor<T>
[[nodiscard]] inline std::expected<std::vector<float>, std::string> overrideToValues(const gr::pmt::Value& value, std::size_t elemCount) {
    auto fromTensor = [elemCount]<typename TTensor>(const TTensor& tensor) -> std::expected<std::vector<float>, std::string> {
        if (tensor.size() != elemCount) {
            return std::unexpected(std::format("size mismatch: override has {} elements, initializer expects {}", tensor.size(), elemCount));
        }
        std::vector<float> converted(tensor.size());
        std::ranges::transform(tensor, converted.begin(), [](auto v) { return static_cast<float>(v); });
        return converted;
    };
    if (auto tensorF = value.get_if<gr::TensorView<float>>()) {
        return fromTensor(*tensorF);
    }
    if (auto tensorD = value.get_if<gr::TensorView<double>>()) {
        return fromTensor(*tensorD);
    }
    auto scalar = [&value]() -> std::optional<float> {
        if (const auto* f = value.get_if<float>()) {
            return *f;
        }
        if (const auto* d = value.get_if<double>()) {
            return static_cast<float>(*d);
        }
        if (const auto* i32 = value.get_if<std::int32_t>()) {
            return static_cast<float>(*i32);
        }
        if (const auto* i64 = value.get_if<std::int64_t>()) {
            return static_cast<float>(*i64);
        }
        if (const auto* u32 = value.get_if<std::uint32_t>()) {
            return static_cast<float>(*u32);
        }
        return std::nullopt;
    }();
    if (scalar) {
        return std::vector<float>(elemCount, *scalar);
    }
    return std::unexpected("unsupported type: expected a scalar (float/double/int32/int64/uint32) or a size-matched Tensor/vector of float/double");
}

// shared by every block exposing a model_overrides setting, so the precedence and rejection rules
// cannot drift apart between them; rejected entries are reported rather than silently dropped
template<typename THigherPrecedence, typename TReport>
[[nodiscard]] std::vector<NamedTensor> buildOverrideInputs(const gr::property_map& overrides, const std::vector<NamedTensor>& initializers, THigherPrecedence&& higherPrecedence, TReport&& report) {
    std::vector<NamedTensor> extras;
    for (std::size_t i = 0; i < initializers.size(); ++i) {
        const NamedTensor& init      = initializers[i];
        const std::size_t  elemCount = std::accumulate(init.shape.begin(), init.shape.end(), 1UZ, std::multiplies<>{});

        std::optional<std::vector<float>> values = higherPrecedence(i, elemCount);
        if (!values) {
            if (auto it = overrides.find(std::pmr::string(init.name)); it != overrides.end()) {
                auto converted = overrideToValues(it->second, elemCount);
                if (!converted) {
                    report(std::format("model_overrides['{}']: {}", init.name, converted.error()));
                    continue;
                }
                values = std::move(*converted);
            }
        }
        if (values) {
            extras.push_back(NamedTensor{.name = init.name, .shape = init.shape, .values = std::move(*values)});
        }
    }
    for (const auto& [name, value] : overrides) {
        if (std::ranges::none_of(initializers, [&name](const NamedTensor& init) { return init.name == std::string_view(name); })) {
            report(std::format("model_overrides['{}'] does not name an overridable initializer of the loaded model", name));
        }
    }
    return extras;
}

} // namespace gr::blocks::onnx

#endif // GR_ONNX_SESSION_HPP
