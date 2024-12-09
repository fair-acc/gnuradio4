#ifndef GR_ONNX_SESSION_HPP
#define GR_ONNX_SESSION_HPP

#include <gnuradio-4.0/Message.hpp>
#include <gnuradio-4.0/algorithm/fileio/FileIo.hpp>
#include <gnuradio-4.0/onnx/OnnxHelper.hpp>

#include <onnxruntime_cxx_api.h>

#include <charconv>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gr::blocks::onnx {

struct ModelMetadata {
    std::size_t              inputSize          = 0; // primary input dimension (N)
    std::size_t              historyDepth       = 1; // M; 1 = no history, >1 = [1, M, N] input
    std::size_t              regressionChannels = 0;
    std::string              architecture;
    std::string              normaliseMode; // auto-detect hint from model metadata
    std::vector<std::size_t> inputShape;    // full input shape from model
    std::vector<std::size_t> outputShape;   // full output shape from model
};

struct OnnxSession {
    Ort::Env                    _env{ORT_LOGGING_LEVEL_WARNING, "gr-onnx"};
    std::optional<Ort::Session> _session;
    Ort::SessionOptions         _sessionOptions;
    Ort::RunOptions             _runOptions;

    std::vector<int64_t> _inputShape; // e.g. [1, 1, N]
    std::string          _inputName;
    std::string          _outputName;
    ModelMetadata        _metadata;
    std::vector<uint8_t> _modelData; // retained for ORT session lifetime

    OnnxSession() {
        _sessionOptions.SetIntraOpNumThreads(1);
        _sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    }

    [[nodiscard]] std::expected<void, gr::Error> load(std::string_view modelUri) {
        // resolve URI → raw bytes via FileIo (supports file:, http://, https://, memory)
        std::string uri(modelUri);
        if (!uri.starts_with("file:") && !uri.starts_with("http://") && !uri.starts_with("https://")) {
            // treat plain filesystem paths as file: URIs
            auto absPath = std::filesystem::absolute(uri).string();
            uri          = "file:" + absPath;
        }

        // validate format before fetching
        auto ext = std::filesystem::path(modelUri).extension().string();
        if (auto valid = gr::onnx::validateModelPath(modelUri); !valid) {
            return std::unexpected(valid.error());
        }

        auto reader = gr::algorithm::fileio::readAsync(uri);
        if (!reader) {
            return std::unexpected(gr::Error{std::format("failed to open model URI: {}", reader.error().message)});
        }

        auto data = reader->get();
        if (!data) {
            return std::unexpected(gr::Error{std::format("failed to read model data: {}", data.error().message)});
        }

        if (data->empty()) {
            return std::unexpected(gr::Error{"model data is empty"});
        }

        return loadFromMemory(std::move(*data), modelUri);
    }

    [[nodiscard]] std::expected<void, gr::Error> loadFromMemory(std::vector<uint8_t> data, std::string_view sourceUri = "<memory>") {
        _modelData = std::move(data);

        try {
            _session.emplace(_env, _modelData.data(), _modelData.size(), _sessionOptions);
        } catch (const Ort::Exception& e) {
            return std::unexpected(gr::Error{std::format("failed to load model: {}", e.what())});
        } catch (const std::exception& e) {
            return std::unexpected(gr::Error{std::format("failed to load model: {}", e.what())});
        }

        auto& session = *_session;

        if (session.GetInputCount() < 1 || session.GetOutputCount() < 1) {
            _session.reset();
            return std::unexpected(gr::Error{"model must have at least one input and one output"});
        }

        Ort::AllocatorWithDefaultOptions alloc;
        _inputName  = session.GetInputNameAllocated(0, alloc).get();
        _outputName = session.GetOutputNameAllocated(0, alloc).get();

        _metadata = {};
        if (!readMetadata()) {
            inferMetadataFromFilename(sourceUri);
        }

        readShapes();

        if (_metadata.inputSize == 0) {
            _session.reset();
            return std::unexpected(gr::Error{"could not determine model input size from metadata or filename"});
        }

        auto m      = static_cast<int64_t>(_metadata.historyDepth);
        auto n      = static_cast<int64_t>(_metadata.inputSize);
        _inputShape = {1, m, n};

        return {};
    }

    [[nodiscard]] std::expected<std::vector<float>, gr::Error> run(std::span<const float> input) {
        if (!_session) {
            return std::unexpected(gr::Error{"no model loaded"});
        }

        std::size_t expectedSize = _metadata.historyDepth * _metadata.inputSize;
        if (input.size() != expectedSize) {
            return std::unexpected(gr::Error{std::format("input size mismatch: expected {} got {}", expectedSize, input.size())});
        }

        auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        auto inputTensor = Ort::Value::CreateTensor<float>(memInfo, const_cast<float*>(input.data()), input.size(), _inputShape.data(), _inputShape.size());

        const char* inNames[]  = {_inputName.c_str()};
        const char* outNames[] = {_outputName.c_str()};

        std::vector<Ort::Value> outputs;
        try {
            outputs = _session->Run(_runOptions, inNames, &inputTensor, 1, outNames, 1);
        } catch (const Ort::Exception& e) {
            return std::unexpected(gr::Error{std::format("inference failed: {}", e.what())});
        } catch (const std::exception& e) {
            return std::unexpected(gr::Error{std::format("inference failed: {}", e.what())});
        }

        auto        info    = outputs[0].GetTensorTypeAndShapeInfo();
        std::size_t outSize = info.GetElementCount();
        const auto* outData = outputs[0].GetTensorData<float>();

        return std::vector<float>(outData, outData + outSize);
    }

    void reset() {
        _session.reset();
        _modelData.clear();
        _inputShape.clear();
        _inputName.clear();
        _outputName.clear();
        _metadata = {};
    }

    [[nodiscard]] std::size_t                     modelN() const noexcept { return _metadata.inputSize; }
    [[nodiscard]] std::size_t                     historyDepth() const noexcept { return _metadata.historyDepth; }
    [[nodiscard]] std::size_t                     regressionChannels() const noexcept { return _metadata.regressionChannels; }
    [[nodiscard]] const ModelMetadata&            metadata() const noexcept { return _metadata; }
    [[nodiscard]] bool                            isLoaded() const noexcept { return _session.has_value(); }
    [[nodiscard]] const std::vector<std::size_t>& inputShape() const noexcept { return _metadata.inputShape; }
    [[nodiscard]] const std::vector<std::size_t>& outputShape() const noexcept { return _metadata.outputShape; }

    template<typename SizeType = std::size_t>
    [[nodiscard]] std::vector<SizeType> modelIoShape() const {
        std::vector<SizeType> shape;
        shape.reserve(_metadata.inputShape.size() + _metadata.outputShape.size());
        for (auto d : _metadata.inputShape) {
            shape.push_back(static_cast<SizeType>(d));
        }
        for (auto d : _metadata.outputShape) {
            shape.push_back(static_cast<SizeType>(d));
        }
        return shape;
    }

private:
    bool readMetadata() {
#if GR_ONNX_MINIMAL_BUILD
        return false;
#else
        if (!_session) {
            return false;
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

            if (auto v = tryRead("input_size").or_else([&]() { return tryRead("fft_size"); })) {
                std::size_t parsed = 0;
                auto [ptr, ec]     = std::from_chars(v->data(), v->data() + v->size(), parsed);
                if (ec == std::errc()) {
                    _metadata.inputSize = parsed;
                }
            }
            if (auto v = tryRead("history_depth")) {
                std::size_t parsed = 0;
                auto [ptr2, ec2]   = std::from_chars(v->data(), v->data() + v->size(), parsed);
                if (ec2 == std::errc() && parsed > 0) {
                    _metadata.historyDepth = parsed;
                }
            }
            if (auto v = tryRead("n_regression_channels")) {
                std::size_t parsed = 0;
                auto [ptr, ec]     = std::from_chars(v->data(), v->data() + v->size(), parsed);
                if (ec == std::errc()) {
                    _metadata.regressionChannels = parsed;
                }
            }
            if (auto v = tryRead("architecture")) {
                _metadata.architecture = *v;
            }
            if (auto v = tryRead("normalise_mode")) {
                _metadata.normaliseMode = *v;
            }

            if (_metadata.regressionChannels == 0) {
                _metadata.regressionChannels = 8;
            }
            return _metadata.inputSize > 0;
        } catch (...) {
            return false;
        }
#endif
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

    void inferMetadataFromFilename(std::string_view modelPath) {
        auto stem = std::filesystem::path(modelPath).stem().string();
        auto nPos = stem.rfind('N');
        if (nPos != std::string::npos && nPos + 1 < stem.size()) {
            std::size_t parsed = 0;
            auto [ptr, ec]     = std::from_chars(stem.data() + nPos + 1, stem.data() + stem.size(), parsed);
            if (ec == std::errc() && parsed > 0) {
                _metadata.inputSize = parsed;
            }
        }
        _metadata.regressionChannels = 8;
    }
};

} // namespace gr::blocks::onnx

#endif // GR_ONNX_SESSION_HPP
