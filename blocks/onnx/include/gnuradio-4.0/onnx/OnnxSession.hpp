#ifndef GR_ONNX_SESSION_HPP
#define GR_ONNX_SESSION_HPP

#include <gnuradio-4.0/onnx/OnnxHelper.hpp>

#include <onnxruntime_cxx_api.h>

#include <charconv>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gr::blocks::onnx {

struct ModelMetadata {
    std::size_t fftSize            = 0;
    std::size_t regressionChannels = 0;
    std::string architecture;
};

struct OnnxSession {
    Ort::Env                    _env{ORT_LOGGING_LEVEL_WARNING, "gr-onnx"};
    std::optional<Ort::Session> _session;
    Ort::SessionOptions         _sessionOptions;
    Ort::RunOptions             _runOptions;

    std::vector<int64_t> _inputShape; // [1, 1, N]
    std::string          _inputName;
    std::string          _outputName;
    ModelMetadata        _metadata;

    OnnxSession() {
        _sessionOptions.SetIntraOpNumThreads(1);
        _sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    }

    [[nodiscard]] std::expected<void, std::string> load(std::string_view modelPath) {
        try {
            gr::onnx::validateModelPath(modelPath);
        } catch (const std::invalid_argument& e) {
            return std::unexpected(e.what());
        }

        try {
            _session.emplace(_env, std::string(modelPath).c_str(), _sessionOptions);
        } catch (const Ort::Exception& e) {
            return std::unexpected(std::string("failed to load model: ") + e.what());
        } catch (const std::exception& e) {
            return std::unexpected(std::string("failed to load model: ") + e.what());
        }

        auto& session = *_session;

        if (session.GetInputCount() < 1 || session.GetOutputCount() < 1) {
            _session.reset();
            return std::unexpected("model must have at least one input and one output");
        }

        // read input/output names
        Ort::AllocatorWithDefaultOptions alloc;
        _inputName  = session.GetInputNameAllocated(0, alloc).get();
        _outputName = session.GetOutputNameAllocated(0, alloc).get();

        // read model metadata — shape introspection via GetInputTypeInfo().GetShape()
        // is unreliable across ORT versions (crashes on ORT ≤ 1.19 with tf2onnx models),
        // so we prefer metadata or filename convention for model dimensions
        _metadata = {};
        if (!readMetadata()) {
            inferMetadataFromFilename(modelPath);
        }

        if (_metadata.fftSize == 0) {
            _session.reset();
            return std::unexpected("could not determine model FFT size from metadata or filename");
        }

        // construct input shape from metadata: [batch=1, channels=1, N]
        auto n      = static_cast<int64_t>(_metadata.fftSize);
        _inputShape = {1, 1, n};

        return {};
    }

    [[nodiscard]] std::expected<std::vector<float>, std::string> run(std::span<const float> input) {
        if (!_session) {
            return std::unexpected("no model loaded");
        }

        std::size_t expectedSize = _metadata.fftSize;
        if (input.size() != expectedSize) {
            return std::unexpected(std::string("input size mismatch: expected ") + std::to_string(expectedSize) + " got " + std::to_string(input.size()));
        }

        auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        auto inputTensor = Ort::Value::CreateTensor<float>(memInfo, const_cast<float*>(input.data()), input.size(), _inputShape.data(), _inputShape.size());

        const char* inNames[]  = {_inputName.c_str()};
        const char* outNames[] = {_outputName.c_str()};

        std::vector<Ort::Value> outputs;
        try {
            outputs = _session->Run(_runOptions, inNames, &inputTensor, 1, outNames, 1);
        } catch (const Ort::Exception& e) {
            return std::unexpected(std::string("inference failed: ") + e.what());
        } catch (const std::exception& e) {
            return std::unexpected(std::string("inference failed: ") + e.what());
        }

        auto        info    = outputs[0].GetTensorTypeAndShapeInfo();
        std::size_t outSize = info.GetElementCount();
        const auto* outData = outputs[0].GetTensorData<float>();

        return std::vector<float>(outData, outData + outSize);
    }

    void reset() {
        _session.reset();
        _inputShape.clear();
        _inputName.clear();
        _outputName.clear();
        _metadata = {};
    }

    [[nodiscard]] std::size_t          modelN() const noexcept { return _metadata.fftSize; }
    [[nodiscard]] std::size_t          regressionChannels() const noexcept { return _metadata.regressionChannels; }
    [[nodiscard]] const ModelMetadata& metadata() const noexcept { return _metadata; }
    [[nodiscard]] bool                 isLoaded() const noexcept { return _session.has_value(); }

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

            if (auto v = tryRead("fft_size")) {
                std::size_t parsed = 0;
                auto [ptr, ec]     = std::from_chars(v->data(), v->data() + v->size(), parsed);
                if (ec == std::errc()) {
                    _metadata.fftSize = parsed;
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

            if (_metadata.regressionChannels == 0) {
                _metadata.regressionChannels = 8;
            }
            return _metadata.fftSize > 0;
        } catch (...) {
            return false;
        }
#endif
    }

    void inferMetadataFromFilename(std::string_view modelPath) {
        // convention: peak_detector_N1024.onnx → N=1024
        auto stem = std::filesystem::path(modelPath).stem().string();
        auto nPos = stem.rfind('N');
        if (nPos != std::string::npos && nPos + 1 < stem.size()) {
            std::size_t parsed = 0;
            auto [ptr, ec]     = std::from_chars(stem.data() + nPos + 1, stem.data() + stem.size(), parsed);
            if (ec == std::errc() && parsed > 0) {
                _metadata.fftSize = parsed;
            }
        }
        _metadata.regressionChannels = 8;
    }
};

} // namespace gr::blocks::onnx

#endif // GR_ONNX_SESSION_HPP
