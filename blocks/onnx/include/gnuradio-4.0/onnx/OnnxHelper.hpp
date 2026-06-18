#ifndef GR_ONNX_HELPER_HPP
#define GR_ONNX_HELPER_HPP

#include <gnuradio-4.0/Message.hpp>

#include <expected>
#include <filesystem>
#include <format>
#include <string>

namespace gr::blocks::onnx {

[[nodiscard]] inline std::expected<void, gr::Error> validateModelPath(const std::filesystem::path& modelPath) {
    const auto ext = modelPath.extension().string();

    if (ext == ".ort") {
        return {};
    }

    if (ext == ".onnx") {
#if GR_ONNX_MINIMAL_BUILD
        return std::unexpected(gr::Error{std::format("this build only supports .ort format models — convert with: "
                                                     "python -m onnxruntime.tools.convert_onnx_models_to_ort {}",
            modelPath.string())});
#else
        return {};
#endif
    }

#if GR_ONNX_MINIMAL_BUILD
    return std::unexpected(gr::Error{std::format("unsupported model format '{}' — supported: .ort", ext)});
#else
    return std::unexpected(gr::Error{std::format("unsupported model format '{}' — supported: .onnx, .ort", ext)});
#endif
}

} // namespace gr::blocks::onnx

#endif // GR_ONNX_HELPER_HPP
