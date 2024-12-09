#ifndef GR_ONNX_HELPER_HPP
#define GR_ONNX_HELPER_HPP

#include <gnuradio-4.0/Message.hpp>

#include <onnxruntime_cxx_api.h>

#include <expected>
#include <filesystem>
#include <format>
#include <string>
#include <string_view>

namespace gr::onnx {

[[nodiscard]] constexpr bool supportsOnnxFormat() noexcept {
#if GR_ONNX_MINIMAL_BUILD
    return false;
#else
    return true;
#endif
}

[[nodiscard]] constexpr bool isMinimalBuild() noexcept {
#if GR_ONNX_MINIMAL_BUILD
    return true;
#else
    return false;
#endif
}

[[nodiscard]] constexpr std::string_view supportedFormats() noexcept {
#if GR_ONNX_MINIMAL_BUILD
    return ".ort";
#else
    return ".onnx, .ort";
#endif
}

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

    return std::unexpected(gr::Error{std::format("unsupported model format '{}' — supported: {}", ext, supportedFormats())});
}

[[nodiscard]] inline bool detectOnnxFormatSupport() {
    const auto* api       = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    const char* buildInfo = api->GetBuildInfoString();
    return std::string_view(buildInfo).find("minimal_build") == std::string_view::npos;
}

struct BuildInfo {
    int                      apiVersion;
    std::string              buildString;
    bool                     minimalBuild;
    bool                     supportsOnnx;
    std::vector<std::string> providers;
};

[[nodiscard]] inline BuildInfo getBuildInfo() {
    const auto* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    BuildInfo info{.apiVersion = ORT_API_VERSION, .buildString = api->GetBuildInfoString(), .minimalBuild = isMinimalBuild(), .supportsOnnx = supportsOnnxFormat(), .providers = Ort::GetAvailableProviders()};

    return info;
}

} // namespace gr::onnx

#endif // GR_ONNX_HELPER_HPP
