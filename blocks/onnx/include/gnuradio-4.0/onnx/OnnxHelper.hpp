#ifndef GR_ONNX_HELPER_HPP
#define GR_ONNX_HELPER_HPP

#include <onnxruntime_cxx_api.h>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

namespace gr::onnx {

/**
 * @brief Check if this build supports full ONNX format (.onnx files)
 *
 * Minimal builds (ENABLE_ONNX_INTEGRATION=on) only support .ort format.
 * Full builds (ENABLE_ONNX_INTEGRATION=opt with system packages) support both.
 */
[[nodiscard]] constexpr bool supportsOnnxFormat() noexcept {
#if GR_ONNX_MINIMAL_BUILD
    return false;
#else
    return true;
#endif
}

/**
 * @brief Check if this build is a minimal/embedded build
 */
[[nodiscard]] constexpr bool isMinimalBuild() noexcept {
#if GR_ONNX_MINIMAL_BUILD
    return true;
#else
    return false;
#endif
}

/**
 * @brief Get the supported model format(s) as a string
 */
[[nodiscard]] constexpr std::string_view supportedFormats() noexcept {
#if GR_ONNX_MINIMAL_BUILD
    return ".ort";
#else
    return ".onnx, .ort";
#endif
}

/**
 * @brief Validate that a model file has a supported extension
 *
 * @throws std::invalid_argument if the format is not supported
 */
inline void validateModelPath(const std::filesystem::path& modelPath) {
    const auto ext = modelPath.extension().string();

    if (ext == ".ort") {
        return; // Always supported
    }

    if (ext == ".onnx") {
#if GR_ONNX_MINIMAL_BUILD
        throw std::invalid_argument("This build only supports .ort format models. "
                                    "Convert your model: python -m onnxruntime.tools.convert_onnx_models_to_ort " +
                                    modelPath.string());
#else
        return; // Full build supports .onnx
#endif
    }

    throw std::invalid_argument("Unsupported model format '" + ext + "'. Supported: " + std::string(supportedFormats()));
}

/**
 * @brief Create a session with automatic format validation
 */
[[nodiscard]] inline Ort::Session createSession(Ort::Env& env, const std::filesystem::path& modelPath, const Ort::SessionOptions& options = Ort::SessionOptions{}) {
    validateModelPath(modelPath);
    return Ort::Session(env, modelPath.c_str(), options);
}

/**
 * @brief Runtime check for .onnx support (tries to detect via build info)
 *
 * This is a heuristic - checks if build info contains "minimal_build".
 * Use the compile-time supportsOnnxFormat() when possible.
 */
[[nodiscard]] inline bool detectOnnxFormatSupport() {
    const auto* api       = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    const char* buildInfo = api->GetBuildInfoString();

    // Minimal builds typically have "minimal_build" in the build string
    // This is not 100% reliable but works as a runtime heuristic
    return std::string_view(buildInfo).find("minimal_build") == std::string_view::npos;
}

/**
 * @brief Get detailed build information
 */
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
