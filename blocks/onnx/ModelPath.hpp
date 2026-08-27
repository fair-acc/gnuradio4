#ifndef GR_ONNX_TEST_MODELPATH_HPP
#define GR_ONNX_TEST_MODELPATH_HPP

#include <filesystem>
#include <string>
#include <string_view>

// test and example scaffolding, deliberately outside include/ so it is not installed
namespace gr::blocks::onnx::test {

/// Model in blocks/onnx/models/, so callers name only the file. GR_ONNX_MODELS_DIR comes from CMake:
/// __FILE__ is relative under the Ninja generator when the build tree sits beside the source tree.
[[nodiscard]] inline std::string modelPath(std::string_view fileName) {
#ifdef __EMSCRIPTEN__
    return (std::filesystem::path("/data") / fileName).string();
#else
    return (std::filesystem::path(GR_ONNX_MODELS_DIR) / fileName).string();
#endif
}

[[nodiscard]] inline std::string deliverableModelPath(std::string_view stem) {
#ifdef __EMSCRIPTEN__
    return modelPath(std::string(stem) + ".ort.gz");
#else
    return modelPath(std::string(stem) + ".onnx.gz");
#endif
}

} // namespace gr::blocks::onnx::test

#endif // GR_ONNX_TEST_MODELPATH_HPP
