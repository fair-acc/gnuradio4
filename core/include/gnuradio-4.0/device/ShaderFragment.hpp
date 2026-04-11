#ifndef GNURADIO_SHADER_FRAGMENT_HPP
#define GNURADIO_SHADER_FRAGMENT_HPP

#include <cstddef>
#include <string>
#include <vector>

namespace gr::device {

/// a baked constant to specialise into the shader source (e.g. #define GAIN 2.5)
struct ShaderConst {
    std::string name;
    float       value;
};

/**
 * @brief Describes a block's GLSL compute shader fragment for device execution and fusion.
 *
 * Blocks implement shaderFragment() to provide their GLSL function body, baked constants,
 * and chunk size requirements. The scheduler uses this to:
 * - dispatch single-block shaders via DeviceContextGLSL
 * - fuse adjacent shader-capable blocks into one combined dispatch
 * - recompile when settings change (constants invalidate the cache)
 *
 * @example element-wise block:
 *   ShaderFragment{ .glslFunction = "float process(float x) { return x * GAIN; }",
 *                   .constants = {{"GAIN", 2.5f}}, .inputChunkSize = 0, .outputChunkSize = 0 }
 *
 * @example chunk-based block (FFT):
 *   ShaderFragment{ .glslFunction = "...", .constants = {...},
 *                   .inputChunkSize = 4096, .outputChunkSize = 4096 }
 */
struct ShaderFragment {
    std::string              glslFunction;        // GLSL function body
    std::vector<ShaderConst> constants;           // baked into #define before compilation
    std::size_t              inputChunkSize  = 0; // 0 = element-wise, N = requires N-sample chunks
    std::size_t              outputChunkSize = 0;
    std::size_t              workgroupSize   = 256;
};

/// generates a complete GLSL compute shader from a ShaderFragment for element-wise processing
inline std::string generateElementWiseShader(const ShaderFragment& frag, std::size_t totalElements) {
    std::string shader = "#version 430\nlayout(local_size_x = " + std::to_string(frag.workgroupSize) + ") in;\n";
    shader += "layout(binding = 0) readonly buffer InBuf  { float data[]; } inBuf;\n";
    shader += "layout(binding = 1) writeonly buffer OutBuf { float data[]; } outBuf;\n";

    for (const auto& c : frag.constants) {
        shader += "#define " + c.name + " " + std::to_string(c.value) + "\n";
    }

    shader += frag.glslFunction + "\n";
    shader += "void main() {\n";
    shader += "    uint i = gl_GlobalInvocationID.x;\n";
    shader += "    if (i < " + std::to_string(totalElements) + ") outBuf.data[i] = process(inBuf.data[i]);\n";
    shader += "}\n";
    return shader;
}

} // namespace gr::device

#endif // GNURADIO_SHADER_FRAGMENT_HPP
