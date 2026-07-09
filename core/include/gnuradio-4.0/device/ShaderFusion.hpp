#ifndef GNURADIO_SHADER_FUSION_HPP
#define GNURADIO_SHADER_FUSION_HPP

#include <numeric>
#include <span>
#include <string>
#include <vector>

#include <gnuradio-4.0/device/ShaderFragment.hpp>

namespace gr::device {

/**
 * @brief Result of fusing a chain of ShaderFragments into one or more dispatch stages.
 *
 * Element-wise blocks are composed into a single shader. Chunk-based blocks (e.g. FFT)
 * break the chain into stages separated by barriers (separate dispatches).
 *
 * Each FusedStage is one GLSL compute dispatch with its own source and workgroup size.
 */
struct FusedStage {
    std::string glslSource;
    std::size_t workgroupSize = 256;
    std::size_t chunkSize     = 0; // 0 = element-wise (processes totalElements), N = chunk-based
};

struct FusedShader {
    std::vector<FusedStage> stages;
    std::size_t             requiredChunkSize = 0; // LCM of all chunk sizes (0 if all element-wise)
};

/**
 * @brief Fuse a chain of ShaderFragments into a FusedShader.
 *
 * Consecutive element-wise fragments (inputChunkSize == 0) are composed into a single shader:
 *   outBuf.data[i] = process2(process1(process0(inBuf.data[i])));
 *
 * A chunk-based fragment (inputChunkSize > 0) breaks the chain:
 * - preceding element-wise fragments become one stage
 * - the chunk-based fragment becomes its own stage
 * - following element-wise fragments start a new fusible group
 *
 * The scheduler must provide totalElements as a multiple of requiredChunkSize.
 */
inline FusedShader fuseShaderFragments(std::span<const ShaderFragment> fragments, std::size_t totalElements) {
    FusedShader result;

    // compute LCM of all non-zero chunk sizes
    for (const auto& f : fragments) {
        if (f.inputChunkSize > 0) {
            result.requiredChunkSize = result.requiredChunkSize == 0 ? f.inputChunkSize : std::lcm(result.requiredChunkSize, f.inputChunkSize);
        }
    }

    // accumulate element-wise fragments into fusible groups
    std::vector<const ShaderFragment*> currentGroup;

    auto flushElementWiseGroup = [&](std::size_t wgSize) {
        if (currentGroup.empty()) {
            return;
        }

        std::string shader = "#version 430\nlayout(local_size_x = " + std::to_string(wgSize) + ") in;\n";
        shader += "layout(binding = 0) readonly buffer InBuf  { float data[]; } inBuf;\n";
        shader += "layout(binding = 1) writeonly buffer OutBuf { float data[]; } outBuf;\n\n";

        // emit #defines for all constants
        for (std::size_t idx = 0; idx < currentGroup.size(); ++idx) {
            for (const auto& c : currentGroup[idx]->constants) {
                shader += "#define " + c.name + "_" + std::to_string(idx) + " " + std::to_string(c.value) + "\n";
            }
        }

        // emit per-block process functions with unique names
        for (std::size_t idx = 0; idx < currentGroup.size(); ++idx) {
            // rename "process(" to "process_N(" and constants "NAME" to "NAME_N"
            std::string func = currentGroup[idx]->glslFunction;

            // replace constant names with indexed versions
            for (const auto& c : currentGroup[idx]->constants) {
                std::string from = c.name;
                std::string to   = c.name + "_" + std::to_string(idx);
                std::size_t pos  = 0;
                while ((pos = func.find(from, pos)) != std::string::npos) {
                    func.replace(pos, from.size(), to);
                    pos += to.size();
                }
            }

            // rename process → process_N
            {
                std::string from = "process(";
                std::string to   = "process_" + std::to_string(idx) + "(";
                std::size_t pos  = 0;
                while ((pos = func.find(from, pos)) != std::string::npos) {
                    func.replace(pos, from.size(), to);
                    pos += to.size();
                }
            }

            shader += func + "\n";
        }

        // emit main() with composed calls
        shader += "\nvoid main() {\n";
        shader += "    uint i = gl_GlobalInvocationID.x;\n";
        shader += "    if (i < " + std::to_string(totalElements) + ") {\n";
        shader += "        float v = inBuf.data[i];\n";
        for (std::size_t idx = 0; idx < currentGroup.size(); ++idx) {
            shader += "        v = process_" + std::to_string(idx) + "(v);\n";
        }
        shader += "        outBuf.data[i] = v;\n";
        shader += "    }\n}\n";

        result.stages.push_back({std::move(shader), wgSize, 0});
        currentGroup.clear();
    };

    std::size_t defaultWgSize = 256;

    for (const auto& frag : fragments) {
        if (frag.inputChunkSize == 0) {
            // element-wise: accumulate into current group
            currentGroup.push_back(&frag);
            defaultWgSize = std::max(defaultWgSize, frag.workgroupSize);
        } else {
            // chunk-based: flush the current element-wise group, then emit the chunk stage
            flushElementWiseGroup(defaultWgSize);

            // emit the chunk-based fragment as its own stage (full shader, not just a function)
            std::string shader = "#version 430\nlayout(local_size_x = " + std::to_string(frag.workgroupSize) + ") in;\n";
            shader += "layout(binding = 0) readonly buffer InBuf  { float data[]; } inBuf;\n";
            shader += "layout(binding = 1) writeonly buffer OutBuf { float data[]; } outBuf;\n\n";
            for (const auto& c : frag.constants) {
                shader += "#define " + c.name + " " + std::to_string(c.value) + "\n";
            }
            shader += frag.glslFunction + "\n";
            shader += "void main() {\n";
            shader += "    uint i = gl_GlobalInvocationID.x;\n";
            shader += "    if (i < " + std::to_string(totalElements) + ") outBuf.data[i] = process(inBuf.data[i]);\n";
            shader += "}\n";

            result.stages.push_back({std::move(shader), frag.workgroupSize, frag.inputChunkSize});
            defaultWgSize = 256;
        }
    }

    // flush any remaining element-wise group
    flushElementWiseGroup(defaultWgSize);

    return result;
}

} // namespace gr::device

#endif // GNURADIO_SHADER_FUSION_HPP
