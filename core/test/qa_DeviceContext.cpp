#include <boost/ut.hpp>

#include <cstring>
#include <format>
#include <numeric>
#include <vector>

#include "device_test_helpers.hpp"
#include <gnuradio-4.0/Complex.hpp>
#include <gnuradio-4.0/device/DeviceContext.hpp>
#include <gnuradio-4.0/device/DeviceContextGLSL.hpp>
#include <gnuradio-4.0/device/GLSL2WGSL.hpp>
#include <gnuradio-4.0/device/ShaderFragment.hpp>
#include <gnuradio-4.0/device/ShaderFusion.hpp>

using namespace boost::ut;
using namespace std::string_view_literals;

const suite<"device::DeviceContext"> tests =
    [] {
        "allocate and deallocate device memory"_test = [] {
            gr::device::DeviceContextCpu ctx;
            auto*                        ptr = ctx.allocateDevice<float>(4096);
            expect(ptr != nullptr);
            ctx.deallocate(ptr);
        };

        "allocate and deallocate shared memory"_test = [] {
            gr::device::DeviceContextCpu ctx;
            auto*                        ptr = ctx.allocateShared<float>(4096);
            expect(ptr != nullptr);
            for (std::size_t i = 0; i < 4096; ++i) {
                ptr[i] = static_cast<float>(i);
            }
            expect(eq(ptr[0], 0.f));
            expect(eq(ptr[4095], 4095.f));
            ctx.deallocate(ptr);
        };

        "copy host to device and back"_test = [] {
            gr::device::DeviceContextCpu ctx;
            constexpr std::size_t        N = 1024;

            std::vector<float> host(N);
            std::iota(host.begin(), host.end(), 1.f);

            auto* device = ctx.allocateDevice<float>(N);
            ctx.copyHostToDevice(host.data(), device, N);

            std::vector<float> result(N, 0.f);
            ctx.copyDeviceToHost(device, result.data(), N);

            for (std::size_t i = 0; i < N; ++i) {
                expect(eq(result[i], host[i]));
            }
            ctx.deallocate(device);
        };

        "parallelFor multiplies array via helper TU"_test = [] {
            constexpr std::size_t N = 512;
            std::vector<float>    input(N);
            std::iota(input.begin(), input.end(), 0.f);
            std::vector<float> output(N, 0.f);

            gr::test::deviceParallelMultiply(input.data(), output.data(), N, 2.f);

            for (std::size_t i = 0; i < N; ++i) {
                expect(eq(output[i], static_cast<float>(i) * 2.f));
            }
        };

        "parallelFor with gr::complex via helper TU"_test = [] {
            constexpr std::size_t           N = 256;
            std::vector<gr::complex<float>> input(N);
            for (std::size_t i = 0; i < N; ++i) {
                input[i] = {static_cast<float>(i), static_cast<float>(i * 2)};
            }

            std::vector<gr::complex<float>> output(N);
            gr::test::deviceParallelComplexRotate(input.data(), output.data(), N, {2.f, 0.f});

            for (std::size_t i = 0; i < N; ++i) {
                expect(eq(output[i].re, static_cast<float>(i) * 2.f));
                expect(eq(output[i].im, static_cast<float>(i * 2) * 2.f));
            }
        };

        "CPU fallback produces correct results"_test = [] {
            gr::device::DeviceContextCpu ctx;
            expect(ctx.backend() == gr::device::DeviceBackend::CPU_Fallback);

            constexpr std::size_t N    = 100;
            auto*                 data = ctx.allocateShared<int>(N);
            for (std::size_t i = 0; i < N; ++i) {
                data[i] = static_cast<int>(i * i);
            }

            for (std::size_t i = 0; i < N; ++i) {
                expect(eq(data[i], static_cast<int>(i * i)));
            }
            ctx.deallocate(data);
        };

        "backend reports CPU_Fallback"_test = [] {
            gr::device::DeviceContextCpu ctx;
            expect(ctx.backend() == gr::device::DeviceBackend::CPU_Fallback);
            expect(ctx.name() == "CPU fallback");
            expect(ctx.shortName() == "CPU");
        };
};

#if GR_DEVICE_HAS_GL_COMPUTE
const suite<"device::DeviceContextGLSL"> glTests =
    [] {
        "GLSL allocate, H2D, dispatch, D2H"_test = [] {
            gr::device::GlComputeContext gl;
            if (!gl.isAvailable()) {
                return; // skip if no EGL/GL
            }

            gr::device::DeviceContextGLSL ctx(gl);
            expect(ctx.backend() == gr::device::DeviceBackend::GLSL);

            constexpr std::size_t N    = 1024;
            auto*                 dIn  = ctx.allocateDevice<float>(N);
            auto*                 dOut = ctx.allocateDevice<float>(N);

            // upload input
            std::vector<float> input(N);
            for (std::size_t i = 0; i < N; ++i) {
                input[i] = static_cast<float>(i);
            }
            ctx.copyHostToDevice(input.data(), dIn, N);

            // compile and dispatch multiply-by-2 shader
            auto prog = ctx.compileOrGetCached(std::format(R"(
                #version 430
                layout(local_size_x = 256) in;
                layout(binding = 0) readonly buffer InBuf  {{ float data[]; }} inBuf;
                layout(binding = 1) writeonly buffer OutBuf {{ float data[]; }} outBuf;
                void main() {{
                    uint i = gl_GlobalInvocationID.x;
                    if (i < {}) outBuf.data[i] = inBuf.data[i] * 2.0;
                }}
            )",
                N));
            expect(prog.has_value());
            ctx.dispatch(*prog, dIn, dOut, N, 256);

            // download and verify
            std::vector<float> output(N);
            ctx.copyDeviceToHost(dOut, output.data(), N);

            bool allCorrect = true;
            for (std::size_t i = 0; i < N; ++i) {
                if (std::abs(output[i] - input[i] * 2.f) > 1e-5f) {
                    allCorrect = false;
                    break;
                }
            }
            expect(allCorrect);

            ctx.deallocate(dIn);
            ctx.deallocate(dOut);
        };

        "GLSL ShaderFragment via builtin_multiply"_test = [] {
            gr::device::GlComputeContext gl;
            if (!gl.isAvailable()) {
                return;
            }

            gr::device::DeviceContextGLSL ctx(gl);
            constexpr std::size_t         N      = 4096;
            constexpr float               factor = 3.5f;

            // generate shader from ShaderFragment
            gr::device::ShaderFragment frag = {.glslFunction = "float process(float x) { return x * FACTOR; }", .constants = {{"FACTOR", factor}}};
            auto                       glsl = gr::device::generateElementWiseShader(frag, N);
            auto                       prog = ctx.compileOrGetCached(glsl);
            expect(prog.has_value());

            // allocate, upload, dispatch, download
            std::vector<float> input(N), output(N);
            for (std::size_t i = 0; i < N; ++i) {
                input[i] = static_cast<float>(i) * 0.01f;
            }

            auto* dIn  = ctx.allocateDevice<float>(N);
            auto* dOut = ctx.allocateDevice<float>(N);
            ctx.copyHostToDevice(input.data(), dIn, N);
            ctx.dispatch(*prog, dIn, dOut, N, frag.workgroupSize);
            ctx.copyDeviceToHost(dOut, output.data(), N);

            bool allCorrect = true;
            for (std::size_t i = 0; i < N; ++i) {
                if (std::abs(output[i] - input[i] * factor) > 1e-4f) {
                    allCorrect = false;
                    break;
                }
            }
            expect(allCorrect);

            ctx.deallocate(dIn);
            ctx.deallocate(dOut);
        };

        "GLSL shader fusion: 3-block element-wise chain"_test = [] {
            gr::device::GlComputeContext gl;
            if (!gl.isAvailable()) {
                return;
            }

            gr::device::DeviceContextGLSL ctx(gl);
            constexpr std::size_t         N = 1024;

            // 3 blocks: multiply(×2) → add(+1) → clamp(0,5)
            std::array<gr::device::ShaderFragment, 3> frags = {{
                {.glslFunction = "float process(float x) { return x * FACTOR; }", .constants = {{"FACTOR", 2.0f}}},
                {.glslFunction = "float process(float x) { return x + OFFSET; }", .constants = {{"OFFSET", 1.0f}}},
                {.glslFunction = "float process(float x) { return clamp(x, LO, HI); }", .constants = {{"LO", 0.0f}, {"HI", 5.0f}}},
            }};

            auto fused = gr::device::fuseShaderFragments(frags, N);
            expect(eq(fused.stages.size(), 1UZ)); // all element-wise → one fused stage
            expect(eq(fused.requiredChunkSize, 0UZ));

            auto prog = ctx.compileOrGetCached(fused.stages[0].glslSource);
            expect(prog.has_value());

            // input: [-2, -1, 0, 1, 2, ...]
            std::vector<float> input(N), output(N);
            for (std::size_t i = 0; i < N; ++i) {
                input[i] = static_cast<float>(i) - 2.f;
            }

            auto* dIn  = ctx.allocateDevice<float>(N);
            auto* dOut = ctx.allocateDevice<float>(N);
            ctx.copyHostToDevice(input.data(), dIn, N);
            ctx.dispatch(*prog, dIn, dOut, N, fused.stages[0].workgroupSize);
            ctx.copyDeviceToHost(dOut, output.data(), N);

            // verify: clamp((x * 2) + 1, 0, 5)
            bool allCorrect = true;
            for (std::size_t i = 0; i < N; ++i) {
                float expected = std::clamp(input[i] * 2.f + 1.f, 0.f, 5.f);
                if (std::abs(output[i] - expected) > 1e-4f) {
                    allCorrect = false;
                    break;
                }
            }
            expect(allCorrect);

            ctx.deallocate(dIn);
            ctx.deallocate(dOut);
        };
        "shader fusion with chunk-based block creates separate stages"_test = [] {
            // multiply(×2) → FFT(N=512) → multiply(×3) should produce 3 stages
            std::array<gr::device::ShaderFragment, 3> frags = {{
                {.glslFunction = "float process(float x) { return x * FACTOR; }", .constants = {{"FACTOR", 2.0f}}},
                {.glslFunction = "float process(float x) { return x; }", .constants = {}, .inputChunkSize = 512, .outputChunkSize = 512},
                {.glslFunction = "float process(float x) { return x * GAIN; }", .constants = {{"GAIN", 3.0f}}},
            }};

            auto fused = gr::device::fuseShaderFragments(frags, 2048);
            expect(eq(fused.stages.size(), 3UZ)); // pre-multiply | FFT | post-multiply
            expect(eq(fused.requiredChunkSize, 512UZ));
            expect(eq(fused.stages[0].chunkSize, 0UZ));   // element-wise
            expect(eq(fused.stages[1].chunkSize, 512UZ)); // chunk-based
            expect(eq(fused.stages[2].chunkSize, 0UZ));   // element-wise
        };
        "GLSL → WGSL transpilation"_test = [] {
            gr::device::ShaderFragment frag = {.glslFunction = "float process(float x) { return x * GAIN; }", .constants = {{"GAIN", 2.5f}}};
            auto                       glsl = gr::device::generateElementWiseShader(frag, 1024);
            auto                       wgsl = gr::device::GLSL2WGSL(glsl);

            // verify key WGSL constructs are present
            expect(wgsl.find("@compute") != std::string::npos);
            expect(wgsl.find("@workgroup_size(256)") != std::string::npos);
            expect(wgsl.find("var<storage, read>") != std::string::npos);
            expect(wgsl.find("var<storage, read_write>") != std::string::npos);
            expect(wgsl.find("gid.x") != std::string::npos);
            expect(wgsl.find("f32") != std::string::npos);
            expect(wgsl.find("const GAIN") != std::string::npos);
            // verify no GLSL remnants
            expect(wgsl.find("#version") == std::string::npos);
            expect(wgsl.find("gl_GlobalInvocationID") == std::string::npos);
            expect(wgsl.find("layout") == std::string::npos);
        };
};
#endif

int main() { /* not needed for UT */ }
