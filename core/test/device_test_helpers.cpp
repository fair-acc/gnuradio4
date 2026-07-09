#include "device_test_helpers.hpp"
#include <gnuradio-4.0/device/DeviceContext.hpp>

namespace gr::test {

void deviceParallelMultiply(const float* hostIn, float* hostOut, std::size_t N, float factor) {
    gr::device::DeviceContextCpu ctx;
    auto*                        dIn  = ctx.allocateShared<float>(N);
    auto*                        dOut = ctx.allocateShared<float>(N);
    ctx.copyHostToDevice(hostIn, dIn, N);
    for (std::size_t i = 0; i < N; ++i) {
        dOut[i] = dIn[i] * factor;
    }
    ctx.copyDeviceToHost(dOut, hostOut, N);
    ctx.deallocate(dIn);
    ctx.deallocate(dOut);
}

void deviceParallelComplexRotate(const gr::complex<float>* hostIn, gr::complex<float>* hostOut, std::size_t N, gr::complex<float> factor) {
    gr::device::DeviceContextCpu ctx;
    auto*                        dIn  = ctx.allocateShared<gr::complex<float>>(N);
    auto*                        dOut = ctx.allocateShared<gr::complex<float>>(N);
    ctx.copyHostToDevice(hostIn, dIn, N);
    for (std::size_t i = 0; i < N; ++i) {
        dOut[i] = dIn[i] * factor;
    }
    ctx.copyDeviceToHost(dOut, hostOut, N);
    ctx.deallocate(dIn);
    ctx.deallocate(dOut);
}

} // namespace gr::test

// GL compute shader helpers — in same TU to keep SYCL and GL kernel code together

#include <format>
#include <gnuradio-4.0/device/GlComputeContext.hpp>

namespace gr::test {

bool glComputeAvailable() {
#if GR_DEVICE_HAS_GL_COMPUTE
    static gr::device::GlComputeContext ctx;
    return ctx.isAvailable();
#else
    return false;
#endif
}

void glShaderMultiply(const float* in, float* out, std::size_t N, float factor) {
#if GR_DEVICE_HAS_GL_COMPUTE
    static gr::device::GlComputeContext ctx;
    if (!ctx.isAvailable()) {
        return;
    }

    auto shader = std::format(R"(
        #version 430
        layout(local_size_x = 256) in;
        layout(binding = 0) readonly buffer InBuf  {{ float data[]; }} inBuf;
        layout(binding = 1) writeonly buffer OutBuf {{ float data[]; }} outBuf;
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            if (i < {}) outBuf.data[i] = inBuf.data[i] * {};
        }}
    )",
        N, factor);

    auto result = ctx.compileOrGetCached(shader);
    if (!result) {
        return;
    }
    ctx.dispatch(*result, in, out, N, 256);
#else
    (void)in;
    (void)out;
    (void)N;
    (void)factor;
#endif
}

} // namespace gr::test
