#ifndef GNURADIO_ALGORITHM_GLSL_FFT_HPP
#define GNURADIO_ALGORITHM_GLSL_FFT_HPP

#include <cassert>
#include <cmath>
#include <format>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

#include <gnuradio-4.0/Complex.hpp>
#include <gnuradio-4.0/device/DeviceContextGLSL.hpp>

namespace gr::device {

#if GR_DEVICE_HAS_GL_COMPUTE

/**
 * @brief Stockham auto-sort FFT via GLSL compute shaders.
 *
 * Mirrors SyclFFT's algorithm: radix-2 Stockham stages, per-stage twiddle sub-tables,
 * ping-pong SSBOs, all dispatched through DeviceContextGLSL.
 *
 * Each stage is one compute shader dispatch with glMemoryBarrier between stages.
 * Twiddle factors stored in a third SSBO (binding 2).
 *
 * @example
 * GlComputeContext gl;
 * DeviceContextGLSL ctx(gl);
 * GlslFFT fft;
 * fft.init(ctx, 1024);
 * fft.forward(ctx, dData, 1024);
 */
struct GlslFFT {
    using C = gr::complex<float>;

    std::vector<C>     _twiddleHost;
    void*              _twiddleSSBO  = nullptr;
    void*              _pingPongSSBO = nullptr;
    std::size_t        _pingPongCap  = 0;
    std::size_t        _fftSize      = 0;
    DeviceContextGLSL* _ctx          = nullptr;

    void init(DeviceContextGLSL& ctx, std::size_t fftSize) {
        if (_fftSize == fftSize && _ctx == &ctx) {
            return;
        }
        cleanup();

        _ctx     = &ctx;
        _fftSize = fftSize;

        // forward twiddle table: exp(-2πi·k/N) for k = 0..N/2-1
        _twiddleHost.resize(fftSize / 2);
        const float theta = -2.f * std::numbers::pi_v<float> / static_cast<float>(fftSize);
        for (std::size_t k = 0; k < fftSize / 2; ++k) {
            _twiddleHost[k] = {std::cos(theta * static_cast<float>(k)), std::sin(theta * static_cast<float>(k))};
        }

        // upload base twiddle table (N/2 entries) to SSBO
        _twiddleSSBO = ctx.allocateDevice<C>(fftSize / 2);
        ctx.copyHostToDevice(_twiddleHost.data(), static_cast<C*>(_twiddleSSBO), fftSize / 2);

        _pingPongCap = 0;
    }

    void cleanup() {
        if (_ctx) {
            if (_twiddleSSBO) {
                _ctx->deallocateRaw(_twiddleSSBO);
                _twiddleSSBO = nullptr;
            }
            if (_pingPongSSBO) {
                _ctx->deallocateRaw(_pingPongSSBO);
                _pingPongSSBO = nullptr;
            }
        }
        _fftSize = 0;
    }

    ~GlslFFT() { cleanup(); }

    GlslFFT()                          = default;
    GlslFFT(const GlslFFT&)            = delete;
    GlslFFT& operator=(const GlslFFT&) = delete;

    GlslFFT(GlslFFT&& o) noexcept : _twiddleHost(std::move(o._twiddleHost)), _twiddleSSBO(std::exchange(o._twiddleSSBO, nullptr)), _pingPongSSBO(std::exchange(o._pingPongSSBO, nullptr)), _pingPongCap(o._pingPongCap), _fftSize(std::exchange(o._fftSize, 0)), _ctx(std::exchange(o._ctx, nullptr)) {}

    GlslFFT& operator=(GlslFFT&& o) noexcept {
        if (this != &o) {
            cleanup();
            new (this) GlslFFT(std::move(o));
        }
        return *this;
    }

    void forward(DeviceContextGLSL& ctx, void* data, std::size_t N) { forward(ctx, data, N, 1); }

    void forward(DeviceContextGLSL& ctx, void* data, std::size_t N, std::size_t nBatches) {
        assert(N == _fftSize);
        const std::size_t nStages    = static_cast<std::size_t>(std::countr_zero(N));
        const std::size_t totalElems = nBatches * N;

        ensurePingPong(ctx, totalElems);

        // compile per-stage shaders with baked nBatches (avoids uniform issues)
        std::vector<unsigned int> programs(nStages);
        for (std::size_t s = 0; s < nStages; ++s) {
            auto glsl = generateStageShader(N, s, nStages, nBatches);
            auto prog = ctx.compileOrGetCached(glsl);
            assert(prog.has_value());
            programs[s] = *prog;
        }

        auto* src = data;
        auto* dst = _pingPongSSBO;

        ctx._gl->_glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, DeviceContextGLSL::toSsbo(_twiddleSSBO));

        for (std::size_t s = 0; s < nStages; ++s) {
            ctx._gl->_glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, DeviceContextGLSL::toSsbo(src));
            ctx._gl->_glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, DeviceContextGLSL::toSsbo(dst));
            ctx._gl->_glUseProgram(programs[s]);

            auto numGroups = static_cast<unsigned int>((nBatches * N / 2 + 255) / 256);
            ctx._gl->_glDispatchCompute(numGroups, 1, 1);
            ctx._gl->_glMemoryBarrier(0x00000002 /*GL_SHADER_STORAGE_BARRIER_BIT*/);

            std::swap(src, dst);
        }

        if (src != data) {
            auto srcSsbo = DeviceContextGLSL::toSsbo(src);
            auto dstSsbo = DeviceContextGLSL::toSsbo(data);
            ctx._gl->_glBindBuffer(0x8F36 /*GL_COPY_READ_BUFFER*/, srcSsbo);
            ctx._gl->_glBindBuffer(0x8F37 /*GL_COPY_WRITE_BUFFER*/, dstSsbo);
            ctx._gl->_glCopyBufferSubData(0x8F36, 0x8F37, 0, 0, static_cast<long>(totalElems * sizeof(C)));
            ctx._gl->_glMemoryBarrier(0x00000002);
        }
    }

private:
    void ensurePingPong(DeviceContextGLSL& ctx, std::size_t elems) {
        if (_pingPongCap >= elems) {
            return;
        }
        if (_pingPongSSBO) {
            ctx.deallocateRaw(_pingPongSSBO);
        }
        _pingPongSSBO = ctx.allocateDevice<C>(elems);
        _pingPongCap  = elems;
    }

    static std::string generateStageShader(std::size_t N, std::size_t stage, [[maybe_unused]] std::size_t nStages, std::size_t nBatches = 1) {
        // Van Loan Stockham radix-2: srcLo = j (sequential read), dst interleaved
        // Ls = 2^(stage+1), Ls2 = 2^stage
        // dstLo = (j >> stage) * Ls + (j & (Ls2-1))
        // twiddle: tw[k * (N/Ls)] where k = j & (Ls2-1)
        std::size_t Ls2       = std::size_t(1) << stage; // half sub-FFT length
        std::size_t Ls        = Ls2 << 1;                // full sub-FFT length
        std::size_t twStride  = N / Ls;                  // twiddle stride in the base table
        std::size_t totalWork = nBatches * N / 2;

        return std::format(R"(#version 430
layout(local_size_x = 256) in;
layout(binding = 0) readonly buffer SrcBuf {{ vec2 data[]; }} src;
layout(binding = 1) writeonly buffer DstBuf {{ vec2 data[]; }} dst;
layout(binding = 2) readonly buffer TwBuf  {{ vec2 data[]; }} tw;

vec2 cmul(vec2 a, vec2 b) {{ return vec2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }}

void main() {{
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= {}u) return;

    uint halfN = {}u;
    uint batch = gid / halfN;
    uint j     = gid - batch * halfN;
    uint off   = batch * {}u;

    uint srcLo = off + j;
    uint srcHi = srcLo + halfN;

    uint group = j >> {}u;
    uint k     = j & {}u;
    uint dstLo = off + group * {}u + k;
    uint dstHi = dstLo + {}u;

    vec2 a = src.data[srcLo];
    vec2 b = src.data[srcHi];
    vec2 w = tw.data[k * {}u];

    vec2 wb = cmul(w, b);
    dst.data[dstLo] = a + wb;
    dst.data[dstHi] = a - wb;
}}
)",
            totalWork, N / 2, N, stage, Ls2 - 1, Ls, Ls2, twStride);
    }
};

#else // no GL compute

struct GlslFFT {
    void init(auto&, std::size_t) {}
    void forward(auto&, void*, std::size_t, std::size_t = 1) {}
    void cleanup() {}
};

#endif // GR_DEVICE_HAS_GL_COMPUTE

} // namespace gr::device

#endif // GNURADIO_ALGORITHM_GLSL_FFT_HPP
