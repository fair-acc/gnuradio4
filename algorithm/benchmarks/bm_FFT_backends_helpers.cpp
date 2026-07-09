// separate TU + SHARED library under acpp — see G10 in claude_wip.md (AdaptiveCpp#2042)
#include "bm_FFT_backends_helpers.hpp"
#include <cstring>
#include <format>

#include <gnuradio-4.0/Complex.hpp>
#include <gnuradio-4.0/algorithm/fourier/GlslFFT.hpp>
#include <gnuradio-4.0/algorithm/fourier/SyclFFT.hpp>
#include <gnuradio-4.0/device/DeviceContextGLSL.hpp>
#include <gnuradio-4.0/device/DeviceContextSycl.hpp>

namespace gr::benchmark_fft {

using GC = gr::complex<float>;

// ── CPU SYCL backend (SimdFFT delegation via SyclFFT wrapper) ──

static FFTBackend makeCpuSyclBackend() {
    struct State {
        device::DeviceContextCpu ctx;
        device::SyclFFT          fft;
        GC*                      dData = nullptr;
        std::size_t              cap   = 0;
    };
    auto state = std::make_shared<State>();

    return {.shortName = "SyclFFT:CPU",
        .init =
            [state](std::size_t N, std::size_t maxBatches) {
                std::size_t total = N * maxBatches;
                state->fft.init(state->ctx, N);
                if (state->cap < total) {
                    if (state->dData) {
                        state->ctx.deallocate(state->dData);
                    }
                    state->dData = state->ctx.allocateShared<GC>(total);
                    state->cap   = total;
                }
                // warmup
                state->fft.forwardBatch(state->ctx, std::span{state->dData, total}, N);
            },
        .compute = [state](const C* input, std::size_t N, std::size_t nBatches) -> float {
            std::size_t total = N * nBatches;
            state->ctx.copyHostToDevice(reinterpret_cast<const GC*>(input), state->dData, total);
            state->fft.forwardBatch(state->ctx, std::span{state->dData, total}, N);
            return state->dData[0].re;
        }};
}

// ── GPU SYCL backend ──

#if GR_DEVICE_HAS_SYCL_IMPL
static std::vector<FFTBackend> makeSyclBackends() {
    std::vector<FFTBackend> result;

    for (const auto& dev : sycl::device::get_devices()) {
        struct State {
            sycl::queue               queue;
            device::DeviceContextSycl ctx;
            device::SyclFFT           fft;
            GC*                       dData  = nullptr;
            GC*                       hStage = nullptr;
            std::size_t               cap    = 0;
            explicit State(const sycl::device& d) : queue(d, sycl::property::queue::in_order{}), ctx(queue) {}
        };

        try {
            auto        state = std::make_shared<State>(dev);
            std::string label = state->ctx.shortName();

            result.push_back(FFTBackend{.shortName = std::move(label),
                .init =
                    [state](std::size_t N, std::size_t maxBatches) {
                        std::size_t total = N * maxBatches;
                        state->fft.init(state->ctx, N);
                        if (state->cap < total) {
                            if (state->dData) {
                                state->ctx.deallocate(state->dData);
                            }
                            if (state->hStage) {
                                sycl::free(state->hStage, state->queue);
                            }
                            state->dData  = state->ctx.allocateDevice<GC>(total);
                            state->hStage = sycl::malloc_host<GC>(total, state->queue);
                            state->cap    = total;
                        }
                        std::memset(state->hStage, 0, total * sizeof(GC));
                        state->queue.memcpy(state->dData, state->hStage, total * sizeof(GC)).wait();
                        state->fft.forwardBatch(state->ctx, std::span{state->dData, total}, N);
                    },
                .compute = [state](const C* input, std::size_t N, std::size_t nBatches) -> float {
                    std::size_t total = N * nBatches;
                    std::memcpy(state->hStage, input, total * sizeof(GC));
                    state->queue.memcpy(state->dData, state->hStage, total * sizeof(GC)).wait();
                    state->fft.forwardBatch(state->ctx, std::span{state->dData, total}, N);
                    GC result;
                    state->queue.memcpy(&result, state->dData, sizeof(GC)).wait();
                    return result.re;
                }});
        } catch (...) {
            // skip devices that fail to create a queue
        }
    }
    return result;
}
#endif

// ── GPU GLSL backend ──

#if GR_DEVICE_HAS_GL_COMPUTE
static std::optional<FFTBackend> makeGpuGlslBackend() {
    static device::GlComputeContext gl;
    if (!gl.isAvailable()) {
        return std::nullopt;
    }

    struct State {
        device::DeviceContextGLSL ctx;
        device::GlslFFT           fft;
        void*                     dData = nullptr;
        std::size_t               cap   = 0;
        explicit State(device::GlComputeContext& g) : ctx(g) {}
    };
    auto state = std::make_shared<State>(gl);

    std::string label = state->ctx.shortName();

    return FFTBackend{.shortName = std::move(label),
        .init =
            [state](std::size_t N, std::size_t maxBatches) {
                std::size_t total = N * maxBatches;
                state->fft.init(state->ctx, N);
                if (state->cap < total) {
                    if (state->dData) {
                        state->ctx.deallocateRaw(state->dData);
                    }
                    state->dData = state->ctx.allocateDevice<GC>(total);
                    state->cap   = total;
                }
                // warmup
                state->fft.forward(state->ctx, state->dData, N, maxBatches);
            },
        .compute = [state](const C* input, std::size_t N, std::size_t nBatches) -> float {
            std::size_t total = N * nBatches;
            state->ctx.copyHostToDevice(reinterpret_cast<const GC*>(input), static_cast<GC*>(state->dData), total);
            state->fft.forward(state->ctx, state->dData, N, nBatches);
            GC result;
            state->ctx.copyDeviceToHost(static_cast<GC*>(state->dData), &result, 1UZ);
            return result.re;
        }};
}
#endif

// ── public API ──

std::vector<FFTBackend> availableBackends() {
    std::vector<FFTBackend> backends;
    backends.push_back(makeCpuSyclBackend());
#if GR_DEVICE_HAS_SYCL_IMPL
    auto syclBackends = makeSyclBackends();
    for (auto& b : syclBackends) {
        backends.push_back(std::move(b));
    }
#endif
#if GR_DEVICE_HAS_GL_COMPUTE
    if (auto b = makeGpuGlslBackend()) {
        backends.push_back(std::move(*b));
    }
#endif
    return backends;
}

std::string deviceInfo() {
    auto        backends = availableBackends();
    std::string info;
    for (const auto& b : backends) {
        info += std::format("  {}\n", b.shortName);
    }
    if (backends.empty()) {
        info = "  (no device backends available)\n";
    }
    return info;
}

} // namespace gr::benchmark_fft
