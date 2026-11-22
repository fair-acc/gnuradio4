// separate TU + shared library: acpp miscompiles these otherwise (AdaptiveCpp#2042)
#include "bm_FFT_backends_helpers.hpp"
#include <cstring>
#include <format>

#include <optional>

#include <gnuradio-4.0/Complex.hpp>
#include <gnuradio-4.0/device/BackendDetect.hpp>
#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/device/DeviceContextSycl.hpp>
#include <gnuradio-4.0/device/DeviceFFT.hpp>
#if GR_DEVICE_HAS_WEBGPU
#include <gnuradio-4.0/device/DeviceContextWebGpu.hpp>
#endif

namespace gr::benchmark_fft {

using GC = gr::complex<float>;

static FFTBackend makeCpuSyclBackend() {
    struct State {
        device::DeviceContextCpu ctx;
        device::DeviceFFT        fft;
        device::DeviceBuffer     dData;
        std::size_t              cap = 0;
    };
    auto state = std::make_shared<State>();

    return {.shortName = "DeviceFFT:CPU",
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
                state->fft.forwardBatch(state->ctx, std::span{state->dData.devicePointer<GC>(), total}, N);
            },
        .compute = [state](const C* input, std::size_t N, std::size_t nBatches) -> float {
            std::size_t total = N * nBatches;
            state->ctx.copyHostToDevice(reinterpret_cast<const GC*>(input), state->dData, total);
            state->fft.forwardBatch(state->ctx, std::span{state->dData.devicePointer<GC>(), total}, N);
            return state->dData.devicePointer<GC>()[0].re;
        }};
}

#if GR_DEVICE_HAS_SYCL
static std::vector<FFTBackend> makeSyclBackends() {
    std::vector<FFTBackend> result;

    for (const auto& dev : sycl::device::get_devices()) {
        struct State {
            sycl::queue               queue;
            device::DeviceContextSycl ctx;
            device::DeviceFFT         fft;
            device::DeviceBuffer      dData;
            GC*                       hStage = nullptr;
            std::size_t               cap    = 0;
            explicit State(const sycl::device& d) : queue(d, sycl::property::queue::in_order{}), ctx(queue) {}
        };

        // "resident": the batch stays on the device between calls, as on a device-only edge of a flow graph
        try {
            for (const bool resident : {false, true}) {
                auto        state = std::make_shared<State>(dev);
                std::string label = std::format("{}{}", state->ctx.shortName(), resident ? " resident" : "");

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
                            GC* devicePtr = state->dData.devicePointer<GC>();
                            std::memset(state->hStage, 0, total * sizeof(GC));
                            state->queue.memcpy(devicePtr, state->hStage, total * sizeof(GC)).wait();
                            state->fft.forwardBatch(state->ctx, std::span{devicePtr, total}, N);
                        },
                    .compute = [state, resident](const C* input, std::size_t N, std::size_t nBatches) -> float {
                        std::size_t total     = N * nBatches;
                        GC*         devicePtr = state->dData.devicePointer<GC>();
                        if (!resident) {
                            std::memcpy(state->hStage, input, total * sizeof(GC));
                            state->queue.memcpy(devicePtr, state->hStage, total * sizeof(GC)).wait();
                        }
                        state->fft.forwardBatch(state->ctx, std::span{devicePtr, total}, N);
                        GC result;
                        state->queue.memcpy(&result, devicePtr, sizeof(GC)).wait();
                        return result.re;
                    }});
            }
        } catch (...) {
            // skip devices that fail to create a queue
        }
    }
    return result;
}
#endif

#if GR_DEVICE_HAS_WEBGPU
// device-only data, so DeviceFFT runs its Stockham stages on the adapter; as the SYCL backends: upload (not when resident), transform, read one element
static std::vector<FFTBackend> makeWebGpuBackends() {
    std::vector<FFTBackend> result;
    std::ignore = device::registerWebGpuRuntime();
    for (const std::string_view domain : {std::string_view{"gpu:webgpu"}, std::string_view{"gpu:swift"}}) {
        if (!device::DeviceContextRegistry::instance().isServedExactly(domain)) {
            continue;
        }
        device::DeviceContext* const ctx = device::DeviceContextRegistry::instance().tryResolve(domain);
        if (ctx == nullptr) {
            continue;
        }
        struct State {
            device::DeviceContext& ctx;
            device::DeviceFFT      fft;
            device::DeviceBuffer   dData;
            std::size_t            cap = 0;
        };
        for (const bool resident : {false, true}) {
            auto state = std::make_shared<State>(State{.ctx = *ctx, .fft = {}, .dData = {}, .cap = 0});
            result.push_back(FFTBackend{.shortName = std::format("DeviceFFT:{}{}", domain, resident ? " resident" : ""),
                .init =
                    [state](std::size_t N, std::size_t maxBatches) {
                        const std::size_t total = N * maxBatches;
                        state->fft.init(state->ctx, N);
                        if (state->cap < total) {
                            if (state->dData) {
                                state->ctx.deallocate(state->dData);
                            }
                            state->dData = state->ctx.allocateDevice<GC>(total);
                            state->cap   = total;
                        }
                        const std::vector<GC> zeros(total);
                        state->ctx.copyHostToDevice(zeros.data(), state->dData, total);
                        state->fft.forwardBatch(state->ctx, std::span{state->dData.devicePointer<GC>(), total}, N);
                        state->ctx.wait(); // the warm-up transform runs here, not in the first timed call
                    },
                .compute = [state, resident](const C* input, std::size_t N, std::size_t nBatches) -> float {
                    const std::size_t total = N * nBatches;
                    if (!resident) {
                        state->ctx.copyHostToDevice(reinterpret_cast<const GC*>(input), state->dData, total);
                    }
                    state->fft.forwardBatch(state->ctx, std::span{state->dData.devicePointer<GC>(), total}, N);
                    GC first{};
                    state->ctx.copyDeviceToHost(state->dData, &first, 1UZ);
                    return first.re;
                }});
        }
    }
    return result;
}
#endif

std::vector<FFTBackend> availableBackends() {
    std::vector<FFTBackend> backends;
    backends.push_back(makeCpuSyclBackend());
#if GR_DEVICE_HAS_SYCL
    auto syclBackends = makeSyclBackends();
    for (auto& b : syclBackends) {
        backends.push_back(std::move(b));
    }
#endif
#if GR_DEVICE_HAS_WEBGPU
    for (auto& b : makeWebGpuBackends()) {
        backends.push_back(std::move(b));
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
