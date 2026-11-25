// separate TU + shared library: acpp miscompiles these otherwise (AdaptiveCpp#2042)
#include "bm_FFT_backends_helpers.hpp"
#include <algorithm>
#include <format>
#include <iterator>

#include <gnuradio-4.0/Complex.hpp>
#include <gnuradio-4.0/device/BackendDetect.hpp>
#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/device/DeviceContextWebGpu.hpp>
#include <gnuradio-4.0/device/DeviceFFT.hpp>
#include <gnuradio-4.0/device/SyclRuntime.hpp>

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

// every device context the registry serves, as on a device-only edge of a flow graph: upload (not when "resident", where
// the batch stays on the device between calls), transform, read one element
static std::vector<FFTBackend> makeDeviceBackends() {
    std::ignore                             = device::registerSyclRuntime();
    std::ignore                             = device::registerWebGpuRuntime();
    device::DeviceContextRegistry& registry = device::DeviceContextRegistry::instance();
    std::vector<std::string>       domains;
    for (const std::string_view kind : {"host", "gpu"}) {
        for (int index = 0; registry.isServedExactly(std::format("{}:sycl:{}", kind, index)); ++index) {
            domains.push_back(std::format("{}:sycl:{}", kind, index));
        }
    }
    for (const std::string_view webGpu : {"gpu:webgpu", "gpu:swift"}) {
        if (registry.isServedExactly(webGpu)) {
            domains.emplace_back(webGpu);
        }
    }

    struct State {
        device::DeviceContext& ctx;
        device::DeviceFFT      fft;
        device::DeviceBuffer   dData;
        std::size_t            cap = 0;
    };
    std::vector<FFTBackend> result;
    for (const std::string& domain : domains) {
        device::DeviceContext& ctx = *registry.tryResolve(domain);
        for (const bool resident : {false, true}) {
            auto state = std::make_shared<State>(State{.ctx = ctx, .fft = {}, .dData = {}, .cap = 0});
            result.push_back(FFTBackend{.shortName = std::format("{}{}", ctx.shortName(), resident ? " resident" : ""),
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

std::vector<FFTBackend> availableBackends() {
    std::vector<FFTBackend> backends;
    backends.push_back(makeCpuSyclBackend());
    std::ranges::move(makeDeviceBackends(), std::back_inserter(backends));
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
