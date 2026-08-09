#include <boost/ut.hpp>

#include <cstring>
#include <format>
#include <numeric>
#include <vector>

#include "device_test_helpers.hpp"
#include <gnuradio-4.0/Complex.hpp>
#include <gnuradio-4.0/device/DeviceContext.hpp>
#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/device/SyclRuntime.hpp>

using namespace boost::ut;
using namespace std::string_view_literals;

#if GR_DEVICE_HAS_SYCL_IMPL
#include <gnuradio-4.0/device/DeviceContextSycl.hpp>

const suite<"device::syclContextFor"> syclContextOwnership = [] {
    using namespace boost::ut;

    "one queue always resolves to one context"_test = [] {
        gr::device::SyclQueue queue{sycl::cpu_selector_v};
        gr::device::SyclQueue sameQueue = queue; // a SYCL queue is a handle; a copy denotes the same queue

        expect(std::addressof(gr::device::syclContextFor(queue)) == std::addressof(gr::device::syclContextFor(sameQueue))) //
            << "a block asking twice, or two blocks sharing a queue, must get the same scratch";
    };

    "a different queue gets its own context"_test = [] {
        gr::device::SyclQueue first{sycl::cpu_selector_v};
        gr::device::SyclQueue second{sycl::cpu_selector_v};

        expect(std::addressof(gr::device::syclContextFor(first)) != std::addressof(gr::device::syclContextFor(second))) //
            << "otherwise the second queue would silently run on the first one's context";
    };

    "the context outlives the caller's queue"_test = [] {
        gr::device::DeviceContextSycl* ctx = nullptr;
        {
            gr::device::SyclQueue local{sycl::cpu_selector_v};
            ctx = std::addressof(gr::device::syclContextFor(local));
        } // the caller's queue dies here; a block freeing scratch in its destructor still needs the context
        expect(ctx != nullptr);
        auto buffer = ctx->allocateShared<float>(4UZ);
        expect(static_cast<bool>(buffer)) << "the context must still be usable after the queue that made it went away";
        ctx->deallocate(buffer);
    };
};

const suite<"device::DeviceContextRegistry"> registryResolution = [] {
    using namespace boost::ut;
    using namespace std::string_literals;

    "every spelling of one device names one context"_test = [] {
        if (!gr::device::registerSyclRuntime()) {
            return;
        }
        gr::device::DeviceContextRegistry& registry = gr::device::DeviceContextRegistry::instance();

        gr::device::DeviceContext* viaCanonical = registry.tryResolve("gpu:sycl");
        if (viaCanonical == nullptr) {
            return; // this machine serves no GPU
        }

        expect(registry.tryResolve("gpu") == viaCanonical) << "'gpu' must name the device that serves it";

        const gr::DomainResolution canonical = registry.resolve("gpu:sycl");
        expect(!canonical.downgraded) << "'gpu:sycl' is served, so nothing was downgraded";
        expect(registry.tryResolve(canonical.resolved) == viaCanonical) << "resolution must name the context it just found";

        expect(eq(registry.resolve("gpu").resolved, canonical.resolved)) << "both spellings must resolve to one name, or edge placement splits them";
        expect(eq(registry.resolve("gpu:sycl:0").resolved, canonical.resolved)) << "an explicit index for the same device must not read as a second domain";
    };

    "the plain host domain is never promoted onto a device"_test = [] {
        if (!gr::device::registerSyclRuntime()) {
            return;
        }
        gr::device::DeviceContextRegistry& registry = gr::device::DeviceContextRegistry::instance();

        const gr::DomainResolution resolution = registry.resolve("host");
        expect(eq(resolution.resolved, "host"s));
        expect(!resolution.downgraded);
        expect(registry.tryResolve("host") == nullptr) << "a host graph must not acquire a device context";
    };

    "an index nobody serves stays on the same device rather than dropping to the host"_test = [] {
        if (!gr::device::registerSyclRuntime()) {
            return;
        }
        gr::device::DeviceContextRegistry& registry = gr::device::DeviceContextRegistry::instance();
        if (registry.tryResolve("gpu:sycl") == nullptr) {
            return; // no GPU on this machine
        }

        const gr::DomainResolution resolution = registry.resolve("gpu:sycl:31");
        expect(resolution.downgraded) << "an index that is not served is a downgrade, and must be announced";
        expect(eq(resolution.declared, "gpu:sycl:31"s)) << "the warning has to name what was asked for";
        expect(registry.tryResolve(resolution.resolved) == registry.tryResolve("gpu:sycl")) //
            << "it must land on the GPU: the host:sycl rung would hand a device-only ring to a CPU kernel";
    };
};
#endif

const suite<"device::DeviceContext"> tests = [] {
    "an over-aligned request comes back over-aligned, on every residency the context serves"_test = [] {
        gr::device::DeviceContextCpu cpuCtx;
#if GR_DEVICE_HAS_SYCL_IMPL
        gr::device::SyclQueue      queue{sycl::cpu_selector_v};
        gr::device::DeviceContext* contexts[] = {&cpuCtx, &gr::device::syclContextFor(queue)};
#else
        gr::device::DeviceContext* contexts[] = {&cpuCtx};
#endif
        for (gr::device::DeviceContext* ctxPtr : contexts) {
            auto& ctx = *ctxPtr;
            for (const std::size_t align : {std::size_t{16}, std::size_t{64}, std::size_t{128}}) {
                for (const gr::device::Residency residency : {gr::device::Residency::host, gr::device::Residency::shared, gr::device::Residency::devicePtr}) {
                    // a contract assertion, not a regression detector: measured 2026-09-02, AdaptiveCpp's host
                    // backend returns the same pointer for aligned_alloc_* as for malloc_* (0/8 satisfied a
                    // 128-byte request in isolation), but whether a given allocation is under-aligned depends on
                    // allocator state, so this cannot be made to fail reliably on one machine
                    for (int attempt = 0; attempt < 8; ++attempt) {
                        gr::device::DeviceBuffer buf = ctx.allocate(256UZ, align, residency);
                        if (!buf) {
                            break; // a context need not serve every residency
                        }
                        expect(eq(buf.token % align, std::uintptr_t{0})) << std::format("residency {} at align {} came back under-aligned", static_cast<int>(residency), align);
                        expect(ge(buf.align, align)) << "the buffer must record at least the alignment it was asked for";
                        ctx.deallocate(buf);
                    }
                }
            }
        }
    };

    "allocate and deallocate host memory"_test = [] {
        gr::device::DeviceContextCpu ctx;
        auto                         buf = ctx.allocateHost<float>(4096);
        expect(static_cast<bool>(buf));
        ctx.deallocate(buf);
    };

    "CPU backend refuses devicePtr residency rather than lying about it"_test = [] {
        gr::device::DeviceContextCpu ctx;
        const auto                   buf = ctx.allocate(4096 * sizeof(float), alignof(float), gr::device::Residency::devicePtr);
        expect(!static_cast<bool>(buf)) << "CPU has no true device memory; invalid, never a lying token";
    };

    "allocate and deallocate shared memory"_test = [] {
        gr::device::DeviceContextCpu ctx;
        auto                         buf = ctx.allocateShared<float>(4096);
        expect(static_cast<bool>(buf));
        float* ptr = buf.devicePointer<float>();
        for (std::size_t i = 0; i < 4096; ++i) {
            ptr[i] = static_cast<float>(i);
        }
        expect(eq(ptr[0], 0.f));
        expect(eq(ptr[4095], 4095.f));
        ctx.deallocate(buf);
    };

    "copy host to device and back"_test = [] {
        gr::device::DeviceContextCpu ctx;
        constexpr std::size_t        N = 1024;

        std::vector<float> host(N);
        std::iota(host.begin(), host.end(), 1.f);

        auto device = ctx.allocateShared<float>(N);
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
        auto                  buf  = ctx.allocateShared<int>(N);
        int*                  data = buf.devicePointer<int>();
        for (std::size_t i = 0; i < N; ++i) {
            data[i] = static_cast<int>(i * i);
        }

        for (std::size_t i = 0; i < N; ++i) {
            expect(eq(data[i], static_cast<int>(i * i)));
        }
        ctx.deallocate(buf);
    };

    "backend reports CPU_Fallback"_test = [] {
        gr::device::DeviceContextCpu ctx;
        expect(ctx.backend() == gr::device::DeviceBackend::CPU_Fallback);
        expect(ctx.name() == "CPU fallback");
        expect(ctx.shortName() == "CPU");
    };
};

int main() { /* not needed for UT */ }
