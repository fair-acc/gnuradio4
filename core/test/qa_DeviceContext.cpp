#include <boost/ut.hpp>

#include <cstring>
#include <format>
#include <numeric>
#include <vector>

#include "device_test_helpers.hpp"
#include <gnuradio-4.0/Complex.hpp>
#include <gnuradio-4.0/device/DeviceContext.hpp>

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
#endif

const suite<"device::DeviceContext"> tests = [] {
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
