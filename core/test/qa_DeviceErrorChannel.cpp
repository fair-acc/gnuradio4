#include <boost/ut.hpp>

#include <cstdio>

#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/device/DeviceContextSycl.hpp>
#include <gnuradio-4.0/device/SyclRuntime.hpp>

#if GR_DEVICE_HAS_SYCL_IMPL

#include <sycl/sycl.hpp>

int main() {
    using namespace boost::ut;

    if (!gr::device::registerSyclRuntime()) {
        std::puts("SKIP: this build has no SYCL backend");
        return 77;
    }

    auto* scheduler = gr::device::DeviceContextRegistry::instance().tryResolve("gpu:sycl");
    if (scheduler == nullptr) {
        std::puts("SKIP: no gpu:sycl device available");
        return 77;
    }
    auto& ctx = static_cast<gr::device::DeviceContextSycl&>(*scheduler); // "gpu:sycl" only ever registers a DeviceContextSycl

    // GPU only: an out-of-bounds USM write on a CPU/host SYCL device is an ordinary host-process
    // memory access, so it either corrupts silently or SIGSEGVs the whole test - neither proves
    // anything about the async error channel. A GPU device reports it as a recoverable async fault.
    "a genuine out-of-bounds device write surfaces through pollDeviceError"_test = [&ctx] {
        expect(!ctx.pollDeviceError().has_value()) << "the context starts clean";

        auto buf = ctx.allocateDevice<int>(1UZ);
        expect(static_cast<bool>(buf)) << "device allocation for the fault-injection buffer";

        int* dBuf = buf.devicePointer<int>(); // resolved once, at the kernel-launch boundary
        ctx.queue->submit([dBuf](sycl::handler& h) { h.parallel_for(sycl::range<1>{1}, [dBuf](sycl::id<1>) { dBuf[1'000'000'000] = 42; }); });
        ctx.queue->wait(); // synchronous; acpp registers the fault in its global async-error list here but does not throw

        expect(ctx.pollDeviceError().has_value()) << "an out-of-bounds device write must surface as a device error";
        expect(ctx.pollDeviceError().has_value()) << "the poisoned latch is sticky: a second poll still reports the fault";
        // the context is now permanently poisoned (acpp exposes no reset) - skip deallocate(buf), it would just re-fault
    };

    return 0;
}

#else

int main() {
    std::puts("SKIP: not built with an AdaptiveCpp SYCL backend");
    return 77;
}

#endif
