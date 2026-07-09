#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <tuple>

#include <unistd.h>

#include <boost/ut.hpp>

#include <format>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Logger.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/device/DeviceContext.hpp>
#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/device/DeviceLog.hpp>
#include <gnuradio-4.0/device/ParallelFor.hpp>
#include <gnuradio-4.0/device/SyclRuntime.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include "device_test_helpers.hpp"

using namespace boost::ut;
using namespace std::string_view_literals;

namespace {

constexpr std::size_t kSlots     = 128UZ;
constexpr std::size_t kWorkItems = 2000UZ;
constexpr std::size_t kEmitted   = (kWorkItems + 7UZ) / 8UZ; // every eighth work item emits

// kernel bodies carry no SYCL types: a POD logger handle and the host's own call syntax
struct WarningKernel {
    gr::log::DeviceLogger log;
    bool                  gpu;

    void operator()(std::size_t) const {
        if (gpu) {
            log.warning("origin=sycl_gpu processed {} samples, rate={:.2f} MS/s", 4096UZ, 61.44);
            log.error("origin=sycl_gpu fft_size={} is not a power of two", 4095);
        } else {
            log.warning("origin=sycl_cpu processed {} samples, rate={:.2f} MS/s", 4096UZ, 61.44);
            log.error("origin=sycl_cpu fft_size={} is not a power of two", 4095);
        }
    }
};

struct BatchKernel {
    gr::log::DeviceLogger log;

    void operator()(std::size_t index) const {
        if ((index & 7UZ) == 0UZ) {
            log.warning("batch item={}", index);
        }
    }
};

struct CountingSink : gr::log::Backend {
    std::size_t count{};

    bool publish(const gr::log::LogRecord&) noexcept override {
        ++count;
        return true;
    }
};

// redirects fd 2 so the console backend's own write() output can be asserted on
class CapturedConsole {
    int         _saved{-1};
    int         _file{-1};
    std::string _path{"/tmp/gr_qa_console_XXXXXX"};

public:
    CapturedConsole() {
        _file = ::mkstemp(_path.data());
        std::fflush(stderr);
        _saved      = ::dup(STDERR_FILENO);
        std::ignore = ::dup2(_file, STDERR_FILENO);
    }
    ~CapturedConsole() { std::ignore = ::unlink(_path.c_str()); }

    CapturedConsole(const CapturedConsole&)            = delete;
    CapturedConsole& operator=(const CapturedConsole&) = delete;

    [[nodiscard]] std::string release() {
        std::fflush(stderr);
        std::ignore = ::dup2(_saved, STDERR_FILENO);
        ::close(_saved);
        std::ignore = ::lseek(_file, 0, SEEK_SET);

        std::string text;
        char        buffer[4096];
        for (ssize_t n = ::read(_file, buffer, sizeof(buffer)); n > 0; n = ::read(_file, buffer, sizeof(buffer))) {
            text.append(buffer, static_cast<std::size_t>(n));
        }
        ::close(_file);
        return text;
    }
};

} // namespace

/// Reports, from inside the block's own processing function, whether that function ran on the host or on a device.
/// `__acpp_if_target_device` is the only thing that can tell them apart: SSCP compiles one generic kernel and the
/// host path runs the identical source. The logger handle is a POD (a slab pointer) and is deliberately NOT
/// reflected -- it is not a setting -- which also keeps the block `DeviceRelocatable`.
[[nodiscard]] constexpr std::string_view executionTarget() noexcept {
    std::string_view target{"host", 4UZ};
#ifdef __acpp_if_target_device
    __acpp_if_target_device(target = std::string_view{"device", 6UZ};)
#endif
        return target;
}

struct ReportingGain : gr::Block<ReportingGain> {
    gr::PortIn<float>  in;
    gr::PortOut<float> out;

    gr::Annotated<float, "gain"> gain = 2.f;
    GR_MAKE_REFLECTABLE(ReportingGain, in, out, gain);

    gr::log::DeviceLogger log{}; // unreflected: a diagnostic handle, not a setting

    [[nodiscard]] constexpr float processOne(float sample) const noexcept {
        log.warning("processOne ran on {}", executionTarget());
        return gain * sample;
    }
};

// Boost.UT has no public dynamic-name test, hence detail::test for the per-device names
int main() {
    if (!gr::device::registerSyclRuntime()) {
        std::puts("no SYCL backend in this build - nothing to exercise");
        return 0;
    }
    boost::ut::expect(!gr::device::registerSyclRuntime() || gr::device::hostSyclIsServed()) //
        << "a build with a SYCL backend must serve 'host:sycl'; without it every device case below skips and asserts nothing";

    for (const std::string_view domain : {"host:sycl"sv, "gpu:sycl"sv}) {
        if (!gr::device::DeviceContextRegistry::instance().isServedExactly(domain)) {
            continue; // no such device
        }
        auto*                      scheduler = gr::device::DeviceContextRegistry::instance().tryResolve(domain);
        gr::device::DeviceContext& context   = *scheduler;
        const bool                 gpu       = context.isGpu();
        const std::string          device    = context.shortName();

        const std::string consoleName     = device + ": host and kernel records reach the console";
        detail::test{"test", consoleName} = [&context, gpu] {
            std::string console;
            std::size_t merged = 0UZ;
            {
                CapturedConsole                 capture;
                gr::device::DeviceLoggerBackend backend(context, 8UZ); // host backend defaults to the console
                gr::log::Backend*               previous = gr::log::setBackend(&backend);

                gr::log::warning("origin=host processed {} samples, rate={:.2f} MS/s", 4096UZ, 61.44);
                gr::device::parallelFor(context, 1UZ, WarningKernel{.log = backend.deviceLogger(), .gpu = gpu});
                context.wait();            // the device produces; the host drains only after this barrier
                merged = gr::log::flush(); // decode -> render -> console

                std::ignore = gr::log::setBackend(previous);
                console     = capture.release();
            }
            std::fputs(console.c_str(), stderr); // echo, so the rendered records stay visible
            std::fflush(stderr);

            const std::string_view origin = gpu ? "origin=sycl_gpu"sv : "origin=sycl_cpu"sv;
            expect(eq(merged, 2UZ)) << "both kernel records merge into the host backend";
            expect(console.contains("origin=host"sv)) << "the host record reaches the console";
            expect(console.contains(origin)) << "the kernel record reaches the console";
            expect(console.contains("rate=61.44 MS/s"sv)) << "kernel arguments render host-side";
            expect(console.contains("fft_size=4095"sv)) << "the kernel error argument renders host-side";
            expect(console.contains("qa_DeviceLoggerBackend.cpp"sv)) << "the kernel call site reaches the console";
        };

        const std::string dropName     = device + ": a full slab drops the surplus and counts it";
        detail::test{"test", dropName} = [&context] {
            CountingSink                    sink;
            gr::device::DeviceLoggerBackend backend(context, kSlots, sink);

            gr::device::parallelFor(context, kWorkItems, BatchKernel{.log = backend.deviceLogger()});
            context.wait();

            const std::uint64_t dropped = backend.droppedDeviceRecords();
            expect(eq(backend.flush(), kSlots)) << "a full slab flushes exactly its capacity";
            expect(eq(dropped, kEmitted - kSlots)) << "the surplus is dropped and counted";
            expect(eq(sink.count, kSlots)) << "every flushed record reaches the host backend";
        };
    }

    // the block path: DeviceLog's reason for existing is that a block can say where its own body ran
    for (const std::string_view domain : {"host:sycl"sv, "gpu:sycl"sv}) {
        if (!gr::device::DeviceContextRegistry::instance().isServedExactly(domain)) {
            continue;
        }
        auto*                      resolved = gr::device::DeviceContextRegistry::instance().tryResolve(domain);
        gr::device::DeviceContext& context  = *resolved;
        const std::string          name     = context.shortName() + ": a block's processing function reports where it ran";

        detail::test{"test", name} = [&context, domain] {
            std::string console;
            std::size_t merged = 0UZ;
            {
                CapturedConsole                 capture;
                gr::device::DeviceLoggerBackend backend(context, 32UZ);
                gr::log::Backend* const         previous = gr::log::setBackend(&backend);

                gr::Graph flow;
                auto&     source = flow.emplaceBlock<gr::testing::TagSource<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(64)}, {"mark_tag", false}});
                auto&     dut    = flow.emplaceBlock<ReportingGain>({{"gr:compute_domain", std::string(domain)}});
                auto&     sink   = flow.emplaceBlock<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_ONE>>({{"n_samples_expected", gr::Size_t(64)}});
                dut.log          = backend.deviceLogger();

                expect(flow.connect<"out", "in">(source, dut).has_value());
                expect(flow.connect<"out", "in">(dut, sink).has_value());

                gr::scheduler::Simple<> sched;
                expect(sched.exchange(std::move(flow)).has_value());
                expect(sched.runAndWait().has_value()) << "the reporting block must run to completion";

                context.wait(); // the kernel writes the slab; nothing may decode it before it has finished
                merged      = gr::log::flush();
                std::ignore = gr::log::setBackend(previous);
                console     = capture.release();
            }

            expect(gt(merged, 0UZ)) << "the block's own processing function must have logged";
            if constexpr (gr::test::kKernelHasDeviceCompilationPass) {
                expect(console.contains("processOne ran on device"sv)) << std::format("'{}' must report DEVICE execution, console was: {}", domain, console);
            }
        };
    }
    return 0;
}
