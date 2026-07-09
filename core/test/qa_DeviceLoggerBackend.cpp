#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <tuple>

#include <unistd.h>

#include <boost/ut.hpp>

#include <gnuradio-4.0/DeviceLog.hpp>
#include <gnuradio-4.0/Logger.hpp>
#include <gnuradio-4.0/device/DeviceContext.hpp>
#include <gnuradio-4.0/device/DeviceLoggerBackend.hpp>
#include <gnuradio-4.0/device/ParallelFor.hpp>
#include <gnuradio-4.0/device/SchedulerRegistry.hpp>
#include <gnuradio-4.0/device/SyclRuntime.hpp>

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

// Kernel submission must happen while the AdaptiveCpp runtime is alive. Boost.UT runs registered
// suites from ~runner, i.e. during static destruction, where the HCF kernel registry is already gone
// and every launch aborts with "Could not obtain hcf kernel info". The tests are therefore registered
// (and run) directly from main(); Boost.UT still owns the exit code and fails the binary on a failed
// expect(). Boost.UT has no public dynamic-name test, hence detail::test for the per-device names.
int main() {
    if (!gr::device::registerSyclRuntime()) {
        std::puts("no SYCL backend in this build - nothing to exercise");
        return 0;
    }

    for (const std::string_view domain : {"cpu:sycl"sv, "gpu:sycl"sv}) {
        auto* scheduler = gr::device::SchedulerRegistry::instance().tryResolve(domain);
        if (scheduler == nullptr) {
            continue; // no such device
        }
        gr::device::DeviceContext& context = scheduler->context();
        const bool                 gpu     = context.isGpu();
        const std::string          device  = context.shortName();

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
    return 0;
}
