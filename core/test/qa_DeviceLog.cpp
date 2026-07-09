#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <tuple>
#include <type_traits>

#include <boost/ut.hpp>

#include <gnuradio-4.0/DeviceLog.hpp>
#include <gnuradio-4.0/Logger.hpp>
#include <gnuradio-4.0/device/DeviceContext.hpp>
#include <gnuradio-4.0/device/DeviceLoggerBackend.hpp>

using namespace boost::ut;
using namespace std::string_view_literals;

static_assert(std::is_trivially_copyable_v<gr::log::DeviceLogSlab>);
static_assert(std::is_trivially_copyable_v<gr::log::DeviceLogger>); // a kernel captures it by value

namespace {

// stores the rendered records so the expected message text can be compared
struct RecordSink : gr::log::Backend {
    std::array<gr::log::LogRecord, 8UZ> records{};
    std::size_t                         count{};

    bool publish(const gr::log::LogRecord& record) noexcept override {
        if (count < records.size()) {
            records[count++] = record;
        }
        return true;
    }

    [[nodiscard]] std::string_view text(std::size_t index) const noexcept { return {records[index].text, records[index].textLength}; }
    [[nodiscard]] std::string_view file(std::size_t index) const noexcept { return {records[index].location, records[index].locationLength}; }
};

// the CPU-fallback kernel body: identical to what a SYCL kernel runs
void emitRecords(const gr::log::DeviceLogger& log, std::size_t count) noexcept {
    for (std::size_t i = 0UZ; i < count; ++i) {
        log.warning("record {}", i);
    }
}

} // namespace

const boost::ut::suite<"gr::log::DeviceLogger"> _deviceLog = [] {
    "renders arguments and format specifications host-side"_test = [] {
        gr::device::DeviceContextCpu    context;
        RecordSink                      sink;
        gr::device::DeviceLoggerBackend backend(context, 8UZ, sink);
        const gr::log::DeviceLogger     log = backend.deviceLogger();
        expect(backend.valid());

        log.warning("processed {} items", 42UZ);
        log.warning("v={:.2f} ok={}", 1.5, true);
        log.error("{} and {} and {}", -7, 8UZ, "str"sv);
        context.wait(); // the device produces; the host decodes only after this barrier

        expect(eq(backend.flush(), 3UZ));
        expect(eq(sink.count, 3UZ));
        expect(eq(sink.text(0), "processed 42 items"sv));
        expect(eq(sink.text(1), "v=1.50 ok=true"sv));
        expect(eq(sink.text(2), "-7 and 8 and str"sv));
        expect(sink.records[2].level == gr::log::Level::error);
    };

    "captures the emitting call site"_test = [] {
        gr::device::DeviceContextCpu    context;
        RecordSink                      sink;
        gr::device::DeviceLoggerBackend backend(context, 8UZ, sink);

        backend.deviceLogger().warning("call site");
        context.wait();

        expect(eq(backend.flush(), 1UZ));
        expect(eq(sink.file(0), "qa_DeviceLog.cpp"sv));
        expect(neq(sink.records[0].line, 0U));
    };

    "a full slab drops instead of blocking"_test = [] {
        gr::device::DeviceContextCpu    context;
        RecordSink                      sink;
        gr::device::DeviceLoggerBackend backend(context, 4UZ, sink);

        emitRecords(backend.deviceLogger(), 10UZ);
        context.wait();

        expect(eq(backend.droppedDeviceRecords(), std::uint64_t{6}));
        expect(eq(backend.flush(), 4UZ));
        expect(eq(sink.count, 4UZ));
    };

    "the slab is reusable after a flush"_test = [] {
        gr::device::DeviceContextCpu    context;
        RecordSink                      sink;
        gr::device::DeviceLoggerBackend backend(context, 4UZ, sink);

        emitRecords(backend.deviceLogger(), 2UZ);
        context.wait();
        expect(eq(backend.flush(), 2UZ));

        emitRecords(backend.deviceLogger(), 2UZ);
        context.wait();
        expect(eq(backend.flush(), 2UZ));
        expect(eq(sink.count, 4UZ));
    };

    "host records bypass the device slab"_test = [] {
        gr::device::DeviceContextCpu    context;
        RecordSink                      sink;
        gr::device::DeviceLoggerBackend backend(context, 8UZ, sink);

        gr::log::Backend* previous = gr::log::setBackend(&backend);
        gr::log::warning("host record");
        std::ignore = gr::log::setBackend(previous);

        expect(eq(backend.flush(), 0UZ)) << "the slab stayed empty";
        expect(eq(sink.count, 1UZ));
        expect(eq(sink.text(0), "host record"sv));
    };

    "an unallocated logger is a safe no-op"_test = [] {
        const gr::log::DeviceLogger log{};
        RecordSink                  sink;

        expect(not log.valid());
        log.warning("must not crash {}", 1);
        log.error("must not crash");

        expect(eq(gr::log::drainDeviceLog(log.slab, sink), 0UZ));
        expect(eq(sink.count, 0UZ));
    };

    "static slab storage needs no allocator"_test = [] {
        gr::log::StaticDeviceLogSlab<2> storage;
        const gr::log::DeviceLogger     log{.slab = storage.view()};
        RecordSink                      sink;

        expect(log.valid());
        log.warning("static {}", 5);

        expect(eq(gr::log::drainDeviceLog(storage.view(), sink), 1UZ));
        expect(eq(sink.text(0), "static 5"sv));
    };
};

int main() { /* statically registered */ }
