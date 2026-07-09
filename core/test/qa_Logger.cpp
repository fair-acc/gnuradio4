#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <semaphore>
#include <source_location>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>

#include <boost/ut.hpp>

#include <gnuradio-4.0/Logger.hpp>

using namespace boost::ut;
using namespace std::string_view_literals;

namespace {

struct Snapshot {
    std::array<gr::log::LogRecord, gr::log::HistoryLoggerBackend::kCapacity> records{};
    std::size_t                                                              count{};
};

void collectRecord(const gr::log::LogRecord& record, void* user) noexcept {
    auto& state = *static_cast<Snapshot*>(user);
    if (state.count < state.records.size()) {
        state.records[state.count++] = record;
    }
}

void discardRecord(const gr::log::LogRecord&, void*) noexcept {}

Snapshot snapshot(gr::log::HistoryLoggerBackend& backend) noexcept {
    Snapshot state;
    std::ignore = backend.snapshot(collectRecord, &state);
    return state;
}

struct ScopedBackend {
    gr::log::Backend* previous;

    explicit ScopedBackend(gr::log::Backend& backend) : previous(gr::log::setBackend(&backend)) {}
    ~ScopedBackend() { std::ignore = gr::log::setBackend(previous); }
};

} // namespace

static_assert(std::same_as<decltype(gr::log::warning(std::string_view{})), void>);
static_assert(std::same_as<decltype(gr::log::error(std::string_view{})), void>);
static_assert(std::same_as<decltype(gr::log::failure(std::string_view{})), void>);

const boost::ut::suite<"gr::log"> _fatal = [] {
    "throws gr::exception"_test = [] {
        gr::log::HistoryLoggerBackend capture;
        ScopedBackend                 scoped(capture);

        expect(throws<gr::exception>([] { gr::log::fatal("oops"); }));
        const auto records = snapshot(capture);
        expect(eq(records.count, 1UZ));
        expect(records.records[0].level == gr::log::Level::fatal);
    };

    "exception carries supplied message"_test = [] {
        gr::log::HistoryLoggerBackend capture;
        ScopedBackend                 scoped(capture);

        try {
            gr::log::fatal("specific-error-string");
            expect(false) << "fatal should have thrown";
        } catch (const gr::exception& e) {
            expect(eq(e.message, "specific-error-string"sv));
        }
    };

    "captures source_location at call site by default"_test = [] {
        gr::log::HistoryLoggerBackend capture;
        ScopedBackend                 scoped(capture);

        try {
            gr::log::fatal("loc test");
            expect(false) << "fatal should have thrown";
        } catch (const gr::exception& e) {
            const std::string_view file = e.sourceLocation.file_name();
            expect(file.contains("qa_Logger.cpp"sv));
        }
    };

    "accepts explicit source_location"_test = [] {
        gr::log::HistoryLoggerBackend capture;
        ScopedBackend                 scoped(capture);

        const auto loc = std::source_location::current();
        try {
            gr::log::fatal("explicit loc", loc);
            expect(false) << "fatal should have thrown";
        } catch (const gr::exception& e) {
            expect(eq(e.sourceLocation.line(), loc.line()));
            expect(std::string_view{e.sourceLocation.function_name()}.contains(std::string_view{loc.function_name()}));
        }
    };

    "what() includes message and location"_test = [] {
        gr::log::HistoryLoggerBackend capture;
        ScopedBackend                 scoped(capture);

        try {
            gr::log::fatal("what-test");
            expect(false) << "fatal should have thrown";
        } catch (const gr::exception& e) {
            const std::string_view rendered = e.what();
            expect(rendered.contains("what-test"sv));
            expect(rendered.contains("qa_Logger.cpp"sv));
        }
    };

    "empty message is permitted"_test = [] {
        gr::log::HistoryLoggerBackend capture;
        ScopedBackend                 scoped(capture);

        try {
            gr::log::fatal({});
            expect(false) << "fatal should have thrown";
        } catch (const gr::exception& e) {
            expect(eq(e.message, ""sv));
        }
    };

    "fatal(fmt, args...) formats the message before throwing"_test = [] {
        gr::log::HistoryLoggerBackend capture;
        ScopedBackend                 scoped(capture);

        expect(throws<gr::exception>([] { gr::log::fatal("fatal {}", 42); }));
        try {
            gr::log::fatal("fatal {}", 42);
            expect(false) << "fatal should have thrown";
        } catch (const gr::exception& e) {
            expect(eq(e.message, "fatal 42"sv));
        }
    };

    "warning publishes message + source_location"_test = [] {
        gr::log::HistoryLoggerBackend capture;
        ScopedBackend                 scoped(capture);

        gr::log::warning("warn-msg");
        const auto records = snapshot(capture);
        expect(eq(records.count, 1UZ));
        expect(records.records[0].level == gr::log::Level::warning);
        expect(std::string_view{records.records[0].location, records.records[0].locationLength}.contains("qa_Logger.cpp"sv));
        expect(std::string_view{records.records[0].text, records.records[0].textLength}.contains("warn-msg"sv));
    };

    "warning honours explicit source_location"_test = [] {
        gr::log::HistoryLoggerBackend capture;
        ScopedBackend                 scoped(capture);

        const auto loc = std::source_location::current();
        gr::log::warning("with-loc", loc);
        const auto records = snapshot(capture);
        expect(eq(records.records[0].line, loc.line()));
    };

    "error publishes message + source_location"_test = [] {
        gr::log::HistoryLoggerBackend capture;
        ScopedBackend                 scoped(capture);

        gr::log::error("err-msg");
        const auto records = snapshot(capture);
        expect(eq(records.count, 1UZ));
        expect(records.records[0].level == gr::log::Level::error);
        expect(std::string_view{records.records[0].location, records.records[0].locationLength}.contains("qa_Logger.cpp"sv));
        expect(std::string_view{records.records[0].text, records.records[0].textLength}.contains("err-msg"sv));
    };

    "error(fmt, args...) formats through gr::log front-end"_test = [] {
        gr::log::HistoryLoggerBackend capture;
        ScopedBackend                 scoped(capture);

        gr::log::error("e {}", 2);
        const auto records = snapshot(capture);
        expect(eq(records.count, 1UZ));
        expect(records.records[0].level == gr::log::Level::error);
        expect(eq(std::string_view{records.records[0].text, records.records[0].textLength}, "e 2"sv));
    };

    "warning/error are not [[noreturn]] — caller continues"_test = [] {
        gr::log::HistoryLoggerBackend capture;
        ScopedBackend                 scoped(capture);

        int reached = 0;
        gr::log::warning("non-fatal warn");
        ++reached;
        gr::log::error("non-fatal err");
        ++reached;
        expect(eq(reached, 2));
    };

    "format-string API renders through gr::log front-end"_test = [] {
        gr::log::HistoryLoggerBackend capture;
        ScopedBackend                 scoped(capture);

        gr::log::warning("value={} name={}", 42, "alpha");
        const auto records = snapshot(capture);
        expect(eq(records.count, 1UZ));
        expect(eq(std::string_view{records.records[0].text, records.records[0].textLength}, "value=42 name=alpha"sv));
    };

    "runtime format strings must be explicit"_test = [] {
        gr::log::HistoryLoggerBackend capture;
        ScopedBackend                 scoped(capture);

        const std::string fmt = "runtime {}";
        gr::log::error(gr::log::runtime(fmt), "message");
        const auto records = snapshot(capture);
        expect(eq(records.count, 1UZ));
        expect(eq(std::string_view{records.records[0].text, records.records[0].textLength}, "runtime message"sv));
    };

    "warning(runtime(fmt), args...) formats through gr::log front-end"_test = [] {
        gr::log::HistoryLoggerBackend capture;
        ScopedBackend                 scoped(capture);

        gr::log::warning(gr::log::runtime("w {}"), 1);
        const auto records = snapshot(capture);
        expect(eq(records.count, 1UZ));
        expect(records.records[0].level == gr::log::Level::warning);
        expect(eq(std::string_view{records.records[0].text, records.records[0].textLength}, "w 1"sv));
    };

    "runtime format failure yields diagnostic record instead of throwing"_test = [] {
        gr::log::HistoryLoggerBackend capture;
        ScopedBackend                 scoped(capture);

        gr::log::error(gr::log::runtime("{:d}"), "not-an-int");
        const auto records = snapshot(capture);
        expect(eq(records.count, 1UZ));
        expect(records.records[0].level == gr::log::Level::error);
        expect(std::string_view{records.records[0].text, records.records[0].textLength}.starts_with("<log format failed"sv));
    };

    "failure publishes formatted failure records"_test = [] {
        gr::log::HistoryLoggerBackend capture;
        ScopedBackend                 scoped(capture);

        gr::log::failure("failure {}", 7);
        const auto records = snapshot(capture);
        expect(eq(records.count, 1UZ));
        expect(records.records[0].level == gr::log::Level::failure);
        expect(eq(std::string_view{records.records[0].text, records.records[0].textLength}, "failure 7"sv));
        expect(std::string_view{records.records[0].location, records.records[0].locationLength}.contains("qa_Logger.cpp"sv));
    };

    "bounded log record truncates long text"_test = [] {
        gr::log::HistoryLoggerBackend capture;
        ScopedBackend                 scoped(capture);

        const std::string longMessage(gr::log::kLogTextCapacity + 32UZ, 'x');
        gr::log::warning(longMessage);
        const auto records = snapshot(capture);
        expect(eq(records.count, 1UZ));
        expect(records.records[0].textTruncated);
        expect(eq(records.records[0].textLength, static_cast<std::uint16_t>(gr::log::kLogTextCapacity - 1UZ)));
        expect(eq(records.records[0].text[records.records[0].textLength], '\0'));
    };

    "info debug trace follow debug-build policy"_test = [] {
        gr::log::HistoryLoggerBackend capture;
        ScopedBackend                 scoped(capture);

        gr::log::info("info {}", 1);
        gr::log::debug("debug {}", 2);
        gr::log::trace("trace {}", 3);
        const auto records = snapshot(capture);
        if constexpr (gr::meta::kDebugBuild) {
            expect(eq(records.count, 3UZ));
        } else {
            expect(eq(records.count, 0UZ));
        }
    };

    "visual default-console formatting examples"_test = [] {
        auto* previous = gr::log::setBackend(nullptr);
        std::ignore    = gr::log::drain(discardRecord);

        gr::log::detail::publish(gr::log::Level::failure, "\"visual failure - device=sycl queue=0\"");
        gr::log::detail::publish(gr::log::Level::error, "\"visual error   - block=fft_backend status=-17\"");
        gr::log::detail::publish(gr::log::Level::warning, "\"visual warning - dropped=128 total=4096\"");
        gr::log::detail::publish(gr::log::Level::info, "\"visual info    - scheduler=default_cpu workers=4\"");
        gr::log::detail::publish(gr::log::Level::debug, "\"visual debug   - span=[0, 1024) rate=1.0\"");

        Snapshot   state;
        const auto drained = gr::log::drain(collectRecord, &state);
        std::ignore        = gr::log::setBackend(previous);

        expect(drained >= 5UZ);
        expect(eq(state.count, 5UZ));
        expect(state.records[0].level == gr::log::Level::failure);
        expect(state.records[1].level == gr::log::Level::error);
        expect(state.records[2].level == gr::log::Level::warning);
        expect(state.records[3].level == gr::log::Level::info);
        expect(state.records[4].level == gr::log::Level::debug);
    };

    "default backend writes through — nothing pending after publish"_test = [] {
        auto* previous = gr::log::setBackend(nullptr); // compiled default: write-through console
        std::ignore    = gr::log::drain(discardRecord);
        std::ignore    = gr::log::flush(); // clear any console backlog

        gr::log::warning("\"visual warning - write-through probe\"");
        const std::size_t pending = gr::log::flush();

        std::ignore = gr::log::setBackend(previous);
        expect(eq(pending, 0UZ)) << "publish must flush the console ring on the same thread — no external pump";
    };

    "history backend keeps newest records and drains once"_test = [] {
        gr::log::HistoryLoggerBackend capture;
        ScopedBackend                 scoped(capture);

        for (std::size_t i = 0UZ; i < gr::log::HistoryLoggerBackend::kCapacity + 5UZ; ++i) {
            gr::log::warning("recent {}", i);
        }

        auto records = snapshot(capture);
        expect(eq(capture.size(), gr::log::HistoryLoggerBackend::kCapacity));
        constexpr std::uint64_t expectedPublished = gr::log::HistoryLoggerBackend::kCapacity + std::uint64_t{5};
        expect(eq(capture.published(), expectedPublished));
        expect(eq(records.count, gr::log::HistoryLoggerBackend::kCapacity));
        expect(eq(std::string_view{records.records[0].text, records.records[0].textLength}, "recent 5"sv));
        expect(eq(std::string_view{records.records[records.count - 1UZ].text, records.records[records.count - 1UZ].textLength}, "recent 24"sv));

        Snapshot drained;
        expect(eq(capture.drain(collectRecord, &drained), gr::log::HistoryLoggerBackend::kCapacity));
        expect(eq(capture.drain(collectRecord, &drained), 0UZ));
    };

    "bounded log record truncates long location"_test = [] {
        gr::log::HistoryLoggerBackend capture;
        ScopedBackend                 scoped(capture);

        gr::log::warning("short location message");
        const auto records = snapshot(capture);
        expect(eq(records.count, 1UZ));
        const auto& record = records.records[0];
        expect(lt(record.locationLength, static_cast<std::uint16_t>(gr::log::kLogLocationCapacity)));
        expect(!record.locationTruncated) << "qa_Logger.cpp is far shorter than kLogLocationCapacity";
        expect(eq(record.location[record.locationLength], '\0'));

        gr::log::LogRecord overflow{};
        overflow.level = gr::log::Level::warning;
        const std::string     longLocation(gr::log::kLogLocationCapacity + 16UZ, 'L');
        constexpr std::size_t kMaxLocationLength = gr::log::kLogLocationCapacity - 1UZ;
        std::copy_n(longLocation.data(), kMaxLocationLength, overflow.location);
        overflow.location[kMaxLocationLength] = '\0';
        overflow.locationLength               = static_cast<std::uint16_t>(kMaxLocationLength);
        overflow.locationTruncated            = true;

        gr::log::HistoryLoggerBackend overflowCapture;
        expect(overflowCapture.publish(overflow));
        const auto overflowRecords = snapshot(overflowCapture);
        expect(eq(overflowRecords.count, 1UZ));
        expect(overflowRecords.records[0].locationTruncated);
        expect(eq(overflowRecords.records[0].locationLength, static_cast<std::uint16_t>(kMaxLocationLength)));
        expect(eq(overflowRecords.records[0].location[overflowRecords.records[0].locationLength], '\0'));
    };

    "history backend records carry droppedBefore after overflow"_test = [] {
        gr::log::HistoryLoggerBackend capture;
        ScopedBackend                 scoped(capture);

        constexpr std::size_t overflowCount = 5UZ;
        for (std::size_t i = 0UZ; i < gr::log::HistoryLoggerBackend::kCapacity + overflowCount; ++i) {
            gr::log::warning("overflow {}", i);
        }

        const auto records = snapshot(capture);
        expect(eq(records.count, gr::log::HistoryLoggerBackend::kCapacity));

        for (std::size_t i = 0UZ; i + overflowCount < records.count; ++i) {
            expect(eq(records.records[i].droppedBefore, std::uint64_t{0})) << "records retained from before capacity was reached must not report drops";
        }
        for (std::size_t i = 0UZ; i < overflowCount; ++i) {
            const auto& record = records.records[records.count - overflowCount + i];
            expect(eq(record.droppedBefore, i + 1UZ)) << "each overflow record reports how many earlier records it displaced";
        }
    };

    "Backend base-class drain and flush are inert defaults"_test = [] {
        struct PublishOnlyBackend : gr::log::Backend {
            bool publish(const gr::log::LogRecord&) noexcept override { return true; }
        };

        PublishOnlyBackend backend;
        ScopedBackend      scoped(backend);

        gr::log::warning("into inert backend");
        expect(eq(gr::log::drain(discardRecord), 0UZ));
        expect(eq(gr::log::flush(), 0UZ));
    };

    "setBackend returns the previously active backend"_test = [] {
        gr::log::HistoryLoggerBackend backendA;
        gr::log::HistoryLoggerBackend backendB;

        auto* original                = gr::log::setBackend(&backendA);
        auto* returnedWhenInstallingB = gr::log::setBackend(&backendB);
        expect(eq(returnedWhenInstallingB, &backendA));

        std::ignore = gr::log::setBackend(original);
    };

    "empty format arguments render literal text"_test = [] {
        gr::log::HistoryLoggerBackend capture;
        ScopedBackend                 scoped(capture);

        gr::log::warning<>("plain literal");
        const auto records = snapshot(capture);
        expect(eq(records.count, 1UZ));
        expect(eq(std::string_view{records.records[0].text, records.records[0].textLength}, "plain literal"sv));
    };

    "defaultBackend() returns a stable singleton"_test = [] { expect(&gr::log::defaultBackend() == &gr::log::defaultBackend()); };

    "visual console dropped-count formatting example"_test = [] {
        gr::log::LogRecord     record{};
        const std::string_view text = "\"visual warning - droppedBefore rendering\"";
        record.level                = gr::log::Level::warning;
        record.timestampNanos       = 1'725'000'000'000'000'000ULL;
        record.droppedBefore        = std::uint64_t{3};
        std::copy_n(text.data(), text.size(), record.text);
        record.textLength = static_cast<std::uint16_t>(text.size());

        expect(gr::log::defaultBackend().publish(record));
    };

    "history publish under contention counts the dropped record"_test = [] {
        gr::log::HistoryLoggerBackend backend;

        gr::log::LogRecord first{};
        expect(backend.publish(first)) << "first publish on an idle backend succeeds";

        struct Sync {
            std::binary_semaphore inside{0};
            std::binary_semaphore resume{0};
        } sync;

        auto park = [](const gr::log::LogRecord&, void* user) noexcept {
            auto& state = *static_cast<Sync*>(user);
            state.inside.release();
            state.resume.acquire();
        };

        std::thread reader([&] { std::ignore = backend.snapshot(park, &sync); });

        sync.inside.acquire(); // reader now holds _busy inside the consumer
        gr::log::LogRecord contended{};
        expect(!backend.publish(contended)) << "publish while the backend is busy is rejected";
        sync.resume.release();
        reader.join();

        gr::log::LogRecord retained{};
        expect(backend.publish(retained)) << "publish after contention succeeds";

        const auto records = snapshot(backend);
        expect(eq(records.records[records.count - 1UZ].droppedBefore, std::uint64_t{1})) << "the retained record folds in the single contention drop";
    };
};

int main() { /* statically registered */ }
