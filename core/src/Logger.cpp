#include <gnuradio-4.0/Logger.hpp>

#include <algorithm>
#include <chrono>
#include <format>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

#if !defined(EMBEDDED)
#include <charconv>
#if __has_include(<unistd.h>)
#include <unistd.h>
#define GR_LOG_HAS_POSIX_WRITE 1
#endif
#endif

#include <gnuradio-4.0/AtomicRef.hpp>
#if !defined(EMBEDDED)
#include <gnuradio-4.0/CircularBuffer.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>
#endif

#if !defined(EMBEDDED)
#ifndef GR_LOG_CONSOLE_SPLIT_STDOUT_STDERR
#define GR_LOG_CONSOLE_SPLIT_STDOUT_STDERR 1
#endif
#endif

namespace gr::log {

std::size_t Backend::drain(RecordConsumer, void*) noexcept { return 0UZ; }
std::size_t Backend::flush() noexcept { return 0UZ; }

namespace {

[[nodiscard]] constexpr std::size_t historyIndex(std::uint64_t sequence) noexcept {
    constexpr auto capacity = static_cast<std::uint64_t>(HistoryLoggerBackend::kCapacity);
    const auto     index    = sequence % capacity;
    if constexpr (sizeof(std::size_t) >= sizeof(std::uint64_t)) {
        return index;
    } else {
        return static_cast<std::size_t>(index);
    }
}

} // namespace

bool HistoryLoggerBackend::publish(const LogRecord& record) noexcept {
    if (gr::atomic_ref(_busy).exchange(true)) {
        gr::atomic_ref(_droppedContended).fetch_add(std::uint64_t{1});
        return false;
    }

    constexpr auto      capacity    = static_cast<std::uint64_t>(HistoryLoggerBackend::kCapacity);
    const std::uint64_t sequence    = gr::atomic_ref(_published).load_relaxed();
    LogRecord           local       = record;
    const std::uint64_t overwritten = sequence >= capacity ? sequence + std::uint64_t{1} - capacity : std::uint64_t{0};
    const std::uint64_t lost        = overwritten + gr::atomic_ref(_droppedContended).load_relaxed();
    if (lost > 0U) {
        local.droppedBefore = std::max(local.droppedBefore, lost);
    }
    _records[historyIndex(sequence)] = local;
    gr::atomic_ref(_published).store_release(sequence + std::uint64_t{1});
    gr::atomic_ref(_busy).store_release(false);
    return true;
}

std::size_t HistoryLoggerBackend::snapshot(RecordConsumer consumer, void* user) noexcept { return read(consumer, user, false); }

std::size_t HistoryLoggerBackend::drain(RecordConsumer consumer, void* user) noexcept { return read(consumer, user, true); }

void HistoryLoggerBackend::clear() noexcept {
    if (gr::atomic_ref(_busy).exchange(true)) {
        return;
    }
    _records = {};
    gr::atomic_ref(_drained).store_release(std::uint64_t{0});
    gr::atomic_ref(_droppedContended).store_release(std::uint64_t{0});
    gr::atomic_ref(_published).store_release(std::uint64_t{0});
    gr::atomic_ref(_busy).store_release(false);
}

std::size_t HistoryLoggerBackend::size() noexcept { return static_cast<std::size_t>(std::min(gr::atomic_ref(_published).load_acquire(), static_cast<std::uint64_t>(kCapacity))); }

std::uint64_t HistoryLoggerBackend::published() noexcept { return gr::atomic_ref(_published).load_acquire(); }

std::size_t HistoryLoggerBackend::read(RecordConsumer consumer, void* user, bool consume) noexcept {
    if (consumer == nullptr) {
        return 0UZ;
    }
    if (gr::atomic_ref(_busy).exchange(true)) {
        return 0UZ;
    }

    constexpr auto      capacity      = static_cast<std::uint64_t>(HistoryLoggerBackend::kCapacity);
    const std::uint64_t end           = gr::atomic_ref(_published).load_acquire();
    const std::uint64_t retainedBegin = end > capacity ? end - capacity : 0U;
    const std::uint64_t drained       = std::min(gr::atomic_ref(_drained).load_acquire(), end);
    const std::uint64_t begin         = consume ? std::max(retainedBegin, drained) : retainedBegin;
    const auto          n             = static_cast<std::size_t>(std::min(end - begin, capacity));

    for (std::size_t i = 0UZ; i < n; ++i) {
        consumer(_records[historyIndex(begin + i)], user);
    }
    if (consume) {
        gr::atomic_ref(_drained).store_release(end);
    }
    gr::atomic_ref(_busy).store_release(false);
    return n;
}

namespace {

using Clock = std::chrono::system_clock;

#if !defined(EMBEDDED)
using Ring = gr::CircularBuffer<LogRecord, std::dynamic_extent, gr::ProducerType::Multi>;

constexpr std::size_t kDefaultCapacity = 1024UZ;

[[nodiscard]] constexpr bool routeToStderr([[maybe_unused]] Level level) noexcept {
#if GR_LOG_CONSOLE_SPLIT_STDOUT_STDERR
    switch (level) {
    case Level::fatal:
    case Level::error:
    case Level::warning: return true;
    case Level::failure:
    case Level::info:
    case Level::debug:
    case Level::trace: return false;
    }
#endif
    return true;
}
#endif

[[nodiscard]] constexpr std::string_view filename(std::string_view path) noexcept {
    const auto slash = path.find_last_of("/\\");
    return slash == std::string_view::npos ? path : path.substr(slash + 1UZ);
}

[[nodiscard]] std::uint64_t nowNanos() noexcept { return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count()); }

LogRecord makeRecordAt(Level level, std::string_view message, std::string_view file, std::uint32_t line, std::uint32_t column, std::uint64_t droppedBefore) noexcept {
    LogRecord  record{};
    const auto store = []<std::size_t N>(char(&dst)[N], std::uint16_t& length, std::string_view text) noexcept {
        static_assert(N > 0UZ);
        const std::size_t n = std::min(text.size(), N - 1UZ);
        if (n > 0UZ) {
            std::copy_n(text.data(), n, dst);
        }
        dst[n] = '\0';
        length = static_cast<std::uint16_t>(n);
        return text.size() > n;
    };

    record.level          = level;
    record.line           = line;
    record.column         = column;
    record.timestampNanos = nowNanos();
    record.droppedBefore  = droppedBefore;
    record.textTruncated  = store(record.text, record.textLength, message);

    record.locationTruncated = store(record.location, record.locationLength, filename(file));
    return record;
}

LogRecord makeRecord(Level level, std::string_view message, std::source_location loc, std::uint64_t droppedBefore) noexcept { return makeRecordAt(level, message, loc.file_name(), loc.line(), loc.column(), droppedBefore); }

#if !defined(EMBEDDED)
class FixedRecordBackend final : public Backend {
    using Reader = decltype(std::declval<Ring&>().new_reader());

    Ring                 _ring;
    Reader               _consoleReader;
    HistoryLoggerBackend _history;
    std::uint64_t        _droppedRecords{};
    bool                 _consoleBusy{};
    bool                 _flushBusy{};

    static void appendUnsigned(char* line, std::size_t capacity, std::size_t& pos, std::uint64_t value) noexcept {
        if (pos >= capacity) {
            return;
        }
        const auto result = std::to_chars(line + pos, line + capacity, value);
        if (result.ec == std::errc{}) {
            pos = static_cast<std::size_t>(result.ptr - line);
        }
    }

    static void appendFixed(char* line, std::size_t capacity, std::size_t& pos, std::uint32_t value, std::uint32_t width) noexcept {
        if (pos >= capacity) {
            return;
        }
        char digits[10]{};
        for (std::uint32_t i = 0U; i < width; ++i) {
            digits[width - i - 1U] = static_cast<char>('0' + (value % 10U));
            value /= 10U;
        }
        const std::size_t n = std::min(static_cast<std::size_t>(width), capacity - pos);
        if (n > 0UZ) {
            std::copy_n(digits, n, line + pos);
            pos += n;
        }
    }

    static void appendTimestamp(char* line, std::size_t capacity, std::size_t& pos, std::uint64_t nanos) noexcept {
        using namespace std::chrono;
        const sys_time<nanoseconds>  timestamp{nanoseconds{nanos}};
        const auto                   dayPoint = floor<days>(timestamp);
        const year_month_day         ymd{dayPoint};
        const hh_mm_ss<milliseconds> tod{floor<milliseconds>(timestamp - dayPoint)};
        const auto                   seconds = tod.seconds();

        appendFixed(line, capacity, pos, static_cast<std::uint32_t>(static_cast<int>(ymd.year())), 4U);
        appendChar(line, capacity, pos, '-');
        appendFixed(line, capacity, pos, static_cast<unsigned>(ymd.month()), 2U);
        appendChar(line, capacity, pos, '-');
        appendFixed(line, capacity, pos, static_cast<unsigned>(ymd.day()), 2U);
        appendChar(line, capacity, pos, 'T');
        appendFixed(line, capacity, pos, static_cast<std::uint32_t>(tod.hours().count()), 2U);
        appendChar(line, capacity, pos, ':');
        appendFixed(line, capacity, pos, static_cast<std::uint32_t>(tod.minutes().count()), 2U);
        appendChar(line, capacity, pos, ':');
        appendFixed(line, capacity, pos, static_cast<std::uint32_t>(seconds.count()), 2U);
        appendChar(line, capacity, pos, '.');
        appendFixed(line, capacity, pos, static_cast<std::uint32_t>(tod.subseconds().count()), 3U);
    }

    static void appendChar(char* line, std::size_t capacity, std::size_t& pos, char value) noexcept {
        if (pos < capacity) {
            line[pos++] = value;
        }
    }

    void writeConsole(const LogRecord& record) noexcept {
        if (gr::atomic_ref(_consoleBusy).exchange(true)) {
            return;
        }

#if defined(GR_LOG_HAS_POSIX_WRITE)
        char        line[768]{};
        std::size_t pos    = 0UZ;
        auto        append = [&](std::string_view text) noexcept {
            const std::size_t n = std::min(text.size(), sizeof(line) - 1UZ - pos);
            if (n > 0UZ) {
                std::copy_n(text.data(), n, line + pos);
                pos += n;
            }
        };
        auto appendLevel = [&]() noexcept {
            char       label[7] = {' ', ' ', ' ', ' ', ' ', ' ', ' '};
            const auto name     = gr::meta::enumName(record.level).value_or("unknown");
            const auto n        = std::min(name.size(), sizeof(label));
            auto       upper    = [](char c) noexcept { return c >= 'a' && c <= 'z' ? static_cast<char>(c - ('a' - 'A')) : c; };
            for (std::size_t i = 0UZ; i < n; ++i) {
                label[i] = upper(name[i]);
            }
            append(std::string_view(label, sizeof(label)));
        };
        appendTimestamp(line, sizeof(line) - 1UZ, pos, record.timestampNanos);
        append(" ");
        appendLevel();
        append(" : ");
        append(std::string_view(record.text, record.textLength));
        if (record.textTruncated) {
            append(" <truncated>");
        }
        if (record.droppedBefore != 0UZ) {
            append(" dropped=");
            appendUnsigned(line, sizeof(line) - 1UZ, pos, record.droppedBefore);
        }
        append(" in: ");
        append(std::string_view(record.location, record.locationLength));
        if (record.line != 0U) {
            append(":");
            appendUnsigned(line, sizeof(line) - 1UZ, pos, record.line);
        }
        append("\n");
        std::ignore = ::write(routeToStderr(record.level) ? STDERR_FILENO : STDOUT_FILENO, line, pos);
#endif

        gr::atomic_ref(_consoleBusy).store_release(false);
    }

public:
    FixedRecordBackend() : _ring(kDefaultCapacity), _consoleReader(_ring.new_reader()) {}

    bool publish(const LogRecord& record) noexcept override {
        LogRecord local      = record;
        local.timestampNanos = local.timestampNanos == 0U ? nowNanos() : local.timestampNanos;
        local.droppedBefore  = std::max(local.droppedBefore, gr::atomic_ref(_droppedRecords).load_relaxed());
        _history.publish(local);

        {
            auto writer = _ring.new_writer();
            auto span   = writer.template tryReserve<gr::SpanReleasePolicy::ProcessAll>(1UZ);
            if (span.size() == 0UZ) { // ring full: the record never reaches the console reader, so emit it directly
                local.droppedBefore = std::max(local.droppedBefore, gr::atomic_ref(_droppedRecords).fetch_add(1UZ) + 1UZ);
                writeConsole(local);
                return false;
            }
            span[0] = local;
        } // ~WriterSpan commits before the console reader can observe the record

        flush(); // write-through: the default backend stays visible without an external pump
        return true;
    }

    std::size_t drain(RecordConsumer consumer, void* user) noexcept override {
        if (consumer == nullptr) {
            return 0UZ;
        }
        return _history.drain(consumer, user);
    }

    std::size_t flush() noexcept override {
        if (gr::atomic_ref(_flushBusy).exchange(true)) {
            return 0UZ;
        }

        auto              span = _consoleReader.get();
        const std::size_t n    = span.size();
        for (std::size_t i = 0UZ; i < n; ++i) {
            writeConsole(span[i]);
        }
        std::ignore = span.consume(n);
        gr::atomic_ref(_flushBusy).store_release(false);
        return n;
    }
};

FixedRecordBackend& defaultFixedBackend() noexcept {
    static FixedRecordBackend backend;
    return backend;
}
#endif

#if defined(EMBEDDED)
HistoryLoggerBackend& defaultHistoryBackend() noexcept {
    static HistoryLoggerBackend backend;
    return backend;
}
#endif

Backend& compiledDefaultBackend() noexcept {
#if defined(EMBEDDED)
    return defaultHistoryBackend();
#else
    return defaultFixedBackend();
#endif
}

Backend*& activeBackendStorage() noexcept {
    static Backend* backend = &compiledDefaultBackend();
    return backend;
}

} // namespace

namespace detail {

Backend& defaultBackend() noexcept { return compiledDefaultBackend(); }

Backend* setBackend(Backend* backend) noexcept { return gr::atomic_ref(activeBackendStorage()).exchange(backend == nullptr ? &compiledDefaultBackend() : backend); }

Backend& activeBackend() noexcept {
    Backend* backend = gr::atomic_ref(activeBackendStorage()).load_acquire();
    return backend == nullptr ? compiledDefaultBackend() : *backend;
}

void publish(Level level, std::string_view message, std::source_location loc) noexcept { activeBackend().publish(makeRecord(level, message, loc, 0UZ)); }

std::string formatMessage(std::string_view fmt, std::format_args args) noexcept {
#if __cpp_exceptions
    try {
        return std::vformat(fmt, args);
    } catch (const std::exception& ex) {
        return std::format("<log format failed: {}>", ex.what());
    } catch (...) {
        return "<log format failed>";
    }
#else
    return std::vformat(fmt, args);
#endif
}

void publishFormatted(Level level, std::string_view fmt, std::format_args args, std::source_location loc) noexcept { publish(level, formatMessage(fmt, args), loc); }

std::size_t drain(RecordConsumer consumer, void* user) noexcept { return activeBackend().drain(consumer, user); }

std::size_t flush() noexcept { return activeBackend().flush(); }

} // namespace detail

} // namespace gr::log

namespace gr {

exception::exception(std::string_view msg, std::source_location location) noexcept : message(msg), sourceLocation(location) {}

const char* exception::what() const noexcept {
    if (_formattedMessage.empty()) {
        _formattedMessage = std::format("{} at {}:{}", message, sourceLocation.file_name(), sourceLocation.line());
    }
    return _formattedMessage.c_str();
}

Error::Error(std::string_view msg, std::source_location location, std::chrono::system_clock::time_point time) noexcept //
    : message(msg), sourceLocation(location), errorTime(time) {}

Error::Error(const std::exception& ex, std::source_location location) noexcept : Error(ex.what(), location) {}

Error::Error(const gr::exception& ex) noexcept : Error(ex.message, ex.sourceLocation, ex.errorTime) {}

std::string Error::srcLoc() const noexcept { return std::format("{}:{}", sourceLocation.file_name(), sourceLocation.line()); }

std::string Error::methodName() const noexcept { return {sourceLocation.function_name()}; }

std::string Error::isoTime() const noexcept { return std::format("{}", errorTime); }

} // namespace gr
