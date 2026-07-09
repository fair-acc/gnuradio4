#ifndef GNURADIO_LOGGER_HPP
#define GNURADIO_LOGGER_HPP

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <format>
#include <source_location>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

#include <gnuradio-4.0/meta/utils.hpp>

namespace gr {

namespace log {

enum class Level : std::uint8_t { fatal, failure, error, warning, info, debug, trace };

inline constexpr std::size_t kLogTextCapacity     = 512UZ;
inline constexpr std::size_t kLogLocationCapacity = 96UZ;

struct LogRecord {
    Level         level{};
    std::uint16_t textLength{};
    std::uint16_t locationLength{};
    std::uint32_t line{};
    std::uint32_t column{};
    std::uint64_t timestampNanos{};
    std::uint64_t droppedBefore{};
    bool          textTruncated{};
    bool          locationTruncated{};
    char          location[kLogLocationCapacity]{};
    char          text[kLogTextCapacity]{};
};

static_assert(std::is_trivially_copyable_v<LogRecord>);

using RecordConsumer = void (*)(const LogRecord&, void*) noexcept;

class Backend {
public:
    virtual ~Backend() = default;

    virtual bool        publish(const LogRecord& record) noexcept = 0;
    virtual std::size_t drain(RecordConsumer consumer, void* user) noexcept;
    virtual std::size_t flush() noexcept;
};

class HistoryLoggerBackend final : public Backend {
public:
    static constexpr std::size_t kCapacity = 20UZ;

    bool                        publish(const LogRecord& record) noexcept override;
    [[nodiscard]] std::size_t   snapshot(RecordConsumer consumer, void* user = nullptr) noexcept;
    [[nodiscard]] std::size_t   drain(RecordConsumer consumer, void* user = nullptr) noexcept override;
    void                        clear() noexcept;
    [[nodiscard]] std::size_t   size() noexcept;
    [[nodiscard]] std::uint64_t published() noexcept;

private:
    std::array<LogRecord, kCapacity> _records{};
    std::uint64_t                    _published{};
    std::uint64_t                    _drained{};
    bool                             _busy{};

    [[nodiscard]] std::size_t read(RecordConsumer consumer, void* user, bool consume) noexcept;
};

namespace detail {

[[nodiscard]] Backend&    defaultBackend() noexcept;
Backend*                  setBackend(Backend* backend) noexcept;
[[nodiscard]] Backend&    activeBackend() noexcept;
void                      publish(Level level, std::string_view message, std::source_location loc = std::source_location::current()) noexcept;
[[nodiscard]] std::size_t drain(RecordConsumer consumer, void* user = nullptr) noexcept;
[[nodiscard]] std::size_t flush() noexcept;
[[nodiscard]] std::string formatMessage(std::string_view fmt, std::format_args args) noexcept;
void                      publishFormatted(Level level, std::string_view fmt, std::format_args args, std::source_location loc) noexcept;

} // namespace detail

template<typename... Args>
class FormatString {
    std::format_string<std::type_identity_t<Args>...> _format;
    std::source_location                              _location;

public:
    template<std::size_t N>
    consteval FormatString(const char (&format)[N], std::source_location location = std::source_location::current()) : _format(format), _location(location) {}

    [[nodiscard]] constexpr std::string_view     text() const noexcept { return _format.get(); }
    [[nodiscard]] constexpr std::source_location location() const noexcept { return _location; }
};

struct RuntimeFormatString {
    std::string_view     format;
    std::source_location location;
};

[[nodiscard]] constexpr RuntimeFormatString runtime(std::string_view format, std::source_location location = std::source_location::current()) noexcept { return {format, location}; }

[[nodiscard]] inline Backend& defaultBackend() noexcept { return detail::defaultBackend(); }

[[nodiscard]] inline Backend& activeBackend() noexcept { return detail::activeBackend(); }

inline Backend* setBackend(Backend* backend) noexcept { return detail::setBackend(backend); }

[[nodiscard]] inline std::size_t drain(RecordConsumer consumer, void* user = nullptr) noexcept { return detail::drain(consumer, user); }

[[nodiscard]] inline std::size_t flush() noexcept { return detail::flush(); }

} // namespace log

/**
 * @brief thrown by `gr::log::fatal` on hosted targets
 *
 * Carries the panic message, the call-site `std::source_location`, and the timestamp
 * at which the exception was constructed. `what()` renders "{message} at {file}:{line}".
 * User code may continue to throw this type directly; the conversion machinery in
 * `Message.hpp` (via `gr::Error(const gr::exception&)`) folds it into the record path.
 */
struct exception : std::exception {
    std::string                           message;
    std::source_location                  sourceLocation;
    std::chrono::system_clock::time_point errorTime = std::chrono::system_clock::now();

    mutable std::string _formattedMessage;

    exception(std::string_view msg = "unknown exception", std::source_location location = std::source_location::current()) noexcept;

    [[nodiscard]] const char* what() const noexcept override;
};

/**
 * @brief structured diagnostic record carrying message + source_location + timestamp
 *
 * Companion to `gr::exception` on the no-throw path: the canonical error type for
 * `std::expected<T, gr::Error>` returns and for `gr::log::{warning,error}` records.
 * Constructible from `gr::exception` or any `std::exception`, so try/catch boundaries
 * can fold into the record path without losing the source location.
 *
 * Helpers `srcLoc()`, `methodName()`, `isoTime()` produce render-ready strings; a
 * `std::formatter<gr::Error>` specialisation lives in `Message.hpp` once the
 * meta-formatter is on the include path of the consumer.
 */
struct Error {
    std::string                           message;
    std::source_location                  sourceLocation;
    std::chrono::system_clock::time_point errorTime = std::chrono::system_clock::now();

    Error(std::string_view msg = "unknown error", std::source_location location = std::source_location::current(), //
        std::chrono::system_clock::time_point time = std::chrono::system_clock::now()) noexcept;

    explicit Error(const std::exception& ex, std::source_location location = std::source_location::current()) noexcept;

    explicit Error(const gr::exception& ex) noexcept;

    [[nodiscard]] std::string srcLoc() const noexcept;
    [[nodiscard]] std::string methodName() const noexcept;
    [[nodiscard]] std::string isoTime() const noexcept; // ms-precision ISO time-format
};

static_assert(std::is_default_constructible_v<Error>);
static_assert(!std::is_trivially_copyable_v<Error>); // because of std::string

namespace log {

#if __cpp_exceptions

/**
 * @brief single quarantine point for unrecoverable invariant violations
 *
 * Hosted (`__cpp_exceptions` defined): throws `gr::exception(msg, loc)` — caught by the existing message-conversion machinery on the runtime path.
 * AOT (`-fno-exceptions`): emits a best-effort diagnostic record and aborts.
 * Both branches are `[[noreturn]]`. Lives in `gr::log` so unqualified `fatal` in test
 * TUs resolves to `boost::ut::fatal` without ambiguity.
 *
 * Siblings: `gr::log::warning`, `gr::log::error` (non-fatal record emitters).
 */
[[noreturn]] inline void fatal(std::string_view msg, std::source_location loc = std::source_location::current()) {
    detail::publish(Level::fatal, msg, loc);
    throw gr::exception(msg, loc);
}

#else

[[noreturn]] inline void fatal(std::string_view msg, std::source_location loc = std::source_location::current()) noexcept {
    detail::publish(Level::fatal, msg, loc);
    std::abort();
}

#endif

template<typename... Args>
[[noreturn]] inline void fatal(FormatString<std::type_identity_t<Args>...> fmt, Args&&... args) {
    std::string message = detail::formatMessage(fmt.text(), std::make_format_args(args...));
    fatal(message, fmt.location());
}

template<typename... Args>
[[noreturn]] inline void fatal(RuntimeFormatString fmt, Args&&... args) {
    std::string message = detail::formatMessage(fmt.format, std::make_format_args(args...));
    fatal(message, fmt.location);
}

/**
 * @brief non-fatal warning record sink
 */
inline void warning(std::string_view msg, std::source_location loc = std::source_location::current()) noexcept { detail::publish(Level::warning, msg, loc); }

template<typename... Args>
inline void warning(FormatString<std::type_identity_t<Args>...> fmt, Args&&... args) {
    detail::publishFormatted(Level::warning, fmt.text(), std::make_format_args(args...), fmt.location());
}

template<typename... Args>
inline void warning(RuntimeFormatString fmt, Args&&... args) {
    detail::publishFormatted(Level::warning, fmt.format, std::make_format_args(args...), fmt.location);
}

/**
 * @brief recoverable-error record sink
 */
inline void error(std::string_view msg, std::source_location loc = std::source_location::current()) noexcept { detail::publish(Level::error, msg, loc); }

template<typename... Args>
inline void error(FormatString<std::type_identity_t<Args>...> fmt, Args&&... args) {
    detail::publishFormatted(Level::error, fmt.text(), std::make_format_args(args...), fmt.location());
}

template<typename... Args>
inline void error(RuntimeFormatString fmt, Args&&... args) {
    detail::publishFormatted(Level::error, fmt.format, std::make_format_args(args...), fmt.location);
}

inline void failure(std::string_view msg, std::source_location loc = std::source_location::current()) noexcept { detail::publish(Level::failure, msg, loc); }

template<typename... Args>
inline void failure(FormatString<std::type_identity_t<Args>...> fmt, Args&&... args) {
    detail::publishFormatted(Level::failure, fmt.text(), std::make_format_args(args...), fmt.location());
}

template<typename... Args>
inline void failure(RuntimeFormatString fmt, Args&&... args) {
    detail::publishFormatted(Level::failure, fmt.format, std::make_format_args(args...), fmt.location);
}

inline void info(std::string_view msg, std::source_location loc = std::source_location::current()) noexcept {
    if constexpr (gr::meta::kDebugBuild) {
        detail::publish(Level::info, msg, loc);
    }
}

template<typename... Args>
inline void info(FormatString<std::type_identity_t<Args>...> fmt, Args&&... args) {
    if constexpr (gr::meta::kDebugBuild) {
        detail::publishFormatted(Level::info, fmt.text(), std::make_format_args(args...), fmt.location());
    }
}

template<typename... Args>
inline void info(RuntimeFormatString fmt, Args&&... args) {
    if constexpr (gr::meta::kDebugBuild) {
        detail::publishFormatted(Level::info, fmt.format, std::make_format_args(args...), fmt.location);
    }
}

inline void debug(std::string_view msg, std::source_location loc = std::source_location::current()) noexcept {
    if constexpr (gr::meta::kDebugBuild) {
        detail::publish(Level::debug, msg, loc);
    }
}

template<typename... Args>
inline void debug(FormatString<std::type_identity_t<Args>...> fmt, Args&&... args) {
    if constexpr (gr::meta::kDebugBuild) {
        detail::publishFormatted(Level::debug, fmt.text(), std::make_format_args(args...), fmt.location());
    }
}

template<typename... Args>
inline void debug(RuntimeFormatString fmt, Args&&... args) {
    if constexpr (gr::meta::kDebugBuild) {
        detail::publishFormatted(Level::debug, fmt.format, std::make_format_args(args...), fmt.location);
    }
}

inline void trace(std::string_view msg, std::source_location loc = std::source_location::current()) noexcept {
    if constexpr (gr::meta::kDebugBuild) {
        detail::publish(Level::trace, msg, loc);
    }
}

template<typename... Args>
inline void trace(FormatString<std::type_identity_t<Args>...> fmt, Args&&... args) {
    if constexpr (gr::meta::kDebugBuild) {
        detail::publishFormatted(Level::trace, fmt.text(), std::make_format_args(args...), fmt.location());
    }
}

template<typename... Args>
inline void trace(RuntimeFormatString fmt, Args&&... args) {
    if constexpr (gr::meta::kDebugBuild) {
        detail::publishFormatted(Level::trace, fmt.format, std::make_format_args(args...), fmt.location);
    }
}

} // namespace log

} // namespace gr

#endif // GNURADIO_LOGGER_HPP
