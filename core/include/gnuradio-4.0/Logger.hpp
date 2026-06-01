#ifndef GNURADIO_LOGGER_HPP
#define GNURADIO_LOGGER_HPP

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <format> //TODO: to be replaced through a circular-buffer-based logging system (e.g. Profiler.hpp-based)
#include <print>  //TODO: to be replaced through a circular-buffer-based logging system (e.g. Profiler.hpp-based)
#include <source_location>
#include <string>
#include <string_view>

#include <gnuradio-4.0/meta/formatter.hpp> // std::formatter<std::source_location>

namespace gr {

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

    exception(std::string_view msg = "unknown exception", std::source_location location = std::source_location::current()) noexcept : message(msg), sourceLocation(location) {
#ifndef NDEBUG
        std::println("Exception thrown: {} at {}:{}", msg, location.file_name(), location.line());
#endif
    }

    [[nodiscard]] const char* what() const noexcept override {
        if (_formattedMessage.empty()) {
            _formattedMessage = std::format("{} at {}:{}", message, sourceLocation.file_name(), sourceLocation.line());
        }
        return _formattedMessage.c_str();
    }
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
        std::chrono::system_clock::time_point time = std::chrono::system_clock::now()) noexcept                    //
        : message(msg), sourceLocation(location), errorTime(time) {}

    explicit Error(const std::exception& ex, std::source_location location = std::source_location::current()) noexcept : Error(ex.what(), location) {}

    explicit Error(const gr::exception& ex) noexcept : Error(ex.message, ex.sourceLocation, ex.errorTime) {}

    [[nodiscard]] std::string srcLoc() const noexcept { return std::format("{}", sourceLocation); }
    [[nodiscard]] std::string methodName() const noexcept { return {sourceLocation.function_name()}; }
    [[nodiscard]] std::string isoTime() const noexcept { return std::format("{}", errorTime); } // ms-precision ISO time-format
};

static_assert(std::is_default_constructible_v<Error>);
static_assert(!std::is_trivially_copyable_v<Error>); // because of std::string

namespace log {

#if __cpp_exceptions

/**
 * @brief single quarantine point for unrecoverable invariant violations
 *
 * Hosted (`__cpp_exceptions` defined): throws `gr::exception(msg, loc)` — caught by the existing message-conversion machinery on the runtime path.
 * AOT (`-fno-exceptions`): writes a one-line diagnostic to stderr and aborts.
 * Both branches are `[[noreturn]]`. Lives in `gr::log` so unqualified `fatal` in test
 * TUs resolves to `boost::ut::fatal` without ambiguity.
 *
 * Siblings: `gr::log::warning`, `gr::log::error` (non-fatal record emitters).
 */
[[noreturn]] inline void fatal(std::string_view msg, std::source_location loc = std::source_location::current()) { throw gr::exception(msg, loc); }

#else

[[noreturn]] inline void fatal(std::string_view msg, std::source_location loc = std::source_location::current()) noexcept {
    std::print(stderr, "[gr::log::fatal] {}:{} {}\n", loc.file_name(), loc.line(), msg);
    std::abort();
}

#endif

/**
 * @brief non-fatal warning record sink — stub, writes to stderr today
 *
 * Returns a `gr::Error` so call sites can compose:
 * `return std::unexpected(gr::log::warning("..."));`
 * Future: routes through a configurable logger backend (severity gates, sinks).
 */
inline gr::Error warning(std::string_view msg, std::source_location loc = std::source_location::current()) noexcept {
    std::print(stderr, "[gr::log::warning] {}:{} {}\n", loc.file_name(), loc.line(), msg);
    return gr::Error{msg, loc};
}

/**
 * @brief recoverable-error record sink — stub, writes to stderr today
 *
 * Pair with `std::expected<T, gr::Error>` at API boundaries:
 * `return std::unexpected(gr::log::error("pool '{}' not found"));`
 */
inline gr::Error error(std::string_view msg, std::source_location loc = std::source_location::current()) noexcept {
    std::print(stderr, "[gr::log::error] {}:{} {}\n", loc.file_name(), loc.line(), msg);
    return gr::Error{msg, loc};
}

} // namespace log

} // namespace gr

#endif // GNURADIO_LOGGER_HPP
