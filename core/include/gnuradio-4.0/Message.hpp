#ifndef GNURADIO_MESSAGE_HPP
#define GNURADIO_MESSAGE_HPP

#include <gnuradio-4.0/Buffer.hpp>
#include <gnuradio-4.0/CircularBuffer.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

#include <pmtv/pmt.hpp>

#include <expected>
#include <source_location>
#include <string_view>

#include <fmt/chrono.h>
#include <fmt/format.h>

namespace gr {

struct exception : public std::exception {
    std::string                           message;
    std::source_location                  sourceLocation;
    std::chrono::system_clock::time_point errorTime = std::chrono::system_clock::now();

    exception(std::string_view msg = "unknown exception", std::source_location location = std::source_location::current()) noexcept : message(msg), sourceLocation(location) {}

    [[nodiscard]] const char* what() const noexcept override {
        if (formattedMessage.empty()) {
            formattedMessage = fmt::format("{} at {}:{}", message, sourceLocation.file_name(), sourceLocation.line());
        }
        return formattedMessage.c_str();
    }

private:
    mutable std::string formattedMessage; // Now storing the formatted message
};

struct Error {
    std::string                           message;
    std::source_location                  sourceLocation;
    std::chrono::system_clock::time_point errorTime = std::chrono::system_clock::now();

    Error(std::string_view msg = "unknown error", std::source_location location = std::source_location::current(), //
        std::chrono::system_clock::time_point time = std::chrono::system_clock::now()) noexcept                    //
        : message(msg), sourceLocation(location), errorTime(time) {}

    explicit Error(const std::exception& ex, std::source_location location = std::source_location::current()) noexcept : Error(ex.what(), location) {}

    explicit Error(const gr::exception& ex) noexcept : Error(ex.message, ex.sourceLocation, ex.errorTime) {}

    [[nodiscard]] std::string srcLoc() const noexcept { return fmt::format("{}", sourceLocation); }
    [[nodiscard]] std::string methodName() const noexcept { return {sourceLocation.function_name()}; }
    [[nodiscard]] std::string isoTime() const noexcept {
        return fmt::format("{:%Y-%m-%dT%H:%M:%S}.{:03}",                     // ms-precision ISO time-format
            fmt::localtime(std::chrono::system_clock::to_time_t(errorTime)), //
            std::chrono::duration_cast<std::chrono::milliseconds>(errorTime.time_since_epoch()).count() % 1000);
    }
};

static_assert(std::is_default_constructible_v<Error>);
static_assert(!std::is_trivially_copyable_v<Error>); // because of the usage of std::string

namespace message {
/**
 * @brief Follows the OpenCMW command structure.
 * https://github.com/fair-acc/opencmw-cpp/blob/main/docs/Majordomo_protocol_comparison.pdf
 * derived from: https://rfc.zeromq.org/spec/7/ and https://rfc.zeromq.org/spec/18/
 */
enum class Command : unsigned char {
    Invalid     = 0x00,
    Get         = 0x01,
    Set         = 0x02,
    Partial     = 0x03,
    Final       = 0x04,
    Ready       = 0x05, ///< optional for client
    Disconnect  = 0x06, ///< optional for client
    Subscribe   = 0x07, ///< client-only
    Unsubscribe = 0x08, ///< client-only
    Notify      = 0x09, ///< worker-only
    Heartbeat   = 0x0a  ///< optional for client
};

template<Command command>
std::string commandName() noexcept {
    return std::string(magic_enum::enum_name<command>());
}

inline static std::string defaultBlockProtocol  = "MDPW03";
inline static std::string defaultClientProtocol = "MDPC03";

} // namespace message

/**
 * @brief Follows OpenCMW's Majordomo protocol frame structure.
 * https://github.com/fair-acc/opencmw-cpp/blob/main/docs/Majordomo_protocol_comparison.pdf
 * derived from: https://rfc.zeromq.org/spec/7/ and https://rfc.zeromq.org/spec/18/
 */
struct Message {
    using enum gr::message::Command;
    using Error = gr::Error;

    std::string                        protocol = message::defaultBlockProtocol; ///< unique protocol name including version (e.g. 'MDPC03' or 'MDPW03')
    message::Command                   cmd      = Notify;                        ///< command type (GET, SET, SUBSCRIBE, UNSUBSCRIBE, PARTIAL, FINAL, NOTIFY, READY, DISCONNECT, HEARTBEAT)
    std::string                        serviceName;                              ///< service/block name (normally the URI path only), or client source ID (for broker/scheduler <-> worker messages) N.B empty string is wildcard
    std::string                        clientRequestID = "";                     ///< stateful: worker mirrors clientRequestID; stateless: worker generates unique increasing IDs (to detect packet loss)
    std::string                        endpoint;                                 ///< URI containing at least <path> and optionally <query> parameters (e.g. property name)
    std::expected<property_map, Error> data;                                     ///< request/reply body and/or Error containing stack-trace
    std::string                        rbac = "";                                ///< optional RBAC meta-info -- may contain token, role, signed message hash (implementation dependent)
};

static_assert(std::is_default_constructible_v<Message>);
static_assert(!std::is_trivially_copyable_v<Message>); // because of the usage of std::string
static_assert(std::is_move_assignable_v<Message>);

namespace detail {
template<message::Command cmd, typename T>
requires(std::is_same_v<T, property_map> || std::is_same_v<T, Error>)
void sendMessage(auto& port, std::string_view serviceName, std::string_view endpoint, T userMessage, std::string_view clientRequestID = "") {
    using namespace gr::message;
    using enum gr::message::Command;

    Message message;
    message.cmd             = cmd;
    message.serviceName     = serviceName;
    message.clientRequestID = clientRequestID;
    message.endpoint        = endpoint;
    message.rbac            = "";

    if constexpr (std::is_same_v<T, property_map>) {
        message.data = std::move(userMessage);
    } else {
        message.data = std::unexpected(userMessage);
    }
    WriterSpanLike auto msgSpan = port.streamWriter().template reserve<SpanReleasePolicy::ProcessAll>(1UZ);
    msgSpan[0]                  = std::move(message);
}
} // namespace detail

template<auto cmd>
void sendMessage(auto& port, std::string_view serviceName, std::string_view endpoint, property_map userMessage, std::string_view clientRequestID = "") {
    detail::sendMessage<cmd>(port, serviceName, endpoint, std::move(userMessage), clientRequestID);
}

template<auto cmd>
void sendMessage(auto& port, std::string_view serviceName, std::string_view endpoint, std::initializer_list<std::pair<const std::string, pmtv::pmt>> userMessage, std::string_view clientRequestID = "") {
    detail::sendMessage<cmd, property_map>(port, serviceName, endpoint, property_map(userMessage), clientRequestID);
}

template<auto cmd>
void sendMessage(auto& port, std::string_view serviceName, std::string_view endpoint, Error userMessage, std::string_view clientRequestID = "") {
    detail::sendMessage<cmd, Error>(port, serviceName, endpoint, std::move(userMessage), clientRequestID);
}

} // namespace gr

template<>
struct fmt::formatter<gr::Error> {
    char presentation = 's';

    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        auto it = ctx.begin(), end = ctx.end();
        if (it != end && (*it == 'f' || *it == 't' || *it == 's')) {
            presentation = *it++;
        }
        if (it != end && *it != '}') {
            throw fmt::format_error("invalid format");
        }
        return it;
    }

    // Formats the source_location, using 'f' for file and 'l' for line
    template<typename FormatContext>
    auto format(const gr::Error& err, FormatContext& ctx) const -> decltype(ctx.out()) {
        switch (presentation) {
        case 't': return fmt::format_to(ctx.out(), "{}: {}: {} in method: {}", err.isoTime(), err.sourceLocation, err.message, err.sourceLocation.function_name());
        case 'f': return fmt::format_to(ctx.out(), "{}: {} in method: {}", err.sourceLocation, err.message, err.sourceLocation.function_name());
        case 's':
        default: return fmt::format_to(ctx.out(), "{}: {}", err.sourceLocation, err.message);
        }
    }
};

template<>
struct fmt::formatter<gr::message::Command> {
    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) { return ctx.begin(); }

    // Formats the source_location, using 'f' for file and 'l' for line
    template<typename FormatContext>
    auto format(const gr::message::Command& command, FormatContext& ctx) const -> decltype(ctx.out()) {
        return fmt::format_to(ctx.out(), "{}", magic_enum::enum_name(command));
    }
};

inline std::ostream& operator<<(std::ostream& os, const gr::message::Command& command) { return os << magic_enum::enum_name(command); }

template<>
struct fmt::formatter<gr::Message> {
    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) { return ctx.begin(); }

    // Formats the source_location, using 'f' for file and 'l' for line
    template<typename FormatContext>
    auto format(const gr::Message& msg, FormatContext& ctx) const -> decltype(ctx.out()) {
        return fmt::format_to(ctx.out(), "{{ protocol: '{}', cmd: {}, serviceName: '{}', clientRequestID: '{}', endpoint: '{}', {}, RBAC: '{}' }}", //
            msg.protocol, msg.cmd, msg.serviceName, msg.clientRequestID, msg.endpoint,                                                              //
            msg.data.has_value() ? fmt::format("data: {}", msg.data.value()) : fmt::format("error: {}", msg.data.error()), msg.rbac);
    }
};

inline std::ostream& operator<<(std::ostream& os, const gr::Message& msg) { return os << fmt::format("{}", msg); }

#endif // include guard
