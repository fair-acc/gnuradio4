#ifndef GNURADIO_MESSAGE_HPP
#define GNURADIO_MESSAGE_HPP

#include <gnuradio-4.0/Buffer.hpp>
#include <gnuradio-4.0/CircularBuffer.hpp>
#include <gnuradio-4.0/Logger.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

#include <expected>
#include <source_location>
#include <string_view>

namespace gr {

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
    return std::string(gr::meta::enumName(command).value_or(""));
}

inline static const std::string defaultBlockProtocol  = "MDPW03";
inline static const std::string defaultClientProtocol = "MDPC03";

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

    constexpr void clear() noexcept {
        protocol.clear();
        cmd = Notify;
        serviceName.clear();
        clientRequestID.clear();
        endpoint.clear();
        data = std::expected<property_map, Error>{property_map{}};
        rbac.clear();
    }

    void shrink_to_fit() {
        protocol.shrink_to_fit();
        serviceName.shrink_to_fit();
        clientRequestID.shrink_to_fit();
        endpoint.shrink_to_fit();
        if (data.has_value()) {
            data->shrink_to_fit();
        }
        rbac.shrink_to_fit();
    }
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
    if (!port.isConnected()) {
        return; // unconnected msg port: silently drop (blocking reserve would spin forever on a zero-capacity buffer)
    }
    WriterSpanLike auto msgSpan = port.streamWriter().template reserve<SpanReleasePolicy::ProcessAll>(1UZ);
    msgSpan[0]                  = std::move(message);
    msgSpan.publish(1UZ);
}
} // namespace detail

template<auto cmd>
void sendMessage(auto& port, std::string_view serviceName, std::string_view endpoint, property_map userMessage, std::string_view clientRequestID = "") {
    detail::sendMessage<cmd>(port, serviceName, endpoint, std::move(userMessage), clientRequestID);
}

template<auto cmd>
void sendMessage(auto& port, std::string_view serviceName, std::string_view endpoint, std::initializer_list<std::pair<std::string_view, Value>> userMessage, std::string_view clientRequestID = "") {
    detail::sendMessage<cmd, property_map>(port, serviceName, endpoint, property_map(userMessage), clientRequestID);
}

template<auto cmd>
void sendMessage(auto& port, std::string_view serviceName, std::string_view endpoint, Error userMessage, std::string_view clientRequestID = "") {
    detail::sendMessage<cmd, Error>(port, serviceName, endpoint, std::move(userMessage), clientRequestID);
}

} // namespace gr

template<>
struct std::formatter<gr::Error> {
    char presentation = 's';

    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        auto it = ctx.begin(), end = ctx.end();
        if (it != end && (*it == 'f' || *it == 't' || *it == 's')) {
            presentation = *it++;
        }
        if (it != end && *it != '}') {
            gr::log::fatal("invalid format specifier for gr::Error");
        }
        return it;
    }

    // Formats the source_location, using 'f' for file and 'l' for line
    template<typename FormatContext>
    auto format(const gr::Error& err, FormatContext& ctx) const {
        const auto& loc = err.sourceLocation;
        switch (presentation) {
        case 'f': return std::format_to(ctx.out(), "{}:{} in {}: {}", loc.file_name(), loc.line(), loc.function_name(), err.message);
        case 't': return std::format_to(ctx.out(), "{}: {}:{}: {} in {}", err.isoTime(), loc.file_name(), loc.line(), err.message, loc.function_name());
        case 's':
        default: return std::format_to(ctx.out(), "{}:{}: {}", loc.file_name(), loc.line(), err.message);
        }
    }
};

template<>
struct std::formatter<gr::message::Command> {
    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) { return ctx.begin(); }

    // Formats the source_location, using 'f' for file and 'l' for line
    template<typename FormatContext>
    auto format(const gr::message::Command& command, FormatContext& ctx) const -> decltype(ctx.out()) {
        return std::format_to(ctx.out(), "{}", gr::meta::enumName(command).value_or(""));
    }
};

inline std::ostream& operator<<(std::ostream& os, const gr::message::Command& command) { return os << gr::meta::enumName(command).value_or(""); }

template<>
struct std::formatter<gr::Message> {
    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) { return ctx.begin(); }

    // Formats the source_location, using 'f' for file and 'l' for line
    template<typename FormatContext>
    auto format(const gr::Message& msg, FormatContext& ctx) const -> decltype(ctx.out()) {
        return std::format_to(ctx.out(), "{{ protocol: '{}', cmd: {}, serviceName: '{}', clientRequestID: '{}', endpoint: '{}', {}, RBAC: '{}' }}", //
            msg.protocol, msg.cmd, msg.serviceName, msg.clientRequestID, msg.endpoint,                                                              //
            msg.data.has_value() ? std::format("data: {}", msg.data.value()) : std::format("error: {}", msg.data.error()), msg.rbac);
    }
};

inline std::ostream& operator<<(std::ostream& os, const gr::Message& msg) { return os << std::format("{}", msg); }

#endif // include guard
