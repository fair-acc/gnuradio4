#ifndef CORE_TEST_MESSAGE_UTILS_HPP
#define CORE_TEST_MESSAGE_UTILS_HPP

#include <chrono>
#include <optional>
#include <string>

#include <boost/ut.hpp>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Message.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Scheduler.hpp>

namespace gr::testing {

using namespace boost::ut;
using namespace gr;

using namespace std::chrono_literals;

/// Sleeps the current thread until condition() is true or timeout
template<typename Condition>
bool awaitCondition(std::chrono::milliseconds timeout, Condition condition) {
    auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < timeout) {
        if (condition()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

inline Message consumeFirstReply(gr::MsgPortIn& port) {
    ReaderSpanLike auto span = port.streamReader().get<SpanReleasePolicy::ProcessAll>(1UZ);
    Message             msg  = span[0];
    expect(span.consume(span.size()));
    return msg;
};

inline std::size_t getNReplyMessages(gr::MsgPortIn& port) { return port.streamReader().available(); };

inline std::vector<Message> consumeAllReplyMessages(gr::MsgPortIn& port, std::source_location sourceLocation = std::source_location::current()) {
    std::vector<Message> msgs;

    {
        ReaderSpanLike auto messages = port.streamReader().get();
        msgs                         = std::vector<Message>(messages.begin(), messages.end());
        const bool isConsumed        = messages.consume(messages.size());
        expect(isConsumed) << std::format("Can not consume reply messages. Requested at:{}\n", sourceLocation);
    }

    expect(eq(getNReplyMessages(port), 0UZ)) << std::format("Unexpected available messages: {}. Requested at:{}\n", getNReplyMessages(port), sourceLocation);
    return msgs;
};

template<typename Condition>
inline std::optional<Message> waitForReply(gr::MsgPortIn& fromGraph, Condition condition, std::chrono::milliseconds maxWaitTime = 1s) {
    auto startedAt = std::chrono::system_clock::now();
    while (true) {
        auto messages = fromGraph.streamReader().get();
        auto it       = std::find_if(messages.begin(), messages.end(), condition);
        if (it != messages.end()) {
            return *it;
        }

        std::this_thread::sleep_for(100ms);
        if (std::chrono::system_clock::now() - startedAt > maxWaitTime) {
            std::println("msg dump for failed test:");
            for (auto& msg : messages) {
                std::println("msg: .endpoint={} .hasData={}", msg.endpoint, msg.data.has_value());
            }
            return std::nullopt;
        }
    }
};

/// Sends a message, waits for its reply and validates it
///
/// This is the preferred way to send a message in a test, as it handles steps
/// that can be either forgotten (like waiting) or that are just verbose boiler-plate
/// Condition can be a lamdba, such as [] (const Message &reply) { return reply.endpoint == myendpoint; }
/// or pass a functor for convenience, like:  ReplyChecker{.expectedEndpoint = block::property::kSetting}
template<message::Command Cmd, typename Condition>
inline std::optional<Message> sendAndWaitForReply(auto& toPort, auto& fromPort, std::string_view serviceName, std::string_view endpoint, //
    property_map data, Condition condition, std::string_view clientRequestID = "", std::source_location sourceLocation = std::source_location::current()) {
    gr::sendMessage<Cmd>(toPort, serviceName, endpoint, std::move(data), clientRequestID);
    auto reply = waitForReply(fromPort, condition);
    expect(reply.has_value()) << std::format("Reply message not received. at {}\n", sourceLocation);

    // consume every message so they dont' leak into the next test. The next test is resilient
    // but it makes debugging harder if we have old messages
    consumeAllReplyMessages(fromPort);
    expect(eq(getNReplyMessages(fromPort), 0UZ));

    return reply;
}

struct ReplyChecker {
    bool operator()(const Message& reply) const { return reply.endpoint == expectedEndpoint && reply.data.has_value() == expectedHasData; }

    std::string_view expectedEndpoint;
    bool             expectedHasData = true;
};

inline std::string sendAndWaitMessageEmplaceBlock(gr::MsgPortOut& toGraph, gr::MsgPortIn& fromGraph, std::string type, property_map properties, std::string serviceName = "", std::source_location sourceLocation = std::source_location::current()) {
    expect(eq(getNReplyMessages(fromGraph), 0UZ)) << std::format("Input port has unconsumed messages. Requested at: {}\n", sourceLocation);
    auto reply = testing::sendAndWaitForReply<gr::message::Command::Set>(toGraph, fromGraph, serviceName, gr::scheduler::property::kEmplaceBlock, //
        {{"type", std::move(type)}, {"properties", std::move(properties)}},                                                                       //
        ReplyChecker{.expectedEndpoint = gr::scheduler::property::kBlockEmplaced});

    return std::get<std::string>(reply.value().data.value().at("uniqueName"s));
};

inline void sendAndWaitMessageEmplaceEdge(gr::MsgPortOut& toGraph, gr::MsgPortIn& fromGraph, std::string sourceBlock, std::string sourcePort, std::string destinationBlock, std::string destinationPort, std::string serviceName = "", std::source_location sourceLocation = std::source_location::current()) {
    expect(eq(getNReplyMessages(fromGraph), 0UZ)) << std::format("Input port has unconsumed messages. Requested at: {}\n", sourceLocation);
    gr::property_map data = {{"sourceBlock", sourceBlock}, {"sourcePort", sourcePort}, {"destinationBlock", destinationBlock}, {"destinationPort", destinationPort}, //
        {"minBufferSize", gr::Size_t()}, {"weight", 0}, {"edgeName", "unnamed edge"}};
    testing::sendAndWaitForReply<gr::message::Command::Set>(toGraph, fromGraph, serviceName, gr::scheduler::property::kEmplaceEdge, data, //
        ReplyChecker{.expectedEndpoint = gr::scheduler::property::kEdgeEmplaced});
};

} // namespace gr::testing

#endif // include guard
