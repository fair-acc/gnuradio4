#ifndef CORE_TEST_MESSAGE_UTILS_HPP
#define CORE_TEST_MESSAGE_UTILS_HPP

#include <chrono>
#include <optional>
#include <string>

#include <boost/ut.hpp>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Message.hpp>
#include <gnuradio-4.0/Port.hpp>

namespace gr::testing {

using namespace boost::ut;
using namespace gr;

using namespace std::chrono_literals;
using enum gr::message::Command;

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

inline Message getAndConsumeFirstReplyMessage(gr::MsgPortIn& port, std::optional<std::string> expectedEndpoint = {}, std::source_location sourceLocation = std::source_location::current()) {
    auto available = port.streamReader().available();
    expect(gt(available, 0UZ)) << fmt::format("No available messages. Requested at:{}\n", sourceLocation);
    ReaderSpanLike auto messages = port.streamReader().get();
    Message             result;

    if (expectedEndpoint) {
        auto it = std::ranges::find_if(messages, [endpoint = *expectedEndpoint](const auto& message) { return message.endpoint == endpoint; });
        if (it == messages.end()) {
            expect(false) << fmt::format("Message with endpoint ({}) not found. Requested at:{}\n", *expectedEndpoint, sourceLocation);
        } else {
            result = *it;
        }
    } else {
        result = messages[0];
    }

    fmt::print("First Reply Message: {}\n", result);
    if (expectedEndpoint) {
        expect(eq(*expectedEndpoint, result.endpoint));
    }

    const bool isConsumed = messages.consume(1UZ);
    if (!isConsumed) {
        expect(false) << fmt::format("Can not consume reply message. Requested at:{}\n", sourceLocation);
    }

    return result;
};

inline std::vector<Message> getAllReplyMessages(gr::MsgPortIn& port) {
    ReaderSpanLike auto  messages = port.streamReader().get();
    std::vector<Message> result(messages.begin(), messages.end());
    return result;
};

inline std::size_t getNReplyMessages(gr::MsgPortIn& port) { return port.streamReader().available(); };

inline void consumeAllReplyMessages(gr::MsgPortIn& port, std::source_location sourceLocation = std::source_location::current()) {
    ReaderSpanLike auto messages   = port.streamReader().get();
    const bool          isConsumed = messages.consume(messages.size());
    if (!isConsumed) {
        expect(false) << fmt::format("Can not consume reply messages. Requested at:{}\n", sourceLocation);
    }
};

template<bool processScheduledMessages = true>
Message awaitReplyMessage(auto& graph, std::chrono::milliseconds timeout, gr::MsgPortIn& port, std::optional<std::string> expectedEndpoint = {}, std::source_location sourceLocation = std::source_location::current()) {
    awaitCondition(timeout, [&port, &graph] {
        if constexpr (processScheduledMessages) {
            graph.processScheduledMessages();
        }
        return port.streamReader().available() > 0;
    });

    return getAndConsumeFirstReplyMessage(port, expectedEndpoint, sourceLocation);
};

inline bool waitForReply(gr::MsgPortIn& fromGraph, std::size_t nReplies = 1UZ, std::chrono::milliseconds maxWaitTime = 1s) {
    auto startedAt = std::chrono::system_clock::now();
    while (fromGraph.streamReader().available() < nReplies) {
        std::this_thread::sleep_for(100ms);
        if (std::chrono::system_clock::now() - startedAt > maxWaitTime) {
            break;
        }
    }
    return fromGraph.streamReader().available() >= nReplies;
};

inline std::string sendAndWaitMessageEmplaceBlock(gr::MsgPortOut& toGraph, gr::MsgPortIn& fromGraph, std::string type, std::string params, property_map properties, std::string serviceName = "", std::source_location sourceLocation = std::source_location::current()) {
    expect(eq(getNReplyMessages(fromGraph), 0UZ)) << fmt::format("Input port has unconsumed messages. Requested at: {}\n", sourceLocation);
    sendMessage<Set>(toGraph, serviceName, graph::property::kEmplaceBlock /* endpoint */, //
        {{"type", std::move(type)}, {"parameters", std::move(params)}, {"properties", std::move(properties)}} /* data */);

    expect(waitForReply(fromGraph)) << fmt::format("Reply message not received. Requested at: {}\n", sourceLocation);

    const Message reply = getAndConsumeFirstReplyMessage(fromGraph);
    if (!reply.data.has_value()) {
        expect(false) << fmt::format("Emplace block failed and returned an error. Requested at: {}. Error: {}\n", sourceLocation, reply.data.error());
    }
    return std::get<std::string>(reply.data.value().at("uniqueName"s));
};

inline void sendAndWaitMessageEmplaceEdge(gr::MsgPortOut& toGraph, gr::MsgPortIn& fromGraph, std::string sourceBlock, std::string sourcePort, std::string destinationBlock, std::string destinationPort, std::string serviceName = "", std::source_location sourceLocation = std::source_location::current()) {
    expect(eq(getNReplyMessages(fromGraph), 0UZ)) << fmt::format("Input port has unconsumed messages. Requested at: {}\n", sourceLocation);
    gr::property_map data = {{"sourceBlock", sourceBlock}, {"sourcePort", sourcePort}, {"destinationBlock", destinationBlock}, {"destinationPort", destinationPort}, //
        {"minBufferSize", gr::Size_t()}, {"weight", 0}, {"edgeName", "unnamed edge"}};
    sendMessage<Set>(toGraph, serviceName, graph::property::kEmplaceEdge /* endpoint */, data /* data */);

    expect(waitForReply(fromGraph)) << fmt::format("Reply message not received. Requested at: {}\n", sourceLocation);
    expect(eq(getNReplyMessages(fromGraph), 1UZ)) << fmt::format("No messages available. Requested at: {}\n", sourceLocation);
    const Message reply = getAndConsumeFirstReplyMessage(fromGraph);

    if (!reply.data.has_value()) {
        expect(false) << fmt::format("Emplace edge failed and returned an error. Requested at: {}. Error: {}\n", sourceLocation, reply.data.error());
    }
};

} // namespace gr::testing

#endif // include guard
