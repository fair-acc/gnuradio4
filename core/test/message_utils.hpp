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

inline Message consumeReplyMsg(gr::MsgPortIn& port) {
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

inline std::string sendAndWaitMessageEmplaceBlock(gr::MsgPortOut& toGraph, gr::MsgPortIn& fromGraph, std::string type, property_map properties, std::string serviceName = "", std::source_location sourceLocation = std::source_location::current()) {
    expect(eq(getNReplyMessages(fromGraph), 0UZ)) << std::format("Input port has unconsumed messages. Requested at: {}\n", sourceLocation);
    sendMessage<Set>(toGraph, serviceName, gr::scheduler::property::kEmplaceBlock /* endpoint */, //
        {{"type", std::move(type)}, {"properties", std::move(properties)}} /* data */);

    expect(waitForReply(fromGraph)) << std::format("Reply message not received. Requested at: {}\n", sourceLocation);

    const Message reply = consumeReplyMsg(fromGraph);
    if (!reply.data.has_value()) {
        expect(false) << std::format("Emplace block failed and returned an error. Requested at: {}. Error: {}\n", sourceLocation, reply.data.error());
    }
    return std::get<std::string>(reply.data.value().at("uniqueName"s));
};

inline void sendAndWaitMessageEmplaceEdge(gr::MsgPortOut& toGraph, gr::MsgPortIn& fromGraph, std::string sourceBlock, std::string sourcePort, std::string destinationBlock, std::string destinationPort, std::string serviceName = "", std::source_location sourceLocation = std::source_location::current()) {
    expect(eq(getNReplyMessages(fromGraph), 0UZ)) << std::format("Input port has unconsumed messages. Requested at: {}\n", sourceLocation);
    gr::property_map data = {{"sourceBlock", sourceBlock}, {"sourcePort", sourcePort}, {"destinationBlock", destinationBlock}, {"destinationPort", destinationPort}, //
        {"minBufferSize", gr::Size_t()}, {"weight", 0}, {"edgeName", "unnamed edge"}};
    sendMessage<Set>(toGraph, serviceName, gr::scheduler::property::kEmplaceEdge /* endpoint */, data /* data */);

    expect(waitForReply(fromGraph)) << std::format("Reply message not received. Requested at: {}\n", sourceLocation);
    expect(eq(getNReplyMessages(fromGraph), 1UZ)) << std::format("No messages available. Requested at: {}\n", sourceLocation);
    const Message reply = consumeReplyMsg(fromGraph);

    if (!reply.data.has_value()) {
        expect(false) << std::format("Emplace edge failed and returned an error. Requested at: {}. Error: {}\n", sourceLocation, reply.data.error());
    }
};

} // namespace gr::testing

#endif // include guard
