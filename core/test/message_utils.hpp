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

inline auto returnReplyMsg(gr::MsgPortIn& port, std::optional<std::string> expectedEndpoint = {}, std::source_location caller = std::source_location::current()) {
    auto available = port.streamReader().available();
    expect(gt(available, 0UZ)) << "didn't receive a reply message, caller: " << caller.file_name() << ":" << caller.line() << "\n";
    ReaderSpanLike auto messages = port.streamReader().get<SpanReleasePolicy::ProcessAll>(available);
    Message             result;

    if (expectedEndpoint) {
        auto it = std::ranges::find_if(messages, [endpoint = *expectedEndpoint](const auto& message) { return message.endpoint == endpoint; });
        if (it == messages.end()) {
            expect(gt(available, 0UZ)) << "didn't receive the expected reply message, caller: " << caller.file_name() << ":" << caller.line() << "\n";
        } else {
            result = *it;
        }
    } else {
        result = messages[0];
    }

    expect(messages.consume(messages.size()));
    fmt::print("Test got a reply: {}\n", result);
    if (expectedEndpoint) {
        expect(eq(*expectedEndpoint, result.endpoint));
    }
    return result;
};

template<bool processScheduledMessages = true>
auto awaitReplyMsg(auto& graph, std::chrono::milliseconds timeout, gr::MsgPortIn& port, std::optional<std::string> expectedEndpoint = {}, std::source_location caller = std::source_location::current()) {
    awaitCondition(timeout, [&port, &graph] {
        if constexpr (processScheduledMessages) {
            graph.processScheduledMessages();
        }
        return port.streamReader().available() > 0;
    });

    return returnReplyMsg(port, expectedEndpoint, caller);
};

inline auto waitForAReply(gr::MsgPortIn& fromGraph, std::chrono::milliseconds maxWait = 1s, std::source_location currentSource = std::source_location::current()) {
    auto startedAt = std::chrono::system_clock::now();
    while (fromGraph.streamReader().available() == 0) {
        std::this_thread::sleep_for(100ms);
        if (std::chrono::system_clock::now() - startedAt > maxWait) {
            break;
        }
    }
    expect(fromGraph.streamReader().available() > 0) << "Caller at" << currentSource.file_name() << ":" << currentSource.line();
    return fromGraph.streamReader().available() > 0;
};

inline auto sendEmplaceTestBlockMsg(gr::MsgPortOut& toGraph, gr::MsgPortIn& fromGraph, std::string type, std::string params, property_map properties) {
    sendMessage<Set>(toGraph, "" /* serviceName */, graph::property::kEmplaceBlock /* endpoint */, //
        {{"type", std::move(type)}, {"parameters", std::move(params)}, {"properties", std::move(properties)}} /* data */);
    expect(waitForAReply(fromGraph)) << "didn't receive a reply message";

    const Message reply = returnReplyMsg(fromGraph);
    expect(reply.data.has_value()) << "emplace block failed and returned an error";
    return reply.data.has_value() ? std::get<std::string>(reply.data.value().at("uniqueName"s)) : std::string{};
};

inline auto sendEmplaceTestEdgeMsg(gr::MsgPortOut& toGraph, gr::MsgPortIn& fromGraph, std::string sourceBlock, std::string sourcePort, std::string destinationBlock, std::string destinationPort) {
    gr::property_map data = {{"sourceBlock", sourceBlock}, {"sourcePort", sourcePort}, //
        {"destinationBlock", destinationBlock}, {"destinationPort", destinationPort},  //
        {"minBufferSize", gr::Size_t()}, {"weight", 0}, {"edgeName", "unnamed edge"}};
    sendMessage<Set>(toGraph, "" /* serviceName */, graph::property::kEmplaceEdge /* endpoint */, data /* data */);
    if (!waitForAReply(fromGraph)) {
        fmt::println("didn't receive a reply message for {}", data);
        return false;
    }

    const Message reply = returnReplyMsg(fromGraph);
    return reply.data.has_value();
};

} // namespace gr::testing

#endif // include guard
