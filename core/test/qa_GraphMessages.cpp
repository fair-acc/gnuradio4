#include <boost/ut.hpp>

#include "gnuradio-4.0/Block.hpp"
#include "gnuradio-4.0/Message.hpp"
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/basic/clock_source.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include <magic_enum.hpp>
#include <magic_enum_utility.hpp>

#include <optional>

using namespace std::chrono_literals;
using namespace std::string_literals;

using namespace gr::message;

namespace gr::testing {

using namespace boost::ut;
using namespace gr;

auto returnReplyMsg(gr::MsgPortIn& port) {
    expect(eq(port.streamReader().available(), 1UZ)) << "didn't receive a reply message";
    ConsumableSpan auto span = port.streamReader().get<SpanReleasePolicy::ProcessAll>(1UZ);
    Message             msg  = span[0];
    expect(span.consume(span.size()));
    fmt::print("Test got a reply: {}\n", msg);
    return msg;
};

const boost::ut::suite NonRunningGraphTests = [] {
    using namespace std::string_literals;
    using namespace boost::ut;
    using namespace gr;
    using enum gr::message::Command;

    "Block addition tests"_test = [] {
        gr::MsgPortOut toGraph;
        gr::Graph      testGraph;
        gr::MsgPortIn  fromGraph;

        expect(eq(ConnectionResult::SUCCESS, toGraph.connect(testGraph.msgIn)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.msgOut.connect(fromGraph)));

        "Add a valid block"_test = [&] {
            sendMessage<Set>(toGraph, "" /* serviceName */, graph::property::kEmplaceBlock /* endpoint */, //
                {{"type", "gr::testing::Copy"}, {"parameters", "float"}, {"properties", property_map{}}} /* data */);
            expect(nothrow([&] { testGraph.processScheduledMessages(); })) << "manually execute processing of messages";

            const Message reply = returnReplyMsg(fromGraph);

            expect(reply.data.has_value());
            expect(eq(testGraph.blocks().size(), 1));
        };

        "Add an invalid block"_test = [&] {
            sendMessage<Set>(toGraph, "" /* serviceName */, graph::property::kEmplaceBlock /* endpoint */, //
                {{"type", "doesnt_exist::multiply"}, {"parameters", "float"}, {"properties", property_map{}}} /* data */);
            expect(nothrow([&] { testGraph.processScheduledMessages(); })) << "manually execute processing of messages";

            const Message reply = returnReplyMsg(fromGraph);

            expect(!reply.data.has_value());
            expect(eq(testGraph.blocks().size(), 1));
        };
    };

    "Block removal tests"_test = [] {
        gr::MsgPortOut toGraph;
        gr::Graph      testGraph;
        gr::MsgPortIn  fromGraph;

        auto& block = testGraph.emplaceBlock("gr::testing::Copy", "float", {});
        expect(eq(testGraph.blocks().size(), 1));

        expect(eq(ConnectionResult::SUCCESS, toGraph.connect(testGraph.msgIn)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.msgOut.connect(fromGraph)));

        "Remove a known block"_test = [&] {
            auto& temporaryBlock = testGraph.emplaceBlock("gr::testing::Copy", "float", {});
            expect(eq(testGraph.blocks().size(), 2));

            sendMessage<Set>(toGraph, "" /* serviceName */, graph::property::kRemoveBlock /* endpoint */, //
                {{"uniqueName", std::string(temporaryBlock.uniqueName())}} /* data */);
            expect(nothrow([&] { testGraph.processScheduledMessages(); })) << "manually execute processing of messages";

            const Message reply = returnReplyMsg(fromGraph);

            expect(reply.data.has_value());
            expect(eq(testGraph.blocks().size(), 1));
        };

        "Remove an unknown block"_test = [&] {
            sendMessage<Set>(toGraph, "" /* serviceName */, graph::property::kRemoveBlock /* endpoint */, //
                {{"uniqueName", "this_block_is_unknown"}} /* data */);
            expect(nothrow([&] { testGraph.processScheduledMessages(); })) << "manually execute processing of messages";

            const Message reply = returnReplyMsg(fromGraph);

            expect(!reply.data.has_value());
            expect(eq(testGraph.blocks().size(), 1));
        };
    };

    "Block replacement tests"_test = [] {
        gr::MsgPortOut toGraph;
        gr::Graph      testGraph;
        gr::MsgPortIn  fromGraph;

        auto& block = testGraph.emplaceBlock("gr::testing::Copy", "float", {});
        expect(eq(testGraph.blocks().size(), 1));

        expect(eq(ConnectionResult::SUCCESS, toGraph.connect(testGraph.msgIn)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.msgOut.connect(fromGraph)));

        "Replace a known block"_test = [&] {
            auto& temporaryBlock = testGraph.emplaceBlock("gr::testing::Copy", "float", {});
            expect(eq(testGraph.blocks().size(), 2));

            sendMessage<Set>(toGraph, "" /* serviceName */, graph::property::kReplaceBlock /* endpoint */, //
                {{"uniqueName", std::string(temporaryBlock.uniqueName())},                                 //
                    {"type", "gr::testing::Copy"}, {"parameters", "float"}, {"properties", property_map{}}} /* data */);
            expect(nothrow([&] { testGraph.processScheduledMessages(); })) << "manually execute processing of messages";

            const Message reply = returnReplyMsg(fromGraph);

            expect(reply.data.has_value());
        };

        "Replace an unknown block"_test = [&] {
            sendMessage<Set>(toGraph, "" /* serviceName */, graph::property::kReplaceBlock /* endpoint */, //
                {{"uniqueName", "this_block_is_unknown"},                                                  //
                    {"type", "gr::testing::Copy"}, {"parameters", "float"}, {"properties", property_map{}}} /* data */);
            expect(nothrow([&] { testGraph.processScheduledMessages(); })) << "manually execute processing of messages";

            const Message reply = returnReplyMsg(fromGraph);

            expect(!reply.data.has_value());
        };

        "Replace with an unknown block"_test = [&] {
            sendMessage<Set>(toGraph, "" /* serviceName */, graph::property::kReplaceBlock /* endpoint */, //
                {{"uniqueName", std::string(block.uniqueName())},                                          //
                    {"type", "doesnt_exist::multiply"}, {"parameters", "float"}, {"properties", property_map{}}} /* data */);
            expect(nothrow([&] { testGraph.processScheduledMessages(); })) << "manually execute processing of messages";

            const Message reply = returnReplyMsg(fromGraph);

            expect(!reply.data.has_value());
        };
    };

    "Edge addition tests"_test = [&] {
        gr::MsgPortOut toGraph;
        gr::Graph      testGraph;
        gr::MsgPortIn  fromGraph;

        auto& blockOut       = testGraph.emplaceBlock("gr::testing::Copy", "float", {});
        auto& blockIn        = testGraph.emplaceBlock("gr::testing::Copy", "float", {});
        auto& blockWrongType = testGraph.emplaceBlock("gr::testing::Copy", "double", {});

        expect(eq(ConnectionResult::SUCCESS, toGraph.connect(testGraph.msgIn)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.msgOut.connect(fromGraph)));

        "Add an edge"_test = [&] {
            sendMessage<Set>(toGraph, "" /* serviceName */, graph::property::kEmplaceEdge /* endpoint */, //
                {{"sourceBlock", std::string(blockOut.uniqueName())}, {"sourcePort", "out"},              //
                    {"destinationBlock", std::string(blockIn.uniqueName())}, {"destinationPort", "in"}} /* data */);
            expect(nothrow([&] { testGraph.processScheduledMessages(); })) << "manually execute processing of messages";

            const Message reply = returnReplyMsg(fromGraph);

            expect(reply.data.has_value());
        };

        "Fail to add an edge because source port is invalid"_test = [&] {
            sendMessage<Set>(toGraph, "" /* serviceName */, graph::property::kEmplaceEdge /* endpoint */, //
                {{"sourceBlock", std::string(blockOut.uniqueName())}, {"sourcePort", "OUTPUT"},           //
                    {"destinationBlock", std::string(blockIn.uniqueName())}, {"destinationPort", "in"}} /* data */);
            expect(nothrow([&] { testGraph.processScheduledMessages(); })) << "manually execute processing of messages";

            const Message reply = returnReplyMsg(fromGraph);
            expect(!reply.data.has_value());
        };

        "Fail to add an edge because destination port is invalid"_test = [&] {
            sendMessage<Set>(toGraph, "" /* serviceName */, graph::property::kEmplaceEdge /* endpoint */, //
                {{"sourceBlock", std::string(blockOut.uniqueName())}, {"sourcePort", "in"},               //
                    {"destinationBlock", std::string(blockIn.uniqueName())}, {"destinationPort", "INPUT"}} /* data */);
            expect(nothrow([&] { testGraph.processScheduledMessages(); })) << "manually execute processing of messages";

            const Message reply = returnReplyMsg(fromGraph);
            expect(!reply.data.has_value());
        };

        "Fail to add an edge because ports are not compatible"_test = [&] {
            sendMessage<Set>(toGraph, "" /* serviceName */, graph::property::kEmplaceEdge /* endpoint */, //
                {{"sourceBlock", std::string(blockOut.uniqueName())}, {"sourcePort", "out"},              //
                    {"destinationBlock", std::string(blockWrongType.uniqueName())}, {"destinationPort", "in"}} /* data */);
            expect(nothrow([&] { testGraph.processScheduledMessages(); })) << "manually execute processing of messages";

            const Message reply = returnReplyMsg(fromGraph);
            expect(!reply.data.has_value());
        };
    };
};

const boost::ut::suite RunningGraphTests = [] {
    using namespace std::string_literals;
    using namespace boost::ut;
    using namespace gr;
    using enum gr::message::Command;

    std::atomic_bool keep_running = true;

    gr::scheduler::Simple scheduler{gr::Graph()};

    // auto& source = scheduler.graph().emplaceBlock<gr::basic::ClockSource<float>>();
    auto& source = scheduler.graph().emplaceBlock<SlowSource<float>>();
    auto& sink   = scheduler.graph().emplaceBlock<CountingSink<float>>();
    expect(eq(ConnectionResult::SUCCESS, scheduler.graph().connect<"out">(source).to<"in">(sink)));

    gr::MsgPortOut toGraph;
    gr::MsgPortIn  fromGraph;
    expect(eq(ConnectionResult::SUCCESS, toGraph.connect(scheduler.graph().msgIn)));
    expect(eq(ConnectionResult::SUCCESS, scheduler.graph().msgOut.connect(fromGraph)));

    auto waitForAReply = [&](std::chrono::milliseconds maxWait = 1s, std::source_location source = std::source_location::current()) {
        auto startedAt = std::chrono::system_clock::now();
        while (fromGraph.streamReader().available() == 0) {
            std::this_thread::sleep_for(100ms);
            if (std::chrono::system_clock::now() - startedAt > maxWait) {
                break;
            }
        }
        expect(fromGraph.streamReader().available() > 0) << "Caller at" << source.file_name() << ":" << source.line();
        return fromGraph.streamReader().available() > 0;
    };

    auto emplaceTestBlock = [&](std::string type, std::string params, property_map properties) {
        sendMessage<Set>(toGraph, "" /* serviceName */, graph::property::kEmplaceBlock /* endpoint */, //
            {{"type", std::move(type)}, {"parameters", std::move(params)}, {"properties", std::move(properties)}} /* data */);
        expect(waitForAReply()) << "didn't receive a reply message";

        const Message reply = returnReplyMsg(fromGraph);
        expect(reply.data.has_value()) << "emplace block failed and returned an error";
        return reply.data.has_value() ? std::get<std::string>(reply.data.value().at("uniqueName"s)) : std::string{};
    };

    auto emplaceTestEdge = [&](std::string sourceBlock, std::string sourcePort, std::string destinationBlock, std::string destinationPort) {
        sendMessage<Set>(toGraph, "" /* serviceName */, graph::property::kEmplaceEdge /* endpoint */, //
            {{"sourceBlock", sourceBlock}, {"sourcePort", sourcePort},                                //
                {"destinationBlock", destinationBlock}, {"destinationPort", destinationPort}} /* data */);
        expect(waitForAReply()) << "didn't receive a reply message";

        const Message reply = returnReplyMsg(fromGraph);
        expect(reply.data.has_value()) << "emplace block failed and returned an error";
    };

    std::thread tester([&] {
        // Adding a few blocks
        auto multiply1 = emplaceTestBlock("gr::testing::Copy"s, "float"s, property_map{});
        auto multiply2 = emplaceTestBlock("gr::testing::Copy"s, "float"s, property_map{});

        expect(eq(scheduler.graph().blocks().size(), 4));

        emplaceTestEdge(source.unique_name, "out", multiply1, "in");
        emplaceTestEdge(multiply1, "out", sink.unique_name, "in");

        // expect(eq(scheduler.graph().edges().size(), 2));
        while (sink.count < 10) {
            std::this_thread::sleep_for(100ms);
        }

        keep_running = false;
        scheduler.requestStop();
    });

    while (keep_running) {
        scheduler.processScheduledMessages();
        std::ignore = scheduler.runAndWait();
        fmt::print("Counting sink counted to {}\n", sink.count);
    }

    tester.join();
};

} // namespace gr::testing

int main() { /* tests are statically executed */ }
