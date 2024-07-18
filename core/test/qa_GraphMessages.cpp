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

const boost::ut::suite<"Graph Formatter Tests"> graphFormatterTests = [] {
    using namespace boost::ut;
    using namespace gr;

    "Edge formatter tests"_test = [] {
        Graph graph;
        auto& source = graph.emplaceBlock<NullSource<float>>();
        auto& sink   = graph.emplaceBlock<NullSink<float>>();
        Edge  edge{graph.blocks()[0UZ].get(), {1}, graph.blocks()[1UZ].get(), {2}, 1024, 1, "test_edge"};

        "default"_test = [&edge] {
            std::string result = fmt::format("{:s}", edge);
            fmt::println("Edge formatter - default:   {}", result);

            expect(result.contains(" ⟶ (name: 'test_edge', size: 1024, weight:  1, connected: false) ⟶")) << result;
        };

        "short names"_test = [&edge] {
            std::string result = fmt::format("{:s}", edge);
            fmt::println("Edge formatter - short 's': {}", result);

            expect(result.contains(" ⟶ (name: 'test_edge', size: 1024, weight:  1, connected: false) ⟶")) << result;
        };

        "long names"_test = [&edge] {
            std::string result = fmt::format("{:l}", edge);
            fmt::println("Edge formatter - long  'l': {}", result);

            expect(result.contains(" ⟶ (name: 'test_edge', size: 1024, weight:  1, connected: false) ⟶")) << result;
        };
    };
};

auto returnReplyMsg(gr::MsgPortIn& port) {
    expect(eq(port.streamReader().available(), 1UZ)) << "didn't receive a reply message";
    ConsumableSpan auto span = port.streamReader().get<SpanReleasePolicy::ProcessAll>(1UZ);
    Message             msg  = span[0];
    expect(span.consume(span.size()));
    fmt::print("Test got a reply: {}\n", msg);
    return msg;
};

bool awaitCondition(std::chrono::milliseconds timeout, std::function<bool()> condition) {
    auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < timeout) {
        if (condition()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

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
            expect(eq(testGraph.blocks().size(), 1UZ));
        };

        "Add an invalid block"_test = [&] {
            sendMessage<Set>(toGraph, "" /* serviceName */, graph::property::kEmplaceBlock /* endpoint */, //
                {{"type", "doesnt_exist::multiply"}, {"parameters", "float"}, {"properties", property_map{}}} /* data */);
            expect(nothrow([&] { testGraph.processScheduledMessages(); })) << "manually execute processing of messages";

            const Message reply = returnReplyMsg(fromGraph);

            expect(!reply.data.has_value());
            expect(eq(testGraph.blocks().size(), 1UZ));
        };
    };

    "Block removal tests"_test = [] {
        gr::MsgPortOut toGraph;
        gr::Graph      testGraph;
        gr::MsgPortIn  fromGraph;

        testGraph.emplaceBlock("gr::testing::Copy", "float", {});
        expect(eq(testGraph.blocks().size(), 1UZ));

        expect(eq(ConnectionResult::SUCCESS, toGraph.connect(testGraph.msgIn)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.msgOut.connect(fromGraph)));

        "Remove a known block"_test = [&] {
            auto& temporaryBlock = testGraph.emplaceBlock("gr::testing::Copy", "float", {});
            expect(eq(testGraph.blocks().size(), 2UZ));

            sendMessage<Set>(toGraph, "" /* serviceName */, graph::property::kRemoveBlock /* endpoint */, //
                {{"uniqueName", std::string(temporaryBlock.uniqueName())}} /* data */);
            expect(nothrow([&] { testGraph.processScheduledMessages(); })) << "manually execute processing of messages";

            const Message reply = returnReplyMsg(fromGraph);

            expect(reply.data.has_value());
            expect(eq(testGraph.blocks().size(), 1UZ));
        };

        "Remove an unknown block"_test = [&] {
            sendMessage<Set>(toGraph, "" /* serviceName */, graph::property::kRemoveBlock /* endpoint */, //
                {{"uniqueName", "this_block_is_unknown"}} /* data */);
            expect(nothrow([&] { testGraph.processScheduledMessages(); })) << "manually execute processing of messages";

            const Message reply = returnReplyMsg(fromGraph);

            expect(!reply.data.has_value());
            expect(eq(testGraph.blocks().size(), 1UZ));
        };
    };

    "Block replacement tests"_test = [] {
        gr::MsgPortOut toGraph;
        gr::Graph      testGraph;
        gr::MsgPortIn  fromGraph;

        auto& block = testGraph.emplaceBlock("gr::testing::Copy", "float", {});
        expect(eq(testGraph.blocks().size(), 1UZ));

        expect(eq(ConnectionResult::SUCCESS, toGraph.connect(testGraph.msgIn)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.msgOut.connect(fromGraph)));

        "Replace a known block"_test = [&] {
            auto& temporaryBlock = testGraph.emplaceBlock("gr::testing::Copy", "float", {});
            expect(eq(testGraph.blocks().size(), 2UZ));

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
            property_map data = {{"sourceBlock", std::string(blockOut.uniqueName())}, {"sourcePort", "out"}, //
                {"destinationBlock", std::string(blockIn.uniqueName())}, {"destinationPort", "in"},          //
                {"minBufferSize", gr::Size_t()}, {"weight", std::int32_t(0)}, {"edgeName", "unnamed edge"}};

            sendMessage<Set>(toGraph, "" /* serviceName */, graph::property::kEmplaceEdge /* endpoint */, data /* data */);
            expect(nothrow([&] { testGraph.processScheduledMessages(); })) << "manually execute processing of messages";

            const Message reply = returnReplyMsg(fromGraph);

            if (!reply.data.has_value()) {
                expect(false) << fmt::format("edge not being placed - error: {}", reply.data.error());
            }
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
        property_map data = {{"sourceBlock", sourceBlock}, {"sourcePort", sourcePort},    //
            {"destinationBlock", destinationBlock}, {"destinationPort", destinationPort}, //
            {"minBufferSize", gr::Size_t()}, {"weight", std::int32_t(0)}, {"edgeName", "unnamed edge"}};
        sendMessage<Set>(toGraph, "" /* serviceName */, graph::property::kEmplaceEdge /* endpoint */, data /* data */);
        if (!waitForAReply()) {
            fmt::println("didn't receive a reply message for {}", data);
            return false;
        }

        const Message reply = returnReplyMsg(fromGraph);
        return reply.data.has_value();
    };

    std::expected<void, Error> schedulerRet;
    auto                       runScheduler = [&scheduler, &schedulerRet] { schedulerRet = scheduler.runAndWait(); };

    std::thread schedulerThread1(runScheduler);

    expect(awaitCondition(1s, [&scheduler] { return scheduler.state() == lifecycle::State::RUNNING; })) << "scheduler thread up and running w/ timeout";
    expect(scheduler.state() == lifecycle::State::RUNNING) << "scheduler thread up and running";
    // FIXME: expect(eq(scheduler.graph().edges().size(), 1UZ)) << "added one new edges";

    expect(awaitCondition(1s, [&sink] { return sink.count >= 10U; })) << "sink received enough data";
    fmt::println("executed basic graph");

    // Adding a few blocks
    auto multiply1 = emplaceTestBlock("gr::testing::Copy"s, "float"s, property_map{});
    auto multiply2 = emplaceTestBlock("gr::testing::Copy"s, "float"s, property_map{});
    scheduler.processScheduledMessages();

    for (const auto& block : scheduler.graph().blocks()) {
        fmt::println("block in list: {} - state() : {}", block->name(), magic_enum::enum_name(block->state()));
    }
    expect(eq(scheduler.graph().blocks().size(), 4UZ)) << "should contain sink->multiply1->multiply2->sink";

    expect(emplaceTestEdge(source.unique_name, "out", multiply1, "in")) << "emplace edge source -> multiply1 failed and returned an error";
    expect(emplaceTestEdge(multiply1, "out", multiply2, "in")) << "emplace edge multiply1 -> multiply2 failed and returned an error";
    expect(emplaceTestEdge(multiply2, "out", sink.unique_name, "in")) << "emplace edge multiply2 -> sink failed and returned an error";
    scheduler.processScheduledMessages();

    scheduler.requestStop();
    schedulerThread1.join();
    if (!schedulerRet.has_value()) {
        expect(false) << fmt::format("scheduler.runAndWait() failed:\n{}\n", schedulerRet.error());
    }

    // return to initial state
    expect(scheduler.changeStateTo(lifecycle::State::INITIALISED).has_value()) << "could switch to INITIALISED?";
    expect(awaitCondition(1s, [&scheduler] { return scheduler.state() == lifecycle::State::INITIALISED; })) << "scheduler INITIALISED w/ timeout";
    expect(scheduler.state() == lifecycle::State::INITIALISED) << fmt::format("scheduler INITIALISED - actual: {}\n", magic_enum::enum_name(scheduler.state()));

    std::thread schedulerThread2(runScheduler);
    expect(awaitCondition(1s, [&scheduler] { return scheduler.state() == lifecycle::State::RUNNING; })) << "scheduler thread up and running w/ timeout";
    expect(scheduler.state() == lifecycle::State::RUNNING) << "scheduler thread up and running";

    for (const auto& edge : scheduler.graph().edges()) {
        fmt::println("edge in list({}): {}", scheduler.graph().edges().size(), edge);
    }
    // FIXME: expect(eq(scheduler.graph().edges().size(), 3UZ)) << "added three new edges";

    // FIXME: edge->connection is not performed
    //    expect(awaitCondition(1s, [&sink] {
    //        std::this_thread::sleep_for(100ms);
    //        fmt::println("sink has received {} samples - parents: {}", sink.count, sink.in.buffer().streamBuffer.n_writers());
    //        return sink.count >= 10U;
    //    })) << "sink received enough data";

    scheduler.requestStop();

    fmt::print("Counting sink counted to {}\n", sink.count);

    schedulerThread2.join();
    if (!schedulerRet.has_value()) {
        expect(false) << fmt::format("scheduler.runAndWait() failed:\n{}\n", schedulerRet.error());
    }
};

} // namespace gr::testing

int main() { /* tests are statically executed */ }
