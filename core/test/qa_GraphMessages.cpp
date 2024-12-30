#include <boost/ut.hpp>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Message.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/basic/clock_source.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include <magic_enum.hpp>
#include <magic_enum_utility.hpp>

#include "message_utils.hpp"

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
        Graph                  graph;
        [[maybe_unused]] auto& source = graph.emplaceBlock<NullSource<float>>();
        [[maybe_unused]] auto& sink   = graph.emplaceBlock<NullSink<float>>();
        Edge                   edge{graph.blocks()[0UZ].get(), {1}, graph.blocks()[1UZ].get(), {2}, 1024, 1, "test_edge"};

        "default"_test = [&edge] {
            std::string result = fmt::format("{:s}", edge);
            fmt::println("Edge formatter - default:   {}", result);

            expect(result.contains(" ⟶ (name: 'test_edge', size: 1024, weight:  1, state: WaitingToBeConnected) ⟶")) << result;
        };

        "short names"_test = [&edge] {
            std::string result = fmt::format("{:s}", edge);
            fmt::println("Edge formatter - short 's': {}", result);

            expect(result.contains(" ⟶ (name: 'test_edge', size: 1024, weight:  1, state: WaitingToBeConnected) ⟶")) << result;
        };

        "long names"_test = [&edge] {
            std::string result = fmt::format("{:l}", edge);
            fmt::println("Edge formatter - long  'l': {}", result);

            expect(result.contains(" ⟶ (name: 'test_edge', size: 1024, weight:  1, state: WaitingToBeConnected) ⟶")) << result;
        };
    };
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

            const Message reply = awaitReplyMsg(testGraph, 100ms, fromGraph, graph::property::kBlockEmplaced);

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
                {"minBufferSize", gr::Size_t()}, {"weight", 0}, {"edgeName", "unnamed edge"}};

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

    "BlockRegistry tests"_test = [] {
        gr::MsgPortOut toGraph;
        gr::Graph      testGraph;
        gr::MsgPortIn  fromGraph;

        expect(eq(ConnectionResult::SUCCESS, toGraph.connect(testGraph.msgIn)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.msgOut.connect(fromGraph)));

        "Get available block types"_test = [&] {
            sendMessage<Get>(toGraph, "" /* serviceName */, graph::property::kRegistryBlockTypes /* endpoint */, {} /* data */);
            expect(nothrow([&] { testGraph.processScheduledMessages(); })) << "manually execute processing of messages";

            const Message reply = awaitReplyMsg(testGraph, 100ms, fromGraph);

            if (reply.data.has_value()) {
                const auto& dataMap    = reply.data.value();
                auto        foundTypes = dataMap.find("types");
                if (foundTypes != dataMap.end() || !std::holds_alternative<property_map>(foundTypes->second)) {
                    PluginLoader& loader             = gr::globalPluginLoader();
                    const auto    expectedBlockTypes = loader.knownBlocks();
                    const auto&   blockTypes         = std::get<property_map>(foundTypes->second);
                    expect(eq(expectedBlockTypes.size(), blockTypes.size()));

                    for (const auto& expectedBlockType : expectedBlockTypes) {
                        auto foundBlockType = blockTypes.find(expectedBlockType);
                        if (foundBlockType != blockTypes.end() && std::holds_alternative<property_map>(foundBlockType->second)) {
                            const property_map& blockMap = std::get<property_map>(foundBlockType->second);

                            if (std::holds_alternative<std::vector<std::string>>(blockMap.at("parametrizations"))) {
                                const std::vector<std::string>      blockParams         = std::get<std::vector<std::string>>(blockMap.at("parametrizations"));
                                const std::vector<std::string_view> expectedBlockParams = loader.knownBlockParameterizations(expectedBlockType);

                                if (blockParams.size() == expectedBlockParams.size()) {
                                    expect(std::ranges::equal(expectedBlockParams, blockParams));
                                } else {
                                    expect(false) << std::format("different number of parametrizations for block `{}`: {} vs. {}", foundBlockType->first, blockParams.size(), expectedBlockParams.size());
                                }
                            } else {
                                expect(false) << std::format("block type ({}) parametrizations is not a std::vector<std::string>", foundBlockType->first);
                            }
                        } else {
                            expect(false) << std::format("block type ({}) not found or pmt type is not correct", foundBlockType->first);
                        }
                    }
                } else {
                    expect(false) << "`types` key not found or data type is not a `property_map`";
                }
            } else {
                expect(false) << fmt::format("data has no value - error: {}", reply.data.error());
            }
        };
    };
};

const boost::ut::suite RunningGraphTests = [] {
    using namespace std::string_literals;
    using namespace boost::ut;
    using namespace gr;
    using enum gr::message::Command;

    gr::scheduler::Simple scheduler{gr::Graph()};

    auto& source = scheduler.graph().emplaceBlock<SlowSource<float>>();
    auto& sink   = scheduler.graph().emplaceBlock<CountingSink<float>>();
    expect(eq(ConnectionResult::SUCCESS, scheduler.graph().connect<"out">(source).to<"in">(sink)));
    expect(eq(scheduler.graph().edges().size(), 1UZ)) << "edge registered with connect";

    gr::MsgPortOut toGraph;
    gr::MsgPortIn  fromGraph;
    expect(eq(ConnectionResult::SUCCESS, toGraph.connect(scheduler.msgIn)));
    expect(eq(ConnectionResult::SUCCESS, scheduler.msgOut.connect(fromGraph)));

    std::expected<void, Error> schedulerRet;
    auto                       runScheduler = [&scheduler, &schedulerRet] { schedulerRet = scheduler.runAndWait(); };

    std::thread schedulerThread1(runScheduler);

    expect(awaitCondition(1s, [&scheduler] { return scheduler.state() == lifecycle::State::RUNNING; })) << "scheduler thread up and running w/ timeout";
    expect(scheduler.state() == lifecycle::State::RUNNING) << "scheduler thread up and running";
    expect(eq(scheduler.graph().edges().size(), 1UZ)) << "added one edge";

    expect(awaitCondition(1s, [&sink] { return sink.count >= 10U; })) << "sink received enough data";
    fmt::println("executed basic graph");

    // Adding a few blocks
    auto multiply1 = sendEmplaceTestBlockMsg(toGraph, fromGraph, "gr::testing::Copy"s, "float"s, property_map{});
    auto multiply2 = sendEmplaceTestBlockMsg(toGraph, fromGraph, "gr::testing::Copy"s, "float"s, property_map{});
    scheduler.processScheduledMessages();

    for (const auto& block : scheduler.graph().blocks()) {
        fmt::println("block in list: {} - state() : {}", block->name(), magic_enum::enum_name(block->state()));
    }
    expect(eq(scheduler.graph().blocks().size(), 4UZ)) << "should contain sink->multiply1->multiply2->sink";

    expect(sendEmplaceTestEdgeMsg(toGraph, fromGraph, source.unique_name, "out", multiply1, "in")) << "emplace edge source -> multiply1 failed and returned an error";
    expect(sendEmplaceTestEdgeMsg(toGraph, fromGraph, multiply1, "out", multiply2, "in")) << "emplace edge multiply1 -> multiply2 failed and returned an error";
    expect(sendEmplaceTestEdgeMsg(toGraph, fromGraph, multiply2, "out", sink.unique_name, "in")) << "emplace edge multiply2 -> sink failed and returned an error";
    scheduler.processScheduledMessages();

    // Get the whole graph
    {
        sendMessage<Set>(toGraph, "" /* serviceName */, graph::property::kGraphInspect /* endpoint */, property_map{} /* data */);
        if (!waitForAReply(fromGraph)) {
            fmt::println("didn't receive a reply message for kGraphInspect");
            expect(false);
        }

        const Message reply = returnReplyMsg(fromGraph);
        expect(reply.data.has_value());

        const auto& data     = reply.data.value();
        const auto& children = std::get<property_map>(data.at("children"s));
        expect(eq(children.size(), 4UZ));

        const auto& edges = std::get<property_map>(data.at("edges"s));
        expect(eq(edges.size(), 4UZ));
    }
    scheduler.processScheduledMessages();

    // Stopping scheduler
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
    expect(eq(scheduler.graph().edges().size(), 4UZ)) << "added three new edges, one previously registered with connect";

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
