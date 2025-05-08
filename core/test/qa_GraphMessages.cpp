#include <boost/ut.hpp>

#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>

#include <GrBasicBlocks.hpp>
#include <GrTestingBlocks.hpp>

#include "TestBlockRegistryContext.hpp"

#include "message_utils.hpp"

#include <filesystem>

using namespace std::chrono_literals;
using namespace std::string_literals;

namespace ut = boost::ut;

// We don't like new, but this will ensure the object is alive
// when ut starts running the tests. It runs the tests when
// its static objects get destroyed, which means other static
// objects might have been destroyed before that.
TestContext* context = new TestContext(paths{}, // plugin paths
    gr::blocklib::initGrBasicBlocks,            //
    gr::blocklib::initGrTestingBlocks);

const boost::ut::suite<"Graph Formatter Tests"> graphFormatterTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;

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
    using namespace gr::testing;
    using enum gr::message::Command;

    expect(fatal(gt(context->registry.knownBlocks().size(), 0UZ))) << "didn't register any blocks";
    fmt::println("registered blocks:");
    for (const auto& blockName : context->registry.knownBlocks()) {
        fmt::println("    block: {}", blockName);
    }

    "Block addition tests"_test = [] {
        gr::MsgPortOut toGraph;
        gr::Graph      testGraph(context->loader);
        gr::MsgPortIn  fromGraph;

        expect(eq(ConnectionResult::SUCCESS, toGraph.connect(testGraph.msgIn)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.msgOut.connect(fromGraph)));

        "Add a valid block"_test = [&] {
            sendMessage<Set>(toGraph, testGraph.unique_name, graph::property::kEmplaceBlock /* endpoint */, //
                {{"type", "gr::testing::Copy<float32>"}, {"properties", property_map{}}} /* data */);
            expect(nothrow([&] { testGraph.processScheduledMessages(); })) << "manually execute processing of messages";

            expect(eq(getNReplyMessages(fromGraph), 1UZ));
            const Message reply = getAndConsumeFirstReplyMessage(fromGraph, graph::property::kBlockEmplaced);
            if (!reply.data.has_value()) {
                expect(false) << fmt::format("reply.data has no value:{}\n", reply.data.error());
            }
            expect(eq(testGraph.blocks().size(), 1UZ));
        };

        "Add an invalid block"_test = [&] {
            sendMessage<Set>(toGraph, testGraph.unique_name, graph::property::kEmplaceBlock /* endpoint */, //
                {{"type", "doesnt_exist::multiply<float32>"}, {"properties", property_map{}}} /* data */);
            expect(nothrow([&] { testGraph.processScheduledMessages(); })) << "manually execute processing of messages";

            expect(eq(getNReplyMessages(fromGraph), 1UZ));
            const Message reply = getAndConsumeFirstReplyMessage(fromGraph);
            expect(eq(getNReplyMessages(fromGraph), 0UZ));
            expect(!reply.data.has_value());
            expect(eq(testGraph.blocks().size(), 1UZ));
        };
    };

    "Block removal tests"_test = [] {
        gr::MsgPortOut toGraph;
        gr::Graph      testGraph(context->loader);
        gr::MsgPortIn  fromGraph;

        expect(eq(ConnectionResult::SUCCESS, toGraph.connect(testGraph.msgIn)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.msgOut.connect(fromGraph)));

        testGraph.emplaceBlock("gr::testing::Copy<float32>", {});
        expect(eq(testGraph.blocks().size(), 1UZ));
        expect(eq(getNReplyMessages(fromGraph), 1UZ)); // emplaceBlock emits message
        consumeAllReplyMessages(fromGraph);
        expect(eq(getNReplyMessages(fromGraph), 0UZ)); // all messages are consumed

        "Remove a known block"_test = [&] {
            auto& temporaryBlock = testGraph.emplaceBlock("gr::testing::Copy<float32>", {});
            expect(eq(testGraph.blocks().size(), 2UZ));
            expect(eq(getNReplyMessages(fromGraph), 1UZ)); // emplaceBlock emits message
            consumeAllReplyMessages(fromGraph);
            expect(eq(getNReplyMessages(fromGraph), 0UZ)); // all messages are consumed

            sendMessage<Set>(toGraph, testGraph.unique_name, graph::property::kRemoveBlock /* endpoint */, //
                {{"uniqueName", std::string(temporaryBlock.uniqueName())}} /* data */);
            expect(nothrow([&] { testGraph.processScheduledMessages(); })) << "manually execute processing of messages";

            expect(eq(getNReplyMessages(fromGraph), 1UZ));
            const Message reply = getAndConsumeFirstReplyMessage(fromGraph);
            if (!reply.data.has_value()) {
                expect(false) << fmt::format("reply.data has no value:{}\n", reply.data.error());
            }
            expect(eq(testGraph.blocks().size(), 1UZ));
        };

        "Remove an unknown block"_test = [&] {
            sendMessage<Set>(toGraph, testGraph.unique_name, graph::property::kRemoveBlock /* endpoint */, //
                {{"uniqueName", "this_block_is_unknown"}} /* data */);
            expect(nothrow([&] { testGraph.processScheduledMessages(); })) << "manually execute processing of messages";

            expect(eq(getNReplyMessages(fromGraph), 1UZ));
            const Message reply = getAndConsumeFirstReplyMessage(fromGraph);
            expect(!reply.data.has_value());
            expect(eq(testGraph.blocks().size(), 1UZ));
        };
    };

    "Block replacement tests"_test = [] {
        gr::MsgPortOut toGraph;
        gr::Graph      testGraph(context->loader);
        gr::MsgPortIn  fromGraph;

        expect(eq(ConnectionResult::SUCCESS, toGraph.connect(testGraph.msgIn)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.msgOut.connect(fromGraph)));

        auto& block = testGraph.emplaceBlock("gr::testing::Copy<float32>", {});
        expect(eq(testGraph.blocks().size(), 1UZ));
        expect(eq(getNReplyMessages(fromGraph), 1UZ)); // emplaceBlock emits message
        consumeAllReplyMessages(fromGraph);
        expect(eq(getNReplyMessages(fromGraph), 0UZ)); // all messages are consumed

        "Replace a known block"_test = [&] {
            auto& temporaryBlock = testGraph.emplaceBlock("gr::testing::Copy<float32>", {});
            expect(eq(testGraph.blocks().size(), 2UZ));
            expect(eq(getNReplyMessages(fromGraph), 1UZ)); // emplaceBlock emits message
            consumeAllReplyMessages(fromGraph);
            expect(eq(getNReplyMessages(fromGraph), 0UZ)); // all messages are consumed

            sendMessage<Set>(toGraph, testGraph.unique_name, graph::property::kReplaceBlock /* endpoint */, //
                {{"uniqueName", std::string(temporaryBlock.uniqueName())},                                  //
                    {"type", "gr::testing::Copy<float32>"}, {"properties", property_map{}}} /* data */);
            expect(nothrow([&] { testGraph.processScheduledMessages(); })) << "manually execute processing of messages";

            expect(eq(getNReplyMessages(fromGraph), 1UZ));
            const Message reply = getAndConsumeFirstReplyMessage(fromGraph);
            if (!reply.data.has_value()) {
                expect(false) << fmt::format("reply.data has no value:{}\n", reply.data.error());
            }
        };

        "Replace an unknown block"_test = [&] {
            sendMessage<Set>(toGraph, testGraph.unique_name, graph::property::kReplaceBlock /* endpoint */, //
                {{"uniqueName", "this_block_is_unknown"},                                                   //
                    {"type", "gr::testing::Copy<float32>"}, {"properties", property_map{}}} /* data */);
            expect(nothrow([&] { testGraph.processScheduledMessages(); })) << "manually execute processing of messages";

            expect(eq(getNReplyMessages(fromGraph), 1UZ));
            const Message reply = getAndConsumeFirstReplyMessage(fromGraph);

            expect(!reply.data.has_value());
        };

        "Replace with an unknown block"_test = [&] {
            sendMessage<Set>(toGraph, testGraph.unique_name, graph::property::kReplaceBlock /* endpoint */, //
                {{"uniqueName", std::string(block.uniqueName())},                                           //
                    {"type", "doesnt_exist::multiply<float32>"}, {"properties", property_map{}}} /* data */);
            expect(nothrow([&] { testGraph.processScheduledMessages(); })) << "manually execute processing of messages";

            expect(eq(getNReplyMessages(fromGraph), 1UZ));
            const Message reply = getAndConsumeFirstReplyMessage(fromGraph);

            expect(!reply.data.has_value());
        };
    };

    "Edge addition tests"_test = [&] {
        gr::MsgPortOut toGraph;
        gr::Graph      testGraph(context->loader);
        gr::MsgPortIn  fromGraph;

        expect(eq(ConnectionResult::SUCCESS, toGraph.connect(testGraph.msgIn)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.msgOut.connect(fromGraph)));

        auto& blockOut       = testGraph.emplaceBlock("gr::testing::Copy<float32>", {});
        auto& blockIn        = testGraph.emplaceBlock("gr::testing::Copy<float32>", {});
        auto& blockWrongType = testGraph.emplaceBlock("gr::testing::Copy<float64>", {});

        expect(eq(getNReplyMessages(fromGraph), 3UZ)); // emplaceBlock emits message
        consumeAllReplyMessages(fromGraph);
        expect(eq(getNReplyMessages(fromGraph), 0UZ)); // all messages are consumed

        "Add an edge"_test = [&] {
            property_map data = {{"sourceBlock", std::string(blockOut.uniqueName())}, {"sourcePort", "out"}, //
                {"destinationBlock", std::string(blockIn.uniqueName())}, {"destinationPort", "in"},          //
                {"minBufferSize", gr::Size_t()}, {"weight", 0}, {"edgeName", "unnamed edge"}};

            sendMessage<Set>(toGraph, testGraph.unique_name, graph::property::kEmplaceEdge /* endpoint */, data /* data */);
            expect(nothrow([&] { testGraph.processScheduledMessages(); })) << "manually execute processing of messages";

            expect(eq(getNReplyMessages(fromGraph), 1UZ));
            const Message reply = getAndConsumeFirstReplyMessage(fromGraph);
            if (!reply.data.has_value()) {
                expect(false) << fmt::format("edge not being placed - error: {}", reply.data.error());
            }
        };

        "Fail to add an edge because source port is invalid"_test = [&] {
            sendMessage<Set>(toGraph, testGraph.unique_name, graph::property::kEmplaceEdge /* endpoint */, //
                {{"sourceBlock", std::string(blockOut.uniqueName())}, {"sourcePort", "OUTPUT"},            //
                    {"destinationBlock", std::string(blockIn.uniqueName())}, {"destinationPort", "in"}} /* data */);
            expect(nothrow([&] { testGraph.processScheduledMessages(); })) << "manually execute processing of messages";

            expect(eq(getNReplyMessages(fromGraph), 1UZ));
            const Message reply = getAndConsumeFirstReplyMessage(fromGraph);
            expect(!reply.data.has_value());
        };

        "Fail to add an edge because destination port is invalid"_test = [&] {
            sendMessage<Set>(toGraph, testGraph.unique_name, graph::property::kEmplaceEdge /* endpoint */, //
                {{"sourceBlock", std::string(blockOut.uniqueName())}, {"sourcePort", "in"},                //
                    {"destinationBlock", std::string(blockIn.uniqueName())}, {"destinationPort", "INPUT"}} /* data */);
            expect(nothrow([&] { testGraph.processScheduledMessages(); })) << "manually execute processing of messages";

            expect(eq(getNReplyMessages(fromGraph), 1UZ));
            const Message reply = getAndConsumeFirstReplyMessage(fromGraph);
            expect(!reply.data.has_value());
        };

        "Fail to add an edge because ports are not compatible"_test = [&] {
            sendMessage<Set>(toGraph, testGraph.unique_name, graph::property::kEmplaceEdge /* endpoint */, //
                {{"sourceBlock", std::string(blockOut.uniqueName())}, {"sourcePort", "out"},               //
                    {"destinationBlock", std::string(blockWrongType.uniqueName())}, {"destinationPort", "in"}} /* data */);
            expect(nothrow([&] { testGraph.processScheduledMessages(); })) << "manually execute processing of messages";

            expect(eq(getNReplyMessages(fromGraph), 1UZ));
            const Message reply = getAndConsumeFirstReplyMessage(fromGraph);
            expect(!reply.data.has_value());
        };
    };

    "BlockRegistry tests"_test = [] {
        gr::MsgPortOut toGraph;
        gr::Graph      testGraph(context->loader);
        gr::MsgPortIn  fromGraph;

        expect(eq(ConnectionResult::SUCCESS, toGraph.connect(testGraph.msgIn)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.msgOut.connect(fromGraph)));

        "Get available block types"_test = [&] {
            sendMessage<Get>(toGraph, testGraph.unique_name, graph::property::kRegistryBlockTypes /* endpoint */, {} /* data */);
            expect(nothrow([&] { testGraph.processScheduledMessages(); })) << "manually execute processing of messages";

            expect(eq(getNReplyMessages(fromGraph), 1UZ));
            const Message reply = getAndConsumeFirstReplyMessage(fromGraph);

            if (reply.data.has_value()) {
                const auto& dataMap    = reply.data.value();
                auto        foundTypes = dataMap.find("types");
                if (foundTypes != dataMap.end() || !std::holds_alternative<std::vector<std::string>>(foundTypes->second)) {
                    PluginLoader& loader             = context->loader;
                    auto          expectedBlockTypes = loader.knownBlocks();
                    std::ranges::sort(expectedBlockTypes);
                    auto blockTypes = std::get<std::vector<std::string>>(foundTypes->second);
                    std::ranges::sort(blockTypes);
                    expect(eq(expectedBlockTypes, blockTypes));
                } else {
                    expect(false) << "`types` key not found or data type is not a `property_map`";
                }
            } else {
                expect(false) << fmt::format("data has no value - error: {}", reply.data.error());
            }
        };
    };

    "GRC tests"_test = [] {
        gr::MsgPortOut toGraph;
        gr::Graph      testGraph(context->loader);
        gr::MsgPortIn  fromGraph;

        expect(eq(ConnectionResult::SUCCESS, toGraph.connect(testGraph.msgIn)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.msgOut.connect(fromGraph)));

        testGraph.emplaceBlock("gr::testing::Copy<float32>", {});
        testGraph.emplaceBlock("gr::testing::Copy<float32>", {});

        expect(eq(getNReplyMessages(fromGraph), 2UZ)); // consume 2 messages from emplace
        consumeAllReplyMessages(fromGraph);
        expect(eq(getNReplyMessages(fromGraph), 0UZ));

        sendMessage<Get>(toGraph, testGraph.unique_name, graph::property::kGraphGRC, {});
        expect(nothrow([&] { testGraph.processScheduledMessages(); })) << "manually execute processing of messages";

        expect(eq(getNReplyMessages(fromGraph), 1UZ));
        const Message reply = getAndConsumeFirstReplyMessage(fromGraph);

        expect(reply.data.has_value()) << "Reply should contain data";
        if (reply.data.has_value()) {
            const auto& data = reply.data.value();
            expect(data.contains("value")) << "Reply should contain 'value' field";
            const auto& yaml = std::get<std::string>(data.at("value"));
            expect(!yaml.empty()) << "YAML string should not be empty";
            fmt::println("YAML content:\n{}", yaml);

            // verify well formed by loading from yaml
            auto graphFromYaml = gr::loadGrc(context->loader, yaml);
            expect(eq(graphFromYaml.blocks().size(), 2UZ)) << fmt::format("Expected 2 blocks in loaded graph, got {} blocks", graphFromYaml.blocks().size());

            "Set GRC YAML"_test = [&] {
                sendMessage<Set>(toGraph, testGraph.unique_name, graph::property::kGraphGRC, {{"value", yaml}});
                expect(nothrow([&] { testGraph.processScheduledMessages(); })) << "manually execute processing of messages";
                expect(eq(testGraph.blocks().size(), 2UZ)) << "Expected 2 blocks after reloading GRC";
            };
        }
    };
};

const boost::ut::suite RunningGraphTests = [] {
    using namespace std::string_literals;
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;
    using enum gr::message::Command;

    gr::scheduler::Simple scheduler{gr::Graph(context->loader)};

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
    auto multiply1 = sendAndWaitMessageEmplaceBlock(toGraph, fromGraph, "gr::testing::Copy<float32>"s, property_map{});
    auto multiply2 = sendAndWaitMessageEmplaceBlock(toGraph, fromGraph, "gr::testing::Copy<float32>"s, property_map{});
    scheduler.processScheduledMessages();

    for (const auto& block : scheduler.graph().blocks()) {
        fmt::println("block in list: {} - state() : {}", block->name(), magic_enum::enum_name(block->state()));
    }
    expect(eq(scheduler.graph().blocks().size(), 4UZ)) << "should contain sink->multiply1->multiply2->sink";

    sendAndWaitMessageEmplaceEdge(toGraph, fromGraph, source.unique_name, "out", multiply1, "in");
    sendAndWaitMessageEmplaceEdge(toGraph, fromGraph, multiply1, "out", multiply2, "in");
    sendAndWaitMessageEmplaceEdge(toGraph, fromGraph, multiply2, "out", sink.unique_name, "in");
    expect(eq(getNReplyMessages(fromGraph), 0UZ));
    scheduler.processScheduledMessages();

    // Get the whole graph
    {
        sendMessage<Set>(toGraph, "" /* serviceName */, graph::property::kGraphInspect /* endpoint */, property_map{} /* data */);
        if (!waitForReply(fromGraph)) {
            expect(false) << "Reply message not received for kGraphInspect.";
        }

        expect(eq(getNReplyMessages(fromGraph), 1UZ));
        const Message reply = getAndConsumeFirstReplyMessage(fromGraph);
        expect(eq(getNReplyMessages(fromGraph), 0UZ));
        if (!reply.data.has_value()) {
            expect(false) << fmt::format("reply.data has no value:{}\n", reply.data.error());
        }

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

int main() { /* tests are statically executed */ }
