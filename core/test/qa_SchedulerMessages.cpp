#include <boost/ut.hpp>

#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>

#include <GrBasicBlocks.hpp>
#include <GrTestingBlocks.hpp>

#include "TestBlockRegistryContext.hpp"

#include "magic_enum.hpp"
#include "message_utils.hpp"
#include "utils.hpp"

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

class TestScheduler {
    using TScheduler = gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::singleThreaded>;
    std::future<std::expected<void, gr::Error>> schedulerRet_;

    gr::Graph& withTestingSourceAndSink(gr::Graph& graph) const noexcept {
        graph.emplaceBlock<gr::testing::SlowSource<float>>();
        graph.emplaceBlock<gr::testing::CountingSink<float>>();
        return graph;
    }

public:
    TScheduler     scheduler_;
    gr::MsgPortOut toScheduler;
    gr::MsgPortIn  fromScheduler;

    TestScheduler(gr::Graph graph, bool addTestSourceAndSink = true) : scheduler_(addTestSourceAndSink ? std::move(withTestingSourceAndSink(graph)) : std::move(graph)) {
        using namespace gr::testing;
        expect(eq(ConnectionResult::SUCCESS, toScheduler.connect(scheduler_.msgIn)));
        expect(eq(ConnectionResult::SUCCESS, scheduler_.msgOut.connect(fromScheduler)));

        run();
    }

    ~TestScheduler() { stop(); }

    TestScheduler(const TestScheduler&)            = delete;
    TestScheduler& operator=(const TestScheduler&) = delete;
    TestScheduler(TestScheduler&&)                 = delete;
    TestScheduler& operator=(TestScheduler&&)      = delete;

    void run() {
        schedulerRet_ = gr::testing::thread_pool::executeScheduler("qa_SchMess::scheduler", scheduler_);

        using namespace boost::ut;
        // Wait for the scheduler to start running
        expect(gr::testing::awaitCondition(1s, [this] { return scheduler_.state() == gr::lifecycle::State::RUNNING; })) << "scheduler thread up and running w/ timeout";
        expect(scheduler_.state() == gr::lifecycle::State::RUNNING) << "scheduler thread up and running";
    }

    void stop() {
        using namespace boost::ut;
        scheduler_.requestStop();

        auto result = schedulerRet_.get(); // this joins the thread
        if (!result.has_value()) {
            expect(false) << std::format("scheduler.runAndWait() failed:\n{}\n", result.error());
        }
    }

    auto&                          scheduler() { return scheduler_; }
    auto&                          scheduler() const { return scheduler_; }
    auto&                          msgIn() { return scheduler_.msgIn; }
    auto&                          msgOut() { return scheduler_.msgOut; }
    auto&                          graph() { return scheduler_.graph(); }
    auto                           state() const { return scheduler_.state(); }
    const std::string&             unique_name() const { return scheduler_.unique_name; }
    void                           processScheduledMessages() { scheduler_.processScheduledMessages(); }
    std::expected<void, gr::Error> changeStateTo(gr::lifecycle::State state) { return scheduler_.changeStateTo(state); }
};

const boost::ut::suite TopologyGraphTests = [] {
    using namespace std::string_literals;
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;
    using namespace gr::test;
    using enum gr::message::Command;

    expect(fatal(gt(context->registry.keys().size(), 0UZ))) << "didn't register any blocks";

    "Block addition tests"_test = [] {
        TestScheduler scheduler(gr::Graph(context->loader));

        "Add a valid block"_test = [&] {
            sendMessage<Set>(scheduler.toScheduler, scheduler.scheduler_.unique_name, scheduler::property::kEmplaceBlock /* endpoint */, //
                {{"type", "gr::testing::Copy<float32>"}, {"properties", property_map{}}} /* data */);

            waitForReply(scheduler.fromScheduler);

            expect(eq(getNReplyMessages(scheduler.fromScheduler), 1UZ));
            const Message reply = getAndConsumeFirstReplyMessage(scheduler.fromScheduler, scheduler::property::kBlockEmplaced);
            if (!reply.data.has_value()) {
                expect(false) << std::format("reply.data has no value:{}\n", reply.data.error());
            }
            expect(eq(scheduler.graph().blocks().size(), 3UZ));
        };

        "Add an invalid block"_test = [&] {
            sendMessage<Set>(scheduler.toScheduler, scheduler.scheduler_.unique_name, scheduler::property::kEmplaceBlock /* endpoint */, //
                {{"type", "doesnt_exist::multiply<float32>"}, {"properties", property_map{}}} /* data */);

            waitForReply(scheduler.fromScheduler);

            expect(eq(getNReplyMessages(scheduler.fromScheduler), 1UZ));

            const Message reply = getAndConsumeFirstReplyMessage(scheduler.fromScheduler);
            expect(eq(getNReplyMessages(scheduler.fromScheduler), 0UZ));
            expect(!reply.data.has_value());
            expect(eq(scheduler.graph().blocks().size(), 3UZ));
        };
    };

    "add block while scheduler is running"_test = [] {
        using namespace gr;
        using namespace gr::testing;

        Graph flow(context->loader);
        auto& source = flow.emplaceBlock<NullSource<float>>();
        auto& sink   = flow.emplaceBlock<NullSink<float>>();
        expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"out">(source).to<"in">(sink)));

        TestScheduler scheduler(std::move(flow));

        MsgPortOut toScheduler;
        MsgPortIn  fromScheduler;
        expect(eq(ConnectionResult::SUCCESS, toScheduler.connect(scheduler.msgIn())));
        expect(eq(ConnectionResult::SUCCESS, scheduler.msgOut().connect(fromScheduler)));

        expect(awaitCondition(2s, [&scheduler] { return scheduler.state() == lifecycle::State::RUNNING; })) << "scheduler thread up and running w/ timeout";

        expect(scheduler.state() == lifecycle::State::RUNNING) << "scheduler is running";

        auto initialBlockCount = scheduler.graph().blocks().size();
        std::println("Initial block count: {}", initialBlockCount);

        for (const auto& block : gr::globalBlockRegistry().keys()) {
            std::println("Block {}", block);
        }

        sendMessage<message::Command::Set>(toScheduler, scheduler.unique_name(), scheduler::property::kEmplaceBlock, property_map{{"type", std::string("builtin_counter<float32>")}, {"properties", property_map{{"disconnect_on_done", false}}}});
        gr::testing::waitForReply(fromScheduler);

        auto messages = fromScheduler.streamReader().get();
        expect(gt(messages.size(), 0UZ)) << "received block emplaced message";
        auto message = messages[0];

        std::println("Got a message {}", message);
        expect(eq(message.endpoint, scheduler::property::kBlockEmplaced)) << "correct message endpoint";
        expect(message.data.has_value()) << "message has data";
        auto consumed = messages.consume(1UZ);
        expect(consumed) << "failed to consume message";

        expect(awaitCondition(2s, [&scheduler, initialBlockCount] { return scheduler.graph().blocks().size() > initialBlockCount; })) << "waiting for block to be added to graph";

        auto finalBlockCount = scheduler.graph().blocks().size();
        std::println("Final block count: {}", finalBlockCount);
        expect(eq(finalBlockCount, initialBlockCount + 1)) << "block was added";

        expect(awaitCondition(2s, [&scheduler] {
            for (const auto& block : scheduler.graph().blocks()) {
                if (block->name() == "builtin_counter<float32>" && block->state() == lifecycle::State::RUNNING) {
                    return true;
                }
            }
            return false;
        })) << "waiting for new block to reach running state";
    };

    "Block removal tests"_test = [] {
        gr::Graph graph(context->loader);
        graph.emplaceBlock("gr::testing::Copy<float32>", {});
        auto& temporaryBlock = graph.emplaceBlock("gr::testing::Copy<float32>", {});

        TestScheduler scheduler(std::move(graph));
        const auto&   testGraph = scheduler.graph();
        expect(eq(testGraph.blocks().size(), 4UZ));
        // expect(eq(getNReplyMessages(fromScheduler), 1UZ)); // emplaceBlock emits message
        consumeAllReplyMessages(scheduler.fromScheduler);
        expect(eq(getNReplyMessages(scheduler.fromScheduler), 0UZ)); // all messages are consumed

        "Remove a known block"_test = [&] {
            expect(eq(testGraph.blocks().size(), 4UZ));
            // expect(eq(getNReplyMessages(fromScheduler), 1UZ)); // emplaceBlock emits message
            consumeAllReplyMessages(scheduler.fromScheduler);
            expect(eq(getNReplyMessages(scheduler.fromScheduler), 0UZ)); // all messages are consumed

            sendMessage<Set>(scheduler.toScheduler, scheduler.scheduler_.unique_name, scheduler::property::kRemoveBlock /* endpoint */, //
                {{"uniqueName", std::string(temporaryBlock->uniqueName())}} /* data */);

            waitForReply(scheduler.fromScheduler);

            expect(eq(getNReplyMessages(scheduler.fromScheduler), 1UZ));
            const Message reply = getAndConsumeFirstReplyMessage(scheduler.fromScheduler);
            if (!reply.data.has_value()) {
                expect(false) << std::format("reply.data has no value:{}\n", reply.data.error());
            }
            expect(eq(testGraph.blocks().size(), 3UZ));
        };

        "Remove an unknown block"_test = [&] {
            sendMessage<Set>(scheduler.toScheduler, scheduler.scheduler_.unique_name, scheduler::property::kRemoveBlock /* endpoint */, //
                {{"uniqueName", "this_block_is_unknown"}} /* data */);
            waitForReply(scheduler.fromScheduler);

            expect(eq(getNReplyMessages(scheduler.fromScheduler), 1UZ));
            const Message reply = getAndConsumeFirstReplyMessage(scheduler.fromScheduler);
            expect(!reply.data.has_value());
            expect(eq(testGraph.blocks().size(), 3UZ));
        };
    };

    "Block replacement tests"_test = [] {
        gr::Graph graph(context->loader);

        auto& block = graph.emplaceBlock("gr::testing::Copy<float32>", {});
        expect(eq(graph.blocks().size(), 1UZ));
        auto& temporaryBlock = graph.emplaceBlock("gr::testing::Copy<float32>", {});

        TestScheduler scheduler(std::move(graph));
        "Replace a known block"_test = [&] {
            expect(eq(scheduler.graph().blocks().size(), 4UZ));

            sendMessage<Set>(scheduler.toScheduler, scheduler.scheduler_.unique_name, scheduler::property::kReplaceBlock /* endpoint */, //
                {{"uniqueName", std::string(temporaryBlock->uniqueName())},                                                              //
                    {"type", "gr::testing::Copy<float32>"}, {"properties", property_map{}}} /* data */);
            waitForReply(scheduler.fromScheduler);

            expect(eq(getNReplyMessages(scheduler.fromScheduler), 1UZ));
            const Message reply = getAndConsumeFirstReplyMessage(scheduler.fromScheduler);
            if (!reply.data.has_value()) {
                expect(false) << std::format("reply.data has no value:{}\n", reply.data.error());
            }
        };

        "Replace an unknown block"_test = [&] {
            sendMessage<Set>(scheduler.toScheduler, scheduler.scheduler_.unique_name, scheduler::property::kReplaceBlock /* endpoint */, //
                {{"uniqueName", "this_block_is_unknown"},                                                                                //
                    {"type", "gr::testing::Copy<float32>"}, {"properties", property_map{}}} /* data */);
            waitForReply(scheduler.fromScheduler);

            expect(eq(getNReplyMessages(scheduler.fromScheduler), 1UZ));
            const Message reply = getAndConsumeFirstReplyMessage(scheduler.fromScheduler);

            expect(!reply.data.has_value());
        };

        "Replace with an unknown block"_test = [&] {
            sendMessage<Set>(scheduler.toScheduler, scheduler.scheduler_.unique_name, scheduler::property::kReplaceBlock /* endpoint */, //
                {{"uniqueName", std::string(block->uniqueName())},                                                                       //
                    {"type", "doesnt_exist::multiply<float32>"}, {"properties", property_map{}}} /* data */);
            waitForReply(scheduler.fromScheduler);

            expect(eq(getNReplyMessages(scheduler.fromScheduler), 1UZ));
            const Message reply = getAndConsumeFirstReplyMessage(scheduler.fromScheduler);

            expect(!reply.data.has_value());
        };
    };

    "Edge addition tests"_test = [&] {
        gr::Graph testGraph(context->loader);

        auto& blockOut       = testGraph.emplaceBlock("gr::testing::Copy<float32>", {});
        auto& blockIn        = testGraph.emplaceBlock("gr::testing::Copy<float32>", {});
        auto& blockWrongType = testGraph.emplaceBlock("gr::testing::Copy<float64>", {});

        TestScheduler scheduler(std::move(testGraph));

        "Add an edge"_test = [&] {
            property_map data = {{"sourceBlock", std::string(blockOut->uniqueName())}, {"sourcePort", "out"}, //
                {"destinationBlock", std::string(blockIn->uniqueName())}, {"destinationPort", "in"},          //
                {"minBufferSize", gr::Size_t()}, {"weight", 0}, {"edgeName", "unnamed edge"}};

            sendMessage<Set>(scheduler.toScheduler, scheduler.scheduler_.unique_name, scheduler::property::kEmplaceEdge /* endpoint */, data /* data */);
            waitForReply(scheduler.fromScheduler);

            expect(eq(getNReplyMessages(scheduler.fromScheduler), 1UZ));
            const Message reply = getAndConsumeFirstReplyMessage(scheduler.fromScheduler);
            if (!reply.data.has_value()) {
                expect(false) << std::format("edge not being placed - error: {}", reply.data.error());
            }
        };

        "Fail to add an edge because source port is invalid"_test = [&] {
            sendMessage<Set>(scheduler.toScheduler, scheduler.scheduler_.unique_name, scheduler::property::kEmplaceEdge /* endpoint */, //
                {{"sourceBlock", std::string(blockOut->uniqueName())}, {"sourcePort", "OUTPUT"},                                        //
                    {"destinationBlock", std::string(blockIn->uniqueName())}, {"destinationPort", "in"}} /* data */);
            waitForReply(scheduler.fromScheduler);

            expect(eq(getNReplyMessages(scheduler.fromScheduler), 1UZ));
            const Message reply = getAndConsumeFirstReplyMessage(scheduler.fromScheduler);
            expect(!reply.data.has_value());
        };

        "Fail to add an edge because destination port is invalid"_test = [&] {
            sendMessage<Set>(scheduler.toScheduler, scheduler.scheduler_.unique_name, scheduler::property::kEmplaceEdge /* endpoint */, //
                {{"sourceBlock", std::string(blockOut->uniqueName())}, {"sourcePort", "in"},                                            //
                    {"destinationBlock", std::string(blockIn->uniqueName())}, {"destinationPort", "INPUT"}} /* data */);
            waitForReply(scheduler.fromScheduler);

            expect(eq(getNReplyMessages(scheduler.fromScheduler), 1UZ));
            const Message reply = getAndConsumeFirstReplyMessage(scheduler.fromScheduler);
            expect(!reply.data.has_value());
        };

        "Fail to add an edge because ports are not compatible"_test = [&] {
            sendMessage<Set>(scheduler.toScheduler, scheduler.scheduler_.unique_name, scheduler::property::kEmplaceEdge /* endpoint */, //
                {{"sourceBlock", std::string(blockOut->uniqueName())}, {"sourcePort", "out"},                                           //
                    {"destinationBlock", std::string(blockWrongType->uniqueName())}, {"destinationPort", "in"}} /* data */);
            waitForReply(scheduler.fromScheduler);

            expect(eq(getNReplyMessages(scheduler.fromScheduler), 1UZ));
            const Message reply = getAndConsumeFirstReplyMessage(scheduler.fromScheduler);
            expect(!reply.data.has_value());
        };
    };

    "Settings change via messages"_test = [] {
        gr::Graph testGraph(context->loader);
        testGraph.emplaceBlock("gr::testing::Copy<float32>", {});
        testGraph.emplaceBlock("gr::testing::Copy<float32>", {});

        TestScheduler scheduler(std::move(testGraph));

        "get scheduler settings"_test = [&] {
            sendMessage<Get>(scheduler.toScheduler, "" /* serviceName */, block::property::kSetting /* endpoint */, {} /* data  */);
            waitForReply(scheduler.fromScheduler);

            bool        atLeastOneReplyFromScheduler = false;
            std::size_t availableMessages            = scheduler.fromScheduler.streamReader().available();
            expect(ge(availableMessages, 1UZ)) << "didn't receive reply message";
            for (const auto& reply : returnReplyMsgs(scheduler.fromScheduler)) {
                if (reply.serviceName != scheduler.scheduler().unique_name) {
                    continue;
                }

                std::println("Got reply: {}", reply);
                expect(reply.cmd == Final) << std::format("mismatch between reply.cmd = {} and expected {} command", reply.cmd, Final);
                expect(eq(reply.endpoint, std::string(block::property::kSetting)));
                expect(reply.data.has_value());
                expect(!reply.data.value().empty());
                expect(reply.data.value().contains("timeout_ms"));
                atLeastOneReplyFromScheduler = true;
            }

            expect(atLeastOneReplyFromScheduler);
        };

        "set scheduler settings"_test = [&] {
            sendMessage<Set>(scheduler.toScheduler, "" /* serviceName */, block::property::kStagedSetting /* endpoint */, {{"timeout_ms", 42}} /* data  */);
            waitForReply(scheduler.fromScheduler);

            bool        atLeastOneReplyFromScheduler = false;
            std::size_t availableMessages            = scheduler.fromScheduler.streamReader().available();
            expect(ge(availableMessages, 1UZ)) << "didn't receive reply message";
            for (const auto& reply : returnReplyMsgs(scheduler.fromScheduler)) {
                if (reply.serviceName != scheduler.scheduler().unique_name) {
                    continue;
                }
                atLeastOneReplyFromScheduler = true;
            }
            expect(!atLeastOneReplyFromScheduler) << "should not receive a reply";
            property_map stagedSettings = scheduler.scheduler().settings().stagedParameters();
            expect(stagedSettings.contains("timeout_ms"));
            expect(eq(42UZ, std::get<gr::Size_t>(stagedSettings.at("timeout_ms"))));

            // setting staged setting via staged setting (N.B. non-real-time <-> real-time setting decoupling
            sendMessage<Set>(scheduler.toScheduler, "" /* serviceName */, block::property::kSetting /* endpoint */, {{"timeout_ms", 43}} /* data  */);
            waitForReply(scheduler.fromScheduler);

            stagedSettings = scheduler.scheduler().settings().stagedParameters();
            expect(stagedSettings.contains("timeout_ms"));
            expect(eq(43UZ, std::get<gr::Size_t>(stagedSettings.at("timeout_ms"))));
            sendMessage<Set>(scheduler.toScheduler, "" /* serviceName */, block::property::kSetting /* endpoint */, {{"timeout_ms", 43}} /* data  */);
        };
    };

    "Get GRC Yaml tests"_test = [] {
        gr::Graph testGraph(context->loader);
        testGraph.emplaceBlock("gr::testing::Copy<float32>", {});
        testGraph.emplaceBlock("gr::testing::Copy<float32>", {});

        TestScheduler scheduler(std::move(testGraph));

        sendMessage<Get>(scheduler.toScheduler, scheduler.scheduler_.unique_name, scheduler::property::kGraphGRC, {});
        waitForReply(scheduler.fromScheduler);

        expect(eq(getNReplyMessages(scheduler.fromScheduler), 1UZ));
        const Message reply = getAndConsumeFirstReplyMessage(scheduler.fromScheduler);

        expect(reply.data.has_value()) << "Reply should contain data";
        if (reply.data.has_value()) {
            const auto& data = reply.data.value();
            expect(data.contains("value")) << "Reply should contain 'value' field";
            const auto& yaml = std::get<std::string>(data.at("value"));
            expect(!yaml.empty()) << "YAML string should not be empty";
            std::println("YAML content:\n{}", yaml);

            // verify well formed by loading from yaml
            auto graphFromYaml = gr::loadGrc(context->loader, yaml);
            expect(eq(graphFromYaml.blocks().size(), 4UZ)) << std::format("Expected 4 blocks in loaded graph, got {} blocks", graphFromYaml.blocks().size());

            // "Set GRC YAML"_test = [&] {
            //     sendMessage<Set>(toGraph, scheduler.scheduler_.unique_name, scheduler::property::kGraphGRC, {{"value", yaml}});
            //     expect(eq(testGraph.blocks().size(), 2UZ)) << "Expected 2 blocks after reloading GRC";
            // };
        }
    };
};

/// old tests, from the time graph handled messages. They're still good
const boost::ut::suite MoreTopologyGraphTests = [] {
    using namespace std::string_literals;
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;
    using enum gr::message::Command;

    gr::Graph graph(context->loader);
    auto&     source = graph.emplaceBlock<SlowSource<float>>();
    auto&     sink   = graph.emplaceBlock<CountingSink<float>>();
    expect(eq(ConnectionResult::SUCCESS, graph.connect<"out">(source).to<"in">(sink)));
    expect(eq(graph.edges().size(), 1UZ)) << "edge registered with connect";

    TestScheduler  scheduler(std::move(graph), /*addTestSourceAndSink=*/false);
    gr::MsgPortOut toScheduler;
    gr::MsgPortIn  fromScheduler;
    expect(eq(ConnectionResult::SUCCESS, toScheduler.connect(scheduler.msgIn())));
    expect(eq(ConnectionResult::SUCCESS, scheduler.msgOut().connect(fromScheduler)));

    expect(awaitCondition(1s, [&scheduler] { return scheduler.state() == lifecycle::State::RUNNING; })) << "scheduler thread up and running w/ timeout";
    expect(scheduler.state() == lifecycle::State::RUNNING) << "scheduler thread up and running";
    expect(eq(scheduler.graph().edges().size(), 1UZ)) << "added one edge";

    expect(awaitCondition(1s, [&sink] { return sink.count >= 10U; })) << "sink received enough data";
    std::println("executed basic graph");

    // Adding a few blocks
    auto multiply1 = sendAndWaitMessageEmplaceBlock(toScheduler, fromScheduler, "gr::testing::Copy<float32>"s, property_map{});
    auto multiply2 = sendAndWaitMessageEmplaceBlock(toScheduler, fromScheduler, "gr::testing::Copy<float32>"s, property_map{});
    scheduler.processScheduledMessages();

    for (const auto& block : scheduler.graph().blocks()) {
        std::println("block in list: {} - state() : {}", block->name(), magic_enum::enum_name(block->state()));
    }
    expect(eq(scheduler.graph().blocks().size(), 4UZ)) << "should contain sink->multiply1->multiply2->sink";

    sendAndWaitMessageEmplaceEdge(toScheduler, fromScheduler, source.unique_name, "out", multiply1, "in");
    sendAndWaitMessageEmplaceEdge(toScheduler, fromScheduler, multiply1, "out", multiply2, "in");
    sendAndWaitMessageEmplaceEdge(toScheduler, fromScheduler, multiply2, "out", sink.unique_name, "in");
    expect(eq(getNReplyMessages(fromScheduler), 0UZ));
    scheduler.processScheduledMessages();

    // Get the whole graph
    {
        sendMessage<Set>(toScheduler, "" /* serviceName */, graph::property::kGraphInspect /* endpoint */, property_map{} /* data */);
        if (!waitForReply(fromScheduler)) {
            expect(false) << "Reply message not received for kGraphInspect.";
        }

        expect(eq(getNReplyMessages(fromScheduler), 1UZ));
        const Message reply = getAndConsumeFirstReplyMessage(fromScheduler);
        expect(eq(getNReplyMessages(fromScheduler), 0UZ));
        if (!reply.data.has_value()) {
            expect(false) << std::format("reply.data has no value:{}\n", reply.data.error());
        }

        const auto& data     = reply.data.value();
        const auto& children = std::get<property_map>(data.at("children"s));
        expect(eq(children.size(), 4UZ));

        const auto& edges = std::get<property_map>(data.at("edges"s));
        expect(eq(edges.size(), 4UZ));
    }
    scheduler.processScheduledMessages();

    scheduler.stop();

    // return to initial state
    expect(scheduler.changeStateTo(lifecycle::State::INITIALISED).has_value()) << "could switch to INITIALISED?";
    expect(awaitCondition(1s, [&scheduler] { return scheduler.state() == lifecycle::State::INITIALISED; })) << "scheduler INITIALISED w/ timeout";
    expect(scheduler.state() == lifecycle::State::INITIALISED) << std::format("scheduler INITIALISED - actual: {}\n", magic_enum::enum_name(scheduler.state()));

    scheduler.run();
    expect(awaitCondition(1s, [&scheduler] { return scheduler.state() == lifecycle::State::RUNNING; })) << "scheduler thread up and running w/ timeout";
    expect(scheduler.state() == lifecycle::State::RUNNING) << "scheduler thread up and running";

    for (const auto& edge : scheduler.graph().edges()) {
        std::println("edge in list({}): {}", scheduler.graph().edges().size(), edge);
    }
    expect(eq(scheduler.graph().edges().size(), 4UZ)) << "added three new edges, one previously registered with connect";

    // FIXME: edge->connection is not performed
    //    expect(awaitCondition(1s, [&sink] {
    //        std::this_thread::sleep_for(100ms);
    //        std::println("sink has received {} samples - parents: {}", sink.count, sink.in.buffer().streamBuffer.n_writers());
    //        return sink.count >= 10U;
    //    })) << "sink received enough data";

    std::print("Counting sink counted to {}\n", sink.count);
};

int main() { /* tests are statically executed */ }
