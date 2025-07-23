#include <boost/ut.hpp>

#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>

#include <GrBasicBlocks.hpp>
#include <GrTestingBlocks.hpp>

#include "TestBlockRegistryContext.hpp"

#include "magic_enum.hpp"
#include "message_utils.hpp"

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
        schedulerRet_ = gr::test::thread_pool::executeScheduler("qa_SchMess::scheduler", scheduler_);

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
    using namespace gr::scheduler;
    using namespace gr::testing;
    using namespace gr::test;
    using enum gr::message::Command;

    expect(fatal(gt(context->registry.keys().size(), 0UZ))) << "didn't register any blocks";

    "Block addition tests"_test = [] {
        TestScheduler scheduler(gr::Graph(context->loader));

        "Add a valid block"_test = [&] {
            testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kEmplaceBlock, //
                {{"type", "gr::testing::Copy<float32>"}, {"properties", property_map{}}},                                                                  //
                ReplyChecker{.expectedEndpoint = scheduler::property::kBlockEmplaced});

            expect(eq(scheduler.graph().blocks().size(), 3UZ));
        };

        "Add an invalid block"_test = [&] {
            testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kEmplaceBlock, //
                {{"type", "doesnt_exist::multiply<float32>"}, {"properties", property_map{}}},                                                             //
                ReplyChecker{.expectedEndpoint = scheduler::property::kEmplaceBlock, .expectedHasData = false});

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

        expect(awaitCondition(2s, [&scheduler] { return scheduler.state() == lifecycle::State::RUNNING; })) << "scheduler thread up and running w/ timeout";

        expect(scheduler.state() == lifecycle::State::RUNNING) << "scheduler is running";

        auto initialBlockCount = scheduler.graph().blocks().size();
        std::println("Initial block count: {}", initialBlockCount);

        for (const auto& block : gr::globalBlockRegistry().keys()) {
            std::println("Block {}", block);
        }

        testing::sendAndWaitForReply<message::Command::Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kEmplaceBlock, //
            property_map{{"type", std::string("builtin_counter<float32>")}, {"properties", property_map{{"disconnect_on_done", false}}}},                                //
            ReplyChecker{.expectedEndpoint = scheduler::property::kBlockEmplaced});

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

        "Remove a known block"_test = [&] {
            expect(eq(testGraph.blocks().size(), 4UZ));
            // expect(eq(getNReplyMessages(fromScheduler), 1UZ)); // emplaceBlock emits message
            consumeAllReplyMessages(scheduler.fromScheduler);

            testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kRemoveBlock, //
                {{"uniqueName", std::string(temporaryBlock->uniqueName())}}, ReplyChecker{.expectedEndpoint = scheduler::property::kBlockRemoved});

            expect(eq(testGraph.blocks().size(), 3UZ));
        };

        "Remove an unknown block"_test = [&] {
            testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kRemoveBlock, //
                {{"uniqueName", "this_block_is_unknown"}}, ReplyChecker{.expectedEndpoint = scheduler::property::kRemoveBlock, .expectedHasData = false});

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

            testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kReplaceBlock, //
                {{"uniqueName", std::string(temporaryBlock->uniqueName())},                                                                                //
                    {"type", "gr::testing::Copy<float32>"}, {"properties", property_map{}}},                                                               //
                ReplyChecker{.expectedEndpoint = scheduler::property::kBlockReplaced});
        };

        "Replace an unknown block"_test = [&] {
            testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kReplaceBlock, //
                {{"uniqueName", "this_block_is_unknown"},                                                                                                  //
                    {"type", "gr::testing::Copy<float32>"}, {"properties", property_map{}}},                                                               //
                ReplyChecker{.expectedEndpoint = scheduler::property::kReplaceBlock, .expectedHasData = false});
        };

        "Replace with an unknown block"_test = [&] {
            testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kReplaceBlock, //
                {{"uniqueName", std::string(block->uniqueName())},                                                                                         //
                    {"type", "doesnt_exist::multiply<float32>"}, {"properties", property_map{}}},
                ReplyChecker{.expectedEndpoint = scheduler::property::kReplaceBlock, .expectedHasData = false});
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

            testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kEmplaceEdge, data, //
                ReplyChecker{.expectedEndpoint = scheduler::property::kEdgeEmplaced});
        };

        "Fail to add an edge because source port is invalid"_test = [&] {
            testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kEmplaceEdge, //
                {{"sourceBlock", std::string(blockOut->uniqueName())}, {"sourcePort", "OUTPUT"},                                                          //
                    {"destinationBlock", std::string(blockIn->uniqueName())}, {"destinationPort", "in"}},
                ReplyChecker{.expectedEndpoint = scheduler::property::kEmplaceEdge, .expectedHasData = false});
        };

        "Fail to add an edge because destination port is invalid"_test = [&] {
            testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kEmplaceEdge, //
                {{"sourceBlock", std::string(blockOut->uniqueName())}, {"sourcePort", "in"},                                                              //
                    {"destinationBlock", std::string(blockIn->uniqueName())}, {"destinationPort", "INPUT"}},
                ReplyChecker{.expectedEndpoint = scheduler::property::kEmplaceEdge, .expectedHasData = false});
        };

        "Fail to add an edge because ports are not compatible"_test = [&] {
            testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kEmplaceEdge, //
                {{"sourceBlock", std::string(blockOut->uniqueName())}, {"sourcePort", "out"},                                                             //
                    {"destinationBlock", std::string(blockWrongType->uniqueName())}, {"destinationPort", "in"}},
                ReplyChecker{.expectedEndpoint = scheduler::property::kEmplaceEdge, .expectedHasData = false});
        };
    };

    "Settings change via messages"_test = [] {
        gr::Graph testGraph(context->loader);
        testGraph.emplaceBlock("gr::testing::Copy<float32>", {});
        testGraph.emplaceBlock("gr::testing::Copy<float32>", {});

        TestScheduler scheduler(std::move(testGraph));

        "get scheduler settings"_test = [&] {
            // TODO: Would like to port to sendAndWaitMessage, but it's logic is looking at the whole message
            // queue, and fails if there's scheduler messages. In the future the scheduler might
            // insert unrelated messages in the queue and this test will fail
            sendMessage<Get>(scheduler.toScheduler, "", block::property::kSetting, {});
            expect(waitForReply(scheduler.fromScheduler, ReplyChecker{.expectedEndpoint = block::property::kSetting}).has_value()) << "expected reply";

            bool        atLeastOneReplyFromScheduler = false;
            std::size_t availableMessages            = scheduler.fromScheduler.streamReader().available();
            expect(ge(availableMessages, 1UZ)) << "didn't receive reply message";
            for (const auto& reply : consumeAllReplyMessages(scheduler.fromScheduler)) {
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
            // See TODO from "get scheduler settings", same case
            sendMessage<Set>(scheduler.toScheduler, "", block::property::kStagedSetting, {{"timeout_ms", 42}});
            expect(waitForReply(scheduler.fromScheduler, ReplyChecker{.expectedEndpoint = block::property::kStagedSetting, .expectedHasData = false}).has_value()) << "expected reply";

            bool        atLeastOneReplyFromScheduler = false;
            std::size_t availableMessages            = scheduler.fromScheduler.streamReader().available();
            expect(ge(availableMessages, 1UZ)) << "didn't receive reply message";
            for (const auto& reply : consumeAllReplyMessages(scheduler.fromScheduler)) {
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
            testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, "", block::property::kSetting, {{"timeout_ms", 43}}, //
                ReplyChecker{.expectedEndpoint = block::property::kSetting, .expectedHasData = false});

            stagedSettings = scheduler.scheduler().settings().stagedParameters();
            expect(stagedSettings.contains("timeout_ms"));
            expect(eq(43UZ, std::get<gr::Size_t>(stagedSettings.at("timeout_ms"))));

            testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, "", block::property::kSetting, {{"timeout_ms", 43}}, //
                ReplyChecker{.expectedEndpoint = block::property::kSetting, .expectedHasData = false});
        };
    };

    "Get GRC Yaml tests"_test = [] {
        gr::Graph testGraph(context->loader);
        testGraph.emplaceBlock("gr::testing::Copy<float32>", {});
        testGraph.emplaceBlock("gr::testing::Copy<float32>", {});

        TestScheduler scheduler(std::move(testGraph));

        testing::sendAndWaitForReply<Get>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), //
            scheduler::property::kGraphGRC, {}, [](const Message& reply) {
                if (reply.endpoint == scheduler::property::kGraphGRC && reply.data.has_value()) {
                    const auto& data = reply.data.value();
                    expect(data.contains("value")) << "Reply should contain 'value' field";
                    const auto& yaml = std::get<std::string>(data.at("value"));
                    expect(!yaml.empty()) << "YAML string should not be empty";
                    std::println("YAML content:\n{}", yaml);

                    // verify well formed by loading from yaml
                    auto graphFromYaml = gr::loadGrc(context->loader, yaml);
                    expect(eq(graphFromYaml.blocks().size(), 4UZ)) << std::format("Expected 4 blocks in loaded graph, got {} blocks", graphFromYaml.blocks().size());

                    return true;
                }

                return false;
            });
    };

    "UI constraints setting test"_test = [] {
        gr::Graph testGraph(context->loader);
        auto&     copy1 = testGraph.emplaceBlock("gr::testing::Copy<float32>", {});
        auto&     copy2 = testGraph.emplaceBlock("gr::testing::Copy<float32>", {});

        TestScheduler scheduler(std::move(testGraph));
        auto          makeUiConstraints = [](float x, float y) { return gr::property_map{{"x", x}, {"y", y}}; };

        {
            // Setting ui_constraints property for all blocks, universal
            sendMessage<Set>(scheduler.toScheduler, "" /* serviceName */, block::property::kSetting /* endpoint */, {{"ui_constraints", makeUiConstraints(43, 7070)}} /* data  */);
            // Setting ui_constraints property for one block
            sendMessage<Set>(scheduler.toScheduler, copy1->uniqueName() /* serviceName */, block::property::kSetting /* endpoint */, {{"ui_constraints", makeUiConstraints(42, 6)}} /* data  */);

            waitForReply(scheduler.fromScheduler);
        }

        const auto replyCount = scheduler.fromScheduler.streamReader().available();
        for (std::size_t replyIndex = 0UZ; replyIndex < replyCount; replyIndex++) {
            const Message reply = getAndConsumeFirstReplyMessage(scheduler.fromScheduler);
            std::println("Got a reply {}:\n{}", replyIndex, reply);
        }

        expect(copy1->settings().applyStagedParameters().forwardParameters.empty());
        expect(copy2->settings().applyStagedParameters().forwardParameters.empty());

        auto uiConstraintsFor = [](const auto& block) {
            return std::visit(meta::overloaded{
                                  //
                                  []<typename... Args>(const std::map<Args...>& map) { return gr::property_map(map); },
                                  //
                                  [](const auto& /*v*/) { return gr::property_map{}; }
                                  //
                              },
                block->settings().get("ui_constraints").value());
        };

        expect(eq(42.f, std::get<float>(uiConstraintsFor(copy1)["x"])));
        expect(eq(43.f, std::get<float>(uiConstraintsFor(copy2)["x"])));

        // Check if block introspection includes ui_constraints

        {
            sendMessage<Get>(scheduler.toScheduler, {}, graph::property::kGraphInspect, {});
            waitForReply(scheduler.fromScheduler);

            expect(ge(getNReplyMessages(scheduler.fromScheduler), 1UZ));
            const Message reply = getAndConsumeFirstReplyMessage(scheduler.fromScheduler);

            expect(reply.data.has_value()) << "Reply should contain data";
            if (reply.data.has_value()) {
                const auto& map = reply.data.value();
                expect(!map.empty()) << "Resulting map should not be empty";

                const auto& children = gr::detail::getOrThrow(gr::detail::getProperty<gr::property_map>(map, "children"s));

                std::set<float> seenUiConstraintsX;
                std::set<float> seenUiConstraintsY;

                for (const auto& child : children) {
                    const auto& uiConstraints = gr::detail::getOrThrow(gr::detail::getProperty<gr::property_map>(std::get<gr::property_map>(child.second), "parameters"s, "ui_constraints"s));
                    seenUiConstraintsX.insert(std::get<float>(uiConstraints.at("x"s)));
                    seenUiConstraintsY.insert(std::get<float>(uiConstraints.at("y"s)));
                }

                expect(seenUiConstraintsX == std::set<float>{42, 43});
                expect(seenUiConstraintsY == std::set<float>{6, 7070});
            }

            scheduler.scheduler().requestStop();

            auto copy1direct = static_cast<gr::testing::Copy<float>*>(copy1->raw());
            auto copy2direct = static_cast<gr::testing::Copy<float>*>(copy2->raw());

            expect(eq(42.f, std::get<float>(copy1direct->ui_constraints["x"])));
            expect(eq(43.f, std::get<float>(copy2direct->ui_constraints["x"])));
        }

        scheduler.scheduler().requestStop();

        auto copy1direct = static_cast<gr::testing::Copy<float>*>(copy1->raw());
        auto copy2direct = static_cast<gr::testing::Copy<float>*>(copy2->raw());

        expect(eq(42.f, std::get<float>(copy1direct->ui_constraints["x"])));
        expect(eq(43.f, std::get<float>(copy2direct->ui_constraints["x"])));
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

    TestScheduler scheduler(std::move(graph), /*addTestSourceAndSink=*/false);

    expect(awaitCondition(1s, [&scheduler] { return scheduler.state() == lifecycle::State::RUNNING; })) << "scheduler thread up and running w/ timeout";
    expect(scheduler.state() == lifecycle::State::RUNNING) << "scheduler thread up and running";
    expect(eq(scheduler.graph().edges().size(), 1UZ)) << "added one edge";

    expect(awaitCondition(1s, [&sink] { return sink.count >= 10U; })) << "sink received enough data";
    std::println("executed basic graph");

    // Adding a few blocks
    auto multiply1 = sendAndWaitMessageEmplaceBlock(scheduler.toScheduler, scheduler.fromScheduler, "gr::testing::Copy<float32>"s, property_map{});
    auto multiply2 = sendAndWaitMessageEmplaceBlock(scheduler.toScheduler, scheduler.fromScheduler, "gr::testing::Copy<float32>"s, property_map{});

    for (const auto& block : scheduler.graph().blocks()) {
        std::println("block in list: {} - state() : {}", block->name(), magic_enum::enum_name(block->state()));
    }
    expect(eq(scheduler.graph().blocks().size(), 4UZ)) << "should contain sink->multiply1->multiply2->sink";

    sendAndWaitMessageEmplaceEdge(scheduler.toScheduler, scheduler.fromScheduler, source.unique_name, "out", multiply1, "in");
    sendAndWaitMessageEmplaceEdge(scheduler.toScheduler, scheduler.fromScheduler, multiply1, "out", multiply2, "in");
    sendAndWaitMessageEmplaceEdge(scheduler.toScheduler, scheduler.fromScheduler, multiply2, "out", sink.unique_name, "in");
    expect(eq(getNReplyMessages(scheduler.fromScheduler), 0UZ));

    // Get the whole graph
    testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, "", graph::property::kGraphInspect, property_map{}, //
        [](const Message& reply) {
            if (reply.endpoint != graph::property::kGraphInspected) {
                return false;
            }

            const auto& data     = reply.data.value();
            const auto& children = std::get<property_map>(data.at("children"s));
            expect(eq(children.size(), 4UZ));

            const auto& edges = std::get<property_map>(data.at("edges"s));
            expect(eq(edges.size(), 4UZ));
            return true;
        });

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
