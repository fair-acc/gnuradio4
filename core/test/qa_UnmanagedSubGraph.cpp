#include <boost/ut.hpp>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/basic/ClockSource.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>
#include <gnuradio-4.0/testing/SettingsChangeRecorder.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include "message_utils.hpp"

namespace gr::subgraph_test {

using namespace std::chrono_literals;
using namespace std::string_literals;

using namespace boost::ut;
using namespace gr;
using namespace gr::message;

using namespace gr::testing;

template<typename T>
struct DemoSubGraphResult {
    using Wrapper = GraphWrapper<Graph>;

    std::shared_ptr<gr::BlockModel> graph;
    std::string                     graphUniqueName;

    Wrapper*                                wrapper          = nullptr;
    gr::testing::Copy<T>*                   pass1            = nullptr;
    gr::testing::Copy<T>*                   pass2            = nullptr;
    gr::testing::SettingsChangeRecorder<T>* settingsRecorder = nullptr;

    DemoSubGraphResult() {}

    void setGraph(gr::Graph&& _graph) {
        graph           = std::static_pointer_cast<BlockModel>(std::make_shared<Wrapper>(std::move(_graph)));
        graphUniqueName = graph->uniqueName();
        wrapper         = static_cast<Wrapper*>(graph.get());
    }
};

template<typename T>
DemoSubGraphResult<T> createDemoSubGraph() {
    DemoSubGraphResult<T> result;
    gr::Graph             graph;
    result.pass1 = std::addressof(graph.template emplaceBlock<gr::testing::Copy<T>>());
    result.pass2 = std::addressof(graph.template emplaceBlock<gr::testing::Copy<T>>());
    expect(eq(ConnectionResult::SUCCESS, graph.connect(*result.pass1, PortDefinition("out"), *result.pass2, PortDefinition("in"))));
    result.setGraph(std::move(graph));
    return result;
}

template<typename T>
DemoSubGraphResult<T> createDemoSubGraphWithSettings() {
    DemoSubGraphResult<T> result;
    gr::Graph             graph;
    result.pass1            = std::addressof(graph.template emplaceBlock<gr::testing::Copy<T>>());
    result.pass2            = std::addressof(graph.template emplaceBlock<gr::testing::Copy<T>>());
    result.settingsRecorder = std::addressof(graph.template emplaceBlock<gr::testing::SettingsChangeRecorder<T>>());
    expect(eq(ConnectionResult::SUCCESS, graph.connect(*result.pass1, PortDefinition("out"), *result.pass2, PortDefinition("in"))));
    result.setGraph(std::move(graph));
    return result;
}

const boost::ut::suite ExportPortsTests_ = [] {
    "Test if port export messages work"_test = [] {
        using namespace std::string_literals;
        using namespace boost::ut;
        using namespace gr;
        using enum gr::message::Command;

        gr::Graph initGraph;

        // Basic source and sink
        auto& source = initGraph.emplaceBlock<SlowSource<float>>();
        auto& sink   = initGraph.emplaceBlock<CountingSink<float>>();

        auto demo = createDemoSubGraph<float>();
        initGraph.addBlock(std::move(demo.graph));

        // Connecting the message ports
        gr::scheduler::Simple scheduler;
        if (auto ret = scheduler.exchange(std::move(initGraph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        const auto&    graph = scheduler.graph();
        gr::MsgPortOut toScheduler;
        gr::MsgPortIn  fromScheduler;
        expect(eq(ConnectionResult::SUCCESS, toScheduler.connect(scheduler.msgIn)));
        expect(eq(ConnectionResult::SUCCESS, scheduler.msgOut.connect(fromScheduler)));

        auto schedulerThreadHandle = gr::test::thread_pool::executeScheduler("qa_HierBlock::scheduler", scheduler);
        expect(awaitCondition(1s, [&scheduler] { return scheduler.state() == lifecycle::State::RUNNING; })) << "scheduler thread up and running w/ timeout";
        expect(scheduler.state() == lifecycle::State::RUNNING) << "scheduler thread up and running";

        testing::sendAndWaitForReply<Set>(toScheduler, fromScheduler, demo.graphUniqueName, graph::property::kSubgraphExportPort,                                                   //
            property_map{{"uniqueBlockName", demo.pass2->unique_name.value()}, {"portDirection", "output"}, {"portName", "out"}, {"exportFlag", true}, {"exportedName", "outExp"}}, //
            ReplyChecker{.expectedEndpoint = graph::property::kSubgraphExportedPort});
        testing::sendAndWaitForReply<Set>(toScheduler, fromScheduler, demo.graphUniqueName, graph::property::kSubgraphExportPort,                                                //
            property_map{{"uniqueBlockName", demo.pass1->unique_name.value()}, {"portDirection", "input"}, {"portName", "in"}, {"exportFlag", true}, {"exportedName", "inExp"}}, //
            ReplyChecker{.expectedEndpoint = graph::property::kSubgraphExportedPort});

        for (const auto& block : graph.blocks()) {
            std::println("block in list: {} - state() : {}", block->name(), magic_enum::enum_name(block->state()));
        }
        expect(eq(graph.blocks().size(), 3UZ)) << "should contain source->(copy->copy)->sink";

        // Make connections
        sendAndWaitMessageEmplaceEdge(toScheduler, fromScheduler, source.unique_name, "out", demo.graphUniqueName, "inExp", scheduler.unique_name);
        sendAndWaitMessageEmplaceEdge(toScheduler, fromScheduler, demo.graphUniqueName, "outExp", sink.unique_name, "in", scheduler.unique_name);

        expect(eq(getNReplyMessages(fromScheduler), 0UZ));

        // Get the whole graph
        {
            testing::sendAndWaitForReply<Set>(toScheduler, fromScheduler, graph.unique_name /* serviceName */, //
                graph::property::kGraphInspect /* endpoint */, property_map{} /* data */, [&](const Message& reply) {
                    if (reply.endpoint != graph::property::kGraphInspected) {
                        return false;
                    }

                    const auto& data     = reply.data.value();
                    const auto& children = get_value_or_fail<property_map>(data.at("children"));
                    expect(eq(children.size(), 3UZ));

                    const auto& edges = get_value_or_fail<property_map>(data.at("edges"));
                    expect(eq(edges.size(), 2UZ));

                    std::size_t subGraphInConnections  = 0UZ;
                    std::size_t subGraphOutConnections = 0UZ;

                    // Check that the subgraph is connected properly

                    for (const auto& [index, edge_] : edges) {
                        const auto& edge = get_value_or_fail<property_map>(edge_);
                        if (get_value_or_fail<std::string>(edge.at("destination_block")) == demo.graphUniqueName) {
                            subGraphInConnections++;
                        }
                        if (get_value_or_fail<std::string>(edge.at("source_block")) == demo.graphUniqueName) {
                            subGraphOutConnections++;
                        }
                    }
                    expect(eq(subGraphInConnections, 1UZ));
                    expect(eq(subGraphOutConnections, 1UZ));

                    // Check subgraph topology
                    const auto& subGraphData     = get_value_or_fail<property_map>(children.at(convert_string_domain(demo.graphUniqueName)));
                    const auto& subGraphChildren = get_value_or_fail<property_map>(subGraphData.at("children"));
                    const auto& subGraphEdges    = get_value_or_fail<property_map>(subGraphData.at("edges"));
                    expect(eq(subGraphChildren.size(), 2UZ));
                    expect(eq(subGraphEdges.size(), 1UZ));
                    return true;
                });
        }

        // Stopping scheduler
        scheduler.requestStop();
        auto schedulerRet = schedulerThreadHandle.get();
        if (!schedulerRet.has_value()) {
            expect(false) << std::format("scheduler.runAndWait() failed:\n{}\n", schedulerRet.error());
        }

        // return to initial state
        const auto initRet = scheduler.changeStateTo(lifecycle::State::INITIALISED);
        expect(initRet.has_value()) << [&initRet] { return std::format("could switch to INITIALISED - error: {}", initRet.error()); };
        expect(awaitCondition(1s, [&scheduler] { return scheduler.state() == lifecycle::State::INITIALISED; })) << "scheduler INITIALISED w/ timeout";
        expect(scheduler.state() == lifecycle::State::INITIALISED) << std::format("scheduler INITIALISED - actual: {}\n", magic_enum::enum_name(scheduler.state()));
    };
};

const boost::ut::suite SchedulerDiveIntoSubgraphTests_ = [] {
    "Test if the blocks in sub-graph get scheduled"_test = [] {
        using namespace std::string_literals;
        using namespace boost::ut;
        using namespace gr;
        using enum gr::message::Command;

        gr::Graph initGraph;
        // auto&     source = graph.emplaceBlock<SlowSource<float>>({{"n_samples_max", 32U}});
        auto& source = initGraph.emplaceBlock<SlowSource<float>>();
        auto& sink   = initGraph.emplaceBlock<CountingSink<float>>();

        auto demo = createDemoSubGraph<float>();
        initGraph.addBlock(demo.graph);

        demo.graph->exportPort(true, demo.pass1->unique_name, PortDirection::INPUT, "in", "inExp");
        demo.graph->exportPort(true, demo.pass2->unique_name, PortDirection::OUTPUT, "out", "outExp");

        expect(eq(ConnectionResult::SUCCESS, initGraph.connect(source, PortDefinition("out"), demo.graph, PortDefinition("inExp"))));
        expect(eq(ConnectionResult::SUCCESS, initGraph.connect(demo.graph, PortDefinition("outExp"), sink, PortDefinition("in"))));
        expect(eq(initGraph.edges().size(), 2UZ));
        expect(eq(demo.graph->edges().size(), 1UZ));

        gr::scheduler::Simple scheduler;
        if (auto ret = scheduler.exchange(std::move(initGraph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }

        auto schedulerThreadHandle = gr::test::thread_pool::executeScheduler("qa_HierBlock::scheduler", scheduler);

        expect(awaitCondition(1s, [&] { return scheduler.state() == lifecycle::State::RUNNING; })) << "scheduler thread up and running w/ timeout";

        expect(scheduler.state() == lifecycle::State::RUNNING) << "scheduler thread up and running";

        expect(awaitCondition(1s, [&] { return sink.count > 0UZ; }));

        expect(source.state() == lifecycle::State::RUNNING);
        expect(sink.state() == lifecycle::State::RUNNING);
        expect(graph::findBlock(*demo.graph, demo.pass1->unique_name).value()->state() == lifecycle::State::RUNNING);
        expect(graph::findBlock(*demo.graph, demo.pass1->unique_name).value()->state() == lifecycle::State::RUNNING);

        // Stopping scheduler
        scheduler.requestStop();
        auto schedulerRet = schedulerThreadHandle.get();
        if (!schedulerRet.has_value()) {
            expect(false) << std::format("scheduler.runAndWait() failed:\n{}\n", schedulerRet.error());
        }

        expect(neq(sink.count, 0UZ)) << "At least one value should have gone through";

        // return to initial state
        expect(scheduler.changeStateTo(lifecycle::State::INITIALISED).has_value()) << "could switch to INITIALISED?";
        expect(awaitCondition(1s, [&scheduler] { return scheduler.state() == lifecycle::State::INITIALISED; })) << "scheduler INITIALISED w/ timeout";
        expect(scheduler.state() == lifecycle::State::INITIALISED) << std::format("scheduler INITIALISED - actual: {}\n", magic_enum::enum_name(scheduler.state()));
    };
};

const boost::ut::suite SubgraphBlockSettingsTests_ = [] {
    "Test changing settings in blocks in subgraph"_test = [] {
        using namespace std::string_literals;
        using namespace boost::ut;
        using namespace gr;
        using enum gr::message::Command;

        gr::Graph initGraph;

        // Basic source and sink
        [[maybe_unused]] auto& source = initGraph.emplaceBlock<SlowSource<float>>();
        [[maybe_unused]] auto& sink   = initGraph.emplaceBlock<CountingSink<float>>();

        // Subgraph with a single block inside
        auto demo = createDemoSubGraphWithSettings<float>();
        initGraph.addBlock(demo.graph);

        // Connecting the message ports
        gr::scheduler::Simple scheduler;
        if (auto ret = scheduler.exchange(std::move(initGraph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        const auto&    graph = scheduler.graph();
        gr::MsgPortOut toScheduler;
        gr::MsgPortIn  fromScheduler;
        expect(eq(ConnectionResult::SUCCESS, toScheduler.connect(scheduler.msgIn)));
        expect(eq(ConnectionResult::SUCCESS, scheduler.msgOut.connect(fromScheduler)));

        auto schedulerThreadHandle = gr::test::thread_pool::executeScheduler("qa_HierBlock::scheduler", scheduler);

        expect(awaitCondition(1s, [&scheduler] { return scheduler.state() == lifecycle::State::RUNNING; })) << "scheduler thread up and running w/ timeout";
        expect(scheduler.state() == lifecycle::State::RUNNING) << "scheduler thread up and running";

        expect(eq(graph.blocks().size(), 3UZ)) << "should contain source->(copy->copy)->sink";

        // Sending messages to blocks in the subgraph
        sendMessage<Set>(toScheduler, std::string(demo.settingsRecorder->unique_name) /* serviceName */, block::property::kStagedSetting /* endpoint */, {{"scaling_factor", 42.0f}} /* data  */);

        // Stopping scheduler
        scheduler.requestStop();
        auto schedulerRet = schedulerThreadHandle.get();
        if (!schedulerRet.has_value()) {
            expect(false) << std::format("scheduler.runAndWait() failed:\n{}\n", schedulerRet.error());
        }

        auto applyResult = demo.settingsRecorder->settings().applyStagedParameters();
        expect(eq(demo.settingsRecorder->scaling_factor, 42.0f)) << "settings didn't change";

        // return to initial state
        expect(scheduler.changeStateTo(lifecycle::State::INITIALISED).has_value()) << "could switch to INITIALISED?";
        expect(awaitCondition(1s, [&scheduler] { return scheduler.state() == lifecycle::State::INITIALISED; })) << "scheduler INITIALISED w/ timeout";
        expect(scheduler.state() == lifecycle::State::INITIALISED) << std::format("scheduler INITIALISED - actual: {}\n", magic_enum::enum_name(scheduler.state()));
    };
};

} // namespace gr::subgraph_test
int main() { /* tests are statically executed */ }
