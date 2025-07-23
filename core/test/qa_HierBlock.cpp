#include <boost/ut.hpp>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/basic/ClockSource.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>
#include <gnuradio-4.0/testing/SettingsChangeRecorder.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include "message_utils.hpp"
#include "utils.hpp"

namespace gr::subgraph_test {

using namespace std::chrono_literals;
using namespace std::string_literals;

using namespace boost::ut;
using namespace gr;
using namespace gr::message;

using namespace gr::testing;

template<typename T>
struct DemoSubGraph : Graph {
    std::string pass1_unique_id;
    std::string pass2_unique_id;

    DemoSubGraph(property_map init = {}) : gr::Graph(std::move(init)) {
        auto& pass1     = emplaceBlock<Copy<T>>();
        auto& pass2     = emplaceBlock<Copy<T>>();
        pass1_unique_id = pass1.unique_name;
        pass2_unique_id = pass2.unique_name;

        expect(eq(ConnectionResult::SUCCESS, gr::Graph::connect(pass1, PortDefinition("out"), pass2, PortDefinition("in"))));
    }
};

template<typename T>
struct DemoSubGraphWithSettings : DemoSubGraph<T> {
    gr::testing::SettingsChangeRecorder<T>* settingsRecorder = nullptr;

    DemoSubGraphWithSettings(property_map init = {}) : DemoSubGraph<T>(std::move(init)) { //
        settingsRecorder = std::addressof(gr::Graph::emplaceBlock<gr::testing::SettingsChangeRecorder<T>>());
    }
};

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

        // Subgraph with a single block inside
        using SubGraphType                         = GraphWrapper<DemoSubGraph<float>>;
        auto                        subGraphDirect = std::make_shared<SubGraphType>();
        std::shared_ptr<BlockModel> subGraph       = initGraph.addBlock(subGraphDirect);

        // Connecting the message ports
        gr::scheduler::Simple scheduler{std::move(initGraph)};
        const auto&           graph = scheduler.graph();
        gr::MsgPortOut        toScheduler;
        gr::MsgPortIn         fromScheduler;
        expect(eq(ConnectionResult::SUCCESS, toScheduler.connect(scheduler.msgIn)));
        expect(eq(ConnectionResult::SUCCESS, scheduler.msgOut.connect(fromScheduler)));

        sendMessage<Set>(toScheduler, subGraph->uniqueName(), graph::property::kSubgraphExportPort, //
            property_map{{"uniqueBlockName"s, subGraphDirect->blockRef().pass2_unique_id}, {"portDirection"s, "output"s}, {"portName"s, "out"s}, {"exportFlag"s, true}});
        sendMessage<Set>(toScheduler, subGraph->uniqueName(), graph::property::kSubgraphExportPort, //
            property_map{{"uniqueBlockName"s, subGraphDirect->blockRef().pass1_unique_id}, {"portDirection"s, "input"s}, {"portName"s, "in"s}, {"exportFlag"s, true}});
        scheduler.processScheduledMessages();

        auto schedulerThreadHandle = gr::testing::thread_pool::executeScheduler("qa_HierBlock::scheduler", scheduler);

        expect(awaitCondition(1s, [&scheduler] { return scheduler.state() == lifecycle::State::RUNNING; })) << "scheduler thread up and running w/ timeout";
        expect(scheduler.state() == lifecycle::State::RUNNING) << "scheduler thread up and running";

        for (const auto& block : graph.blocks()) {
            std::println("block in list: {} - state() : {}", block->name(), magic_enum::enum_name(block->state()));
        }
        expect(eq(graph.blocks().size(), 3UZ)) << "should contain source->(copy->copy)->sink";

        // 2 export ports from the sub-graph
        if (!waitForReply(fromScheduler, 2UZ)) {
            expect(false) << "Reply messages not received for kSubgraphExportPort.";
        }
        expect(ge(getNReplyMessages(fromScheduler), 2UZ));
        consumeAllReplyMessages(fromScheduler);

        // Make connections
        sendAndWaitMessageEmplaceEdge(toScheduler, fromScheduler, source.unique_name, "out", std::string(subGraph->uniqueName()), "in", scheduler.unique_name);
        sendAndWaitMessageEmplaceEdge(toScheduler, fromScheduler, std::string(subGraph->uniqueName()), "out", sink.unique_name, "in", scheduler.unique_name);

        expect(eq(getNReplyMessages(fromScheduler), 0UZ));

        // Get the whole graph
        {
            testing::sendAndWaitMessage<Set>(toScheduler, fromScheduler, graph.unique_name /* serviceName */, //
                graph::property::kGraphInspect /* endpoint */, property_map{} /* data */, [&](const Message& reply) {
                    if (reply.endpoint != graph::property::kGraphInspected) {
                        return false;
                    }

                    const auto& data     = reply.data.value();
                    const auto& children = std::get<property_map>(data.at("children"s));
                    expect(eq(children.size(), 3UZ));

                    const auto& edges = std::get<property_map>(data.at("edges"s));
                    expect(eq(edges.size(), 2UZ));

                    std::size_t subGraphInConnections  = 0UZ;
                    std::size_t subGraphOutConnections = 0UZ;

                    // Check that the subgraph is connected properly

                    for (const auto& [index, edge_] : edges) {
                        const auto& edge = std::get<property_map>(edge_);
                        if (std::get<std::string>(edge.at("destinationBlock")) == subGraph->uniqueName()) {
                            subGraphInConnections++;
                        }
                        if (std::get<std::string>(edge.at("sourceBlock")) == subGraph->uniqueName()) {
                            subGraphOutConnections++;
                        }
                    }
                    expect(eq(subGraphInConnections, 1UZ));
                    expect(eq(subGraphOutConnections, 1UZ));

                    // Check subgraph topology
                    const auto& subGraphData     = std::get<property_map>(children.at(std::string(subGraph->uniqueName())));
                    const auto& subGraphChildren = std::get<property_map>(subGraphData.at("children"s));
                    const auto& subGraphEdges    = std::get<property_map>(subGraphData.at("edges"s));
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

        // Subgraph with a single block inside
        using SubGraphType = GraphWrapper<DemoSubGraph<float>>;
        static_assert(gr::GraphLike<SubGraphType>);
        auto  subGraphDirect = std::make_shared<SubGraphType>();
        auto& subGraph       = initGraph.addBlock(subGraphDirect);

        subGraphDirect->exportPort(true, subGraphDirect->blockRef().pass1_unique_id, PortDirection::INPUT, "in");
        subGraphDirect->exportPort(true, subGraphDirect->blockRef().pass2_unique_id, PortDirection::OUTPUT, "out");

        expect(eq(ConnectionResult::SUCCESS, initGraph.connect(source, PortDefinition("out"), subGraph, PortDefinition("in"))));
        expect(eq(ConnectionResult::SUCCESS, initGraph.connect(subGraph, PortDefinition("out"), sink, PortDefinition("in"))));
        expect(eq(initGraph.edges().size(), 2UZ));
        expect(eq(subGraphDirect->blockRef().edges().size(), 1UZ));

        gr::scheduler::Simple scheduler{std::move(initGraph)};

        auto schedulerThreadHandle = gr::testing::thread_pool::executeScheduler("qa_HierBlock::scheduler", scheduler);

        expect(awaitCondition(1s, [&] { return scheduler.state() == lifecycle::State::RUNNING; })) << "scheduler thread up and running w/ timeout";

        expect(scheduler.state() == lifecycle::State::RUNNING) << "scheduler thread up and running";

        expect(awaitCondition(1s, [&] { return sink.count > 0UZ; }));

        expect(source.state() == lifecycle::State::RUNNING);
        expect(sink.state() == lifecycle::State::RUNNING);
        expect(graph::findBlock(*subGraphDirect, subGraphDirect->blockRef().pass1_unique_id).value()->state() == lifecycle::State::RUNNING);
        expect(graph::findBlock(*subGraphDirect, subGraphDirect->blockRef().pass2_unique_id).value()->state() == lifecycle::State::RUNNING);

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
        using SubGraphType                         = GraphWrapper<DemoSubGraphWithSettings<float>>;
        auto                        subGraphDirect = std::make_shared<SubGraphType>();
        std::shared_ptr<BlockModel> subGraph       = initGraph.addBlock(subGraphDirect);

        // Connecting the message ports
        gr::scheduler::Simple scheduler{std::move(initGraph)};
        const auto&           graph = scheduler.graph();
        gr::MsgPortOut        toScheduler;
        gr::MsgPortIn         fromScheduler;
        expect(eq(ConnectionResult::SUCCESS, toScheduler.connect(scheduler.msgIn)));
        expect(eq(ConnectionResult::SUCCESS, scheduler.msgOut.connect(fromScheduler)));

        auto schedulerThreadHandle = gr::testing::thread_pool::executeScheduler("qa_HierBlock::scheduler", scheduler);

        expect(awaitCondition(1s, [&scheduler] { return scheduler.state() == lifecycle::State::RUNNING; })) << "scheduler thread up and running w/ timeout";
        expect(scheduler.state() == lifecycle::State::RUNNING) << "scheduler thread up and running";

        expect(eq(graph.blocks().size(), 3UZ)) << "should contain source->(copy->copy)->sink";

        // Sending messages to blocks in the subgraph
        sendMessage<Set>(toScheduler, std::string(subGraphDirect->blockRef().settingsRecorder->unique_name) /* serviceName */, block::property::kStagedSetting /* endpoint */, {{"scaling_factor", 42.0f}} /* data  */);

        // Stopping scheduler
        scheduler.requestStop();
        auto schedulerRet = schedulerThreadHandle.get();
        if (!schedulerRet.has_value()) {
            expect(false) << std::format("scheduler.runAndWait() failed:\n{}\n", schedulerRet.error());
        }

        auto applyResult = subGraphDirect->blockRef().settingsRecorder->settings().applyStagedParameters();
        expect(eq(subGraphDirect->blockRef().settingsRecorder->scaling_factor, 42.0f)) << "settings didn't change";

        // return to initial state
        expect(scheduler.changeStateTo(lifecycle::State::INITIALISED).has_value()) << "could switch to INITIALISED?";
        expect(awaitCondition(1s, [&scheduler] { return scheduler.state() == lifecycle::State::INITIALISED; })) << "scheduler INITIALISED w/ timeout";
        expect(scheduler.state() == lifecycle::State::INITIALISED) << std::format("scheduler INITIALISED - actual: {}\n", magic_enum::enum_name(scheduler.state()));
    };
};

} // namespace gr::subgraph_test
int main() { /* tests are statically executed */ }
