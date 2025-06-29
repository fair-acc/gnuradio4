#include <boost/ut.hpp>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/basic/ClockSource.hpp>
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
struct DemoSubGraph : public gr::Graph {
public:
    gr::testing::Copy<T>* pass1 = nullptr;
    gr::testing::Copy<T>* pass2 = nullptr;

    DemoSubGraph(const gr::property_map& init) : gr::Graph(init) {
        pass1 = std::addressof(emplaceBlock<gr::testing::Copy<T>>());
        pass2 = std::addressof(emplaceBlock<gr::testing::Copy<T>>());

        expect(eq(ConnectionResult::SUCCESS, gr::Graph::connect(*pass1, PortDefinition("out"), *pass2, PortDefinition("in"))));
    }
};

template<typename T>
struct DemoSubGraphWithSettings : public DemoSubGraph<T> {
public:
    gr::testing::SettingsChangeRecorder<T>* settingsRecorder = nullptr;

    DemoSubGraphWithSettings(const gr::property_map& init) : DemoSubGraph<T>(init) { //
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
        using SubGraphType   = GraphWrapper<DemoSubGraph<float>>;
        auto& subGraph       = initGraph.addBlock(std::make_unique<SubGraphType>());
        auto* subGraphDirect = dynamic_cast<SubGraphType*>(&subGraph);

        // Connecting the message ports
        gr::scheduler::Simple scheduler{std::move(initGraph)};
        const auto&           graph = scheduler.graph();
        gr::MsgPortOut        toScheduler;
        gr::MsgPortIn         fromScheduler;
        expect(eq(ConnectionResult::SUCCESS, toScheduler.connect(scheduler.msgIn)));
        expect(eq(ConnectionResult::SUCCESS, scheduler.msgOut.connect(fromScheduler)));

        std::expected<void, Error> schedulerRet;
        auto                       runScheduler = [&scheduler, &schedulerRet] { schedulerRet = scheduler.runAndWait(); };

        sendMessage<Set>(toScheduler, subGraph.uniqueName(), graph::property::kSubgraphExportPort, //
            property_map{{"uniqueBlockName"s, subGraphDirect->blockRef().pass2->unique_name}, {"portDirection"s, "output"}, {"portName"s, "out"}, {"exportFlag"s, true}});
        sendMessage<Set>(toScheduler, subGraph.uniqueName(), graph::property::kSubgraphExportPort, //
            property_map{{"uniqueBlockName"s, subGraphDirect->blockRef().pass1->unique_name}, {"portDirection"s, "input"}, {"portName"s, "in"}, {"exportFlag"s, true}});
        scheduler.processScheduledMessages();

        std::thread schedulerThread1(runScheduler);

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
        expect(eq(getNReplyMessages(fromScheduler), 0UZ));

        // Make connections
        sendAndWaitMessageEmplaceEdge(toScheduler, fromScheduler, source.unique_name, "out", std::string(subGraph.uniqueName()), "in", scheduler.unique_name);
        sendAndWaitMessageEmplaceEdge(toScheduler, fromScheduler, std::string(subGraph.uniqueName()), "out", sink.unique_name, "in", scheduler.unique_name);

        expect(eq(getNReplyMessages(fromScheduler), 0UZ));

        // Get the whole graph
        {
            sendMessage<Set>(toScheduler, graph.unique_name /* serviceName */, graph::property::kGraphInspect /* endpoint */, property_map{} /* data */);
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
            expect(eq(children.size(), 3UZ));

            const auto& edges = std::get<property_map>(data.at("edges"s));
            expect(eq(edges.size(), 2UZ));

            std::size_t subGraphInConnections  = 0UZ;
            std::size_t subGraphOutConnections = 0UZ;

            // Check that the subgraph is connected properly

            for (const auto& [index, edge_] : edges) {
                const auto& edge = std::get<property_map>(edge_);
                if (std::get<std::string>(edge.at("destinationBlock")) == subGraph.uniqueName()) {
                    subGraphInConnections++;
                }
                if (std::get<std::string>(edge.at("sourceBlock")) == subGraph.uniqueName()) {
                    subGraphOutConnections++;
                }
            }
            expect(eq(subGraphInConnections, 1UZ));
            expect(eq(subGraphOutConnections, 1UZ));

            // Check subgraph topology
            const auto& subGraphData     = std::get<property_map>(children.at(std::string(subGraph.uniqueName())));
            const auto& subGraphChildren = std::get<property_map>(subGraphData.at("children"s));
            const auto& subGraphEdges    = std::get<property_map>(subGraphData.at("edges"s));
            expect(eq(subGraphChildren.size(), 2UZ));
            expect(eq(subGraphEdges.size(), 1UZ));
        }

        // Stopping scheduler
        scheduler.requestStop();
        schedulerThread1.join();
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
        using SubGraphType   = GraphWrapper<DemoSubGraph<float>>;
        auto& subGraph       = initGraph.addBlock(std::make_unique<SubGraphType>());
        auto* subGraphDirect = dynamic_cast<SubGraphType*>(&subGraph);
        subGraphDirect->exportPort(true, subGraphDirect->blockRef().pass1->unique_name, PortDirection::INPUT, "in");
        subGraphDirect->exportPort(true, subGraphDirect->blockRef().pass2->unique_name, PortDirection::OUTPUT, "out");

        expect(eq(ConnectionResult::SUCCESS, initGraph.connect(source, PortDefinition("out"), subGraph, PortDefinition("in"))));
        expect(eq(ConnectionResult::SUCCESS, initGraph.connect(subGraph, PortDefinition("out"), sink, PortDefinition("in"))));
        expect(eq(initGraph.edges().size(), 2UZ));
        expect(eq(subGraphDirect->blockRef().edges().size(), 1UZ));

        gr::scheduler::Simple      scheduler{std::move(initGraph)};
        std::expected<void, Error> schedulerRet;
        auto                       runScheduler = [&scheduler, &schedulerRet] { schedulerRet = scheduler.runAndWait(); };

        std::thread schedulerThread1(runScheduler);

        expect(awaitCondition(1s, [&] { return scheduler.state() == lifecycle::State::RUNNING; })) << "scheduler thread up and running w/ timeout";

        expect(scheduler.state() == lifecycle::State::RUNNING) << "scheduler thread up and running";

        expect(awaitCondition(1s, [&] { return sink.count > 0UZ; }));

        expect(source.state() == lifecycle::State::RUNNING);
        expect(sink.state() == lifecycle::State::RUNNING);
        expect(subGraphDirect->blockRef().pass1->state() == lifecycle::State::RUNNING);
        expect(subGraphDirect->blockRef().pass2->state() == lifecycle::State::RUNNING);

        // Stopping scheduler
        scheduler.requestStop();
        schedulerThread1.join();
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
        using SubGraphType   = GraphWrapper<DemoSubGraphWithSettings<float>>;
        auto& subGraph       = initGraph.addBlock(std::make_unique<SubGraphType>());
        auto* subGraphDirect = dynamic_cast<SubGraphType*>(&subGraph);

        // Connecting the message ports
        gr::scheduler::Simple scheduler{std::move(initGraph)};
        const auto&           graph = scheduler.graph();
        gr::MsgPortOut        toScheduler;
        gr::MsgPortIn         fromScheduler;
        expect(eq(ConnectionResult::SUCCESS, toScheduler.connect(scheduler.msgIn)));
        expect(eq(ConnectionResult::SUCCESS, scheduler.msgOut.connect(fromScheduler)));

        std::expected<void, Error> schedulerRet;
        auto                       runScheduler = [&scheduler, &schedulerRet] { schedulerRet = scheduler.runAndWait(); };

        std::thread schedulerThread1(runScheduler);

        expect(awaitCondition(1s, [&scheduler] { return scheduler.state() == lifecycle::State::RUNNING; })) << "scheduler thread up and running w/ timeout";
        expect(scheduler.state() == lifecycle::State::RUNNING) << "scheduler thread up and running";

        expect(eq(graph.blocks().size(), 3UZ)) << "should contain source->(copy->copy)->sink";

        // Sending messages to blocks in the subgraph
        sendMessage<Set>(toScheduler, subGraphDirect->blockRef().settingsRecorder->unique_name /* serviceName */, block::property::kStagedSetting /* endpoint */, {{"scaling_factor", 42.0f}} /* data  */);

        // Stopping scheduler
        scheduler.requestStop();
        schedulerThread1.join();
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
