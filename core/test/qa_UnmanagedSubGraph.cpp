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
struct DemoSubGraphResult {
    using Wrapper = GraphWrapper<gr::Graph>;
    DemoSubGraphResult() : _graph(std::make_unique<Wrapper>()), wrapper(static_cast<Wrapper*>(_graph.get())) {}

    std::unique_ptr<gr::BlockModel> _graph;
    std::unique_ptr<gr::BlockModel> graph() && { return std::move(_graph); }

    Wrapper*                                wrapper          = nullptr;
    gr::testing::Copy<T>*                   pass1            = nullptr;
    gr::testing::Copy<T>*                   pass2            = nullptr;
    gr::testing::SettingsChangeRecorder<T>* settingsRecorder = nullptr;
};

template<typename T>
DemoSubGraphResult<T> createDemoSubGraph() {
    DemoSubGraphResult<T> result;
    result.pass1 = std::addressof(result.wrapper->graph().template emplaceBlock<gr::testing::Copy<T>>());
    result.pass2 = std::addressof(result.wrapper->graph().template emplaceBlock<gr::testing::Copy<T>>());
    expect(eq(ConnectionResult::SUCCESS, result.wrapper->graph().connect(*result.pass1, PortDefinition("out"), *result.pass2, PortDefinition("in"))));
    return result;
}

template<typename T>
DemoSubGraphResult<T> createDemoSubGraphWithSettings() {
    auto result             = createDemoSubGraph<T>();
    result.settingsRecorder = std::addressof(result.wrapper->graph().template emplaceBlock<gr::testing::SettingsChangeRecorder<T>>());
    return result;
}

const boost::ut::suite ExportPortsTests_ = [] {
    "Test if port export messages work"_test = [] {
        using namespace std::string_literals;
        using namespace boost::ut;
        using namespace gr;
        using enum gr::message::Command;

        gr::scheduler::Simple scheduler{gr::Graph()};
        auto&                 graph = scheduler.graph();

        // Basic source and sink
        auto& source = graph.emplaceBlock<SlowSource<float>>();
        auto& sink   = graph.emplaceBlock<CountingSink<float>>();

        // Subgraph with a single block inside
        auto demo = createDemoSubGraph<float>();
        graph.addBlock(std::move(demo).graph());

        // Connecting the message ports
        gr::MsgPortOut toScheduler;
        gr::MsgPortIn  fromScheduler;
        expect(eq(ConnectionResult::SUCCESS, toScheduler.connect(scheduler.msgIn)));
        expect(eq(ConnectionResult::SUCCESS, scheduler.msgOut.connect(fromScheduler)));

        std::expected<void, Error> schedulerRet;
        auto                       runScheduler = [&scheduler, &schedulerRet] { schedulerRet = scheduler.runAndWait(); };

        sendMessage<Set>(toScheduler, demo.wrapper->uniqueName(), graph::property::kSubgraphExportPort, //
            property_map{{"uniqueBlockName"s, demo.pass2->unique_name}, {"portDirection"s, "output"}, {"portName"s, "out"}, {"exportFlag"s, true}});
        sendMessage<Set>(toScheduler, demo.wrapper->uniqueName(), graph::property::kSubgraphExportPort, //
            property_map{{"uniqueBlockName"s, demo.pass1->unique_name}, {"portDirection"s, "input"}, {"portName"s, "in"}, {"exportFlag"s, true}});
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
        sendAndWaitMessageEmplaceEdge(toScheduler, fromScheduler, source.unique_name, "out", std::string(demo.wrapper->uniqueName()), "in", scheduler.unique_name);
        sendAndWaitMessageEmplaceEdge(toScheduler, fromScheduler, std::string(demo.wrapper->uniqueName()), "out", sink.unique_name, "in", scheduler.unique_name);

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
                if (std::get<std::string>(edge.at("destinationBlock")) == demo.wrapper->uniqueName()) {
                    subGraphInConnections++;
                }
                if (std::get<std::string>(edge.at("sourceBlock")) == demo.wrapper->uniqueName()) {
                    subGraphOutConnections++;
                }
            }
            expect(eq(subGraphInConnections, 1UZ));
            expect(eq(subGraphOutConnections, 1UZ));

            // Check subgraph topology
            const auto& subGraphData     = std::get<property_map>(children.at(std::string(demo.wrapper->uniqueName())));
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
        expect(scheduler.changeStateTo(lifecycle::State::INITIALISED).has_value()) << "could switch to INITIALISED?";
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

        gr::Graph graph;
        auto&     source = graph.emplaceBlock<SlowSource<float>>();
        auto&     sink   = graph.emplaceBlock<CountingSink<float>>();

        // Subgraph with a single block inside
        auto demo = createDemoSubGraph<float>();
        graph.addBlock(std::move(demo).graph());
        demo.wrapper->exportPort(true, demo.pass1->unique_name, PortDirection::INPUT, "in");
        demo.wrapper->exportPort(true, demo.pass2->unique_name, PortDirection::OUTPUT, "out");

        expect(eq(ConnectionResult::SUCCESS, graph.connect(source, PortDefinition("out"), demo.wrapper->graph(), PortDefinition("in"))));
        expect(eq(ConnectionResult::SUCCESS, graph.connect(demo.wrapper->graph(), PortDefinition("out"), sink, PortDefinition("in"))));
        expect(eq(graph.edges().size(), 2UZ));
        expect(eq(demo.wrapper->blockRef().edges().size(), 1UZ));

        gr::scheduler::Simple scheduler{std::move(graph)};

        std::expected<void, Error> schedulerRet;
        auto                       runScheduler = [&scheduler, &schedulerRet] { schedulerRet = scheduler.runAndWait(); };

        std::thread schedulerThread1(runScheduler);

        expect(awaitCondition(1s, [&] { return scheduler.state() == lifecycle::State::RUNNING; })) << "scheduler thread up and running w/ timeout";

        expect(scheduler.state() == lifecycle::State::RUNNING) << "scheduler thread up and running";

        expect(awaitCondition(1s, [&] { return sink.count > 0UZ; }));

        expect(source.state() == lifecycle::State::RUNNING);
        expect(sink.state() == lifecycle::State::RUNNING);
        expect(demo.pass1->state() == lifecycle::State::RUNNING);
        expect(demo.pass2->state() == lifecycle::State::RUNNING);

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

        gr::scheduler::Simple scheduler{gr::Graph()};
        auto&                 graph = scheduler.graph();

        // Basic source and sink
        [[maybe_unused]] auto& source = graph.emplaceBlock<SlowSource<float>>();
        [[maybe_unused]] auto& sink   = graph.emplaceBlock<CountingSink<float>>();

        // Subgraph with a single block inside
        auto demo = createDemoSubGraphWithSettings<float>();
        graph.addBlock(std::move(demo).graph());

        // Connecting the message ports
        gr::MsgPortOut toScheduler;
        gr::MsgPortIn  fromScheduler;
        expect(eq(ConnectionResult::SUCCESS, toScheduler.connect(scheduler.msgIn)));
        expect(eq(ConnectionResult::SUCCESS, scheduler.msgOut.connect(fromScheduler)));

        std::expected<void, Error> schedulerRet;
        auto                       runScheduler = [&scheduler, &schedulerRet] { schedulerRet = scheduler.runAndWait(); };

        std::thread schedulerThread1(runScheduler);

        expect(awaitCondition(1s, [&scheduler] { return scheduler.state() == lifecycle::State::RUNNING; })) << "scheduler thread up and running w/ timeout";
        expect(scheduler.state() == lifecycle::State::RUNNING) << "scheduler thread up and running";

        expect(eq(graph.blocks().size(), 3UZ)) << "should contain source->(copy->copy)->sink";

        // Sending messages to blocks in the subgraph
        sendMessage<Set>(toScheduler, demo.settingsRecorder->unique_name /* serviceName */, block::property::kStagedSetting /* endpoint */, {{"scaling_factor", 42.0f}} /* data  */);

        // Stopping scheduler
        scheduler.requestStop();
        schedulerThread1.join();
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
