#include <boost/ut.hpp>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Graph_yaml_importer.hpp>
#include <gnuradio-4.0/PluginLoader.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/SchedulerModel.hpp>
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
struct DemoSubSchedulerResult {
    using Scheduler = gr::scheduler::Simple<scheduler::ExecutionPolicy::multiThreaded>;
    using Wrapper   = SchedulerWrapper<Scheduler>;

    DemoSubSchedulerResult() {}

    void setGraph(gr::Graph&& graph) {
        scheduler = std::unique_ptr<BlockModel>(std::make_unique<Wrapper>().release());
        wrapper   = static_cast<Wrapper*>(scheduler.get());
        wrapper->setGraph(std::move(graph));
    }

    std::unique_ptr<gr::BlockModel>         scheduler;
    Wrapper*                                wrapper          = nullptr;
    gr::testing::Copy<T>*                   pass1            = nullptr;
    gr::testing::Copy<T>*                   pass2            = nullptr;
    gr::testing::SettingsChangeRecorder<T>* settingsRecorder = nullptr;
};

template<typename T>
DemoSubSchedulerResult<T> createDemoSubScheduler() {
    DemoSubSchedulerResult<T> result;
    gr::Graph                 graph;
    result.pass1 = std::addressof(graph.template emplaceBlock<gr::testing::Copy<T>>());
    result.pass2 = std::addressof(graph.template emplaceBlock<gr::testing::Copy<T>>());
    expect(eq(ConnectionResult::SUCCESS, graph.connect(*result.pass1, PortDefinition("out"), *result.pass2, PortDefinition("in"))));
    result.setGraph(std::move(graph));
    return result;
}

template<typename T>
DemoSubSchedulerResult<T> createDemoSubSchedulerWithSettings() {
    DemoSubSchedulerResult<T> result;
    gr::Graph                 graph;
    result.pass1            = std::addressof(graph.template emplaceBlock<gr::testing::Copy<T>>());
    result.pass2            = std::addressof(graph.template emplaceBlock<gr::testing::Copy<T>>());
    result.settingsRecorder = std::addressof(graph.template emplaceBlock<gr::testing::SettingsChangeRecorder<T>>());
    expect(eq(ConnectionResult::SUCCESS, graph.connect(*result.pass1, PortDefinition("out"), *result.pass2, PortDefinition("in"))));
    result.setGraph(std::move(graph));
    return result;
}

const boost::ut::suite BasicSchedulerWrapperTests = [] {
    using namespace gr;
    using Scheduler = gr::scheduler::Simple<scheduler::ExecutionPolicy::multiThreaded>;

    "Instantiate scheduler wrapper"_test = [&] {
        SchedulerWrapper<Scheduler> scheduler;
        scheduler.setGraph(gr::Graph());
    };

    "Add scheduler wrapper to a graph"_test = [&] {
        auto scheduler = std::unique_ptr<BlockModel>(std::make_unique<SchedulerWrapper<Scheduler>>());

        gr::Graph graph;
        graph.addBlock(std::move(scheduler));
    };
};

void setCustomDefaultThreadPool() {
    auto cpu = std::make_shared<thread_pool::ThreadPoolWrapper>(std::make_unique<thread_pool::BasicThreadPool>(std::string(thread_pool::kDefaultCpuPoolId), thread_pool::TaskType::CPU_BOUND, 2U, 2U), "CPU");
    gr::thread_pool::Manager::instance().replacePool(std::string(thread_pool::kDefaultCpuPoolId), std::move(cpu));
}

const boost::ut::suite ManagedSubGraph = [] {
    setCustomDefaultThreadPool();

    using Scheduler = gr::scheduler::Simple<scheduler::ExecutionPolicy::multiThreaded>;

    "lifecycle_subscheduler"_test = [] {
        using namespace std::string_literals;
        using namespace boost::ut;
        using namespace gr;
        using enum gr::message::Command;

        gr::Graph graph;

        // Basic source and sink
        [[maybe_unused]] auto& source = graph.emplaceBlock<SlowSource<float>>();
        [[maybe_unused]] auto& sink   = graph.emplaceBlock<CountingSink<float>>();

        // Sub-scheduler with a single block inside
        auto       demo              = createDemoSubScheduler<float>();
        auto       subSchedulerBlock = graph.addBlock(std::move(demo.scheduler));
        gr::Graph* subGraph          = subSchedulerBlock->graph();
        auto       schedModel        = dynamic_cast<SchedulerModel*>(subSchedulerBlock.get());

        expect(schedModel != nullptr) << "schedModel should not be null";
        expect(subGraph != nullptr) << "subGraph should not be null";
        expect(subGraph->state() == lifecycle::State::IDLE) << std::format("subGraph state should be IDLE - actual: {}", magic_enum::enum_name(subGraph->state()));
        expect(subSchedulerBlock->state() == lifecycle::State::IDLE) << std::format("subSchedulerBlock state should be IDLE - actual: {}", magic_enum::enum_name(subSchedulerBlock->state()));

        // Connecting the message ports
        Scheduler scheduler;
        if (auto ret = scheduler.exchange(std::move(graph)); !ret) {
            expect(false) << std::format("couldn't initialise scheduler. error: {}", ret.error()) << fatal;
        }

        gr::MsgPortOut toScheduler;
        gr::MsgPortIn  fromScheduler;
        expect(eq(ConnectionResult::SUCCESS, toScheduler.connect(scheduler.msgIn)));
        expect(eq(ConnectionResult::SUCCESS, scheduler.msgOut.connect(fromScheduler)));

        expect(scheduler.state() == lifecycle::State::IDLE) << std::format("scheduler state should be IDLE - actual: {}", magic_enum::enum_name(scheduler.state()));

        auto schedulerThreadHandle = gr::test::thread_pool::executeScheduler("qa_ManagedSubGraph", scheduler);

        expect(awaitCondition(2s, [&scheduler] { return scheduler.state() == lifecycle::State::RUNNING; })) << "scheduler is not running";

        // Check the state of the sub-blocks before the state of the sub-scheduler, as the scheduler might be running
        // but still updating children states
        expect(eq(subGraph->blocks().size(), 2UZ)) << "subGraph should have 2 blocks";
        expect(eq(scheduler.graph().blocks().size(), 3UZ)) << "Graph should contain source->(copy->copy)->sink";
        expect(awaitCondition(2s, [&subGraph] { return subGraph->blocks()[0]->state() == lifecycle::State::RUNNING; })) << "block 0 is not running";
        expect(awaitCondition(2s, [&subGraph] { return subGraph->blocks()[1]->state() == lifecycle::State::RUNNING; })) << "block 1 is not running";

        expect(subSchedulerBlock->state() == lifecycle::State::RUNNING) << "sub-scheduler is not running";

        std::expected<void, Error> schedulerRet;

        testing::sendAndWaitForReply<Set>(toScheduler, fromScheduler, demo.wrapper->uniqueName(), graph::property::kSubgraphExportPort, //
            property_map{{"uniqueBlockName"s, std::string(demo.pass2->unique_name)}, {"portDirection"s, "output"}, {"portName"s, "out"}, {"exportFlag"s, true}}, [](const Message& reply) { return reply.endpoint == graph::property::kSubgraphExportedPort; });

        testing::sendAndWaitForReply<Set>(toScheduler, fromScheduler, demo.wrapper->uniqueName(), graph::property::kSubgraphExportPort, //
            property_map{{"uniqueBlockName"s, std::string(demo.pass1->unique_name)}, {"portDirection"s, "input"}, {"portName"s, "in"}, {"exportFlag"s, true}}, [](const Message& reply) { return reply.endpoint == graph::property::kSubgraphExportedPort; });

        expect(eq(demo.wrapper->dynamicInputPortsSize(), 1UZ));
        expect(eq(demo.wrapper->dynamicOutputPortsSize(), 1UZ));

        // Make connections
        sendAndWaitMessageEmplaceEdge(toScheduler, fromScheduler, source.unique_name, "out", std::string(demo.wrapper->uniqueName()), "in", scheduler.unique_name);
        sendAndWaitMessageEmplaceEdge(toScheduler, fromScheduler, std::string(demo.wrapper->uniqueName()), "out", sink.unique_name, "in", scheduler.unique_name);
        expect(eq(getNReplyMessages(fromScheduler), 0UZ));

        // Get the whole graph
        {
            auto msg = testing::sendAndWaitForReply<Get>(toScheduler, fromScheduler, scheduler.graph().unique_name /* serviceName */, graph::property::kGraphInspect /* endpoint */, property_map{} /* data */, [](const Message& reply) { return reply.endpoint == graph::property::kGraphInspected; });
            expect(msg.has_value()) << "msg should have a value";
            const auto& data     = msg.value().data.value();
            const auto& children = std::get<property_map>(data.at("children"s));
            expect(eq(children.size(), 3UZ));

            const auto& edges = std::get<property_map>(data.at("edges"s));
            expect(eq(edges.size(), 2UZ));

            std::size_t subGraphInConnections  = 0UZ;
            std::size_t subGraphOutConnections = 0UZ;

            for (const auto& [index, edge_] : edges) {
                const auto& edge = std::get<property_map>(edge_);
                if (std::get<std::string>(edge.at("destination_block")) == demo.wrapper->uniqueName()) {
                    subGraphInConnections++;
                }
                if (std::get<std::string>(edge.at("source_block")) == demo.wrapper->uniqueName()) {
                    subGraphOutConnections++;
                }
            }

            // Check that the subgraph is connected properly
            expect(eq(subGraphInConnections, 1UZ));
            expect(eq(subGraphOutConnections, 1UZ));

            // Check subgraph topology
            const auto& subGraphData     = std::get<property_map>(children.at(std::string(demo.wrapper->uniqueName())));
            const auto& subGraphChildren = std::get<property_map>(subGraphData.at("children"s));
            const auto& subGraphEdges    = std::get<property_map>(subGraphData.at("edges"s));
            expect(eq(subGraphChildren.size(), 2UZ));
            expect(eq(subGraphEdges.size(), 1UZ));
        }

        // Pause scheduler
        expect(scheduler.state() == lifecycle::State::RUNNING) << std::format("scheduler should be running before pause - actual: {}", magic_enum::enum_name(scheduler.state()));
        expect(scheduler.changeStateTo(lifecycle::State::REQUESTED_PAUSE).has_value()) << "could switch to REQUESTED_PAUSE?";

        expect(awaitCondition(2s, [&scheduler] { return scheduler.state() == lifecycle::State::PAUSED; })) << std::format("scheduler should be paused - actual: {}", magic_enum::enum_name(scheduler.state()));
        expect(awaitCondition(2s, [&subGraph] { return subGraph->blocks()[0]->state() == lifecycle::State::PAUSED; })) << "block 0 is not paused";
        expect(awaitCondition(2s, [&subGraph] { return subGraph->blocks()[1]->state() == lifecycle::State::PAUSED; })) << "block 1 is not paused";

        // Resume scheduler
        expect(scheduler.changeStateTo(lifecycle::State::RUNNING).has_value()) << "could switch to RUNNING?";
        expect(awaitCondition(2s, [&subGraph] { return subGraph->blocks()[0]->state() == lifecycle::RUNNING; })) << "block 0 is not running";
        expect(awaitCondition(2s, [&subGraph] { return subGraph->blocks()[1]->state() == lifecycle::State::RUNNING; })) << "block 1 is not running";
        expect(scheduler.state() == lifecycle::State::RUNNING) << std::format("scheduler should be running - actual: {}", magic_enum::enum_name(scheduler.state()));
        expect(subSchedulerBlock->state() == lifecycle::State::RUNNING) << std::format("sub-scheduler should be running - actual: {}", magic_enum::enum_name(subSchedulerBlock->state()));

        // Stop scheduler

        scheduler.requestStop();

        expect(awaitCondition(1s, [&scheduler] { return scheduler.state() == lifecycle::State::STOPPED; })) << std::format("scheduler should be stopped - actual: {}", magic_enum::enum_name(scheduler.state()));
        expect(subSchedulerBlock->state() == lifecycle::State::STOPPED) << std::format("sub-scheduler should be stopped - actual: {}", magic_enum::enum_name(subSchedulerBlock->state()));

        expect(awaitCondition(2s, [&subGraph] { return subGraph->blocks()[0]->state() == lifecycle::State::STOPPED; })) << "block 0 is not stopped";
        expect(awaitCondition(2s, [&subGraph] { return subGraph->blocks()[1]->state() == lifecycle::State::STOPPED; })) << "block 1 is not stopped";

        // return to initial state
        expect(scheduler.changeStateTo(lifecycle::State::INITIALISED).has_value()) << "could not switch to INITIALISED?";
        expect(awaitCondition(1s, [&scheduler] { return scheduler.state() == lifecycle::State::INITIALISED; })) << "scheduler INITIALISED w/ timeout";

        expect(awaitCondition(2s, [&subGraph] { return subGraph->blocks()[0]->state() == lifecycle::State::INITIALISED; })) << "block 0 is not initialized";
        expect(subGraph->blocks()[1]->state() == lifecycle::State::INITIALISED) << "block 1 is not initialized";
        expect(scheduler.state() == lifecycle::State::INITIALISED) << std::format("scheduler INITIALISED - actual: {}\n", magic_enum::enum_name(scheduler.state()));
        expect(subSchedulerBlock->state() == lifecycle::State::INITIALISED) << std::format("sub-scheduler should be stopped - actual: {}", magic_enum::enum_name(subSchedulerBlock->state()));
    };
};

} // namespace gr::subgraph_test
int main() { /* tests are statically executed */ }
