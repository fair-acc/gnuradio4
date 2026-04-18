#include <boost/ut.hpp>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Graph_yaml_importer.hpp>
#include <gnuradio-4.0/PluginLoader.hpp>
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
    expect(graph.connect(*result.pass1, "out", *result.pass2, "in").has_value());
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
    expect(graph.connect(*result.pass1, "out", *result.pass2, "in").has_value());
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
        expect(toScheduler.connect(scheduler.msgIn).has_value());
        expect(scheduler.msgOut.connect(fromScheduler).has_value());

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
                    const auto& children = gr::test::get_value_or_fail<property_map>(data.at("children"));
                    expect(eq(children.size(), 3UZ));

                    const auto& edges = gr::test::get_value_or_fail<property_map>(data.at("edges"));
                    expect(eq(edges.size(), 2UZ));

                    std::size_t subGraphInConnections  = 0UZ;
                    std::size_t subGraphOutConnections = 0UZ;

                    // Check that the subgraph is connected properly

                    for (const auto& [index, edge_] : edges) {
                        const auto& edge = gr::test::get_value_or_fail<property_map>(edge_);
                        if (gr::test::get_value_or_fail<std::string>(edge.at("destination_block")) == demo.graphUniqueName) {
                            subGraphInConnections++;
                        }
                        if (gr::test::get_value_or_fail<std::string>(edge.at("source_block")) == demo.graphUniqueName) {
                            subGraphOutConnections++;
                        }
                    }
                    expect(eq(subGraphInConnections, 1UZ));
                    expect(eq(subGraphOutConnections, 1UZ));

                    // Check subgraph topology
                    const auto& subGraphData   = gr::test::get_value_or_fail<property_map>(children.at(convert_string_domain(demo.graphUniqueName)));
                    const auto& subGraphGraph  = gr::test::get_value_or_fail<property_map>(subGraphData.at("graph"));
                    const auto& subGraphBlocks = gr::test::get_value_or_fail<Tensor<pmt::Value>>(subGraphGraph.at("blocks"));
                    const auto& subGraphConns  = gr::test::get_value_or_fail<Tensor<pmt::Value>>(subGraphGraph.at("connections"));
                    expect(eq(subGraphBlocks.size(), 2UZ));
                    expect(eq(subGraphConns.size(), 1UZ));
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

        expect(demo.graph->exportPort(true, demo.pass1->unique_name, PortDirection::INPUT, "in", "inExp").has_value());
        expect(demo.graph->exportPort(true, demo.pass2->unique_name, PortDirection::OUTPUT, "out", "outExp").has_value());

        auto sourceBlock = gr::graph::findBlock(initGraph, source).value();
        auto sinkBlock   = gr::graph::findBlock(initGraph, sink).value();
        expect(initGraph.connect(sourceBlock, "out", demo.graph, "inExp").has_value());
        expect(initGraph.connect(demo.graph, "outExp", sinkBlock, "in").has_value());
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
        expect(toScheduler.connect(scheduler.msgIn).has_value());
        expect(scheduler.msgOut.connect(fromScheduler).has_value());

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

const boost::ut::suite GraphInspectYamlTests_ = [] {
    "kGraphInspect returns yaml when serialization_format is yaml"_test = [] {
        using enum gr::message::Command;

        gr::Graph              initGraph;
        [[maybe_unused]] auto& source = initGraph.emplaceBlock<SlowSource<float>>();
        [[maybe_unused]] auto& sink   = initGraph.emplaceBlock<CountingSink<float>>();

        auto demo = createDemoSubGraph<float>();
        expect(demo.graph->exportPort(true, demo.pass1->unique_name, PortDirection::INPUT, "in", "inExp").has_value());
        expect(demo.graph->exportPort(true, demo.pass2->unique_name, PortDirection::OUTPUT, "out", "outExp").has_value());
        initGraph.addBlock(std::move(demo.graph));

        gr::scheduler::Simple scheduler;
        if (auto ret = scheduler.exchange(std::move(initGraph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        const auto& graph = scheduler.graph();

        gr::MsgPortOut toScheduler;
        gr::MsgPortIn  fromScheduler;
        expect(toScheduler.connect(scheduler.msgIn).has_value());
        expect(scheduler.msgOut.connect(fromScheduler).has_value());

        auto schedulerThreadHandle = gr::test::thread_pool::executeScheduler("qa_GraphInspectYaml::scheduler", scheduler);
        expect(awaitCondition(1s, [&scheduler] { return scheduler.state() == lifecycle::State::RUNNING; })) << "scheduler running";

        testing::sendAndWaitForReply<Set>(toScheduler, fromScheduler, graph.unique_name,     //
            graph::property::kGraphInspect, property_map{{"serialization_format", "yaml"s}}, //
            [](const Message& reply) {
                if (reply.endpoint != graph::property::kGraphInspected) {
                    return false;
                }
                const auto& data     = reply.data.value();
                const auto  yamlData = data.find("yamlData");
                expect(yamlData != data.cend()) << "yamlData key must be present";
                if (yamlData != data.cend()) {
                    const auto yamlStr = gr::test::get_value_or_fail<std::string>(yamlData->second);
                    expect(!yamlStr.empty()) << "yamlData must not be empty";
                }
                return true;
            });

        scheduler.requestStop();
        schedulerThreadHandle.get();
        expect(scheduler.changeStateTo(lifecycle::State::INITIALISED).has_value());
        expect(awaitCondition(1s, [&scheduler] { return scheduler.state() == lifecycle::State::INITIALISED; }));
    };
};

const boost::ut::suite SchedulerInspectTests_ = [] {
    "kSchedulerInspect returns non-yaml graph structure"_test = [] {
        using enum gr::message::Command;

        gr::Graph              initGraph;
        [[maybe_unused]] auto& source = initGraph.emplaceBlock<SlowSource<float>>();
        [[maybe_unused]] auto& sink   = initGraph.emplaceBlock<CountingSink<float>>();

        auto demo = createDemoSubGraph<float>();
        initGraph.addBlock(std::move(demo.graph));

        gr::scheduler::Simple scheduler;
        if (auto ret = scheduler.exchange(std::move(initGraph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        const auto& graph = scheduler.graph();

        gr::MsgPortOut toScheduler;
        gr::MsgPortIn  fromScheduler;
        expect(toScheduler.connect(scheduler.msgIn).has_value());
        expect(scheduler.msgOut.connect(fromScheduler).has_value());

        auto schedulerThreadHandle = gr::test::thread_pool::executeScheduler("qa_SchedulerInspect::scheduler", scheduler);
        expect(awaitCondition(1s, [&scheduler] { return scheduler.state() == lifecycle::State::RUNNING; })) << "scheduler running";

        testing::sendAndWaitForReply<Set>(toScheduler, fromScheduler, scheduler.unique_name, //
            scheduler::property::kSchedulerInspect, property_map{},                          //
            [&](const Message& reply) {
                if (reply.endpoint != scheduler::property::kSchedulerInspected) {
                    return false;
                }
                const auto& data     = reply.data.value();
                const auto& children = gr::test::get_value_or_fail<property_map>(data.at("children"));
                expect(eq(children.size(), 1UZ)) << "scheduler children should contain the graph";

                const auto& graphData     = gr::test::get_value_or_fail<property_map>(children.at(std::pmr::string(graph.unique_name)));
                const auto& graphChildren = gr::test::get_value_or_fail<property_map>(graphData.at("children"));
                expect(eq(graphChildren.size(), 3UZ)) << "graph has source, sink, subgraph";

                const auto& graphEdges = gr::test::get_value_or_fail<property_map>(graphData.at("edges"));
                expect(eq(graphEdges.size(), 0UZ)) << "no edges (not connected in this test)";
                return true;
            });

        scheduler.requestStop();
        schedulerThreadHandle.get();
        expect(scheduler.changeStateTo(lifecycle::State::INITIALISED).has_value());
        expect(awaitCondition(1s, [&scheduler] { return scheduler.state() == lifecycle::State::INITIALISED; }));
    };

    "kSchedulerInspect returns yaml when serialization_format is yaml"_test = [] {
        using enum gr::message::Command;

        gr::Graph              initGraph;
        [[maybe_unused]] auto& source = initGraph.emplaceBlock<SlowSource<float>>();
        [[maybe_unused]] auto& sink   = initGraph.emplaceBlock<CountingSink<float>>();

        gr::scheduler::Simple scheduler;
        if (auto ret = scheduler.exchange(std::move(initGraph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }

        gr::MsgPortOut toScheduler;
        gr::MsgPortIn  fromScheduler;
        expect(toScheduler.connect(scheduler.msgIn).has_value());
        expect(scheduler.msgOut.connect(fromScheduler).has_value());

        auto schedulerThreadHandle = gr::test::thread_pool::executeScheduler("qa_SchedulerInspectYaml::scheduler", scheduler);
        expect(awaitCondition(1s, [&scheduler] { return scheduler.state() == lifecycle::State::RUNNING; })) << "scheduler running";

        testing::sendAndWaitForReply<Set>(toScheduler, fromScheduler, scheduler.unique_name,         //
            scheduler::property::kSchedulerInspect, property_map{{"serialization_format", "yaml"s}}, //
            [](const Message& reply) {
                if (reply.endpoint != scheduler::property::kSchedulerInspected) {
                    return false;
                }
                const auto& data     = reply.data.value();
                const auto  yamlData = data.find("yamlData");
                expect(yamlData != data.cend()) << "yamlData key must be present";
                if (yamlData != data.cend()) {
                    const auto yamlStr = gr::test::get_value_or_fail<std::string>(yamlData->second);
                    expect(!yamlStr.empty()) << "yamlData must not be empty";
                    expect(yamlStr.find("ui_constraints") != std::string::npos);
                }
                return true;
            });

        scheduler.requestStop();
        schedulerThreadHandle.get();
        expect(scheduler.changeStateTo(lifecycle::State::INITIALISED).has_value());
        expect(awaitCondition(1s, [&scheduler] { return scheduler.state() == lifecycle::State::INITIALISED; }));
    };
};

} // namespace gr::subgraph_test
int main() { return boost::ut::cfg<boost::ut::override>.run(); }
