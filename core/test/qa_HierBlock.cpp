#include <boost/ut.hpp>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/basic/clock_source.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include "message_utils.hpp"

namespace gr::testing {

using namespace std::chrono_literals;
using namespace std::string_literals;

using namespace boost::ut;
using namespace gr;
using namespace gr::message;

template<typename T>
struct DemoSubGraph : public gr::Graph {
public:
    gr::testing::Copy<T>* pass1 = nullptr;
    gr::testing::Copy<T>* pass2 = nullptr;

    DemoSubGraph(const gr::property_map& init) : gr::Graph(init) {
        pass1 = std::addressof(emplaceBlock<gr::testing::Copy<T>>());
        pass2 = std::addressof(emplaceBlock<gr::testing::Copy<T>>());

        std::ignore = gr::Graph::connect(*pass1, PortDefinition("out"), *pass2, PortDefinition("in"));
    }
};

const boost::ut::suite ExportPortsTests_ = [] {
    using namespace std::string_literals;
    using namespace boost::ut;
    using namespace gr;
    using enum gr::message::Command;

    gr::scheduler::Simple scheduler{gr::Graph()};
    auto&                 graph = scheduler.graph();
    fmt::print(">>> Starting test with root graph {} <<<\n", graph.unique_name);

    // Basic source and sink
    auto& source = graph.emplaceBlock<SlowSource<float>>();
    auto& sink   = graph.emplaceBlock<CountingSink<float>>();

    // Subgraph with a single block inside
    using SubGraphType   = GraphWrapper<DemoSubGraph<float>>;
    auto& subGraph       = graph.addBlock(std::make_unique<SubGraphType>());
    auto* subGraphDirect = dynamic_cast<SubGraphType*>(&subGraph);

    // Connecting the message ports
    gr::MsgPortOut toGraph;
    gr::MsgPortIn  fromGraph;
    expect(eq(ConnectionResult::SUCCESS, toGraph.connect(scheduler.msgIn)));
    expect(eq(ConnectionResult::SUCCESS, scheduler.msgOut.connect(fromGraph)));

    std::expected<void, Error> schedulerRet;
    auto                       runScheduler = [&scheduler, &schedulerRet] { schedulerRet = scheduler.runAndWait(); };

    std::thread schedulerThread1(runScheduler);

    expect(awaitCondition(1s, [&scheduler] { return scheduler.state() == lifecycle::State::RUNNING; })) << "scheduler thread up and running w/ timeout";
    expect(scheduler.state() == lifecycle::State::RUNNING) << "scheduler thread up and running";

    for (const auto& block : graph.blocks()) {
        fmt::println("block in list: {} - state() : {}", block->name(), magic_enum::enum_name(block->state()));
    }
    expect(eq(graph.blocks().size(), 3UZ)) << "should contain source->(copy->copy)->sink";

    // Export ports from the sub-graph

    sendMessage<Set>(toGraph, subGraph.uniqueName(), graph::property::kSubgraphExportPort,
        property_map{
            {"uniqueBlockName"s, subGraphDirect->blockRef().pass2->unique_name}, //
            {"portDirection"s, "output"},                                        //
            {"portName"s, "out"},                                                //
            {"exportFlag"s, true}                                                //
        });
    sendMessage<Set>(toGraph, subGraph.uniqueName(), graph::property::kSubgraphExportPort,
        property_map{
            {"uniqueBlockName"s, subGraphDirect->blockRef().pass1->unique_name}, //
            {"portDirection"s, "input"},                                         //
            {"portName"s, "in"},                                                 //
            {"exportFlag"s, true}                                                //
        });
    if (!waitForAReply(fromGraph)) {
        fmt::println("didn't receive a reply message for kSubgraphExportPort");
        expect(false);
    }

    // Make connections

    expect(sendEmplaceTestEdgeMsg(toGraph, fromGraph, source.unique_name, "out", std::string(subGraph.uniqueName()), "in")) << "emplace edge source -> group failed and returned an error";
    expect(sendEmplaceTestEdgeMsg(toGraph, fromGraph, std::string(subGraph.uniqueName()), "out", sink.unique_name, "in")) << "emplace edge multiply2 -> sink failed and returned an error";

    // Get the whole graph
    {
        sendMessage<Set>(toGraph, graph.unique_name /* serviceName */, graph::property::kGraphInspect /* endpoint */, property_map{} /* data */);
        if (!waitForAReply(fromGraph)) {
            fmt::println("didn't receive a reply message for kGraphInspect");
            expect(false);
        }

        const Message reply = returnReplyMsg(fromGraph);
        expect(reply.data.has_value());

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
        expect(false) << fmt::format("scheduler.runAndWait() failed:\n{}\n", schedulerRet.error());
    }

    // return to initial state
    expect(scheduler.changeStateTo(lifecycle::State::INITIALISED).has_value()) << "could switch to INITIALISED?";
    expect(awaitCondition(1s, [&scheduler] { return scheduler.state() == lifecycle::State::INITIALISED; })) << "scheduler INITIALISED w/ timeout";
    expect(scheduler.state() == lifecycle::State::INITIALISED) << fmt::format("scheduler INITIALISED - actual: {}\n", magic_enum::enum_name(scheduler.state()));
};

} // namespace gr::testing
int main() { /* tests are statically executed */ }
