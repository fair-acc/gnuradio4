#include <boost/ut.hpp>

#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>

#include <GrBasicBlocks.hpp>
#include <GrTestingBlocks.hpp>

#include "TestBlockRegistryContext.hpp"

#include "message_utils.hpp"

#include <gnuradio-4.0/meta/UnitTestHelper.hpp>

using namespace std::chrono_literals;
using namespace std::string_literals;

namespace ut = boost::ut;

// For messages that change graph topology, see qa_SchedulerMessages.cpp instead

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
    using namespace gr::test;

    "Edge formatter tests"_test = [] {
        Graph                  graph;
        [[maybe_unused]] auto& source = graph.emplaceBlock<NullSource<float>>();
        [[maybe_unused]] auto& sink   = graph.emplaceBlock<NullSink<float>>();
        Edge                   edge{graph.blocks()[0UZ].get(), {1}, graph.blocks()[1UZ].get(), {2}, 1024, 1, "test_edge"};

        "default"_test = [&edge] {
            std::string result = std::format("{:s}", edge);
            std::println("Edge formatter - default:   {}", result);

            expect(result.contains(" ⟶ (name: 'test_edge', size: 1024, weight:  1, state: WaitingToBeConnected) ⟶")) << result;
        };

        "short names"_test = [&edge] {
            std::string result = std::format("{:s}", edge);
            std::println("Edge formatter - short 's': {}", result);

            expect(result.contains(" ⟶ (name: 'test_edge', size: 1024, weight:  1, state: WaitingToBeConnected) ⟶")) << result;
        };

        "long names"_test = [&edge] {
            std::string result = std::format("{:l}", edge);
            std::println("Edge formatter - long  'l': {}", result);

            expect(result.contains(" ⟶ (name: 'test_edge', size: 1024, weight:  1, state: WaitingToBeConnected) ⟶")) << result;
        };
    };
};

const boost::ut::suite GraphMessageTests = [] {
    using namespace std::string_literals;
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;
    using namespace gr::test;
    using enum gr::message::Command;

    expect(fatal(gt(context->registry.keys().size(), 0UZ))) << "didn't register any blocks";
    std::println("registered blocks:");
    for (const auto& blockName : context->registry.keys()) {
        std::println("    block: {}", blockName);
    }

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
                    auto          expectedBlockTypes = loader.availableBlocks();
                    std::ranges::sort(expectedBlockTypes);
                    auto blockTypes = std::get<std::vector<std::string>>(foundTypes->second);
                    std::ranges::sort(blockTypes);
                    expect(eq(expectedBlockTypes, blockTypes));
                } else {
                    expect(false) << "`types` key not found or data type is not a `property_map`";
                }
            } else {
                expect(false) << std::format("data has no value - error: {}", reply.data.error());
            }
        };
    };
};

int main() { /* tests are statically executed */ }
