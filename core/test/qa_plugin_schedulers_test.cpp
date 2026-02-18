#include "TestBlockRegistryContext.hpp"

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/basic/CommonBlocks.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>

#include <boost/ut.hpp>

#include <cassert>

using namespace std::chrono_literals;

namespace ut = boost::ut;

auto makeTestContext() {
    return std::make_unique<TestContext>( //
        paths{"core/test/plugins", "test/plugins", "plugins"});
}

const boost::ut::suite PluginSchedulerTests = [] {
    auto context = makeTestContext();

    using namespace boost::ut;
    using namespace gr;

    "AvailableSchedulersList"_test = [&] {
        auto availableSchedulers = context->loader.availableSchedulers();

        expect(std::ranges::find(availableSchedulers, "good::GoodMathScheduler") != availableSchedulers.end()) << "The good::GoodMathScheduler should be available in the schedulers list";
        expect(availableSchedulers.size() == 1_u);
    };

    "SchedulerInstantiation"_test = [&] {
        gr::Graph testGraph(context->loader);

        auto& source = testGraph.emplaceBlock("good::fixed_source<float64>", {});
        auto& sink   = testGraph.emplaceBlock("good::cout_sink<float64>", {});

        auto connection = testGraph.connect(source, 0, sink, 0);
        expect(connection == gr::ConnectionResult::SUCCESS);

        auto scheduler = context->loader.instantiateScheduler("good::GoodMathScheduler");
        scheduler->setGraph(std::move(testGraph));

        expect(scheduler != nullptr) << "The good::GoodMathScheduler should be instantiated successfully\n";
        auto schedulerBlock = SchedulerModel::asBlockModelPtr(scheduler);
        expect(schedulerBlock != nullptr) << "The block model cast should succeed";

        expect(schedulerBlock->graph()->blocks().size() == 2_u) << "Graph should contain 2 blocks";
    };

    "NonExistentScheduler"_test = [&] {
        gr::Graph testGraph(context->loader);

        gr::Graph graph(context->loader);
        auto      scheduler = context->loader.instantiateScheduler("NonExistentScheduler");

        expect(scheduler == nullptr) << "Requesting a non-existent scheduler should return nullptr";
    };
};

int main() { /* not needed for UT */ }
