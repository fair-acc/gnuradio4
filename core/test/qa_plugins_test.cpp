#include <array>
#include <cassert>
#include <iostream>

#include <boost/ut.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/basic/CommonBlocks.hpp>

#include <GrBasicBlocks>
#include <GrTestingBlocks>

#include "TestBlockRegistryContext.hpp"

using namespace std::chrono_literals;

namespace ut = boost::ut;

auto makeTestContext() {
    return std::make_unique<TestContext>(                      //
        paths{"core/test/plugins", "test/plugins", "plugins"}, // plugin paths
        gr::blocklib::initGrBasicBlocks,                       //
        gr::blocklib::initGrTestingBlocks);
}

const boost::ut::suite PluginLoaderTests = [] {
    auto context = makeTestContext();

    using namespace boost::ut;
    using namespace gr;

    "GoodPlugins"_test = [&] {
        expect(!context->loader.plugins().empty());
        for (const auto& plugin : context->loader.plugins()) {
            expect(plugin->metadata.plugin_name.starts_with("Good"));
        }
    };

    "BadPlugins"_test = [&] {
        expect(!context->loader.failed_plugins().empty());
        for (const auto& plugin : context->loader.failed_plugins()) {
            expect(plugin.first.ends_with("bad_plugin.so"));
        }
    };

    "KnownBlocksList"_test = [&] {
        auto       known = context->loader.knownBlocks();
        std::array requireds{"good::cout_sink<float64>", "good::cout_sink<float32>", "good::fixed_source<float64>", "good::fixed_source<float32>", "good::divide<float64>", "good::divide<float32>", "builtin_multiply<float64>", "builtin_multiply<float32>"};

        for (const auto& required : requireds) {
            expect(std::ranges::find(known, required) != known.end());
        }
    };
};

const boost::ut::suite BlockInstantiationTests = [] {
    using namespace boost::ut;
    using namespace gr;
    auto context = makeTestContext();

    "KnownBlocksInstantiate"_test = [&] {
        expect(context->loader.instantiate("good::fixed_source<float64>") != nullptr);
        expect(context->loader.instantiate("good::cout_sink<float64>") != nullptr);
        expect(context->loader.instantiate("good::multiply<float64>") != nullptr);
        expect(context->loader.instantiate("good::divide<float64>") != nullptr);
        expect(context->loader.instantiate("good::convert<float64, float32>") != nullptr);

        expect(context->loader.instantiate("good::fixed_source<something>") == nullptr);
        expect(context->loader.instantiate("good::cout_sink<something>") == nullptr);
        expect(context->loader.instantiate("good::multiply<something>") == nullptr);
        expect(context->loader.instantiate("good::divide<something>") == nullptr);
        expect(context->loader.instantiate("good::convert<float32, float32>") == nullptr);
    };

    "UnknownBlocks"_test = [&] { expect(context->loader.instantiate("ThisBlockDoesNotExist<float64>") == nullptr); };
};

const boost::ut::suite BasicPluginBlocksConnectionTests = [] {
    using namespace boost::ut;
    using namespace gr;
    auto context = makeTestContext();

    "FixedSourceToSink"_test = [&] {
        auto block_source = context->loader.instantiate("good::fixed_source<float64>");
        assert(block_source != nullptr);
        auto block_sink = context->loader.instantiate("good::cout_sink<float64>");
        assert(block_sink != nullptr);
        auto connection_1 = block_source->dynamicOutputPort(0).connect(block_sink->dynamicInputPort(0));
        expect(connection_1 == gr::ConnectionResult::SUCCESS);
    };

    "LongerPipeline"_test = [&] {
        auto block_source = context->loader.instantiate("good::fixed_source<float64>");

        gr::property_map block_multiply_params;
        block_multiply_params["factor"] = 2.0;
        auto block_multiply             = context->loader.instantiate("good::multiply<float64>", block_multiply_params);

        std::size_t      repeats = 10;
        gr::property_map block_sink_params;
        block_sink_params["total_count"] = gr::Size_t(100);
        auto block_sink                  = context->loader.instantiate("good::cout_sink<float64>");

        auto connection_1 = block_source->dynamicOutputPort(0).connect(block_multiply->dynamicInputPort(0));
        auto connection_2 = block_multiply->dynamicOutputPort(0).connect(block_sink->dynamicInputPort(0));

        expect(connection_1 == gr::ConnectionResult::SUCCESS);
        expect(connection_2 == gr::ConnectionResult::SUCCESS);

        for (std::size_t i = 0; i < repeats; ++i) {
            std::ignore = block_source->work(std::numeric_limits<std::size_t>::max());
            std::ignore = block_multiply->work(std::numeric_limits<std::size_t>::max());
            std::ignore = block_sink->work(std::numeric_limits<std::size_t>::max());
        }
    };

    "Graph"_test = [&] {
        gr::Graph testGraph(context->loader);

        // Instantiate the node that is defined in a plugin
        auto& block_source = testGraph.emplaceBlock("good::fixed_source<float64>", {});

        // Instantiate a built-in node in a static way
        gr::property_map block_multiply_1_params;
        block_multiply_1_params["factor"] = 2.0;
        auto& block_multiply_double       = testGraph.emplaceBlock<builtin_multiply<double>>(block_multiply_1_params);

        // Instantiate a built-in node via the plugin loader
        auto& block_multiply_float = testGraph.emplaceBlock("builtin_multiply<float32>", {});

        auto& block_convert_to_float  = testGraph.emplaceBlock("good::convert<float64, float32>", {});
        auto& block_convert_to_double = testGraph.emplaceBlock("good::convert<float32, float64>", {});

        //
        std::size_t      repeats = 10;
        gr::property_map block_sink_params;
        block_sink_params["total_count"] = gr::Size_t(100);
        auto  block_sink_load            = context->loader.instantiate("good::cout_sink<float64>", block_sink_params);
        auto& block_sink                 = testGraph.addBlock(std::move(block_sink_load));

        auto connection_1 = testGraph.connect(block_source, 0, block_multiply_double, 0);
        auto connection_2 = testGraph.connect(block_multiply_double, 0, block_convert_to_float, 0);
        auto connection_3 = testGraph.connect(block_convert_to_float, 0, block_multiply_float, 0);
        auto connection_4 = testGraph.connect(block_multiply_float, 0, block_convert_to_double, 0);
        auto connection_5 = testGraph.connect(block_convert_to_double, 0, block_sink, 0);

        expect(connection_1 == gr::ConnectionResult::SUCCESS);
        expect(connection_2 == gr::ConnectionResult::SUCCESS);
        expect(connection_3 == gr::ConnectionResult::SUCCESS);
        expect(connection_4 == gr::ConnectionResult::SUCCESS);
        expect(connection_5 == gr::ConnectionResult::SUCCESS);

        for (std::size_t i = 0; i < repeats; ++i) {
            std::ignore = block_source.work(std::numeric_limits<std::size_t>::max());
            std::ignore = block_multiply_double.work(std::numeric_limits<std::size_t>::max());
            std::ignore = block_convert_to_float.work(std::numeric_limits<std::size_t>::max());
            std::ignore = block_multiply_float.work(std::numeric_limits<std::size_t>::max());
            std::ignore = block_convert_to_double.work(std::numeric_limits<std::size_t>::max());
            std::ignore = block_sink.work(std::numeric_limits<std::size_t>::max());
        }
    };
};

int main() { /* not needed for UT */ }
