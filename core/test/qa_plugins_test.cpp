#include <array>
#include <cassert>
#include <iostream>

#include <boost/ut.hpp>

#include <gnuradio-4.0/meta/formatter.hpp>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/basic/CommonBlocks.hpp>

#include <gnuradio-4.0/GrBasicBlocks.hpp>
#include <gnuradio-4.0/GrTestingBlocks.hpp>

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
        expect(!context->loader.failedPlugins().empty());
        for (const auto& plugin : context->loader.failedPlugins()) {
#if defined(_WIN32)
            expect(plugin.first.ends_with("bad_plugin.dll"));
#else
            expect(plugin.first.ends_with("bad_plugin.so"));
#endif
        }
    };

    "AvailableBlocksList"_test = [&] {
        auto       known = context->loader.availableBlocks();
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

    "AvailableBlocksInstantiate"_test = [&] {
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

    "NotAvailableBlocks"_test = [&] { expect(context->loader.instantiate("ThisBlockDoesNotExist<float64>") == nullptr); };
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
        auto srcPort_1 = block_source->dynamicOutputPort(0);
        auto dstPort_1 = block_sink->dynamicInputPort(0);
        expect(srcPort_1.has_value()) << "source port lookup";
        expect(dstPort_1.has_value()) << "destination port lookup";
        auto connection_1 = srcPort_1.value()->connect(*dstPort_1.value());
        expect(connection_1.has_value());
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

        auto srcPort1 = block_source->dynamicOutputPort(0);
        auto dstPort1 = block_multiply->dynamicInputPort(0);
        auto srcPort2 = block_multiply->dynamicOutputPort(0);
        auto dstPort2 = block_sink->dynamicInputPort(0);
        expect(srcPort1.has_value() && dstPort1.has_value() && srcPort2.has_value() && dstPort2.has_value()) << "port lookups";
        auto connection_1 = srcPort1.value()->connect(*dstPort1.value());
        auto connection_2 = srcPort2.value()->connect(*dstPort2.value());

        expect(connection_1.has_value());
        expect(connection_2.has_value());

        for (std::size_t i = 0; i < repeats; ++i) {
            std::ignore = block_source->work(std::numeric_limits<std::size_t>::max(), gr::device::hostBackend());
            std::ignore = block_multiply->work(std::numeric_limits<std::size_t>::max(), gr::device::hostBackend());
            std::ignore = block_sink->work(std::numeric_limits<std::size_t>::max(), gr::device::hostBackend());
        }
    };

    "Graph"_test = [&] {
        gr::Graph testGraph(context->loader);

        // Instantiate the node that is defined in a plugin
        auto block_source = testGraph.emplaceBlock("good::fixed_source<float64>", {}).value();

        // Instantiate a built-in node in a static way
        gr::property_map block_multiply_1_params;
        block_multiply_1_params["factor"]  = 2.0;
        auto& block_multiply_double_direct = testGraph.emplaceBlock<builtin_multiply<double>>(block_multiply_1_params);
        auto  block_multiply_double        = gr::graph::findBlock(testGraph, block_multiply_double_direct);

        // Instantiate a built-in node via the plugin loader
        auto block_multiply_float = testGraph.emplaceBlock("builtin_multiply<float32>", {}).value();

        auto block_convert_to_float  = testGraph.emplaceBlock("good::convert<float64, float32>", {}).value();
        auto block_convert_to_double = testGraph.emplaceBlock("good::convert<float32, float64>", {}).value();

        //
        std::size_t      repeats = 10;
        gr::property_map block_sink_params;
        block_sink_params["total_count"] = gr::Size_t(100);
        auto  block_sink_load            = context->loader.instantiate("good::cout_sink<float64>", block_sink_params);
        auto& block_sink                 = testGraph.addBlock(std::move(block_sink_load));

        auto connection_1 = testGraph.connect(block_source, 0, *block_multiply_double, 0);
        auto connection_2 = testGraph.connect(*block_multiply_double, 0, block_convert_to_float, 0);
        auto connection_3 = testGraph.connect(block_convert_to_float, 0, block_multiply_float, 0);
        auto connection_4 = testGraph.connect(block_multiply_float, 0, block_convert_to_double, 0);
        auto connection_5 = testGraph.connect(block_convert_to_double, 0, block_sink, 0);

        expect(connection_1.has_value());
        expect(connection_2.has_value());
        expect(connection_3.has_value());
        expect(connection_4.has_value());
        expect(connection_5.has_value());

        for (std::size_t i = 0; i < repeats; ++i) {
            std::ignore = block_source->work(std::numeric_limits<std::size_t>::max(), gr::device::hostBackend());
            std::ignore = block_multiply_double.value()->work(std::numeric_limits<std::size_t>::max(), gr::device::hostBackend());
            std::ignore = block_convert_to_float->work(std::numeric_limits<std::size_t>::max(), gr::device::hostBackend());
            std::ignore = block_multiply_float->work(std::numeric_limits<std::size_t>::max(), gr::device::hostBackend());
            std::ignore = block_convert_to_double->work(std::numeric_limits<std::size_t>::max(), gr::device::hostBackend());
            std::ignore = block_sink->work(std::numeric_limits<std::size_t>::max(), gr::device::hostBackend());
        }
    };
};

const boost::ut::suite EmbeddedVersionConsistencyTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using Definition = gr::detail::YamlDefinitionsLoader::Definition;

    const auto asValue = [](property_map map) { return gr::pmt::Value(std::move(map)); };

    const auto innerBlock = [](std::string id, std::optional<std::string> embeddedVersion) {
        property_map block{{"id", std::move(id)}};
        if (embeddedVersion) {
            block.insert_or_assign("yaml_definition_information", property_map{{"PLUGIN_VERSION", *embeddedVersion}});
        }
        return block;
    };

    // a definition holding one outer block whose graph holds the given inner blocks
    const auto definitionWithInnerBlocks = [&](Tensor<gr::pmt::Value> innerBlocks) {
        Tensor<gr::pmt::Value> outerBlocks;
        outerBlocks.push_back(asValue(property_map{{"graph", property_map{{"blocks", std::move(innerBlocks)}}}}));
        Definition definition;
        definition.metadata.block_type = "outer::Block";
        definition.definition          = property_map{{"blocks", std::move(outerBlocks)}};
        return definition;
    };

    std::unordered_map<std::string, Definition> knownDefinitions;
    knownDefinitions["inner::Block"].metadata.plugin_version = "2.0";
    knownDefinitions["unversioned::Block"];

    "definitions without nested blocks are consistent"_test = [&] {
        expect(gr::detail::checkEmbeddedVersionConsistency(knownDefinitions, Definition{}).has_value()) << "no blocks at all";

        Definition blocksNotAList;
        blocksNotAList.definition = property_map{{"blocks", 5}};
        expect(gr::detail::checkEmbeddedVersionConsistency(knownDefinitions, blocksNotAList).has_value()) << "blocks that is not a list";

        Definition             outerBlockNotAMap;
        Tensor<gr::pmt::Value> outerEntries;
        outerEntries.push_back(gr::pmt::Value(5));
        outerEntries.push_back(gr::pmt::Value(std::string("text")));
        outerBlockNotAMap.definition = property_map{{"blocks", std::move(outerEntries)}};
        expect(gr::detail::checkEmbeddedVersionConsistency(knownDefinitions, outerBlockNotAMap).has_value()) << "outer blocks that are not maps";

        const auto outerWith = [&](property_map outerBlock) {
            Tensor<gr::pmt::Value> outerBlocks;
            outerBlocks.push_back(asValue(std::move(outerBlock)));
            Definition definition;
            definition.definition = property_map{{"blocks", std::move(outerBlocks)}};
            return definition;
        };
        expect(gr::detail::checkEmbeddedVersionConsistency(knownDefinitions, outerWith(property_map{{"id", "plain"}})).has_value()) << "an outer block without a graph";
        expect(gr::detail::checkEmbeddedVersionConsistency(knownDefinitions, outerWith(property_map{{"graph", 5}})).has_value()) << "a graph that is not a map";
        expect(gr::detail::checkEmbeddedVersionConsistency(knownDefinitions, outerWith(property_map{{"graph", property_map{{"connections", 1}}}})).has_value()) << "a graph without blocks";
        expect(gr::detail::checkEmbeddedVersionConsistency(knownDefinitions, outerWith(property_map{{"graph", property_map{{"blocks", 5}}}})).has_value()) << "graph blocks that are not a list";
    };

    "inner blocks that cannot disagree are consistent"_test = [&] {
        Tensor<gr::pmt::Value> inner;
        inner.push_back(asValue(property_map{{"parameters", property_map{}}})); // no id
        inner.push_back(asValue(innerBlock("inner::Block", std::nullopt)));     // no embedded version
        inner.push_back(asValue(innerBlock("never::Registered", "9.9")));       // not a known definition
        inner.push_back(asValue(innerBlock("unversioned::Block", "9.9")));      // known, but it carries no version
        inner.push_back(asValue(innerBlock("inner::Block", "2.0")));            // same version
        expect(gr::detail::checkEmbeddedVersionConsistency(knownDefinitions, definitionWithInnerBlocks(std::move(inner))).has_value());
    };

    "an inner block authored against another version is reported"_test = [&] {
        Tensor<gr::pmt::Value> inner;
        inner.push_back(asValue(innerBlock("inner::Block", "1.0")));
        const auto result = gr::detail::checkEmbeddedVersionConsistency(knownDefinitions, definitionWithInnerBlocks(std::move(inner)));
        expect(!result.has_value());
        if (!result.has_value()) {
            const std::string message = std::string(result.error().message);
            expect(message.find("inner::Block") != std::string::npos) << message;
            expect(message.find("1.0") != std::string::npos && message.find("2.0") != std::string::npos) << "both versions are named: " << message;
            expect(message.find("outer::Block") != std::string::npos) << "the definition is named: " << message;
        }
    };
};

const boost::ut::suite ReadUriTests = [] {
    using namespace boost::ut;

    "an unreadable location is an error"_test = [] {
        const auto result = gr::detail::readUriToString("/this/path/does/not/exist/qa_plugins_test.yaml");
        expect(!result.has_value());
        if (!result.has_value()) {
            expect(result.error().message.find("Failed to read URI") != std::string::npos);
        }
    };

    "a readable file is returned verbatim"_test = [] {
        const auto result = gr::detail::readUriToString(__FILE__);
        expect(result.has_value());
        if (result.has_value()) {
            expect(result->find("readUriToString") != std::string::npos) << "the file is this source file";
        }
    };
};

int main() { /* not needed for UT */ }
