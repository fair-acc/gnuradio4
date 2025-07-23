#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <build_configure.hpp>

#include <gnuradio-4.0/Graph_yaml_importer.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/YamlPmt.hpp>
#include <gnuradio-4.0/basic/CommonBlocks.hpp>

#include <boost/ut.hpp>

#include "CollectionTestBlocks.hpp"

#include <GrBasicBlocks.hpp>
#include <GrTestingBlocks.hpp>
#include <qa_grc.hpp>

#include "TestBlockRegistryContext.hpp"
#include "gnuradio-4.0/BlockModel.hpp"

#include <gnuradio-4.0/meta/UnitTestHelper.hpp>

namespace ut = boost::ut;

auto makeTestContext() {
    return std::make_unique<TestContext>(                      //
        paths{"core/test/plugins", "test/plugins", "plugins"}, // plugin paths
        gr::blocklib::initGrBasicBlocks,                       //
        gr::blocklib::initGrTestingBlocks,                     //
        gr::blocklib::initqa_grc);
}

namespace {
auto collectBlocks(const gr::Graph& graph) {
    std::set<std::string> result;
    gr::graph::forEachBlock<gr::block::Category::NormalBlock>(graph, [&](const auto node) { result.insert(std::format("{}-{}", node->name(), node->typeName())); });
    return result;
}

auto collectEdges(const gr::Graph& graph) {
    std::set<std::string> result;
    gr::graph::forEachEdge<gr::block::Category::NormalBlock>(graph, [&](const auto& edge) {
        auto portDefinitionToString = [](const gr::PortDefinition& definition) {
            return std::visit(gr::meta::overloaded(                                                                                                                   //
                                  [](const gr::PortDefinition::IndexBased& _definition) { return std::format("{}#{}", _definition.topLevel, _definition.subIndex); }, //
                                  [](const gr::PortDefinition::StringBased& _definition) { return _definition.name; }),                                               //
                definition.definition);
        };
        result.insert(std::format("{}#{} - {}#{}",                                           //
            edge.sourceBlock()->name(), portDefinitionToString(edge.sourcePortDefinition()), //
            edge.destinationBlock()->name(), portDefinitionToString(edge.destinationPortDefinition())));
    });
    return result;
}

bool checkAndPrintMissingBlocks(const std::string& first, const std::string& second) {
    gr::property_map firstYaml  = *pmtv::yaml::deserialize(first);
    gr::property_map secondYaml = *pmtv::yaml::deserialize(second);

    // Basic check for blocks
    using BlockMinData = std::pair<std::string, std::string>;
    std::vector<BlockMinData> firstBlocks;
    std::vector<BlockMinData> secondBlocks;
    std::set<BlockMinData>    seenBlocks;

    for (const auto& block : std::get<std::vector<pmtv::pmt>>(firstYaml.at("blocks"))) {
        const auto&  blockMap = std::get<gr::property_map>(block);
        BlockMinData data{std::get<std::string>(blockMap.at("id"s)), gr::detail::getProperty<std::string>(blockMap, "properties"sv, "name"sv).value_or(""s)};
        firstBlocks.push_back(data);
        seenBlocks.insert(data);
    }

    for (const auto& block : std::get<std::vector<pmtv::pmt>>(secondYaml.at("blocks"))) {
        const auto& blockMap = std::get<gr::property_map>(block);
        std::println("Current block {}", blockMap);
        BlockMinData data{std::get<std::string>(blockMap.at("id"s)), gr::detail::getProperty<std::string>(blockMap, "properties"sv, "name"sv).value_or(""s)};
        secondBlocks.push_back(data);
        seenBlocks.erase(data);
    }

    for (const auto& block : seenBlocks) {
        std::print("Missing id={} name={}\n", block.first, block.second);
    }

    if (seenBlocks.empty() && (std::get<std::vector<pmtv::pmt>>(firstYaml.at("connections")).size() == std::get<std::vector<pmtv::pmt>>(secondYaml.at("connections")).size())) {
        return true;
    }

    std::print("Blocks in first:\n");
    for (const auto& data : firstBlocks) {
        std::print("    id={} name={}\n", data.first, data.second);
    }
    std::print("Blocks in second:\n");
    for (const auto& data : secondBlocks) {
        std::print("    id={} name={}\n", data.first, data.second);
    }

    return false;
}

} // namespace

using namespace boost::ut;

namespace gr::qa_grc_test {

template<pmtv::yaml::TypeTagMode tagMode = pmtv::yaml::TypeTagMode::Auto>
std::string ymlDecodeEncode(std::string_view yml, std::source_location location = std::source_location::current()) {
    const auto yaml = pmtv::yaml::deserialize(yml);
    if (!yaml) {
        throw gr::exception(std::format("Could not parse yaml: \n{}", pmtv::yaml::formatAsLines(yml, yaml.error())), location);
    }

    return pmtv::yaml::serialize<tagMode>(yaml.value());
}

const boost::ut::suite BasicGrcTests = [] {
    using namespace gr::test;

    auto context = makeTestContext();

    constexpr std::string_view testGrc = R"(
blocks:
  - id: gr::testing::ArraySink<float64>
    parameters:
      name: ArraySinkImpl<float64, true, 42>
  - id: gr::testing::ArraySource<float64>
    parameters:
      name: ArraySourceOne<float64>
  - id: gr::testing::ArraySource<float64>
    ui_constraints:
      x: !!float32 43
      y: !!float32 7070
    parameters:
      name: ArraySource<float64>

connections:
  - [ArraySourceOne<float64>, [0, 0], 'ArraySinkImpl<float64, true, 42>', [1, 1]]
  - [ArraySourceOne<float64>, [0, 1], 'ArraySinkImpl<float64, true, 42>', [1, 0]]
  - [ArraySource<float64>, [1, 0], 'ArraySinkImpl<float64, true, 42>', [0, 0]]
  - [ArraySource<float64>, [1, 1], 'ArraySinkImpl<float64, true, 42>', [0, 1]]
)";

    "Basic graph loading and storing"_test = [&] {
        try {
            using namespace gr;
            for (const auto& block : context->loader.availableBlocks()) {
                std::print("Block {} is known\n", block);
            }

            const auto graphSrc = ymlDecodeEncode(testGrc);
            auto       graph    = gr::loadGrc(context->loader, graphSrc);

            for (const auto& block : graph.blocks()) {
                if (block->name() == "ArraySource<float64>") {
                    expect(std::get<property_map>(block->settings().get("ui_constraints").value()) == gr::property_map{{"x", 43.f}, {"y", 7070.f}});
                }
            }

            auto graphSavedSrc = gr::saveGrc(context->loader, graph);
            expect(checkAndPrintMissingBlocks(graphSrc, graphSavedSrc));
        } catch (const std::string& e) {
            std::println(std::cerr, "Unexpected exception: {}", e);
            expect(false);
        }
    };

    "Save and load"_test = [&] {
        // Test if we get the same graph when saving it and loading the saved
        // data into another graph
        using namespace gr;

        try {
            auto graph1        = gr::loadGrc(context->loader, testGrc);
            auto graphSavedSrc = gr::saveGrc(context->loader, graph1);
            auto graph2        = gr::loadGrc(context->loader, graphSavedSrc);
            expect(eq(collectBlocks(graph1), collectBlocks(graph2)));
            expect(eq(collectEdges(graph1), collectEdges(graph2)));
        } catch (const std::string& e) {
            std::println(std::cerr, "Unexpected exception: {}", e);
            expect(false);
        }
    };
};

#if not defined(__EMSCRIPTEN__) // && not defined(__APPLE__)
const boost::ut::suite PluginsGrcTests = [] {
    auto context = makeTestContext();

    "Basic graph loading and storing using plugins"_test = [&] {
        try {
            using namespace gr;

            constexpr std::string_view pluginsTestGrc = R"(
blocks:
  - id: good::fixed_source<float64>
    parameters:
      name: main_source
      event_count: 100
      unknown_property: 42
  - id: good::multiply<float64>
    parameters:
      name: multiplier
  - id: builtin_counter<float64>
    parameters:
      name: counter
  - id: good::cout_sink<float64>
    parameters:
      name: sink
      total_count: 100
      unknown_property: 42

connections:
  - [main_source, 0, multiplier, 0]
  - [multiplier, 0, counter, 0]
  - [counter, 0, sink, 0]
)";

            const auto graphSrc1 = ymlDecodeEncode<pmtv::yaml::TypeTagMode::Auto>(pluginsTestGrc);
            const auto graphSrc2 = ymlDecodeEncode<pmtv::yaml::TypeTagMode::None>(pluginsTestGrc);
            std::println("yml-before:\n {}\nwith type-tags:\n{}\nwithout type tags:\n{}", pluginsTestGrc, graphSrc1, graphSrc2);

            auto graph  = gr::loadGrc(context->loader, graphSrc1);
            auto graph2 = gr::loadGrc(context->loader, pluginsTestGrc);

            expect(eq(graph.blocks().size(), 4UZ));

            auto graphSavedSrc = gr::saveGrc(context->loader, graph);

            expect(checkAndPrintMissingBlocks(graphSrc2, graphSavedSrc));

            gr::scheduler::Simple scheduler(std::move(graph));
            expect(scheduler.runAndWait().has_value());
        } catch (const std::string& e) {
            std::println(std::cerr, "Unexpected exception: {}", e);
            expect(false);
        }
    };

    "Basic graph and subgraph loading and storing using plugins"_test = [&] {
        try {
            using namespace gr;

            constexpr std::string_view pluginsTestGrc = R"(
blocks:
  - id: good::fixed_source<float64>
    parameters:
      name: main_source
      event_count: 100
      unknown_property: 42
  - id: good::multiply<float64>
    parameters:
      name: multiplier
  - id: builtin_counter<float64>
    parameters:
      name: counter
  - id: good::cout_sink<float64>
    parameters:
      name: sink
      total_count: 100
      unknown_property: 42
  - id: SUBGRAPH
    parameters:
      name: chained_multiplier
    graph:
      blocks:
        - id: good::multiply<float64>
          parameters:
            name: multiplier1
        - id: good::multiply<float64>
          parameters:
            name: multiplier2
      connections:
        - [multiplier1, 0, multiplier2, 0]
      exported_ports:
        - [multiplier1, INPUT, in]
        - [multiplier2, OUTPUT, out]

connections:
  - [main_source, 0, multiplier, 0]
  - [multiplier, 0, chained_multiplier, 0]
  - [chained_multiplier, 0, counter, 0]
  - [counter, 0, sink, 0]
)";

            const auto graphSrc = ymlDecodeEncode(pluginsTestGrc);
            auto       graph    = gr::loadGrc(context->loader, graphSrc);

            expect(eq(graph.blocks().size(), 5UZ));

            auto graphSavedSrc = gr::saveGrc(context->loader, graph);

            expect(checkAndPrintMissingBlocks(graphSrc, graphSavedSrc));
        } catch (const std::string& e) {
            std::println(std::cerr, "Unexpected exception: {}", e);
            expect(false);
        }
    };
};
#endif

#if not defined(__EMSCRIPTEN__) // && not defined(__APPLE__)
const boost::ut::suite PortTests = [] {
    using namespace boost::ut;
    using namespace boost::ext::ut;
    using namespace gr::test;

    auto context = makeTestContext();

    "Port buffer sizes"_test = [&] {
        constexpr std::string_view testGrc = R"(
blocks:
  - id: good::fixed_source<float64>
    parameters:
      name: main_source
      event_count: 100
      unknown_property: 42
  - id: good::multiply<float64>
    parameters:
      name: multiplier
  - id: builtin_counter<float64>
    parameters:
      name: counter
  - id: good::cout_sink<float64>
    parameters:
      name: sink
      total_count: 100
      unknown_property: 42

connections:
  - [main_source, 0, multiplier, 0, 1024]
  - [multiplier, 0, counter, 0, 2048]
  - [counter, 0, sink, 0, 8192]
)";

        try {
            using namespace gr;
            const auto graphSrc = ymlDecodeEncode(testGrc);

            auto graph = gr::loadGrc(context->loader, graphSrc);

            {
                std::unordered_set expectedSizes{1024UZ, 2048UZ, 8192UZ};
                gr::graph::forEachEdge<gr::block::Category::NormalBlock>(graph, [&expectedSizes](const auto& edge) {
                    auto it = expectedSizes.find(edge.minBufferSize());
                    if (it != expectedSizes.end()) {
                        expectedSizes.erase(it);
                    } else {
                        expect(false);
                    }
                });
                expect(expectedSizes.empty());

                expect(graph.connectPendingEdges());

                std::size_t thresholdSize = 2 * 8192UZ;
                gr::graph::forEachEdge<gr::block::Category::NormalBlock>(graph, [&thresholdSize](const auto& edge) { //
                    expect(thresholdSize >= edge.bufferSize());
                });
            }

            auto graphSavedSrc = gr::saveGrc(context->loader, graph);

            {
                auto               graphDuplicate = gr::loadGrc(context->loader, graphSrc);
                std::unordered_set expectedSizes{1024UZ, 2048UZ, 8192UZ};
                gr::graph::forEachEdge<gr::block::Category::NormalBlock>(graph, [&expectedSizes](const auto& edge) {
                    auto it = expectedSizes.find(edge.minBufferSize());
                    if (it != expectedSizes.end()) {
                        expectedSizes.erase(it);
                    } else {
                        expect(false);
                    }
                });
                expect(expectedSizes.empty());
            }
        } catch (const std::string& e) {
            std::println(std::cerr, "Unexpected exception: {}", e);
            expect(false);
        }
    };

    "Array of Ports"_test = [&] {
        using namespace gr;

        gr::Graph graph1;
        auto&     arraySink    = graph1.emplaceBlock<gr::testing::ArraySink<double>>();
        auto&     arraySource0 = graph1.emplaceBlock<gr::testing::ArraySource<double>>({{"name", "ArraySource0"}});
        auto&     arraySource1 = graph1.emplaceBlock<gr::testing::ArraySource<double>>({{"name", "ArraySource1"}});

        expect(eq(ConnectionResult::SUCCESS, graph1.connect<"outA", 0>(arraySource0).to<"inB", 1>(arraySink)));
        expect(eq(ConnectionResult::SUCCESS, graph1.connect<"outA", 1>(arraySource1).to<"inB", 0>(arraySink)));
        expect(eq(ConnectionResult::SUCCESS, graph1.connect<"outB", 0>(arraySource0).to<"inA", 0>(arraySink)));
        expect(eq(ConnectionResult::SUCCESS, graph1.connect<"outB", 1>(arraySource1).to<"inA", 1>(arraySink)));

        expect(graph1.reconnectAllEdges());

        expect(eq(graph1.edges().size(), 4UZ));
        for (const auto& edge : graph1.edges()) {
            expect(edge.state() == Edge::EdgeState::Connected);
        }

        const auto graph1Saved = gr::saveGrc(context->loader, graph1);
        const auto graph2      = gr::loadGrc(context->loader, graph1Saved);

        expect(eq(collectBlocks(graph1), collectBlocks(graph2)));
        expect(eq(collectEdges(graph1), collectEdges(graph2)));
    };

    "Vector of Ports"_test = [&] {
        using namespace gr;

        gr::Graph graph1;
        auto&     vectorSink    = graph1.emplaceBlock<gr::testing::VectorSink<double>>();
        auto&     vectorSource0 = graph1.emplaceBlock<gr::testing::VectorSource<double>>({{"name", "VectorSource0"}});
        auto&     vectorSource1 = graph1.emplaceBlock<gr::testing::VectorSource<double>>({{"name", "VectorSource1"}});

        expect(eq(ConnectionResult::SUCCESS, graph1.connect(vectorSource0, {0UZ, 0UZ}, vectorSink, {1UZ, 1UZ})));
        expect(eq(ConnectionResult::SUCCESS, graph1.connect(vectorSource0, {1UZ, 0UZ}, vectorSink, {0UZ, 0UZ})));
        expect(eq(ConnectionResult::SUCCESS, graph1.connect(vectorSource1, {0UZ, 1UZ}, vectorSink, {1UZ, 0UZ})));
        expect(eq(ConnectionResult::SUCCESS, graph1.connect(vectorSource1, {1UZ, 1UZ}, vectorSink, {0UZ, 1UZ})));

        expect(graph1.reconnectAllEdges());

        expect(eq(graph1.edges().size(), 4UZ));
        for (const auto& edge : graph1.edges()) {
            expect(edge.state() == Edge::EdgeState::Connected);
        }

        const auto graph1Saved = gr::saveGrc(context->loader, graph1);
        const auto graph2      = gr::loadGrc(context->loader, graph1Saved);

        expect(eq(collectBlocks(graph1), collectBlocks(graph2)));
        expect(eq(collectEdges(graph1), collectEdges(graph2)));
    };
};
#endif

const boost::ut::suite SettingsTests = [] {
    using namespace gr::test;
    auto context = makeTestContext();

    "Settings serialization"_test = [&] {
        try {
            using namespace gr;

            gr::Graph  graph1;
            const auto expectedString        = std::string("abc");
            const bool expectedBool          = true;
            const auto expectedComplex       = std::complex<double>(1., 1.);
            const auto expectedStringVector  = std::vector<std::string>{"a", "b", "c"};
            const auto expectedBoolVector    = std::vector<bool>{true, false, true};
            const auto expectedDoubleVector  = std::vector<double>{1., 2., 3.};
            const auto expectedInt16Vector   = std::vector<int16_t>{1, 2, 3};
            const auto expectedComplexVector = std::vector<std::complex<double>>{{1., 1.}, {2., 2.}, {3., 3.}};

            std::ignore = graph1.emplaceBlock<gr::testing::ArraySink<double>>({{"bool_setting", bool(expectedBool)}, {"string_setting", expectedString}, {"complex_setting", expectedComplex}, //
                {"bool_vector", expectedBoolVector}, {"string_vector", expectedStringVector}, {"double_vector", expectedDoubleVector}, {"int16_vector", expectedInt16Vector}, {"complex_vector", expectedComplexVector}});

            const auto graph1Saved = gr::saveGrc(context->loader, graph1);
            const auto graph2      = gr::loadGrc(context->loader, graph1Saved);
            gr::graph::forEachBlock<gr::block::Category::NormalBlock>(graph2, [&](const auto node) {
                const auto settings = node->settings().get();
                expect(eq(std::get<bool>(settings.at("bool_setting")), expectedBool));
                expect(eq(std::get<std::string>(settings.at("string_setting")), expectedString));
                expect(eq(std::get<std::complex<double>>(settings.at("complex_setting")), expectedComplex));
                expect(eq(std::get<std::vector<bool>>(settings.at("bool_vector")), expectedBoolVector));
                expect(eq(std::get<std::vector<std::string>>(settings.at("string_vector")), expectedStringVector));
                expect(eq(std::get<std::vector<double>>(settings.at("double_vector")), expectedDoubleVector));
                expect(eq(std::get<std::vector<int16_t>>(settings.at("int16_vector")), expectedInt16Vector));
                expect(eq(std::get<std::vector<std::complex<double>>>(settings.at("complex_vector")), expectedComplexVector));
            });

            expect(eq(collectBlocks(graph1), collectBlocks(graph2)));
            expect(eq(collectEdges(graph1), collectEdges(graph2)));
        } catch (const std::string& e) {
            std::println(std::cerr, "Unexpected exception: {}", e);
            expect(false);
        }
    };

    "Context settings"_test = [&] {
        try {
            using namespace gr;

            gr::Graph  graph1;
            auto&      block = graph1.emplaceBlock<gr::testing::ArraySink<double>>({{"name", "ArraySink0"}});
            const auto now   = settings::convertTimePointToUint64Ns(std::chrono::system_clock::now());
            expect(block.settings().set({{"name", "ArraySink1"}}, SettingsCtx{now, "1"}).empty());
            expect(block.settings().set({{"name", "ArraySink1+10"}}, SettingsCtx{now + 10'000'000'000ULL, "1"}).empty());
            expect(block.settings().set({{"name", "ArraySink1+20"}}, SettingsCtx{now + 20'000'000'000ULL, "1"}).empty());
            expect(block.settings().set({{"name", "ArraySink2"}}, SettingsCtx{now, "2"}).empty());
            expect(block.settings().set({{"name", "ArraySink3"}}, SettingsCtx{now, 3}).empty()); // int as context name

            const auto graph1Saved = gr::saveGrc(context->loader, graph1);
            const auto graph2      = gr::loadGrc(context->loader, graph1Saved);

            gr::graph::forEachBlock<gr::block::Category::NormalBlock>(graph2, [&](const auto node) {
                const auto& stored = node->settings().getStoredAll();
                expect(eq(node->settings().getNStoredParameters(), 6UZ));
                for (const auto& [ctx, ctxParameters] : stored) {
                    for (const auto& [ctxTime, settingsMap] : ctxParameters) {
                        std::string expectedName = "ArraySink0";
                        if (ctxTime.context == "1" && ctxTime.time == now) {
                            expectedName = "ArraySink1";
                        } else if (ctxTime.context == "1" && ctxTime.time == now + 10'000'000'000ULL) {
                            expectedName = "ArraySink1+10";
                        } else if (ctxTime.context == "1" && ctxTime.time == now + 20'000'000'000ULL) {
                            expectedName = "ArraySink1+20";
                        } else if (ctxTime.context == "2" && ctxTime.time == now) {
                            expectedName = "ArraySink2";
                        } else if (ctxTime.context == "3" && ctxTime.time == now) {
                            expectedName = "ArraySink3";
                        }

                        expect(eq(std::get<std::string>(settingsMap.at("name")), expectedName));
                    }
                }
            });

            expect(eq(collectBlocks(graph1), collectBlocks(graph2)));
            expect(eq(collectEdges(graph1), collectEdges(graph2)));
        } catch (const std::string& e) {
            std::println(std::cerr, "Unexpected exception: {}", e);
            expect(false);
        }
    };
};

} // namespace gr::qa_grc_test

int main() { return boost::ut::cfg<boost::ut::override>.run(); }
