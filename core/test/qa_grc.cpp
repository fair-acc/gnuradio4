#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <build_configure.hpp>

#include <gnuradio-4.0/Graph_yaml_importer.hpp>
#include <gnuradio-4.0/MemoryAllocators.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/YamlPmt.hpp>
#include <gnuradio-4.0/basic/CommonBlocks.hpp>

#include <boost/ut.hpp>

#include "CollectionTestBlocks.hpp"

#include <gnuradio-4.0/GrBasicBlocks.hpp>
#include <gnuradio-4.0/GrTestingBlocks.hpp>
#include <gnuradio-4.0/qa_grc.hpp>

#include "TestBlockRegistryContext.hpp"
#include <gnuradio-4.0/BlockModel.hpp>

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
    using namespace gr;
    gr::property_map firstYaml  = *pmt::yaml::deserialize(first);
    gr::property_map secondYaml = *pmt::yaml::deserialize(second);

    // Basic check for blocks
    using BlockMinData = std::pair<std::string, std::string>;
    std::vector<BlockMinData> firstBlocks;
    std::vector<BlockMinData> secondBlocks;
    std::set<BlockMinData>    seenBlocks;

    for (const auto& block : test::get_value_or_fail<Tensor<Value>>(firstYaml.find_value("blocks").value())) {
        const auto&  blockMap = test::get_value_or_fail<gr::property_map>(block);
        BlockMinData data{test::get_value_or_fail<std::string>(blockMap.find_value("id").value()), gr::detail::getProperty<std::string>(blockMap, "properties"sv, "name"sv).value_or(""s)};
        firstBlocks.push_back(data);
        seenBlocks.insert(data);
    }

    for (const auto& block : test::get_value_or_fail<Tensor<Value>>(secondYaml.find_value("blocks").value())) {
        const auto& blockMap = test::get_value_or_fail<gr::property_map>(block);
        std::println("Current block {}", blockMap);
        BlockMinData data{test::get_value_or_fail<std::string>(blockMap.find_value("id").value()), gr::detail::getProperty<std::string>(blockMap, "properties"sv, "name"sv).value_or(""s)};
        secondBlocks.push_back(data);
        seenBlocks.erase(data);
    }

    for (const auto& block : seenBlocks) {
        std::print("Missing id={} name={}\n", block.first, block.second);
    }

    if (seenBlocks.empty() && (test::get_value_or_fail<Tensor<Value>>(firstYaml.find_value("connections").value()).size() == test::get_value_or_fail<Tensor<Value>>(secondYaml.find_value("connections").value()).size())) {
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

template<pmt::yaml::TypeTagMode tagMode = pmt::yaml::TypeTagMode::Auto>
std::string ymlDecodeEncode(std::string_view yml, std::source_location location = std::source_location::current()) {
    const auto yaml = pmt::yaml::deserialize(yml);
    if (!yaml) {
        throw gr::exception(std::format("Could not parse yaml: \n{}", pmt::yaml::formatAsLines(yml, yaml.error())), location);
    }

    return pmt::yaml::serialize<tagMode>(yaml.value());
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
    parameters:
      name: ArraySource<float64>
      ui_constraints:
        x: !!float32 43
        y: !!float32 7070

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

            for (const auto& block : graph->blocks()) {
                if (block->name() == "ArraySource<float64>") {
                    expect(block->settings().applyStagedParameters().forwardParameters.empty());
                    expect(test::get_value_or_fail<property_map>(block->settings().get("ui_constraints").value()) == gr::property_map{{"x", 43.f}, {"y", 7070.f}});
                }
            }

            auto graphSavedSrc = gr::saveGrc(context->loader, *graph);
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
            auto graphSavedSrc = gr::saveGrc(context->loader, *graph1);
            auto graph2        = gr::loadGrc(context->loader, graphSavedSrc);
            expect(eq(collectBlocks(*graph1), collectBlocks(*graph2)));
            expect(eq(collectEdges(*graph1), collectEdges(*graph2)));
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

            const auto graphSrc1 = ymlDecodeEncode<pmt::yaml::TypeTagMode::Auto>(pluginsTestGrc);
            const auto graphSrc2 = ymlDecodeEncode<pmt::yaml::TypeTagMode::None>(pluginsTestGrc);
            std::println("yml-before:\n {}\nwith type-tags:\n{}\nwithout type tags:\n{}", pluginsTestGrc, graphSrc1, graphSrc2);

            auto graph  = gr::loadGrc(context->loader, graphSrc1);
            auto graph2 = gr::loadGrc(context->loader, pluginsTestGrc);

            expect(eq(graph->blocks().size(), 4UZ));

            auto graphSavedSrc = gr::saveGrc(context->loader, *graph);

            expect(checkAndPrintMissingBlocks(graphSrc2, graphSavedSrc));

            gr::scheduler::Simple sched;
            if (auto ret = sched.exchange(std::move(graph)); !ret) {
                throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
            }
            expect(sched.runAndWait().has_value());
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
        - [multiplier1, INPUT, in, in]
        - [multiplier2, OUTPUT, out, out]

connections:
  - [main_source, 0, multiplier, 0]
  - [multiplier, 0, chained_multiplier, 0]
  - [chained_multiplier, 0, counter, 0]
  - [counter, 0, sink, 0]
)";

            const auto graphSrc = ymlDecodeEncode(pluginsTestGrc);
            auto       graph    = gr::loadGrc(context->loader, graphSrc);

            expect(eq(graph->blocks().size(), 5UZ));

            auto graphSavedSrc = gr::saveGrc(context->loader, *graph);

            expect(checkAndPrintMissingBlocks(graphSrc, graphSavedSrc));
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
        - [multiplier1, INPUT, in, in]
        - [multiplier2, OUTPUT, out, out]

connections:
  - [main_source, 0, multiplier, 0]
  - [multiplier, 0, chained_multiplier, 0]
  - [chained_multiplier, 0, counter, 0]
  - [counter, 0, sink, 0]
)";

            const auto graphSrc = ymlDecodeEncode(pluginsTestGrc);
            auto       graph    = gr::loadGrc(context->loader, graphSrc);

            expect(eq(graph->blocks().size(), 5UZ));

            auto graphSavedSrc = gr::saveGrc(context->loader, *graph);

            expect(checkAndPrintMissingBlocks(graphSrc, graphSavedSrc));
        } catch (const std::string& e) {
            std::println(std::cerr, "Unexpected exception: {}", e);
            expect(false);
        }
    };

    "Basic graph and managed subgraph loading and storing using plugins"_test = [&] {
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
        scheduler:
          id: good::GoodMathScheduler
          parameters:
            defaultPoolName: default_cpu
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
            - [multiplier1, INPUT, in, in]
            - [multiplier2, OUTPUT, out, out]

    connections:
      - [main_source, 0, multiplier, 0]
      - [multiplier, 0, chained_multiplier, 0]
      - [chained_multiplier, 0, counter, 0]
      - [counter, 0, sink, 0]
    )";

            const auto graphSrc = ymlDecodeEncode(pluginsTestGrc);
            auto       graph    = gr::loadGrc(context->loader, graphSrc);

            expect(eq(graph->blocks().size(), 5UZ));

            auto graphSavedSrc = gr::saveGrc(context->loader, *graph);

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
                gr::graph::forEachEdge<gr::block::Category::NormalBlock>(*graph, [&expectedSizes](const auto& edge) {
                    auto it = expectedSizes.find(edge.minBufferSize());
                    if (it != expectedSizes.end()) {
                        expectedSizes.erase(it);
                    } else {
                        expect(false);
                    }
                });
                expect(expectedSizes.empty());

                expect(graph->connectPendingEdges());

                std::size_t thresholdSize = 2 * 8192UZ;
                gr::graph::forEachEdge<gr::block::Category::NormalBlock>(*graph, [&thresholdSize](const auto& edge) { //
                    expect(thresholdSize >= edge.bufferSize());
                });
            }

            auto graphSavedSrc = gr::saveGrc(context->loader, *graph);

            {
                auto               graphDuplicate = gr::loadGrc(context->loader, graphSrc);
                std::unordered_set expectedSizes{1024UZ, 2048UZ, 8192UZ};
                gr::graph::forEachEdge<gr::block::Category::NormalBlock>(*graph, [&expectedSizes](const auto& edge) {
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

        expect(graph1.connect(arraySource0, "outA#0"s, arraySink, "inB#1"s).has_value());
        expect(graph1.connect(arraySource1, "outA#1"s, arraySink, "inB#0"s).has_value());
        expect(graph1.connect(arraySource0, "outB#0"s, arraySink, "inA#0"s).has_value());
        expect(graph1.connect(arraySource1, "outB#1"s, arraySink, "inA#1"s).has_value());

        expect(graph1.reconnectAllEdges());

        expect(eq(graph1.edges().size(), 4UZ));
        for (const auto& edge : graph1.edges()) {
            expect(edge.state() == Edge::EdgeState::Connected);
        }

        const auto graph1Saved = gr::saveGrc(context->loader, graph1);
        const auto graph2      = gr::loadGrc(context->loader, graph1Saved);

        expect(eq(collectBlocks(graph1), collectBlocks(*graph2)));
        expect(eq(collectEdges(graph1), collectEdges(*graph2)));
    };

    "Vector of Ports"_test = [&] {
        using namespace gr;

        gr::Graph graph1;
        auto&     vectorSink    = graph1.emplaceBlock<gr::testing::VectorSink<double>>();
        auto&     vectorSource0 = graph1.emplaceBlock<gr::testing::VectorSource<double>>({{"name", "VectorSource0"}});
        auto&     vectorSource1 = graph1.emplaceBlock<gr::testing::VectorSource<double>>({{"name", "VectorSource1"}});

        expect(graph1.connect(vectorSource0, "outA#0"s, vectorSink, "inB#1"s).has_value());
        expect(graph1.connect(vectorSource0, "outB#0"s, vectorSink, "inA#0"s).has_value());
        expect(graph1.connect(vectorSource1, "outA#1"s, vectorSink, "inB#0"s).has_value());
        expect(graph1.connect(vectorSource1, "outB#1"s, vectorSink, "inA#1"s).has_value());

        expect(graph1.reconnectAllEdges());

        expect(eq(graph1.edges().size(), 4UZ));
        for (const auto& edge : graph1.edges()) {
            expect(edge.state() == Edge::EdgeState::Connected);
        }

        const auto graph1Saved = gr::saveGrc(context->loader, graph1);
        const auto graph2      = gr::loadGrc(context->loader, graph1Saved);

        expect(eq(collectBlocks(graph1), collectBlocks(*graph2)));
        expect(eq(collectEdges(graph1), collectEdges(*graph2)));
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
            const auto expectedString       = std::string("abc");
            const bool expectedBool         = true;
            const auto expectedComplex      = std::complex<double>(1., 1.);
            const auto expectedStringVector = std::vector<std::string>{"a", "b", "c"};
            const auto expectedBoolVector   = std::vector<bool>{true, false, true};
            const auto expectedDoubleVector = std::vector{1., 2., 3.};
            const auto expectedInt16Vector  = std::vector<std::int16_t>{1, 2, 3};

            using cd                         = std::complex<double>;
            const auto expectedComplexVector = std::vector<std::complex<double>>{cd{1., 1.}, cd{2., 2.}, cd{3., 3.}};

            std::ignore = graph1.emplaceBlock<gr::testing::ArraySink<double>>({{"bool_setting", bool(expectedBool)}, {"string_setting", expectedString}, {"complex_setting", expectedComplex}, //
                {"bool_vector", expectedBoolVector}, {"string_vector", expectedStringVector}, {"double_vector", expectedDoubleVector}, {"int16_vector", expectedInt16Vector}, {"complex_vector", expectedComplexVector}});

            const auto graph1Saved = gr::saveGrc(context->loader, graph1);
            const auto graph2      = gr::loadGrc(context->loader, graph1Saved);
            gr::graph::forEachBlock<gr::block::Category::NormalBlock>(*graph2, [&](const auto node) {
                const auto settings = node->settings().get();
                expect(eq(test::get_value_or_fail<bool>(settings.find_value("bool_setting").value()), expectedBool));
                expect(eq(test::get_value_or_fail<std::string>(settings.find_value("string_setting").value()), expectedString));
                expect(eq(test::get_value_or_fail<std::complex<double>>(settings.find_value("complex_setting").value()), expectedComplex));
                expect(eq(test::get_value_or_fail<std::vector<bool>>(settings.find_value("bool_vector").value()), expectedBoolVector));
                expect(eq(test::get_value_or_fail<std::vector<double>>(settings.find_value("double_vector").value()), expectedDoubleVector));
                expect(eq(test::get_value_or_fail<std::vector<int16_t>>(settings.find_value("int16_vector").value()), expectedInt16Vector));
                expect(eq(test::get_value_or_fail<std::vector<std::complex<double>>>(settings.find_value("complex_vector").value()), expectedComplexVector));
            });

            expect(eq(collectBlocks(graph1), collectBlocks(*graph2)));
            expect(eq(collectEdges(graph1), collectEdges(*graph2)));
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
            expect(block.settings().getStoredAll().size() == 2);
            expect(block.settings().set({{"name", "ArraySink1+10"}}, SettingsCtx{now + 10'000'000'000ULL, "1"}).empty());
            expect(block.settings().getStoredAll().size() == 2);
            expect(block.settings().set({{"name", "ArraySink1+20"}}, SettingsCtx{now + 20'000'000'000ULL, "1"}).empty());
            expect(block.settings().getStoredAll().size() == 2);
            expect(block.settings().set({{"name", "ArraySink2"}}, SettingsCtx{now, "2"}).empty());
            expect(block.settings().getStoredAll().size() == 3);
            expect(block.settings().set({{"name", "ArraySink3"}}, SettingsCtx{now, "3"}).empty()); // string-encoded int
            expect(block.settings().getStoredAll().size() == 4);

            const auto graph1Saved = gr::saveGrc(context->loader, graph1);

            const auto graph2 = gr::loadGrc(context->loader, graph1Saved);

            gr::graph::forEachBlock<gr::block::Category::NormalBlock>(*graph2, [&](const auto node) {
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

                        expect(eq(test::get_value_or_fail<std::string>(settingsMap.find_value("name").value()), expectedName));
                    }
                }
            });

            expect(eq(collectBlocks(graph1), collectBlocks(*graph2)));
            expect(eq(collectEdges(graph1), collectEdges(*graph2)));
        } catch (const std::string& e) {
            std::println(std::cerr, "Unexpected exception: {}", e);
            expect(false);
        }
    };
};

const boost::ut::suite GetPropertyTests = [] {
    using namespace boost::ut;
    using namespace std::string_view_literals;

    "getProperty<property_map>: returned map survives source destruction"_test = [] {
        // Pre-fix this returned a view-mode ValueMap aliasing the source's freed bytes.
        gr::property_map result;
        {
            gr::property_map nested;
            nested.insert_or_assign("k1", std::int64_t{100});
            nested.insert_or_assign("k2", std::int64_t{200});
            gr::property_map parent;
            parent.insert_or_assign("nested", nested);

            auto exp = gr::detail::getProperty<gr::property_map>(parent, "nested"sv);
            expect(exp.has_value()) << "getProperty must succeed";
            result = std::move(*exp);
        }
        expect(eq(result.size(), 2UZ));
        expect(result.contains("k1"));
        expect(result.contains("k2"));
        expect(eq(test::get_value_or_fail<std::int64_t>(result.find_value("k1").value()), std::int64_t{100}));
        expect(eq(test::get_value_or_fail<std::int64_t>(result.find_value("k2").value()), std::int64_t{200}));
    };

    "getProperty<property_map>: two-level recursive lookup returns owning copy"_test = [] {
        gr::property_map result;
        {
            gr::property_map leaf;
            leaf.insert_or_assign("alpha", float{1.5f});
            gr::property_map mid;
            mid.insert_or_assign("inner", leaf);
            gr::property_map outer;
            outer.insert_or_assign("mid", mid);

            auto exp = gr::detail::getProperty<gr::property_map>(outer, "mid"sv, "inner"sv);
            expect(exp.has_value());
            result = std::move(*exp);
        }
        expect(eq(result.size(), 1UZ));
        expect(result.contains("alpha"));
        expect(eq(test::get_value_or_fail<float>(result.find_value("alpha").value()), 1.5f));
    };

    "getProperty<property_map>: nested Tensor<Value> entries survive"_test = [] {
        gr::property_map result;
        {
            Tensor<gr::Value> entries;
            entries.emplace_back(gr::Value{std::string_view{"first"}});
            entries.emplace_back(gr::Value{std::string_view{"second"}});

            gr::property_map nested;
            nested.insert_or_assign("items", std::move(entries));
            gr::property_map parent;
            parent.insert_or_assign("nested", nested);

            auto exp = gr::detail::getProperty<gr::property_map>(parent, "nested"sv);
            expect(exp.has_value());
            result = std::move(*exp);
        }
        const Value itemsValue = result.find_value("items").value();
        const auto  items      = itemsValue.value_or(Tensor<gr::Value>{});
        expect(eq(items.size(), 2UZ));
        expect(eq(items.data()[0].value_or(std::string_view{}), std::string_view{"first"}));
        expect(eq(items.data()[1].value_or(std::string_view{}), std::string_view{"second"}));
    };

    "getProperty<std::string>: returns owning string"_test = [] {
        std::string result;
        {
            gr::property_map parent;
            parent.insert_or_assign("name", std::string_view{"alpha"});
            auto exp = gr::detail::getProperty<std::string>(parent, "name"sv);
            expect(exp.has_value());
            result = std::move(*exp);
        }
        expect(eq(result, std::string{"alpha"}));
    };

    "getProperty<property_map>: returned map is owning (not view-mode)"_test = [] {
        gr::property_map parent;
        gr::property_map nested;
        nested.insert_or_assign("k", std::int32_t{1});
        parent.insert_or_assign("nested", nested);

        const auto exp = gr::detail::getProperty<gr::property_map>(parent, "nested"sv);
        expect(exp.has_value());
        expect(!exp->is_view()) << "returned property_map must own its bytes";
    };

    "getProperty: missing key returns unexpected"_test = [] {
        const gr::property_map parent;
        const auto             exp = gr::detail::getProperty<gr::property_map>(parent, "missing"sv);
        expect(!exp.has_value());
    };

    "getProperty<property_map>: returned map inherits source map's PMR allocator"_test = [] {
        gr::allocator::pmr::CountingResource mr;
        gr::property_map                     parent{&mr};
        gr::property_map                     nested{&mr};
        nested.insert_or_assign("k", std::int32_t{7});
        parent.insert_or_assign("nested", nested);

        const auto allocsBefore = mr.allocCount;

        const auto exp = gr::detail::getProperty<gr::property_map>(parent, "nested"sv);
        expect(exp.has_value());
        expect(exp->resource() == &mr) << "returned property_map must inherit the source map's PMR resource";
        expect(gt(mr.allocCount, allocsBefore)) << "the owning materialisation must allocate on the source's arena";
    };
};

} // namespace gr::qa_grc_test

int main() { return boost::ut::cfg<boost::ut::override>.run(); }
