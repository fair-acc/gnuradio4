#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <build_configure.hpp>

#include <gnuradio-4.0/Graph_yaml_importer.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/YamlPmt.hpp>
#include <gnuradio-4.0/basic/common_blocks.hpp>

#include <boost/ut.hpp>

template<typename T>
struct ArraySource : public gr::Block<ArraySource<T>> {
    std::array<gr::PortOut<T>, 2> outA{};
    std::array<gr::PortOut<T>, 2> outB{};

    GR_MAKE_REFLECTABLE(ArraySource, outA, outB);

    template<gr::OutputSpanLike TOutputSpan1, gr::OutputSpanLike TOutputSpan2, gr::OutputSpanLike TOutputSpan3, gr::OutputSpanLike TOutputSpan4>
    gr::work::Status processBulk(TOutputSpan1&, TOutputSpan2&, TOutputSpan3&, TOutputSpan4&) {
        return gr::work::Status::OK;
    }
};

template<typename T>
struct ArraySink : public gr::Block<ArraySink<T>> {
    std::array<gr::PortIn<T>, 2>                                                    inA;
    std::array<gr::PortIn<T>, 2>                                                    inB;
    gr::Annotated<bool, "bool setting">                                             bool_setting{false};
    gr::Annotated<std::string, "String setting">                                    string_setting;
    gr::Annotated<std::complex<double>, "std::complex settings">                    complex_setting;
    gr::Annotated<std::vector<bool>, "Bool vector setting">                         bool_vector;
    gr::Annotated<std::vector<std::string>, "String vector setting">                string_vector;
    gr::Annotated<std::vector<double>, "Double vector setting">                     double_vector;
    gr::Annotated<std::vector<int16_t>, "int16_t vector setting">                   int16_vector;
    gr::Annotated<std::vector<std::complex<double>>, "std::complex vector setting"> complex_vector;

    GR_MAKE_REFLECTABLE(ArraySink, inA, inB, bool_setting, string_setting, complex_setting, bool_vector, string_vector, double_vector, int16_vector, complex_vector);

    template<gr::InputSpanLike TInputSpan1, gr::InputSpanLike TInputSpan2, gr::InputSpanLike TInputSpan3, gr::InputSpanLike TInputSpan4>
    gr::work::Status processBulk(TInputSpan1&, TInputSpan2&, TInputSpan3&, TInputSpan4&) {
        return gr::work::Status::OK;
    }
};

struct TestContext {
    explicit TestContext(std::vector<std::filesystem::path> paths) : loader(registry, std::move(paths)) {}

    gr::BlockRegistry registry;
    gr::PluginLoader  loader;
};

namespace {
auto collectBlocks(const gr::Graph& graph) {
    std::set<std::string> result;
    graph.forEachBlock([&](const auto& node) { result.insert(fmt::format("{}-{}", node.name(), node.typeName())); });
    return result;
}

auto collectEdges(const gr::Graph& graph) {
    std::set<std::string> result;
    graph.forEachEdge([&](const auto& edge) {
        auto portDefinitionToString = [](const gr::PortDefinition& definition) {
            return std::visit(gr::meta::overloaded(                                                                                                                   //
                                  [](const gr::PortDefinition::IndexBased& _definition) { return fmt::format("{}#{}", _definition.topLevel, _definition.subIndex); }, //
                                  [](const gr::PortDefinition::StringBased& _definition) { return _definition.name; }),                                               //
                definition.definition);
        };
        result.insert(fmt::format("{}#{} - {}#{}",                                          //
            edge.sourceBlock().name(), portDefinitionToString(edge.sourcePortDefinition()), //
            edge.destinationBlock().name(), portDefinitionToString(edge.destinationPortDefinition())));
    });
    return result;
}

bool checkAndPrintMissingLines(const std::string& first, const std::string& second) {
    std::istringstream              ssSecond(second);
    std::unordered_set<std::string> linesSecond;
    std::string                     line;
    while (std::getline(ssSecond, line)) {
        linesSecond.insert(line);
    }

    std::istringstream ssFirst(first);
    bool               allLinesFound = true;
    size_t             lineNumber    = 0;
    while (std::getline(ssFirst, line)) {
        ++lineNumber;
        if (std::ranges::all_of(line, [](char c) { return std::isspace(c); })) {
            continue;
        }
        if (!linesSecond.contains(line)) {
            fmt::println(stderr, "missing line {}:\n{}", lineNumber, line);
            allLinesFound = false;
        }
    }
    if (!allLinesFound) {
        fmt::println(stderr, "\nin:\n{}", second);
    }

    return allLinesFound;
}

auto getContext() {
    static auto ctx = [] {
        auto context = std::make_shared<TestContext>(std::vector<std::filesystem::path>{TESTS_BINARY_PATH "/plugins"});
        gr::registerBlock<builtin_counter, double>(context->loader.registry());
        gr::registerBlock<ArraySource, double>(context->loader.registry());
        gr::registerBlock<ArraySink, double>(context->loader.registry());
        return context;
    }();
    return ctx;
};

} // namespace

using namespace boost::ut;

namespace gr::qa_grc_test {

constexpr std::string_view testGrc = R"(
blocks:
  - name: ArraySink<double>
    id: ArraySink
    parameters:
      name: ArraySink<double>
  - name: ArraySource<double>
    id: ArraySource
    parameters:
      name: ArraySource<double>
  - name: ArraySource<double>
    id: ArraySource
    parameters:
      name: ArraySource<double>
connections:
  - [ArraySource<double>, [0, 0], ArraySink<double>, [1, 1]]
  - [ArraySource<double>, [0, 1], ArraySink<double>, [1, 0]]
  - [ArraySource<double>, [1, 0], ArraySink<double>, [0, 0]]
  - [ArraySource<double>, [1, 1], ArraySink<double>, [0, 1]]
)";

std::string ymlDecodeEncode(std::string_view yml) {
    const auto yaml = pmtv::yaml::deserialize(yml);
    if (!yaml) {
        throw gr::exception(fmt::format("Could not parse yaml: {}:{} content:\n{}", yaml.error().message, yaml.error().line, yml));
    }

    return pmtv::yaml::serialize(yaml.value());
}

const boost::ut::suite GrcTests = [] {
    "Basic graph loading and storing"_test = [] {
        try {
            using namespace gr;
            const auto context       = getContext();
            const auto graphSrc      = ymlDecodeEncode(testGrc);
            auto       graph         = gr::loadGrc(context->loader, graphSrc);
            auto       graphSavedSrc = gr::saveGrc(graph);
            expect(checkAndPrintMissingLines(graphSrc, graphSavedSrc));
        } catch (const std::string& e) {
            fmt::println(std::cerr, "Unexpected exception: {}", e);
            expect(false);
        }
    };

#if not defined(__EMSCRIPTEN__) // && not defined(__APPLE__)
    "Basic graph loading and storing using plugins"_test = [] {
        try {
            using namespace gr;

            constexpr std::string_view pluginsTestGrc = R"(
blocks:
  - name: main_source
    id: good::fixed_source
    parameters:
      event_count: !!uint32 100
      unknown_property: 42
  - name: multiplier
    id: good::multiply
  - name: counter
    id: builtin_counter
  - name: sink
    id: good::cout_sink
    parameters:
      total_count: !!uint32 100
      unknown_property: 42
connections:
  - [main_source, 0, multiplier, 0]
  - [multiplier, 0, counter, 0]
  - [counter, 0, sink, 0]
)";

            const auto context       = getContext();
            const auto graphSrc      = ymlDecodeEncode(pluginsTestGrc);
            auto       graph         = gr::loadGrc(context->loader, graphSrc);
            auto       graphSavedSrc = gr::saveGrc(graph);

            expect(checkAndPrintMissingLines(graphSrc, graphSavedSrc));

            gr::scheduler::Simple scheduler(std::move(graph));
            expect(scheduler.runAndWait().has_value());
        } catch (const std::string& e) {
            fmt::println(std::cerr, "Unexpected exception: {}", e);
            expect(false);
        }
    };
#endif

    "Save and load"_test = [] {
        // Test if we get the same graph when saving it and loading the saved
        // data into another graph
        using namespace gr;

        try {
            const auto context       = getContext();
            auto       graph1        = gr::loadGrc(context->loader, testGrc);
            auto       graphSavedSrc = gr::saveGrc(graph1);
            auto       graph2        = gr::loadGrc(context->loader, graphSavedSrc);
            expect(eq(collectBlocks(graph1), collectBlocks(graph2)));
            expect(eq(collectEdges(graph1), collectEdges(graph2)));
        } catch (const std::string& e) {
            fmt::println(std::cerr, "Unexpected exception: {}", e);
            expect(false);
        }
    };

    "Port collections"_test = [] {
        try {
            using namespace gr;

            const auto context = getContext();
            gr::Graph  graph1;
            auto&      arraySink    = graph1.emplaceBlock<ArraySink<double>>();
            auto&      arraySource0 = graph1.emplaceBlock<ArraySource<double>>();
            auto&      arraySource1 = graph1.emplaceBlock<ArraySource<double>>();

            expect(eq(ConnectionResult::SUCCESS, graph1.connect<"outA0">(arraySource0).to<"inB1">(arraySink)));
            expect(eq(ConnectionResult::SUCCESS, graph1.connect<"outA1">(arraySource1).to<"inB0">(arraySink)));
            expect(eq(ConnectionResult::SUCCESS, graph1.connect<"outB0">(arraySource0).to<"inA0">(arraySink)));
            expect(eq(ConnectionResult::SUCCESS, graph1.connect<"outB1">(arraySource1).to<"inA1">(arraySink)));

            expect(graph1.reconnectAllEdges());

            const auto graph1Saved = gr::saveGrc(graph1);
            const auto graph2      = gr::loadGrc(context->loader, graph1Saved);

            expect(eq(collectBlocks(graph1), collectBlocks(graph2)));
            expect(eq(collectEdges(graph1), collectEdges(graph2)));
        } catch (const std::string& e) {
            fmt::println(std::cerr, "Unexpected exception: {}", e);
            expect(false);
        }
    };

    "Settings serialization"_test = [] {
        try {
            using namespace gr;

            const auto context = getContext();
            gr::Graph  graph1;
            const auto expectedString        = std::string("abc");
            const bool expectedBool          = true;
            const auto expectedComplex       = std::complex<double>(1., 1.);
            const auto expectedStringVector  = std::vector<std::string>{"a", "b", "c"};
            const auto expectedBoolVector    = std::vector<bool>{true, false, true};
            const auto expectedDoubleVector  = std::vector<double>{1., 2., 3.};
            const auto expectedInt16Vector   = std::vector<int16_t>{1, 2, 3};
            const auto expectedComplexVector = std::vector<std::complex<double>>{{1., 1.}, {2., 2.}, {3., 3.}};

            std::ignore = graph1.emplaceBlock<ArraySink<double>>({{"bool_setting", bool(expectedBool)}, {"string_setting", expectedString}, {"complex_setting", expectedComplex}, //
                {"bool_vector", expectedBoolVector}, {"string_vector", expectedStringVector}, {"double_vector", expectedDoubleVector}, {"int16_vector", expectedInt16Vector}, {"complex_vector", expectedComplexVector}});

            const auto graph1Saved = gr::saveGrc(graph1);
            const auto graph2      = gr::loadGrc(context->loader, graph1Saved);
            graph2.forEachBlock([&](const auto& node) {
                const auto settings = node.settings().get();
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
            fmt::println(std::cerr, "Unexpected exception: {}", e);
            expect(false);
        }
    };

    "Context settings"_test = [] {
        try {
            using namespace gr;

            const auto context = getContext();
            gr::Graph  graph1;
            auto&      block = graph1.emplaceBlock<ArraySink<double>>({{"name", "ArraySink0"}});
            const auto now   = settings::convertTimePointToUint64Ns(std::chrono::system_clock::now());
            expect(block.settings().set({{"name", "ArraySink1"}}, SettingsCtx{now, "1"}).empty());
            expect(block.settings().set({{"name", "ArraySink1+10"}}, SettingsCtx{now + 10'000'000'000ULL, "1"}).empty());
            expect(block.settings().set({{"name", "ArraySink1+20"}}, SettingsCtx{now + 20'000'000'000ULL, "1"}).empty());
            expect(block.settings().set({{"name", "ArraySink2"}}, SettingsCtx{now, "2"}).empty());
            expect(block.settings().set({{"name", "ArraySink3"}}, SettingsCtx{now, 3}).empty()); // int as context name

            const auto graph1Saved = gr::saveGrc(graph1);
            const auto graph2      = gr::loadGrc(context->loader, graph1Saved);

            graph2.forEachBlock([&](const auto& node) {
                const auto& stored = node.settings().getStoredAll();
                expect(eq(node.settings().getNStoredParameters(), 6UZ));
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
            fmt::println(std::cerr, "Unexpected exception: {}", e);
            expect(false);
        }
    };
};

} // namespace gr::qa_grc_test

int main() { /* tests are statically executed */ }
