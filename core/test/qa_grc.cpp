#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <build_configure.hpp>

#include <gnuradio-4.0/basic/common_blocks.hpp>
#include <gnuradio-4.0/Graph_yaml_importer.hpp>
#include <gnuradio-4.0/Scheduler.hpp>

#include <boost/ut.hpp>

template<typename T>
struct ArraySource : public gr::Block<ArraySource<T>> {
    std::array<gr::PortOut<T>, 2> outA;
    std::array<gr::PortOut<T>, 2> outB;

    template<typename PublishableSpan1, typename PublishableSpan2>
    gr::work::Status
    processBulk(PublishableSpan1 &, PublishableSpan2 &) { // TODO: needs proper explicit signature
        return gr::work::Status::OK;
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE(ArraySource, outA, outB);

template<typename T>
struct ArraySink : public gr::Block<ArraySink<T>> {
    std::array<gr::PortIn<T>, 2>                                     inA;
    std::array<gr::PortIn<T>, 2>                                     inB;
    gr::Annotated<bool, "bool setting">                              bool_setting = false;
    gr::Annotated<std::string, "String setting">                     string_setting;
    gr::Annotated<std::vector<bool>, "Bool vector setting">          bool_vector;
    gr::Annotated<std::vector<std::string>, "String vector setting"> string_vector;
    gr::Annotated<std::vector<double>, "Double vector setting">      double_vector;
    gr::Annotated<std::vector<int16_t>, "int16_t vector setting">    int16_vector;

    template<typename ConsumableSpan1, typename ConsumableSpan2>
    gr::work::Status
    processBulk(ConsumableSpan1 &, ConsumableSpan2 &) { // TODO: needs proper explicit signature
        return gr::work::Status::OK;
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE(ArraySink, inA, inB, bool_setting, string_setting, bool_vector, string_vector, double_vector, int16_vector);

struct TestContext {
    explicit TestContext(std::vector<std::filesystem::path> paths) : loader(registry, std::move(paths)) {}

    gr::BlockRegistry registry;
    gr::PluginLoader  loader;
};

namespace {
auto
collectBlocks(const gr::Graph &graph) {
    std::set<std::string> result;
    graph.forEachBlock([&](const auto &node) { result.insert(fmt::format("{}-{}", node.name(), node.typeName())); });
    return result;
}

auto
collectEdges(const gr::Graph &graph) {
    std::set<std::string> result;
    graph.forEachEdge([&](const auto &edge) {
        result.insert(fmt::format("{}#{}#{} - {}#{}#{}", edge.sourceBlock().name(), edge.sourcePortDefinition().topLevel, edge.sourcePortDefinition().subIndex, edge.destinationBlock().name(),
                                  edge.destinationPortDefinition().topLevel, edge.destinationPortDefinition().subIndex));
    });
    return result;
}

bool
checkAndPrintMissingLines(const std::string &first, const std::string &second) {
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

auto
getContext() {
    static auto ctx = [] {
        auto context = std::make_shared<TestContext>(std::vector<std::filesystem::path>{ TESTS_BINARY_PATH "/plugins" });
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

constexpr std::string_view test_grc = R"(
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

const boost::ut::suite GrcTests = [] {
    "Basic graph loading and storing"_test = [] {
        try {
            using namespace gr;
            const auto context            = getContext();
            const auto graph_source       = std::string(test_grc);
            auto       graph              = gr::load_grc(context->loader, graph_source);
            auto       graph_saved_source = gr::save_grc(graph);
            expect(checkAndPrintMissingLines(graph_source, graph_saved_source));
        } catch (const std::string &e) {
            fmt::println(std::cerr, "Unexpected exception: {}", e);
            expect(false);
        }
    };

#ifndef __EMSCRIPTEN__
    "Basic graph loading and storing using plugins"_test = [] {
        try {
            using namespace gr;

            constexpr std::string_view plugins_test_grc = R"(
blocks:
  - name: main_source
    id: good::fixed_source
    parameters:
      event_count: 100
      unknown_property: 42
  - name: multiplier
    id: good::multiply
  - name: counter
    id: builtin_counter
  - name: sink
    id: good::cout_sink
    parameters:
      total_count: 100
      unknown_property: 42
connections:
  - [main_source, 0, multiplier, 0]
  - [multiplier, 0, counter, 0]
  - [counter, 0, sink, 0]
)";

            const auto context            = getContext();
            const auto graph_source       = std::string(plugins_test_grc);
            auto       graph              = gr::load_grc(context->loader, graph_source);
            auto       graph_saved_source = gr::save_grc(graph);

            expect(checkAndPrintMissingLines(graph_source, graph_saved_source));

            gr::scheduler::Simple scheduler(std::move(graph));
            expect(scheduler.runAndWait().has_value());
        } catch (const std::string &e) {
            fmt::println(std::cerr, "Unexpected exception: {}", e);
            expect(false);
        }
    };
#endif

    "Save and load"_test = [] {
        // Test if we get the same graph when saving it and loading the saved
        // data into another graph
        using namespace gr;
        const auto graph_source = std::string(test_grc);

        try {
            const auto context = getContext();

            auto graph_1            = gr::load_grc(context->loader, graph_source);
            auto graph_saved_source = gr::save_grc(graph_1);
            auto graph_2            = gr::load_grc(context->loader, graph_saved_source);
            expect(eq(collectBlocks(graph_1), collectBlocks(graph_2)));
            expect(eq(collectEdges(graph_1), collectEdges(graph_2)));
        } catch (const std::string &e) {
            fmt::println(std::cerr, "Unexpected exception: {}", e);
            expect(false);
        }
    };

    "Port collections"_test = [] {
        try {
            using namespace gr;

            const auto context = getContext();
            gr::Graph  graph_1;
            auto      &arraySink    = graph_1.emplaceBlock<ArraySink<double>>();
            auto      &arraySource0 = graph_1.emplaceBlock<ArraySource<double>>();
            auto      &arraySource1 = graph_1.emplaceBlock<ArraySource<double>>();

            expect(eq(ConnectionResult::SUCCESS, graph_1.connect<"outA", 0>(arraySource0).to<"inB", 1>(arraySink)));
            expect(eq(ConnectionResult::SUCCESS, graph_1.connect<"outA", 1>(arraySource1).to<"inB", 0>(arraySink)));
            expect(eq(ConnectionResult::SUCCESS, graph_1.connect<"outB", 0>(arraySource0).to<"inA", 0>(arraySink)));
            expect(eq(ConnectionResult::SUCCESS, graph_1.connect<"outB", 1>(arraySource1).to<"inA", 1>(arraySink)));

            expect(graph_1.performConnections());

            const auto graph_1_saved = gr::save_grc(graph_1);
            const auto graph_2       = gr::load_grc(context->loader, graph_1_saved);

            expect(eq(collectBlocks(graph_1), collectBlocks(graph_2)));
            expect(eq(collectEdges(graph_1), collectEdges(graph_2)));
        } catch (const std::string &e) {
            fmt::println(std::cerr, "Unexpected exception: {}", e);
            expect(false);
        }
    };

    "Settings serialization"_test = [] {
        try {
            using namespace gr;

            const auto context = getContext();
            gr::Graph  graph_1;
            const auto expectedString       = std::string("abc");
            const bool expectedBool         = true;
            const auto expectedStringVector = std::vector<std::string>{ "a", "b", "c" };
            const auto expectedBoolVector   = std::vector<bool>{ true, false, true };
            const auto expectedDoubleVector = std::vector<double>{ 1., 2., 3. };
            const auto expectedInt16Vector  = std::vector<int16_t>{ 1, 2, 3 };
            std::ignore                     = graph_1.emplaceBlock<ArraySink<double>>({ { "bool_setting", bool(expectedBool) },
                                                                                        { "string_setting", expectedString },
                                                                                        { "bool_vector", expectedBoolVector },
                                                                                        { "string_vector", expectedStringVector },
                                                                                        { "double_vector", expectedDoubleVector },
                                                                                        { "int16_vector", expectedInt16Vector } });

            const auto graph_1_saved = gr::save_grc(graph_1);
            const auto graph_2       = gr::load_grc(context->loader, graph_1_saved);
            graph_2.forEachBlock([&](const auto &node) {
                const auto settings = node.settings().get();
                expect(eq(std::get<bool>(settings.at("bool_setting")), expectedBool));
                expect(eq(std::get<std::string>(settings.at("string_setting")), expectedString));
                expect(eq(std::get<std::vector<bool>>(settings.at("bool_vector")), expectedBoolVector));
                expect(eq(std::get<std::vector<std::string>>(settings.at("string_vector")), expectedStringVector));
                expect(eq(std::get<std::vector<double>>(settings.at("double_vector")), expectedDoubleVector));
                expect(eq(std::get<std::vector<int16_t>>(settings.at("int16_vector")), expectedInt16Vector));
            });

            expect(eq(collectBlocks(graph_1), collectBlocks(graph_2)));
            expect(eq(collectEdges(graph_1), collectEdges(graph_2)));
        } catch (const std::string &e) {
            fmt::println(std::cerr, "Unexpected exception: {}", e);
            expect(false);
        }
    };
};

} // namespace gr::qa_grc_test

int
main() { /* tests are statically executed */
}
