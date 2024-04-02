#include <fstream>
#include <iostream>
#include <ranges>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <gnuradio-4.0/Graph_yaml_importer.hpp>
#include <gnuradio-4.0/Scheduler.hpp>

#include <build_configure.hpp>
#include <gnuradio-4.0/basic/common_blocks.hpp>

#include <boost/ut.hpp>

#if defined(__clang__) && __clang_major__ >= 16
// for clang there seems to be some static initialisation problem which leads to segfaults in gr::registerBlock
template<>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif

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
    explicit TestContext(std::vector<std::filesystem::path> paths = {}) : loader(gr::globalBlockRegistry(), std::move(paths)) {}

    gr::PluginLoader loader;
};

namespace {
auto
collectBlocks(const gr::Graph &graph) {
    std::set<std::string> result;
    graph.forEachBlock([&](const auto &node) { result.insert(fmt::format("{}-{}", node.name(), node.typeName())); });
    return result;
};

auto
collectEdges(const gr::Graph &graph) {
    std::set<std::string> result;
    graph.forEachEdge([&](const auto &edge) {
        result.insert(fmt::format("{}#{}#{} - {}#{}#{}", edge.sourceBlock().name(), edge.sourcePortDefinition().topLevel, edge.sourcePortDefinition().subIndex, edge.destinationBlock().name(),
                                  edge.destinationPortDefinition().topLevel, edge.destinationPortDefinition().subIndex));
    });
    return result;
};

auto
readFile(const auto &path) {
    std::ifstream     input(path);
    std::stringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
};

} // namespace

using namespace boost::ut;

namespace gr::qa_grc_test {

const boost::ut::suite GrcTests = [] {
    static TestContext context = [] {
        TestContext ctx({ TESTS_BINARY_PATH "/plugins" });
        gr::registerBlock<ArraySource, double>(gr::globalBlockRegistry());
        gr::registerBlock<ArraySink, double>(gr::globalBlockRegistry());
        return ctx;
    }();

    "Basic graph loading and storing"_test = [] {
        using namespace gr;
        auto graph_source = readFile(TESTS_SOURCE_PATH "/grc/test.grc");

        auto graph              = gr::load_grc(context.loader, graph_source);
        auto graph_saved_source = gr::save_grc(graph);

        auto checkAndPrintMissingLines = [](const std::string& first, const std::string& second) -> bool {
            std::istringstream ssSecond(second);
            std::unordered_set<std::string> linesSecond;
            std::string line;
            while (std::getline(ssSecond, line)) {
                linesSecond.insert(line);
            }

            std::istringstream ssFirst(first);
            bool allLinesFound = true;
            size_t lineNumber = 0;
            while (std::getline(ssFirst, line)) {
                ++lineNumber;
                if (!linesSecond.contains(line)) {
                    fmt::println(stderr, "missing line {}:\n{}", lineNumber, line);
                    allLinesFound = false;
                }
            }
            if (!allLinesFound) {
                fmt::println(stderr, "\nin:\n{}", second);
            }

            return allLinesFound;
        };
        expect(checkAndPrintMissingLines(graph_source, graph_saved_source));

        gr::scheduler::Simple scheduler(std::move(graph));
        scheduler.runAndWait();
    };

    "Save and load"_test = [] {
        // Test if we get the same graph when saving it and loading the saved
        // data into another graph
        using namespace gr;
        auto graph_source = readFile(TESTS_SOURCE_PATH "/grc/test.grc");

        try {
            auto graph_1            = gr::load_grc(context.loader, graph_source);
            auto graph_saved_source = gr::save_grc(graph_1);
            auto graph_2            = gr::load_grc(context.loader, graph_saved_source);
            expect(eq(collectBlocks(graph_1), collectBlocks(graph_2)));
            expect(eq(collectEdges(graph_1), collectEdges(graph_2)));
        } catch (const std::string &e) {
            fmt::println(std::cerr, "Unexpected exception: {}", e);
            expect(false);
        }
    };

    "Port collections"_test = [] {
        using namespace gr;

        gr::Graph graph_1;
        auto     &arraySink    = graph_1.emplaceBlock<ArraySink<double>>();
        auto     &arraySource0 = graph_1.emplaceBlock<ArraySource<double>>();
        auto     &arraySource1 = graph_1.emplaceBlock<ArraySource<double>>();

        expect(eq(ConnectionResult::SUCCESS, graph_1.connect<"outA", 0>(arraySource0).to<"inB", 1>(arraySink)));
        expect(eq(ConnectionResult::SUCCESS, graph_1.connect<"outA", 1>(arraySource1).to<"inB", 0>(arraySink)));
        expect(eq(ConnectionResult::SUCCESS, graph_1.connect<"outB", 0>(arraySource0).to<"inA", 0>(arraySink)));
        expect(eq(ConnectionResult::SUCCESS, graph_1.connect<"outB", 1>(arraySource1).to<"inA", 1>(arraySink)));

        expect(graph_1.performConnections());

        const auto graph_1_saved = gr::save_grc(graph_1);
        const auto graph_2       = gr::load_grc(context.loader, graph_1_saved);

        expect(eq(collectBlocks(graph_1), collectBlocks(graph_2)));
        expect(eq(collectEdges(graph_1), collectEdges(graph_2)));
    };

    "Settings serialization"_test = [] {
        using namespace gr;

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
        const auto graph_2       = gr::load_grc(context.loader, graph_1_saved);
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
    };
};

} // namespace gr::qa_grc_test

int
main() { /* tests are statically executed */
}
