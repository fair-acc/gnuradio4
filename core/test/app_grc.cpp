// TODO: This is a test application that doesn't use ut framework.
// Once all problems with ut and plugins have been resolved,
// implement a unit testing suite

#include <fstream>
#include <sstream>

#include <gnuradio-4.0/Graph_yaml_importer.hpp>
#include <gnuradio-4.0/Scheduler.hpp>

#include <build_configure.hpp>
#include <gnuradio-4.0/basic/common_blocks.hpp>

template<typename T>
struct ArraySource : public gr::Block<ArraySource<T>> {
    std::array<gr::PortOut<T>, 2> outA;
    std::array<gr::PortOut<T>, 2> outB;
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
};

ENABLE_REFLECTION_FOR_TEMPLATE(ArraySink, inA, inB, bool_setting, string_setting, bool_vector, string_vector, double_vector, int16_vector);

struct TestContext {
    TestContext(std::vector<std::filesystem::path> paths) : registry(), loader(&registry, std::move(paths)) {}

    gr::BlockRegistry registry;
    gr::plugin_loader loader;
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
} // namespace

int
main(int argc, char *argv[]) {
    std::vector<std::filesystem::path> paths;
    if (argc < 2) {
        paths.push_back(TESTS_BINARY_PATH "/plugins");
    } else {
        for (int i = 1; i < argc; ++i) {
            paths.push_back(argv[i]);
        }
    }

    auto read_file = [](const auto &path) {
        std::ifstream     input(path);
        std::stringstream buffer;
        buffer << input.rdbuf();
        return buffer.str();
    };

    TestContext context(std::move(paths));

    // Test the basic graph loading and storing
    {
        using namespace gr;
        registerBuiltinBlocks(&context.registry);

        auto graph_source          = read_file(TESTS_SOURCE_PATH "/grc/test.grc");

        auto graph                 = gr::load_grc(context.loader, graph_source);
        auto graph_saved_source    = gr::save_grc(graph);

        auto graph_expected_source = read_file(TESTS_SOURCE_PATH "/grc/test.grc.expected");
        assert(graph_saved_source + "\n"
               == graph_expected_source); // TODO: this is not a good assert since we will add new parameters regularly... should not be identity but checking critical parameter/aspects

        gr::scheduler::Simple scheduler(std::move(graph));
        scheduler.runAndWait();
    }

    // Test if we get the same graph when saving it and loading the saved
    // data into another graph
    {
        using namespace gr;
        registerBuiltinBlocks(&context.registry);

        auto graph_source       = read_file(TESTS_SOURCE_PATH "/grc/test.grc");

        auto graph_1            = gr::load_grc(context.loader, graph_source);
        auto graph_saved_source = gr::save_grc(graph_1);

        auto graph_2            = gr::load_grc(context.loader, graph_saved_source);

        assert(collectBlocks(graph_1) == collectBlocks(graph_2));
        assert(collectEdges(graph_1) == collectEdges(graph_2));
    }

    // Test that connections involving port collections are handled correctly
    {
        using namespace gr;
        registerBuiltinBlocks(&context.registry);
        GP_REGISTER_NODE_RUNTIME(&context.registry, ArraySource, double);
        GP_REGISTER_NODE_RUNTIME(&context.registry, ArraySink, double);

        gr::Graph graph_1;
        auto     &arraySink    = graph_1.emplaceBlock<ArraySink<double>>();
        auto     &arraySource0 = graph_1.emplaceBlock<ArraySource<double>>();
        auto     &arraySource1 = graph_1.emplaceBlock<ArraySource<double>>();

        graph_1.connect<"outA", 0>(arraySource0).to<"inB", 1>(arraySink);
        graph_1.connect<"outA", 1>(arraySource1).to<"inB", 0>(arraySink);
        graph_1.connect<"outB", 0>(arraySource0).to<"inA", 0>(arraySink);
        graph_1.connect<"outB", 1>(arraySource1).to<"inA", 1>(arraySink);

        assert(graph_1.performConnections());

        const auto graph_1_saved = gr::save_grc(graph_1);
        const auto graph_2       = gr::load_grc(context.loader, graph_1_saved);

        assert(collectBlocks(graph_1) == collectBlocks(graph_2));
        assert(collectEdges(graph_1) == collectEdges(graph_2));
    }

    // Test settings serialization
    {
        using namespace gr;
        registerBuiltinBlocks(&context.registry);
        GP_REGISTER_NODE_RUNTIME(&context.registry, ArraySink, double);

        gr::Graph  graph_1;
        const auto expectedString       = std::string("abc");
        const auto expectedBool         = true;
        const auto expectedStringVector = std::vector<std::string>{ "a", "b", "c" };
        const auto expectedBoolVector   = std::vector<bool>{ true, false, true };
        const auto expectedDoubleVector = std::vector<double>{ 1., 2., 3. };
        const auto expectedInt16Vector  = std::vector<int16_t>{ 1, 2, 3 };
        auto      &arraySink            = graph_1.emplaceBlock<ArraySink<double>>({ { "bool_setting", expectedBool },
                                                                                    { "string_setting", expectedString },
                                                                                    { "bool_vector", expectedBoolVector },
                                                                                    { "string_vector", expectedStringVector },
                                                                                    { "double_vector", expectedDoubleVector },
                                                                                    { "int16_vector", expectedInt16Vector } });

        const auto graph_1_saved        = gr::save_grc(graph_1);
        const auto graph_2              = gr::load_grc(context.loader, graph_1_saved);
        graph_2.forEachBlock([&](const auto &node) {
            const auto settings = node.settings().get();
            assert(std::get<bool>(settings.at("bool_setting")) == expectedBool);
            assert(std::get<std::string>(settings.at("string_setting")) == expectedString);
            assert(std::get<std::vector<bool>>(settings.at("bool_vector")) == expectedBoolVector);
            assert(std::get<std::vector<std::string>>(settings.at("string_vector")) == expectedStringVector);
            assert(std::get<std::vector<double>>(settings.at("double_vector")) == expectedDoubleVector);
            assert(std::get<std::vector<int16_t>>(settings.at("int16_vector")) == expectedInt16Vector);
        });

        assert(collectBlocks(graph_1) == collectBlocks(graph_2));
        assert(collectEdges(graph_1) == collectEdges(graph_2));
    }
}
