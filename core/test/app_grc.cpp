// TODO: This is a test application that doesn't use ut framework.
// Once all problems with ut and plugins have been resolved,
// implement a unit testing suite

#include <fstream>
#include <sstream>

#include <gnuradio-4.0/Graph_yaml_importer.hpp>
#include <gnuradio-4.0/Scheduler.hpp>

#include <build_configure.hpp>
#include <gnuradio-4.0/basic/common_blocks.hpp>

namespace grg = gr;

template<typename T>
struct ArraySource : public grg::Block<ArraySource<T>> {
    std::array<grg::PortOut<T>, 2> outA;
    std::array<grg::PortOut<T>, 2> outB;
};

ENABLE_REFLECTION_FOR_TEMPLATE(ArraySource, outA, outB);

template<typename T>
struct ArraySink : public grg::Block<ArraySink<T>> {
    std::array<grg::PortIn<T>, 2> inA;
    std::array<grg::PortIn<T>, 2> inB;
};

ENABLE_REFLECTION_FOR_TEMPLATE(ArraySink, inA, inB);

struct test_context {
    test_context(std::vector<std::filesystem::path> paths) : registry(), loader(&registry, std::move(paths)) {}

    grg::BlockRegistry registry;
    grg::plugin_loader loader;
};

namespace {
    auto collectBlocks(const grg::Graph &graph) {
        std::set<std::string> result;
        graph.forEachBlock([&](const auto &node) { result.insert(fmt::format("{}-{}", node.name(), node.typeName())); });
        return result;
    };

    auto collectEdges(const grg::Graph &graph) {
        std::set<std::string> result;
        graph.forEachEdge([&](const auto &edge) {
            result.insert(fmt::format("{}#{}#{} - {}#{}#{}", edge.sourceBlock().name(), edge.sourcePortDefinition().topLevel, edge.sourcePortDefinition().subIndex, edge.destinationBlock().name(), edge.destinationPortDefinition().topLevel, edge.destinationPortDefinition().subIndex));
        });
        return result;
    };
}

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

    test_context context(std::move(paths));

    // Test the basic graph loading and storing
    {
        using namespace gr;
        registerBuiltinBlocks(&context.registry);

        auto graph_source          = read_file(TESTS_SOURCE_PATH "/grc/test.grc");

        auto graph                 = grg::load_grc(context.loader, graph_source);
        auto graph_saved_source    = grg::save_grc(graph);

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

        auto                  graph_source       = read_file(TESTS_SOURCE_PATH "/grc/test.grc");

        auto                  graph_1            = grg::load_grc(context.loader, graph_source);
        auto                  graph_saved_source = grg::save_grc(graph_1);

        auto                  graph_2            = grg::load_grc(context.loader, graph_saved_source);

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
        auto &arraySink    = graph_1.emplaceBlock<ArraySink<double>>();
        auto &arraySource0 = graph_1.emplaceBlock<ArraySource<double>>();
        auto &arraySource1 = graph_1.emplaceBlock<ArraySource<double>>();

        graph_1.connect<"outA", 0>(arraySource0).to<"inB", 1>(arraySink);
        graph_1.connect<"outA", 1>(arraySource1).to<"inB", 0>(arraySink);
        graph_1.connect<"outB", 0>(arraySource0).to<"inA", 0>(arraySink);
        graph_1.connect<"outB", 1>(arraySource1).to<"inA", 1>(arraySink);

        assert(graph_1.performConnections());

        const auto graph_1_saved = grg::save_grc(graph_1);
        const auto graph_2       = grg::load_grc(context.loader, graph_1_saved);

        assert(collectBlocks(graph_1) == collectBlocks(graph_2));
        assert(collectEdges(graph_1) == collectEdges(graph_2));
    }
}
