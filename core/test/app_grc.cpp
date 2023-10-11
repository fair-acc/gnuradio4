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

struct test_context {
    test_context(std::vector<std::filesystem::path> paths) : registry(), loader(&registry, std::move(paths)) {}

    grg::BlockRegistry registry;
    grg::plugin_loader loader;
};

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

        [[maybe_unused]] auto collectBlocks      = [](grg::Graph &graph) {
            std::set<std::string> result;
            graph.forEachBlock([&](const auto &node) { result.insert(fmt::format("{}-{}", node.name(), node.typeName())); });
            return result;
        };

        assert(collectBlocks(graph_1) == collectBlocks(graph_2));

        [[maybe_unused]] auto collect_edges = [](grg::Graph &graph) {
            std::set<std::string> result;
            graph.forEachEdge([&](const auto &edge) { result.insert(fmt::format("{}#{} - {}#{}", edge.src_block().name(), edge.src_port_index(), edge.dst_block().name(), edge.dst_port_index())); });
            return result;
        };

        assert(collect_edges(graph_1) == collect_edges(graph_2));
    }
}
