// TODO: This is a test application that doesn't use ut framework.
// Once all problems with ut and plugins have been resolved,
// implement a unit testing suite

#include <fstream>
#include <sstream>

#include <gnuradio-4.0/graph_yaml_importer.hpp>
#include <gnuradio-4.0/scheduler.hpp>

#include <gnuradio-4.0/basic/common_nodes.hpp>
#include <build_configure.hpp>

namespace fg = fair::graph;

struct test_context {
    test_context(std::vector<std::filesystem::path> paths) : registry(), loader(&registry, std::move(paths)) {}

    fg::node_registry registry;
    fg::plugin_loader loader;
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
        using namespace fair::graph;
        register_builtin_nodes(&context.registry);

        auto graph_source          = read_file(TESTS_SOURCE_PATH "/grc/test.grc");

        auto graph                 = fg::load_grc(context.loader, graph_source);
        auto graph_saved_source    = fg::save_grc(graph);

        auto graph_expected_source = read_file(TESTS_SOURCE_PATH "/grc/test.grc.expected");
        assert(graph_saved_source + "\n"
               == graph_expected_source); // TODO: this is not a good assert since we will add new parameters regularly... should not be identity but checking critical parameter/aspects

        fair::graph::scheduler::simple scheduler(std::move(graph));
        scheduler.run_and_wait();
    }

    // Test if we get the same graph when saving it and loading the saved
    // data into another graph
    {
        using namespace fair::graph;
        register_builtin_nodes(&context.registry);

        auto                  graph_source       = read_file(TESTS_SOURCE_PATH "/grc/test.grc");

        auto                  graph_1            = fg::load_grc(context.loader, graph_source);
        auto                  graph_saved_source = fg::save_grc(graph_1);

        auto                  graph_2            = fg::load_grc(context.loader, graph_saved_source);

        [[maybe_unused]] auto collect_nodes      = [](fg::graph &graph) {
            std::set<std::string> result;
            graph.for_each_node([&](const auto &node) { result.insert(fmt::format("{}-{}", node.name(), node.type_name())); });
            return result;
        };

        assert(collect_nodes(graph_1) == collect_nodes(graph_2));

        [[maybe_unused]] auto collect_edges = [](fg::graph &graph) {
            std::set<std::string> result;
            graph.for_each_edge([&](const auto &edge) { result.insert(fmt::format("{}#{} - {}#{}", edge.src_node().name(), edge.src_port_index(), edge.dst_node().name(), edge.dst_port_index())); });
            return result;
        };

        assert(collect_edges(graph_1) == collect_edges(graph_2));
    }
}
