// TODO: This is a test application that doesn't use ut framework.
// Once all problems with ut and plugins have been resolved,
// implement a unit testing suite

#include <graph_yaml_importer.hpp>
#include <scheduler.hpp>

#include <fstream>
#include <sstream>

#include "blocklib/core/unit-test/common_blocks.hpp"
#include "build_configure.hpp"

namespace fg = fair::graph;

struct test_context {
    test_context(std::vector<std::filesystem::path> paths) : registry(), loader(&registry, std::move(paths)) {}

    fg::block_registry registry;
    fg::plugin_loader  loader;
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
        register_builtin_blocks(&context.registry);

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
        register_builtin_blocks(&context.registry);

        auto                  graph_source       = read_file(TESTS_SOURCE_PATH "/grc/test.grc");

        auto                  graph_1            = fg::load_grc(context.loader, graph_source);
        auto                  graph_saved_source = fg::save_grc(graph_1);

        auto                  graph_2            = fg::load_grc(context.loader, graph_saved_source);

        [[maybe_unused]] auto collect_blocks     = [](fg::graph &graph) {
            std::set<std::string> result;
            graph.for_each_block([&](const auto &block) { result.insert(fmt::format("{}-{}", block.name(), block.type_name())); });
            return result;
        };

        assert(collect_blocks(graph_1) == collect_blocks(graph_2));

        [[maybe_unused]] auto collect_edges = [](fg::graph &graph) {
            std::set<std::string> result;
            graph.for_each_edge([&](const auto &edge) { result.insert(fmt::format("{}#{} - {}#{}", edge.src_block().name(), edge.src_port_index(), edge.dst_block().name(), edge.dst_port_index())); });
            return result;
        };

        assert(collect_edges(graph_1) == collect_edges(graph_2));
    }
}
