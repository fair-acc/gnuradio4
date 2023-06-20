#include <node_registry.hpp>
#include <plugin_loader.hpp>
#include <utils.hpp>

#include <array>
#include <cassert>
#include <iostream>

#include <fmt/format.h>

#include "blocklib/core/unit-test/common_nodes.hpp"

using namespace std::chrono_literals;
using namespace fair::literals;

namespace fg = fair::graph;

struct test_context {
    explicit test_context(std::vector<std::filesystem::path> paths) : registry(), loader(&registry, std::move(paths)) {}

    fg::node_registry registry;
    fg::plugin_loader loader;
};

namespace names {
const auto fixed_source     = "good::fixed_source"s;
const auto cout_sink        = "good::cout_sink"s;
const auto multiply         = "good::multiply"s;
const auto divide           = "good::divide"s;
const auto builtin_multiply = "builtin_multiply"s;
const auto builtin_counter  = "builtin_counter"s;
} // namespace names

int
main(int argc, char *argv[]) {
    std::vector<std::filesystem::path> paths;
    if (argc < 2) {
        paths.emplace_back("test/plugins");
        paths.emplace_back("plugins");
    } else {
        for (int i = 1; i < argc; ++i) {
            paths.emplace_back(argv[i]);
        }
    }

    test_context context(std::move(paths));
    register_builtin_nodes(&context.registry);

    fmt::print("PluginLoaderTests\n");
    using namespace gr;

    for (const auto &plugin [[maybe_unused]] : context.loader.plugins()) {
        assert(plugin->metadata->plugin_name.starts_with("Good"));
    }

    for (const auto &plugin [[maybe_unused]] : context.loader.failed_plugins()) {
        assert(plugin.first.ends_with("bad_plugin.so"));
    }

    auto        known = context.loader.known_nodes();
    std::vector requireds{ names::cout_sink, names::fixed_source, names::divide, names::multiply };

    for (const auto &required [[maybe_unused]] : requireds) {
        assert(std::ranges::find(known, required) != known.end());
    }

    fg::graph flow_graph;

    // Instantiate the node that is defined in a plugin
    auto &node_source = context.loader.instantiate_in_graph(flow_graph, names::fixed_source, "double");

    // Instantiate a built-in node in a static way
    fair::graph::property_map node_multiply_1_params;
    node_multiply_1_params["factor"] = 2.0;
    auto &node_multiply_1            = flow_graph.make_node<builtin_multiply<double>>(node_multiply_1_params);

    // Instantiate a built-in node via the plugin loader
    auto &node_multiply_2 = context.loader.instantiate_in_graph(flow_graph, names::builtin_multiply, "double");
    auto &node_counter    = context.loader.instantiate_in_graph(flow_graph, names::builtin_counter, "double");

    //
    const std::size_t         repeats = 100;
    fair::graph::property_map node_sink_params;
    node_sink_params["total_count"] = 100_UZ;
    auto node_sink_load             = context.loader.instantiate(names::cout_sink, "double", node_sink_params);

    assert(node_sink_load);
    auto &node_sink                     = flow_graph.add_node(std::move(node_sink_load));

    auto  connection_1 [[maybe_unused]] = flow_graph.dynamic_connect(node_source, 0, node_multiply_1, 0);
    auto  connection_2 [[maybe_unused]] = flow_graph.dynamic_connect(node_multiply_1, 0, node_multiply_2, 0);
    auto  connection_3 [[maybe_unused]] = flow_graph.dynamic_connect(node_multiply_2, 0, node_counter, 0);
    auto  connection_4 [[maybe_unused]] = flow_graph.dynamic_connect(node_counter, 0, node_sink, 0);

    assert(connection_1 == fg::connection_result_t::SUCCESS);
    assert(connection_2 == fg::connection_result_t::SUCCESS);
    assert(connection_3 == fg::connection_result_t::SUCCESS);
    assert(connection_4 == fg::connection_result_t::SUCCESS);

    for (std::size_t i = 0; i < repeats; ++i) {
        std::ignore = node_source.work();
        std::ignore = node_multiply_1.work();
        std::ignore = node_multiply_2.work();
        std::ignore = node_counter.work();
        std::ignore = node_sink.work();
    }

    fmt::print("repeats {} event_count {}\n", repeats, builtin_counter<double>::s_event_count);
    assert(builtin_counter<double>::s_event_count == repeats);
}
