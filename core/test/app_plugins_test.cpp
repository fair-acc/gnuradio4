#include <array>
#include <cassert>
#include <iostream>

#include <fmt/format.h>

#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/plugin_loader.hpp>

#include <gnuradio-4.0/meta/utils.hpp>

#include <gnuradio-4.0/basic/common_blocks.hpp>

using namespace std::chrono_literals;
using namespace gr::literals;

namespace grg = gr;

struct test_context {
    explicit test_context(std::vector<std::filesystem::path> paths) : registry(), loader(&registry, std::move(paths)) {}

    grg::BlockRegistry registry;
    grg::plugin_loader loader;
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
    registerBuiltinBlocks(&context.registry);

    fmt::print("PluginLoaderTests\n");
    using namespace gr;

    for (const auto &plugin [[maybe_unused]] : context.loader.plugins()) {
        assert(plugin->metadata->plugin_name.starts_with("Good"));
    }

    for (const auto &plugin [[maybe_unused]] : context.loader.failed_plugins()) {
        assert(plugin.first.ends_with("bad_plugin.so"));
    }

    auto        known = context.loader.knownBlocks();
    std::vector requireds{ names::cout_sink, names::fixed_source, names::divide, names::multiply };

    for (const auto &required [[maybe_unused]] : requireds) {
        assert(std::ranges::find(known, required) != known.end());
    }

    grg::graph testGraph;

    // Instantiate the node that is defined in a plugin
    auto &block_source = context.loader.instantiate_in_graph(testGraph, names::fixed_source, "double");

    // Instantiate a built-in node in a static way
    gr::property_map block_multiply_1_params;
    block_multiply_1_params["factor"] = 2.0;
    auto &block_multiply_1            = testGraph.emplaceBlock<builtin_multiply<double>>(block_multiply_1_params);

    // Instantiate a built-in node via the plugin loader
    auto &block_multiply_2 = context.loader.instantiate_in_graph(testGraph, names::builtin_multiply, "double");
    auto &block_counter    = context.loader.instantiate_in_graph(testGraph, names::builtin_counter, "double");

    //
    const std::size_t repeats = 100;
    gr::property_map  block_sink_params;
    block_sink_params["total_count"] = 100_UZ;
    auto block_sink_load             = context.loader.instantiate(names::cout_sink, "double", block_sink_params);

    assert(block_sink_load);
    auto &block_sink                    = testGraph.add_block(std::move(block_sink_load));

    auto  connection_1 [[maybe_unused]] = testGraph.dynamic_connect(block_source, 0, block_multiply_1, 0);
    auto  connection_2 [[maybe_unused]] = testGraph.dynamic_connect(block_multiply_1, 0, block_multiply_2, 0);
    auto  connection_3 [[maybe_unused]] = testGraph.dynamic_connect(block_multiply_2, 0, block_counter, 0);
    auto  connection_4 [[maybe_unused]] = testGraph.dynamic_connect(block_counter, 0, block_sink, 0);

    assert(connection_1 == grg::connection_result_t::SUCCESS);
    assert(connection_2 == grg::connection_result_t::SUCCESS);
    assert(connection_3 == grg::connection_result_t::SUCCESS);
    assert(connection_4 == grg::connection_result_t::SUCCESS);

    for (std::size_t i = 0; i < repeats; ++i) {
        std::ignore = block_source.work(std::numeric_limits<std::size_t>::max());
        std::ignore = block_multiply_1.work(std::numeric_limits<std::size_t>::max());
        std::ignore = block_multiply_2.work(std::numeric_limits<std::size_t>::max());
        std::ignore = block_counter.work(std::numeric_limits<std::size_t>::max());
        std::ignore = block_sink.work(std::numeric_limits<std::size_t>::max());
    }

    fmt::print("repeats {} event_count {}\n", repeats, builtin_counter<double>::s_event_count);
    assert(builtin_counter<double>::s_event_count == repeats);
}
