#include <node_registry.hpp>
#include <plugin_loader.hpp>

#include <array>
#include <cassert>
#include <iostream>

#include <fmt/format.h>
#include <fmt/ranges.h>

using namespace std::chrono_literals;
namespace fg = fair::graph;

struct test_context {
    test_context(std::vector<std::filesystem::path> paths) : registry(), loader(&registry, std::move(paths)) {}

    fg::node_registry registry;
    fg::plugin_loader loader;
};

template<typename T>
class builtin_multiply : public fg::node<builtin_multiply<T>> {
    T _factor = static_cast<T>(1.0f);

public:
    fg::IN<T>  in;
    fg::OUT<T> out;

    builtin_multiply() = delete;

    template<typename Arg, typename ArgV = std::remove_cvref_t<Arg>>
        requires(not std::is_same_v<Arg, T> and not std::is_same_v<Arg, builtin_multiply<T>>)
    explicit builtin_multiply(Arg &&) {}

    explicit builtin_multiply(T factor, std::string name = fg::this_source_location()) : _factor(factor) { this->set_name(name); }

    [[nodiscard]] constexpr auto
    process_one(T a) const noexcept {
        return a * _factor;
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE(builtin_multiply, in, out);

template<typename T>
class builtin_counter : public fg::node<builtin_counter<T>> {
public:
    static std::size_t s_event_count;

    fg::IN<T>          in;
    fg::OUT<T>         out;

    [[nodiscard]] constexpr auto
    process_one(T a) const noexcept {
        s_event_count++;
        return a;
    }
};

template<typename T>
std::size_t builtin_counter<T>::s_event_count = 0;

ENABLE_REFLECTION_FOR_TEMPLATE(builtin_counter, in, out);

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
        paths.push_back("test/plugins");
        paths.push_back("plugins");
    } else {
        for (int i = 1; i < argc; ++i) {
            paths.push_back(argv[i]);
        }
    }

    test_context context(std::move(paths));
    GP_REGISTER_NODE(&context.registry, builtin_multiply, double, float);
    GP_REGISTER_NODE(&context.registry, builtin_counter, double, float);

    fmt::print("PluginLoaderTests\n");
    using namespace gr;

    for (const auto &plugin : context.loader.plugins()) {
        assert(plugin->metadata().plugin_name.starts_with("Good"));
    }

    for (const auto &plugin : context.loader.failed_plugins()) {
        assert(plugin.first.ends_with("bad_plugin.so"));
    }

    auto        known = context.loader.known_nodes();
    std::vector requireds{ names::cout_sink, names::fixed_source, names::divide, names::multiply };

    for (const auto &required : requireds) {
        assert(std::ranges::find(known, required) != known.end());
    }

    fg::graph flow_graph;

    auto      node_source_load = context.loader.instantiate(names::fixed_source, "double");
    assert(node_source_load);
    auto &node_source          = flow_graph.add_node(std::move(node_source_load));

    auto &node_multiply_1      = flow_graph.make_node<builtin_multiply<double>>(2.0);

    auto  node_multiply_2_load = context.loader.instantiate(names::builtin_multiply, "double");
    assert(node_multiply_2_load);
    auto &node_multiply_2   = flow_graph.add_node(std::move(node_multiply_2_load));

    auto  node_counter_load = context.loader.instantiate(names::builtin_counter, "double");
    assert(node_counter_load);
    auto             &node_counter = flow_graph.add_node(std::move(node_counter_load));

    const std::size_t repeats      = 100;
    std::array        node_sink_params{ fair::graph::node_construction_param{ "total_count", "100" } };
    auto              node_sink_load = context.loader.instantiate(names::cout_sink, "double", node_sink_params);
    assert(node_sink_load);
    auto &node_sink    = flow_graph.add_node(std::move(node_sink_load));

    auto  connection_1 = flow_graph.dynamic_connect(node_source, 0, node_multiply_1, 0);
    auto  connection_2 = flow_graph.dynamic_connect(node_multiply_1, 0, node_multiply_2, 0);
    auto  connection_3 = flow_graph.dynamic_connect(node_multiply_2, 0, node_counter, 0);
    auto  connection_4 = flow_graph.dynamic_connect(node_counter, 0, node_sink, 0);

    assert(connection_1 == fg::connection_result_t::SUCCESS);
    assert(connection_2 == fg::connection_result_t::SUCCESS);
    assert(connection_3 == fg::connection_result_t::SUCCESS);
    assert(connection_4 == fg::connection_result_t::SUCCESS);

    for (std::size_t i = 0; i < repeats; ++i) {
        node_source.work();
        node_multiply_1.work();
        node_multiply_2.work();
        node_counter.work();
        node_sink.work();
    }

    fmt::print("repeats {} event_count {}\n", repeats, builtin_counter<double>::s_event_count);
    assert(builtin_counter<double>::s_event_count == repeats);
}
