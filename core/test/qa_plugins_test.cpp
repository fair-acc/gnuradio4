#include <array>
#include <cassert>
#include <iostream>

#include <boost/ut.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <gnuradio-4.0/node_registry.hpp>
#include <gnuradio-4.0/plugin_loader.hpp>

#if defined(__clang__) && __clang_major__ >= 16
// clang 16 does not like ut's default reporter_junit due to some issues with stream buffers and output redirection
template<>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif

using namespace std::chrono_literals;
using namespace gr::literals;

namespace grg = gr;

struct test_context {
    grg::node_registry registry;
    grg::plugin_loader loader;

    test_context() : loader(&registry, std::vector<std::filesystem::path>{ "test/plugins", "plugins" }) {}
};

test_context &
context() {
    static test_context instance;
    return instance;
}

template<typename T>
class builtin_multiply : public grg::node<builtin_multiply<T>> {
    T _factor = static_cast<T>(1.0f);

public:
    grg::PortIn<T>  in;
    grg::PortOut<T> out;

    builtin_multiply() = delete;

    template<typename Arg, typename ArgV = std::remove_cvref_t<Arg>>
        requires(not std::is_same_v<Arg, T> and not std::is_same_v<Arg, builtin_multiply<T>>)
    explicit builtin_multiply(Arg &&) {}

    explicit builtin_multiply(T factor, std::string name = grg::this_source_location()) : _factor(factor) { this->set_name(name); }

    [[nodiscard]] constexpr auto
    process_one(T a) const noexcept {
        return a * _factor;
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE(builtin_multiply, in, out);
GP_REGISTER_NODE(&context().registry, builtin_multiply, double, float);

namespace names {
const auto fixed_source     = "good::fixed_source"s;
const auto cout_sink        = "good::cout_sink"s;
const auto multiply         = "good::multiply"s;
const auto divide           = "good::divide"s;
const auto convert          = "good::convert"s;
const auto builtin_multiply = "builtin_multiply"s;
} // namespace names

const boost::ut::suite PluginLoaderTests = [] {
    using namespace boost::ut;
    using namespace gr;

    "GoodPlugins"_test = [] {
        expect(!context().loader.plugins().empty());
        for (const auto &plugin : context().loader.plugins()) {
            expect(plugin->metadata->plugin_name.starts_with("Good"));
        }
    };

    "BadPlugins"_test = [] {
        expect(!context().loader.failed_plugins().empty());
        for (const auto &plugin : context().loader.failed_plugins()) {
            expect(plugin.first.ends_with("bad_plugin.so"));
        }
    };

    "KnownNodesList"_test = [] {
        auto        known = context().loader.known_nodes();
        std::vector requireds{ names::cout_sink, names::fixed_source, names::divide, names::multiply };

        for (const auto &required : requireds) {
            expect(std::ranges::find(known, required) != known.end());
        }
    };
};

const boost::ut::suite NodeInstantiationTests = [] {
    using namespace boost::ut;
    using namespace gr;

    "KnownNodesInstantiate"_test = [] {
        expect(context().loader.instantiate(names::fixed_source, "double") != nullptr);
        expect(context().loader.instantiate(names::cout_sink, "double") != nullptr);
        expect(context().loader.instantiate(names::multiply, "double") != nullptr);
        expect(context().loader.instantiate(names::divide, "double") != nullptr);
        expect(context().loader.instantiate(names::convert, "double;float") != nullptr);

        expect(context().loader.instantiate(names::fixed_source, "something") == nullptr);
        expect(context().loader.instantiate(names::cout_sink, "something") == nullptr);
        expect(context().loader.instantiate(names::multiply, "something") == nullptr);
        expect(context().loader.instantiate(names::divide, "something") == nullptr);
        expect(context().loader.instantiate(names::convert, "float;float") == nullptr);
    };

    "UnknownNodes"_test = [] { expect(context().loader.instantiate("ThisNodeDoesNotExist", "double") == nullptr); };
};

const boost::ut::suite BasicPluginNodesConnectionTests = [] {
    using namespace boost::ut;
    using namespace gr;

    "FixedSourceToSink"_test = [] {
        auto node_source  = context().loader.instantiate(names::fixed_source, "double");
        auto node_sink    = context().loader.instantiate(names::cout_sink, "double");
        auto connection_1 = node_source->dynamic_output_port(0).connect(node_sink->dynamic_input_port(0));
        expect(connection_1 == grg::connection_result_t::SUCCESS);
    };

    "LongerPipeline"_test = [] {
        auto                      node_source = context().loader.instantiate(names::fixed_source, "double");

        gr::property_map node_multiply_params;
        node_multiply_params["factor"]          = 2.0;
        auto                      node_multiply = context().loader.instantiate(names::multiply, "double", node_multiply_params);

        std::size_t               repeats       = 10;
        gr::property_map node_sink_params;
        node_sink_params["total_count"] = 100_UZ;
        auto node_sink                  = context().loader.instantiate(names::cout_sink, "double");

        auto connection_1               = node_source->dynamic_output_port(0).connect(node_multiply->dynamic_input_port(0));
        auto connection_2               = node_multiply->dynamic_output_port(0).connect(node_sink->dynamic_input_port(0));

        expect(connection_1 == grg::connection_result_t::SUCCESS);
        expect(connection_2 == grg::connection_result_t::SUCCESS);

        for (std::size_t i = 0; i < repeats; ++i) {
            std::ignore = node_source->work(std::numeric_limits<std::size_t>::max());
            std::ignore = node_multiply->work(std::numeric_limits<std::size_t>::max());
            std::ignore = node_sink->work(std::numeric_limits<std::size_t>::max());
        }
    };

    "Graph"_test = [] {
        grg::graph flow_graph;

        // Instantiate the node that is defined in a plugin
        auto &node_source = context().loader.instantiate_in_graph(flow_graph, names::fixed_source, "double");

        // Instantiate a built-in node in a static way
        gr::property_map node_multiply_1_params;
        node_multiply_1_params["factor"] = 2.0;
        auto &node_multiply_double       = flow_graph.make_node<builtin_multiply<double>>(node_multiply_1_params);

        // Instantiate a built-in node via the plugin loader
        auto &node_multiply_float    = context().loader.instantiate_in_graph(flow_graph, names::builtin_multiply, "float");

        auto &node_convert_to_float  = context().loader.instantiate_in_graph(flow_graph, names::convert, "double;float");
        auto &node_convert_to_double = context().loader.instantiate_in_graph(flow_graph, names::convert, "float;double");

        //
        std::size_t               repeats = 10;
        gr::property_map node_sink_params;
        node_sink_params["total_count"] = 100_UZ;
        auto  node_sink_load            = context().loader.instantiate(names::cout_sink, "double", node_sink_params);
        auto &node_sink                 = flow_graph.add_node(std::move(node_sink_load));

        auto  connection_1              = flow_graph.dynamic_connect(node_source, 0, node_multiply_double, 0);
        auto  connection_2              = flow_graph.dynamic_connect(node_multiply_double, 0, node_convert_to_float, 0);
        auto  connection_3              = flow_graph.dynamic_connect(node_convert_to_float, 0, node_multiply_float, 0);
        auto  connection_4              = flow_graph.dynamic_connect(node_multiply_float, 0, node_convert_to_double, 0);
        auto  connection_5              = flow_graph.dynamic_connect(node_convert_to_double, 0, node_sink, 0);

        expect(connection_1 == grg::connection_result_t::SUCCESS);
        expect(connection_2 == grg::connection_result_t::SUCCESS);
        expect(connection_3 == grg::connection_result_t::SUCCESS);
        expect(connection_4 == grg::connection_result_t::SUCCESS);
        expect(connection_5 == grg::connection_result_t::SUCCESS);

        for (std::size_t i = 0; i < repeats; ++i) {
            std::ignore = node_source.work(std::numeric_limits<std::size_t>::max());
            std::ignore = node_multiply_double.work(std::numeric_limits<std::size_t>::max());
            std::ignore = node_convert_to_float.work(std::numeric_limits<std::size_t>::max());
            std::ignore = node_multiply_float.work(std::numeric_limits<std::size_t>::max());
            std::ignore = node_convert_to_double.work(std::numeric_limits<std::size_t>::max());
            std::ignore = node_sink.work(std::numeric_limits<std::size_t>::max());
        }
    };
};

int
main() { /* not needed for UT */
}
