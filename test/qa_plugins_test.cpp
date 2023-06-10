// import boost.ut;
#include <boost/ut.hpp>

#include <node_registry.hpp>
#include <plugin_loader.hpp>

#include <array>
#include <cassert>
#include <iostream>

#include <fmt/format.h>
#include <fmt/ranges.h>

#if defined(__clang__) && __clang_major__ >= 16
// clang 16 does not like ut's default reporter_junit due to some issues with stream buffers and output redirection
template<>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif

using namespace std::chrono_literals;
using namespace fair::literals;

namespace fg = fair::graph;

struct test_context {
    fg::node_registry registry;
    fg::plugin_loader loader;

    test_context() : loader(&registry, std::vector<std::filesystem::path>{ "test/plugins", "plugins" }) {}
};

test_context &
context() {
    static test_context instance;
    return instance;
}

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
GP_REGISTER_NODE(&context().registry, builtin_multiply, double, float);

namespace names {
const auto fixed_source     = "good::fixed_source"s;
const auto cout_sink        = "good::cout_sink"s;
const auto multiply         = "good::multiply"s;
const auto divide           = "good::divide"s;
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
        expect(connection_1 == fg::connection_result_t::SUCCESS);
    };

    "LongerPipeline"_test = [] {
        auto                                  node_source = context().loader.instantiate(names::fixed_source, "double");

        fair::graph::node_construction_params node_multiply_params;
        node_multiply_params["factor"]                      = 2.0;
        auto                                  node_multiply = context().loader.instantiate(names::multiply, "double", node_multiply_params);

        std::size_t                           repeats       = 10;
        fair::graph::node_construction_params node_sink_params;
        node_sink_params["total_count"] = 100_UZ;
        auto node_sink                  = context().loader.instantiate(names::cout_sink, "double");

        auto connection_1               = node_source->dynamic_output_port(0).connect(node_multiply->dynamic_input_port(0));
        auto connection_2               = node_multiply->dynamic_output_port(0).connect(node_sink->dynamic_input_port(0));

        expect(connection_1 == fg::connection_result_t::SUCCESS);
        expect(connection_2 == fg::connection_result_t::SUCCESS);

        for (std::size_t i = 0; i < repeats; ++i) {
            node_source->work();
            node_multiply->work();
            node_sink->work();
        }
    };

    "Graph"_test = [] {
        fg::graph                             flow_graph;

        auto                                  node_source_load     = context().loader.instantiate(names::fixed_source, "double");
        auto                                 &node_source          = flow_graph.add_node(std::move(node_source_load));

        auto                                 &node_multiply_1      = flow_graph.make_node<builtin_multiply<double>>(2.0);

        auto                                  node_multiply_2_load = context().loader.instantiate(names::builtin_multiply, "double");
        auto                                 &node_multiply_2      = flow_graph.add_node(std::move(node_multiply_2_load));

        std::size_t                           repeats              = 10;
        fair::graph::node_construction_params node_sink_params;
        node_sink_params["total_count"] = 100_UZ;
        auto  node_sink_load            = context().loader.instantiate(names::cout_sink, "double", node_sink_params);
        auto &node_sink                 = flow_graph.add_node(std::move(node_sink_load));

        auto  connection_1              = flow_graph.dynamic_connect(node_source, 0, node_multiply_1, 0);
        auto  connection_2              = flow_graph.dynamic_connect(node_multiply_1, 0, node_multiply_2, 0);
        auto  connection_3              = flow_graph.dynamic_connect(node_multiply_2, 0, node_sink, 0);

        expect(connection_1 == fg::connection_result_t::SUCCESS);
        expect(connection_2 == fg::connection_result_t::SUCCESS);
        expect(connection_3 == fg::connection_result_t::SUCCESS);

        for (std::size_t i = 0; i < repeats; ++i) {
            node_source.work();
            node_multiply_1.work();
            node_multiply_2.work();
            node_sink.work();
        }
    };
};

int
main() { /* not needed for UT */
}
