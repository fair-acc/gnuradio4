// import boost.ut;
#include <boost/ut.hpp>

#include <block_registry.hpp>
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
    fg::block_registry registry;
    fg::plugin_loader  loader;

    test_context() : loader(&registry, std::vector<std::filesystem::path>{ "test/plugins", "plugins" }) {}
};

test_context &
context() {
    static test_context instance;
    return instance;
}

template<typename T>
class builtin_multiply : public fg::block<builtin_multiply<T>> {
    T _factor = static_cast<T>(1.0f);

public:
    fg::PortIn<T>  in;
    fg::PortOut<T> out;

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
GP_REGISTER_BLOCK(&context().registry, builtin_multiply, double, float);

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

    "KnownBlocksList"_test = [] {
        auto        known = context().loader.known_blocks();
        std::vector requireds{ names::cout_sink, names::fixed_source, names::divide, names::multiply };

        for (const auto &required : requireds) {
            expect(std::ranges::find(known, required) != known.end());
        }
    };
};

const boost::ut::suite BlockInstantiationTests = [] {
    using namespace boost::ut;
    using namespace gr;

    "KnownBlocksInstantiate"_test = [] {
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

    "UnknownBlocks"_test = [] { expect(context().loader.instantiate("ThisBlockDoesNotExist", "double") == nullptr); };
};

const boost::ut::suite BasicPluginBlocksConnectionTests = [] {
    using namespace boost::ut;
    using namespace gr;

    "FixedSourceToSink"_test = [] {
        auto block_source = context().loader.instantiate(names::fixed_source, "double");
        auto block_sink   = context().loader.instantiate(names::cout_sink, "double");
        auto connection_1 = block_source->dynamic_output_port(0).connect(block_sink->dynamic_input_port(0));
        expect(connection_1 == fg::connection_result_t::SUCCESS);
    };

    "LongerPipeline"_test = [] {
        auto                      block_source = context().loader.instantiate(names::fixed_source, "double");

        fair::graph::property_map block_multiply_params;
        block_multiply_params["factor"]          = 2.0;
        auto                      block_multiply = context().loader.instantiate(names::multiply, "double", block_multiply_params);

        std::size_t               repeats        = 10;
        fair::graph::property_map block_sink_params;
        block_sink_params["total_count"] = 100_UZ;
        auto block_sink                  = context().loader.instantiate(names::cout_sink, "double");

        auto connection_1                = block_source->dynamic_output_port(0).connect(block_multiply->dynamic_input_port(0));
        auto connection_2                = block_multiply->dynamic_output_port(0).connect(block_sink->dynamic_input_port(0));

        expect(connection_1 == fg::connection_result_t::SUCCESS);
        expect(connection_2 == fg::connection_result_t::SUCCESS);

        for (std::size_t i = 0; i < repeats; ++i) {
            std::ignore = block_source->work(std::numeric_limits<std::size_t>::max());
            std::ignore = block_multiply->work(std::numeric_limits<std::size_t>::max());
            std::ignore = block_sink->work(std::numeric_limits<std::size_t>::max());
        }
    };

    "Graph"_test = [] {
        fg::graph flow_graph;

        // Instantiate the block that is defined in a plugin
        auto &block_source = context().loader.instantiate_in_graph(flow_graph, names::fixed_source, "double");

        // Instantiate a built-in block in a static way
        fair::graph::property_map block_multiply_1_params;
        block_multiply_1_params["factor"] = 2.0;
        auto &block_multiply_double       = flow_graph.make_block<builtin_multiply<double>>(block_multiply_1_params);

        // Instantiate a built-in block via the plugin loader
        auto &block_multiply_float    = context().loader.instantiate_in_graph(flow_graph, names::builtin_multiply, "float");

        auto &block_convert_to_float  = context().loader.instantiate_in_graph(flow_graph, names::convert, "double;float");
        auto &block_convert_to_double = context().loader.instantiate_in_graph(flow_graph, names::convert, "float;double");

        //
        std::size_t               repeats = 10;
        fair::graph::property_map block_sink_params;
        block_sink_params["total_count"] = 100_UZ;
        auto  block_sink_load            = context().loader.instantiate(names::cout_sink, "double", block_sink_params);
        auto &block_sink                 = flow_graph.add_block(std::move(block_sink_load));

        auto  connection_1               = flow_graph.dynamic_connect(block_source, 0, block_multiply_double, 0);
        auto  connection_2               = flow_graph.dynamic_connect(block_multiply_double, 0, block_convert_to_float, 0);
        auto  connection_3               = flow_graph.dynamic_connect(block_convert_to_float, 0, block_multiply_float, 0);
        auto  connection_4               = flow_graph.dynamic_connect(block_multiply_float, 0, block_convert_to_double, 0);
        auto  connection_5               = flow_graph.dynamic_connect(block_convert_to_double, 0, block_sink, 0);

        expect(connection_1 == fg::connection_result_t::SUCCESS);
        expect(connection_2 == fg::connection_result_t::SUCCESS);
        expect(connection_3 == fg::connection_result_t::SUCCESS);
        expect(connection_4 == fg::connection_result_t::SUCCESS);
        expect(connection_5 == fg::connection_result_t::SUCCESS);

        for (std::size_t i = 0; i < repeats; ++i) {
            std::ignore = block_source.work(std::numeric_limits<std::size_t>::max());
            std::ignore = block_multiply_double.work(std::numeric_limits<std::size_t>::max());
            std::ignore = block_convert_to_float.work(std::numeric_limits<std::size_t>::max());
            std::ignore = block_multiply_float.work(std::numeric_limits<std::size_t>::max());
            std::ignore = block_convert_to_double.work(std::numeric_limits<std::size_t>::max());
            std::ignore = block_sink.work(std::numeric_limits<std::size_t>::max());
        }
    };
};

int
main() { /* not needed for UT */
}
