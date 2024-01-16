#include <array>
#include <cassert>
#include <iostream>

#include <boost/ut.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/PluginLoader.hpp>

#if defined(__clang__) && __clang_major__ >= 16
// clang 16 does not like ut's default reporter_junit due to some issues with stream buffers and output redirection
template<>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif

using namespace std::chrono_literals;

template<typename T>
class builtin_multiply : public gr::Block<builtin_multiply<T>> {
    T _factor = static_cast<T>(1.0f);

public:
    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    builtin_multiply() = delete;

    template<typename Arg, typename ArgV = std::remove_cvref_t<Arg>>
        requires(not std::is_same_v<Arg, T> and not std::is_same_v<Arg, builtin_multiply<T>>)
    explicit builtin_multiply(Arg &&) {}

    explicit builtin_multiply(T factor, std::string name = gr::this_source_location()) : _factor(factor) { this->set_name(name); }

    [[nodiscard]] constexpr auto
    processOne(T a) const noexcept {
        return a * _factor;
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE(builtin_multiply, in, out);

struct TestContext {
    gr::BlockRegistry registry;
    gr::PluginLoader  loader;

    TestContext() : loader(&registry, std::vector<std::filesystem::path>{ "core/test/plugins", "test/plugins", "plugins" }) { GP_REGISTER_BLOCK_RUNTIME(&registry, builtin_multiply, double, float); }
};

TestContext &
context() {
    static TestContext instance;
    return instance;
}

namespace names {
constexpr auto fixed_source     = "good::fixed_source"sv;
constexpr auto cout_sink        = "good::cout_sink"sv;
constexpr auto multiply         = "good::multiply"sv;
constexpr auto divide           = "good::divide"sv;
constexpr auto convert          = "good::convert"sv;
constexpr auto builtin_multiply = "builtin_multiply"sv;
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
        auto       known = context().loader.knownBlocks();
        std::array requireds{ names::cout_sink, names::fixed_source, names::divide, names::multiply };

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
        auto connection_1 = block_source->dynamicOutputPort(0).connect(block_sink->dynamicInputPort(0));
        expect(connection_1 == gr::ConnectionResult::SUCCESS);
    };

    "LongerPipeline"_test = [] {
        auto block_source = context().loader.instantiate(names::fixed_source, "double");

        gr::property_map block_multiply_params;
        block_multiply_params["factor"] = 2.0;
        auto block_multiply             = context().loader.instantiate(names::multiply, "double", block_multiply_params);

        std::size_t      repeats = 10;
        gr::property_map block_sink_params;
        block_sink_params["total_count"] = 100UZ;
        auto block_sink                  = context().loader.instantiate(names::cout_sink, "double");

        auto connection_1 = block_source->dynamicOutputPort(0).connect(block_multiply->dynamicInputPort(0));
        auto connection_2 = block_multiply->dynamicOutputPort(0).connect(block_sink->dynamicInputPort(0));

        expect(connection_1 == gr::ConnectionResult::SUCCESS);
        expect(connection_2 == gr::ConnectionResult::SUCCESS);

        for (std::size_t i = 0; i < repeats; ++i) {
            std::ignore = block_source->work(std::numeric_limits<std::size_t>::max());
            std::ignore = block_multiply->work(std::numeric_limits<std::size_t>::max());
            std::ignore = block_sink->work(std::numeric_limits<std::size_t>::max());
        }
    };

    "Graph"_test = [] {
        gr::Graph testGraph;

        // Instantiate the node that is defined in a plugin
        auto &block_source = context().loader.instantiateInGraph(testGraph, names::fixed_source, "double");

        // Instantiate a built-in node in a static way
        gr::property_map block_multiply_1_params;
        block_multiply_1_params["factor"] = 2.0;
        auto &block_multiply_double       = testGraph.emplaceBlock<builtin_multiply<double>>(block_multiply_1_params);

        // Instantiate a built-in node via the plugin loader
        auto &block_multiply_float = context().loader.instantiateInGraph(testGraph, names::builtin_multiply, "float");

        auto &block_convert_to_float  = context().loader.instantiateInGraph(testGraph, names::convert, "double;float");
        auto &block_convert_to_double = context().loader.instantiateInGraph(testGraph, names::convert, "float;double");

        //
        std::size_t      repeats = 10;
        gr::property_map block_sink_params;
        block_sink_params["total_count"] = 100UZ;
        auto  block_sink_load            = context().loader.instantiate(names::cout_sink, "double", block_sink_params);
        auto &block_sink                 = testGraph.addBlock(std::move(block_sink_load));

        auto connection_1 = testGraph.connect(block_source, 0, block_multiply_double, 0);
        auto connection_2 = testGraph.connect(block_multiply_double, 0, block_convert_to_float, 0);
        auto connection_3 = testGraph.connect(block_convert_to_float, 0, block_multiply_float, 0);
        auto connection_4 = testGraph.connect(block_multiply_float, 0, block_convert_to_double, 0);
        auto connection_5 = testGraph.connect(block_convert_to_double, 0, block_sink, 0);

        expect(connection_1 == gr::ConnectionResult::SUCCESS);
        expect(connection_2 == gr::ConnectionResult::SUCCESS);
        expect(connection_3 == gr::ConnectionResult::SUCCESS);
        expect(connection_4 == gr::ConnectionResult::SUCCESS);
        expect(connection_5 == gr::ConnectionResult::SUCCESS);

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
