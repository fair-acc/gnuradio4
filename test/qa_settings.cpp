#include <boost/ut.hpp>

#include <buffer.hpp>
#include <graph.hpp>
#include <node.hpp>
#include <reflection.hpp>
#include <scheduler.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#if defined(__clang__) && __clang_major__ >= 16
// clang 16 does not like ut's default reporter_junit due to some issues with stream buffers and output redirection
template<>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif

template<typename T>
struct fmt::formatter<std::complex<T>> {
    template<typename ParseContext>
    auto
    parse(ParseContext &ctx) {
        return std::begin(ctx);
    }

    template<typename FormatContext>
    auto
    format(const std::complex<T> value, FormatContext &ctx) const {
        return fmt::format_to(ctx.out(), "({}+{}i)", value.real(), value.imag());
    }
};

namespace fair::graph::setting_test {

namespace utils {

std::string
format_variant(const auto &value) noexcept {
    return std::visit(
            [](auto &&arg) {
                using Type = std::decay_t<decltype(arg)>;
                if constexpr (std::is_arithmetic_v<Type> || std::is_same_v<Type, std::string> || std::is_same_v<Type, std::complex<float>> || std::is_same_v<Type, std::complex<double>>) {
                    return fmt::format("{}", arg);
                } else if constexpr (std::is_same_v<Type, std::monostate>) {
                    return fmt::format("monostate");
                } else if constexpr (std::is_same_v<Type, std::vector<std::complex<float>>> || std::is_same_v<Type, std::vector<std::complex<double>>>) {
                    return fmt::format("[{}]", fmt::join(arg, ", "));
                } else if constexpr (std::is_same_v<Type, std::vector<bool>> || std::is_same_v<Type, std::vector<unsigned char>> || std::is_same_v<Type, std::vector<unsigned short>>
                                     || std::is_same_v<Type, std::vector<unsigned int>> || std::is_same_v<Type, std::vector<unsigned long>> || std::is_same_v<Type, std::vector<signed char>>
                                     || std::is_same_v<Type, std::vector<short>> || std::is_same_v<Type, std::vector<int>> || std::is_same_v<Type, std::vector<long>>
                                     || std::is_same_v<Type, std::vector<float>> || std::is_same_v<Type, std::vector<double>>) {
                    return fmt::format("[{}]", fmt::join(arg, ", "));
                } else {
                    return fmt::format("not-yet-supported type {}", fair::meta::type_name<Type>());
                }
            },
            value);
};

void
printChanges(const property_map &oldMap, const property_map &newMap) noexcept {
    for (const auto &[key, newValue] : newMap) {
        if (!oldMap.contains(key)) {
            fmt::print("    key added '{}` = {}\n", key, format_variant(newValue));
        } else {
            const auto &oldValue = oldMap.at(key);
            const bool  areEqual = std::visit(
                    [](auto &&arg1, auto &&arg2) {
                        if constexpr (std::is_same_v<std::decay_t<decltype(arg1)>, std::decay_t<decltype(arg2)>>) {
                            // compare values if they are of the same type
                            return arg1 == arg2;
                        } else {
                            return false; // values are of different type
                        }
                    },
                    oldValue, newValue);

            if (!areEqual) {
                fmt::print("    key value changed: '{}` = {} -> {}\n", key, format_variant(oldValue), format_variant(newValue));
            }
        }
    }
};
} // namespace utils

template<typename T>
struct Source : public node<Source<T>> {
    OUT<T>       out;
    std::int32_t n_samples_produced = 0;
    std::int32_t n_samples_max      = 1024;
    std::int32_t n_tag_offset       = 0;
    float        sample_rate        = 1000.0f;

    void
    settings_changed(const property_map & /*old_settings*/, const property_map & /*new_settings*/) {
        // optional init function that is called after construction and whenever settings change
        fair::graph::publish_tag(out, { { "n_samples_max", n_samples_max } }, static_cast<std::size_t>(n_tag_offset));
    }

    constexpr std::make_signed_t<std::size_t>
    available_samples(const Source & /*self*/) const noexcept {
        const auto ret = static_cast<std::make_signed_t<std::size_t>>(n_samples_max - n_samples_produced);
        return ret > 0 ? ret : -1; // '-1' -> DONE, produced enough samples
    }

    [[nodiscard]] constexpr T
    process_one() noexcept {
        n_samples_produced++;
        T x{};
        return x;
    }
};

// optional shortening
template<typename T, fair::meta::fixed_string description = "", typename... Arguments>
using A            = Annotated<T, description, Arguments...>;

using TestBlockDoc = Doc<R""(
some test doc documentation
)"">;

template<typename T>
struct TestBlock : public node<TestBlock<T>, BlockingIO, TestBlockDoc, SupportedTypes<float, double>> {
    IN<T>  in{};
    OUT<T> out{};
    // parameters
    A<T, "scaling factor", Visible, Doc<"y = a * x">, Unit<"As">> scaling_factor = static_cast<T>(1); // N.B. unit 'As' = 'Coulomb'
    A<std::string, "context information", Visible>                context{};
    std::int32_t                                                  n_samples_max = -1;
    float                                                         sample_rate   = 1000.0f;
    std::vector<T>                                                vector_setting{ T(3), T(2), T(1) };
    int                                                           update_count = 0;
    bool                                                          debug        = false;

    void
    settings_changed(const property_map &old_settings, const property_map &new_settings) noexcept {
        // optional function that is called whenever settings change
        update_count++;

        if (debug) {
            fmt::print("settings changed - update_count: {}\n", update_count);
            utils::printChanges(old_settings, new_settings);
        }
    }

    template<fair::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr V
    process_one(const V &a) const noexcept {
        return a * scaling_factor;
    }
};

static_assert(NodeType<TestBlock<int>>);
static_assert(NodeType<TestBlock<float>>);
static_assert(NodeType<TestBlock<double>>);

template<typename T>
struct Sink : public node<Sink<T>> {
    IN<T>        in;
    std::int32_t n_samples_consumed = 0;
    std::int32_t n_samples_max      = -1;
    int64_t      last_tag_position  = -1;
    float        sample_rate        = -1.0f;

    template<fair::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    process_one(V) noexcept {
        // alt: optional user-level tag processing
        /*
        if (this->input_tags_present()) {
            if (this->input_tags_present() && this->input_tags()[0].map.contains("n_samples_max")) {
                const auto value = this->input_tags()[0].map.at("n_samples_max");
                assert(std::holds_alternative<std::int32_t>(value));
                n_samples_max = std::get<std::int32_t>(value);
                last_tag_position        = in.streamReader().position();
                this->forward_tags(); // clears further notifications and forward tags to output ports
            }
        }
        */

        if constexpr (fair::meta::any_simd<V>) {
            n_samples_consumed += static_cast<std::int32_t>(V::size());
        } else {
            n_samples_consumed++;
        }
    }
};
} // namespace fair::graph::setting_test

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (fair::graph::setting_test::Source<T>), out, n_samples_produced, n_samples_max, n_tag_offset, sample_rate);
ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (fair::graph::setting_test::TestBlock<T>), in, out, scaling_factor, context, n_samples_max, sample_rate, vector_setting);
ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (fair::graph::setting_test::Sink<T>), in, n_samples_consumed, n_samples_max, last_tag_position, sample_rate);

const boost::ut::suite SettingsTests = [] {
    using namespace boost::ut;
    using namespace fair::graph;
    using namespace fair::graph::setting_test;
    using namespace std::string_view_literals;

    "basic node settings tag"_test = [] {
        graph                  flow_graph;
        constexpr std::int32_t n_samples = gr::util::round_up(1'000'000, 1024);
        // define basic Sink->TestBlock->Sink flow graph
        auto &src = flow_graph.make_node<Source<float>>({ { "n_samples_max", n_samples } });
        expect(eq(src.n_samples_max, n_samples)) << "check map constructor";
        expect(eq(src.settings().auto_update_parameters().size(), 4UL));
        expect(eq(src.settings().auto_forward_parameters().size(), 1UL)); // sample_rate
        auto &block1 = flow_graph.make_node<TestBlock<float>>();
        auto &block2 = flow_graph.make_node<TestBlock<float>>();
        auto &sink   = flow_graph.make_node<Sink<float>>();
        expect(eq(sink.settings().auto_update_parameters().size(), 5UL));
        expect(eq(sink.settings().auto_forward_parameters().size(), 1UL)); // sample_rate

        block1.context = "Test Context";
        block1.settings().update_active_parameters();
        expect(eq(block1.settings().auto_update_parameters().size(), 6UL));
        expect(eq(block1.settings().auto_forward_parameters().size(), 2UL));

        expect(block1.settings().get("context").has_value());
        expect(block1.settings().get({ "context" }).has_value());
        expect(not block1.settings().get({ "test" }).has_value());

        std::vector<std::string>   keys1{ "key1", "key2", "key3" };
        std::span<std::string>     keys2{ keys1 };
        std::array<std::string, 3> keys3{ "key1", "key2", "key3" };
        expect(block1.settings().get(keys1).empty());
        expect(block1.settings().get(keys2).empty());
        expect(block1.settings().get(keys3).empty());
        expect(eq(block1.settings().get().size(), 7UL));

        // set non-existent setting
        expect(not block1.settings().changed()) << "settings not changed";
        auto ret1 = block1.settings().set({ { "unknown", "random value" } });
        expect(eq(ret1.size(), 1U)) << "setting one unknown parameter";
        expect(eq(std::get<std::string>(static_cast<property_map>(block1.meta_information).at("unknown")), "random value"sv)) << "setting one unknown parameter";

        expect(not block1.settings().changed());
        auto ret2 = block1.settings().set({ { "context", "alt context" } });
        expect(not block1.settings().staged_parameters().empty());
        expect(ret2.empty()) << "setting one known parameter";
        expect(block1.settings().changed()) << "settings changed";
        auto forwarding_parameter = block1.settings().apply_staged_parameters();
        expect(eq(forwarding_parameter.size(), 1u)) << "initial forward declarations";
        block1.settings().update_active_parameters();

        // src -> block1 -> block2 -> sink
        expect(eq(connection_result_t::SUCCESS, flow_graph.connect<"out">(src).to<"in">(block1)));
        expect(eq(connection_result_t::SUCCESS, flow_graph.connect<"out">(block1).to<"in">(block2)));
        expect(eq(connection_result_t::SUCCESS, flow_graph.connect<"out">(block2).to<"in">(sink)));

        auto thread_pool = std::make_shared<fair::thread_pool::BasicThreadPool>("custom pool", fair::thread_pool::CPU_BOUND, 2, 2); // use custom pool to limit number of threads for emscripten
        fair::graph::scheduler::simple sched{ std::move(flow_graph), thread_pool };
        expect(src.settings().auto_update_parameters().contains("sample_rate"));
        std::ignore = src.settings().set({ { "sample_rate", 49000.0f } });
        sched.run_and_wait();
        expect(eq(src.n_samples_produced, n_samples)) << "did not produce enough output samples";
        expect(eq(sink.n_samples_consumed, n_samples)) << "did not consume enough input samples";

        expect(eq(src.n_samples_max, n_samples)) << "receive tag announcing max samples";
        expect(eq(block1.n_samples_max, n_samples)) << "receive tag announcing max samples";
        expect(eq(sink.n_samples_max, n_samples)) << "receive tag announcing max samples";

        // expect(eq(src.sample_rate, 49000.0f)) << "src matching sample_rate";
        expect(eq(block1.sample_rate, 49000.0f)) << "block1 matching sample_rate";
        expect(eq(block2.sample_rate, 49000.0f)) << "block2 matching sample_rate";
        expect(eq(sink.sample_rate, 49000.0f)) << "sink matching src sample_rate";

        // check auto-update flags
        expect(!src.settings().auto_update_parameters().contains("sample_rate")) << "src should not retain auto-update flag (was manually set)";
        expect(block1.settings().auto_update_parameters().contains("sample_rate")) << "block1 retained auto-update flag";
        expect(block2.settings().auto_update_parameters().contains("sample_rate")) << "block1 retained auto-update flag";
        expect(sink.settings().auto_update_parameters().contains("sample_rate")) << "sink retained auto-update flag";
    };

    "constructor"_test = [] {
        "empty"_test = [] {
            auto block = TestBlock<float>();
            block.init();
            expect(eq(block.settings().get().size(), 7UL));
            expect(eq(std::get<float>(*block.settings().get("scaling_factor")), 1.f));
        };

#if !defined(__clang_major__) && __clang_major__ <= 15
        "with init parameter"_test = [] {
            auto block = TestBlock<float>({ { "scaling_factor", 2.f } });
            expect(eq(block.settings().staged_parameters().size(), 1u));
            block.init();
            expect(eq(block.settings().staged_parameters().size(), 0u));
            block.settings().update_active_parameters();
            expect(eq(block.settings().get().size(), 7UL));
            expect(eq(block.scaling_factor, 2.f));
            expect(eq(std::get<float>(*block.settings().get("scaling_factor")), 2.f));
        };
#endif

        "empty via graph"_test = [] {
            graph flow_graph;
            auto &block = flow_graph.make_node<TestBlock<float>>();
            expect(eq(block.settings().get().size(), 7UL));
            expect(eq(block.scaling_factor, 1.f));
            expect(eq(std::get<float>(*block.settings().get("scaling_factor")), 1.f));
        };

        "with init parameter via graph"_test = [] {
            graph flow_graph;
            auto &block = flow_graph.make_node<TestBlock<float>>({ { "scaling_factor", 2.f } });
            expect(eq(block.settings().get().size(), 7UL));
            expect(eq(block.scaling_factor, 2.f));
            expect(eq(std::get<float>(*block.settings().get("scaling_factor")), 2.f));
        };
    };

    "vector-type support"_test = [] {
        graph flow_graph;
        auto &block = flow_graph.make_node<TestBlock<float>>();
        block.settings().update_active_parameters();
        expect(eq(block.settings().get().size(), 7UL));

        block.debug    = true;
        const auto val = block.settings().set({ { "vector_setting", std::vector{ 42.f, 2.f, 3.f } } });
        expect(val.empty()) << "unable to stage settings";
        block.init();
        expect(eq(block.vector_setting, std::vector{ 42.f, 2.f, 3.f }));
        expect(eq(block.update_count, 1)) << fmt::format("actual update count: {}\n", block.update_count);
    };

    "unique ID"_test = [] {
        graph flow_graph;
        auto &block1 = flow_graph.make_node<TestBlock<float>>();
        auto &block2 = flow_graph.make_node<TestBlock<float>>();
        expect(not eq(block1.unique_id, block2.unique_id)) << "unique per-type block id (size_t)";
        expect(not eq(block1.unique_name, block2.unique_name)) << "unique per-type block id (string)";

        auto merged1 = merge<"out", "in">(TestBlock<float>(), TestBlock<float>());
        auto merged2 = merge<"out", "in">(TestBlock<float>(), TestBlock<float>());
        expect(not eq(merged1.unique_id, merged2.unique_id)) << "unique per-type block id (size_t) ";
        expect(not eq(merged1.unique_name, merged2.unique_name)) << "unique per-type block id (string) ";
    };

    "run-time type-erased node setter/getter"_test = [] {
        auto wrapped1 = node_wrapper<TestBlock<float>>();
        wrapped1.init();
        wrapped1.set_name("test_name");
        expect(eq(wrapped1.name(), "test_name"sv)) << "node_model wrapper name";
        expect(not wrapped1.unique_name().empty()) << "unique name";
        std::ignore                          = wrapped1.settings().set({ { "context", "a string" } });
        (wrapped1.meta_information())["key"] = "value";
        expect(eq(std::get<std::string>(wrapped1.meta_information().at("key")), "value"sv)) << "node_model meta-information";

        // via constructor
        auto wrapped2 = node_wrapper<TestBlock<float>>({ { "name", "test_name" } });
        std::ignore   = wrapped2.settings().set({ { "context", "a string" } });
        wrapped2.init();
        expect(eq(wrapped2.name(), "test_name"sv)) << "node_model wrapper name";
        expect(not wrapped2.unique_name().empty()) << "unique name";
        std::ignore                          = wrapped2.settings().set({ { "context", "a string" } });
        (wrapped2.meta_information())["key"] = "value";
        expect(eq(std::get<std::string>(wrapped2.meta_information().at("key")), "value"sv)) << "node_model meta-information";
    };
};

const boost::ut::suite AnnotationTests = [] {
    using namespace boost::ut;
    using namespace fair::graph;
    using namespace fair::graph::setting_test;
    using namespace std::literals;

    "basic node annotations"_test = [] {
        graph             flow_graph;
        TestBlock<float> &block = flow_graph.make_node<TestBlock<float>>();
        expect(fair::graph::node_description<TestBlock<float>>().find(std::string_view(TestBlockDoc::value)) != std::string_view::npos);
        expect(eq(std::get<std::string>(block.meta_information.value.at("description")), std::string(TestBlockDoc::value))) << "type-erased block description";
        expect(eq(std::get<std::string>(block.meta_information.value.at("scaling_factor::description")), "scaling factor"sv));
        expect(eq(std::get<std::string>(block.meta_information.value.at("scaling_factor::documentation")), "y = a * x"sv));
        expect(eq(std::get<std::string>(block.meta_information.value.at("scaling_factor::unit")), "As"sv));
        expect(std::get<bool>(block.meta_information.value.at("scaling_factor::visible"))) << "visible being true";
        expect(block.scaling_factor.visible());
        expect(eq(block.scaling_factor.description(), std::string_view{ "scaling factor" }));
        expect(eq(block.scaling_factor.unit(), std::string_view{ "As" }));
        expect(eq(block.context.unit(), std::string_view{ "" }));
        expect(block.context.visible());
        expect(block.is_blocking());

        block.scaling_factor = 42.f; // test wrapper assignment operator
        expect(block.scaling_factor == 42.f) << "the answer to everything failed -- equal operator";
        expect(eq(block.scaling_factor.value, 42.f)) << "the answer to everything failed -- by value";
        expect(eq(block.scaling_factor, 42.f)) << "the answer to everything failed -- direct";

        // fmt::print("description:\n {}", fair::graph::node_description<TestBlock<float>>());
    };
};

int
main() { /* tests are statically executed */
}
