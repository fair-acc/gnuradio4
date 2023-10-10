#include <stdexcept>
#include <string>

#include <boost/ut.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/buffer.hpp>
#include <gnuradio-4.0/graph.hpp>
#include <gnuradio-4.0/reflection.hpp>
#include <gnuradio-4.0/scheduler.hpp>
#include <gnuradio-4.0/tag.hpp>
#include <gnuradio-4.0/transactions.hpp>

using namespace std::string_literals;

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

namespace gr::setting_test {

namespace utils {

std::string
format_variant(const auto &value) noexcept {
    return std::visit(
            [](auto &arg) {
                using Type = std::decay_t<decltype(arg)>;
                if constexpr (std::is_arithmetic_v<Type> || std::is_same_v<Type, std::string> || std::is_same_v<Type, std::complex<float>> || std::is_same_v<Type, std::complex<double>>) {
                    return fmt::format("{}", arg);
                } else if constexpr (std::is_same_v<Type, std::monostate>) {
                    return fmt::format("monostate");
                } else if constexpr (std::is_same_v<Type, std::vector<std::complex<float>>> || std::is_same_v<Type, std::vector<std::complex<double>>>) {
                    return fmt::format("[{}]", fmt::join(arg, ", "));
                } else if constexpr (std::is_same_v<Type, std::vector<std::string>> || std::is_same_v<Type, std::vector<bool>> || std::is_same_v<Type, std::vector<unsigned char>>
                                     || std::is_same_v<Type, std::vector<unsigned short>> || std::is_same_v<Type, std::vector<unsigned int>> || std::is_same_v<Type, std::vector<unsigned long>>
                                     || std::is_same_v<Type, std::vector<signed char>> || std::is_same_v<Type, std::vector<short>> || std::is_same_v<Type, std::vector<int>>
                                     || std::is_same_v<Type, std::vector<long>> || std::is_same_v<Type, std::vector<float>> || std::is_same_v<Type, std::vector<double>>) {
                    return fmt::format("[{}]", fmt::join(arg, ", "));
                } else {
                    return fmt::format("not-yet-supported type {}", gr::meta::type_name<Type>());
                }
            },
            value);
}

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
struct Source : public Block<Source<T>> {
    PortOut<T>   out;
    std::int32_t n_samples_produced = 0;
    std::int32_t n_samples_max      = 1024;
    std::int32_t n_tag_offset       = 0;
    float        sample_rate        = 1000.0f;

    void
    settings_changed(const property_map & /*old_settings*/, property_map & /*new_settings*/) {
        // optional init function that is called after construction and whenever settings change
        gr::publish_tag(out, { { "n_samples_max", n_samples_max } }, static_cast<std::size_t>(n_tag_offset));
    }

    constexpr std::make_signed_t<std::size_t>
    available_samples(const Source & /*self*/) noexcept {
        const auto ret = static_cast<std::make_signed_t<std::size_t>>(n_samples_max - n_samples_produced);
        return ret > 0 ? ret : -1; // '-1' -> DONE, produced enough samples
    }

    [[nodiscard]] constexpr T
    processOne() noexcept {
        n_samples_produced++;
        T x{};
        return x;
    }
};

// optional shortening
template<typename T, gr::meta::fixed_string description = "", typename... Arguments>
using A            = Annotated<T, description, Arguments...>;

using TestBlockDoc = Doc<R""(
some test doc documentation
)"">;

template<typename T>
struct TestBlock : public Block<TestBlock<T>, BlockingIO<true>, TestBlockDoc, SupportedTypes<float, double>> {
    PortIn<T>  in{};
    PortOut<T> out{};
    // parameters
    A<T, "scaling factor", Visible, Doc<"y = a * x">, Unit<"As">>                    scaling_factor = static_cast<T>(1); // N.B. unit 'As' = 'Coulomb'
    A<std::string, "context information", Visible>                                   context{};
    std::int32_t                                                                     n_samples_max = -1;
    A<float, "sample rate", Limits<int64_t(0), std::numeric_limits<int64_t>::max()>> sample_rate   = 1000.0f;
    std::vector<T>                                                                   vector_setting{ T(3), T(2), T(1) };
    A<std::vector<std::string>, "string vector">                                     string_vector_setting = {};
    int                                                                              update_count          = 0;
    bool                                                                             debug                 = false;
    bool                                                                             resetCalled           = false;

    void
    settings_changed(const property_map &old_settings, property_map &new_settings) noexcept {
        // optional function that is called whenever settings change
        update_count++;

        if (debug) {
            fmt::print("settings changed - update_count: {}\n", update_count);
            utils::printChanges(old_settings, new_settings);
        }
    }

    void
    reset() {
        // optional reset function
        resetCalled = true;
    }

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr V
    processOne(const V &a) const noexcept {
        return a * scaling_factor;
    }
};

static_assert(BlockLike<TestBlock<int>>);
static_assert(BlockLike<TestBlock<float>>);
static_assert(BlockLike<TestBlock<double>>);

template<typename T, bool Average = false>
struct Decimate : public Block<Decimate<T, Average>, SupportedTypes<float, double>, ResamplingRatio<>, Doc<R""(
@brief reduces sample rate by given fraction controlled by denominator
)"">> {
    PortIn<T>                        in{};
    PortOut<T>                       out{};
    A<float, "sample rate", Visible> sample_rate = 1.f;

    void
    settings_changed(const property_map & /*old_settings*/, property_map &new_settings, property_map &fwd_settings) noexcept {
        if (new_settings.contains(std::string(gr::tag::SIGNAL_RATE.shortKey())) || new_settings.contains("denominator")) {
            const float fwdSampleRate                                  = sample_rate / static_cast<float>(this->denominator);
            fwd_settings[std::string(gr::tag::SIGNAL_RATE.shortKey())] = fwdSampleRate; // TODO: handle 'gr:sample_rate' vs 'sample_rate';
        }
    }

    constexpr WorkReturnStatus
    processBulk(std::span<const T> input, std::span<T> output) noexcept {
        assert(this->numerator == std::size_t(1) && "block implements only basic decimation");
        assert(this->denominator != std::size_t(0) && "denominator must be non-zero");

        auto outputIt = output.begin();
        if constexpr (Average) {
            for (std::size_t start = 0; start < input.size(); start += this->denominator) {
                constexpr auto chunk_begin = input.begin() + start;
                constexpr auto chunk_end   = chunk_begin + std::min(this->denominator, std::distance(chunk_begin, input.end()));
                *outputIt++                = std::reduce(chunk_begin, chunk_end, T(0)) / static_cast<T>(this->denominator);
            }
        } else {
            for (std::size_t i = 0; i < input.size(); i += this->denominator) {
                *outputIt++ = input[i];
            }
        }

        return WorkReturnStatus::OK;
    }
};

static_assert(BlockLike<Decimate<int>>);
static_assert(BlockLike<Decimate<float>>);
static_assert(BlockLike<Decimate<double>>);

template<typename T>
struct Sink : public Block<Sink<T>> {
    PortIn<T>    in;
    std::int32_t n_samples_consumed = 0;
    std::int32_t n_samples_max      = -1;
    int64_t      last_tag_position  = -1;
    float        sample_rate        = -1.0f;

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    processOne(V) noexcept {
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

        if constexpr (gr::meta::any_simd<V>) {
            n_samples_consumed += static_cast<std::int32_t>(V::size());
        } else {
            n_samples_consumed++;
        }
    }
};
} // namespace gr::setting_test

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (gr::setting_test::Source<T>), out, n_samples_produced, n_samples_max, n_tag_offset, sample_rate);
ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (gr::setting_test::TestBlock<T>), in, out, scaling_factor, context, n_samples_max, sample_rate, vector_setting, string_vector_setting);
ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, bool Average), (gr::setting_test::Decimate<T, Average>), in, out, sample_rate);
ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (gr::setting_test::Sink<T>), in, n_samples_consumed, n_samples_max, last_tag_position, sample_rate);

const boost::ut::suite SettingsTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::setting_test;
    using namespace std::string_view_literals;

    "basic node settings tag"_test = [] {
        graph                  testGraph;
        constexpr std::int32_t n_samples = gr::util::round_up(1'000'000, 1024);
        // define basic Sink->TestBlock->Sink flow graph
        auto &src = testGraph.emplaceBlock<Source<float>>({ { "n_samples_max", n_samples } });
        expect(eq(src.n_samples_max, n_samples)) << "check map constructor";
        expect(eq(src.settings().auto_update_parameters().size(), 4UL));
        expect(eq(src.settings().auto_forward_parameters().size(), 1UL)); // sample_rate
        auto &block1 = testGraph.emplaceBlock<TestBlock<float>>({ { "name", "TestBlock#1" } });
        auto &block2 = testGraph.emplaceBlock<TestBlock<float>>({ { "name", "TestBlock#2" } });
        auto &sink   = testGraph.emplaceBlock<Sink<float>>();
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
        expect(eq(block1.settings().get().size(), 11UL));

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
        expect(eq(connection_result_t::SUCCESS, testGraph.connect<"out">(src).to<"in">(block1)));
        expect(eq(connection_result_t::SUCCESS, testGraph.connect<"out">(block1).to<"in">(block2)));
        expect(eq(connection_result_t::SUCCESS, testGraph.connect<"out">(block2).to<"in">(sink)));

        auto thread_pool = std::make_shared<gr::thread_pool::BasicThreadPool>("custom pool", gr::thread_pool::CPU_BOUND, 2, 2); // use custom pool to limit number of threads for emscripten
        gr::scheduler::simple sched{ std::move(testGraph), thread_pool };
        expect(src.settings().auto_update_parameters().contains("sample_rate"));
        std::ignore = src.settings().set({ { "sample_rate", 49000.0f } });
        sched.run_and_wait();
        expect(eq(src.n_samples_produced, n_samples)) << "did not produce enough output samples";
        expect(eq(sink.n_samples_consumed, n_samples)) << "did not consume enough input samples";

        expect(eq(src.n_samples_max, n_samples)) << "receive tag announcing max samples";
        expect(eq(block1.n_samples_max, n_samples)) << "receive tag announcing max samples";
        expect(eq(sink.n_samples_max, n_samples)) << "receive tag announcing max samples";

        expect(eq(src.sample_rate, 49000.0f)) << "src matching sample_rate";
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
            block.init(block.progress, block.ioThreadPool); // N.B. self-assign existing progress and thread-pool (just for unit-tests)
            expect(eq(block.settings().get().size(), 11UL));
            expect(eq(std::get<float>(*block.settings().get("scaling_factor")), 1.f));
        };

#if !defined(__clang_major__) && __clang_major__ <= 15
        "with init parameter"_test = [] {
            auto block = TestBlock<float>({ { "scaling_factor", 2.f } });
            expect(eq(block.settings().staged_parameters().size(), 1u));
            block.init(block.progress, block.ioThreadPool); // N.B. self-assign existing progress and thread-pool (just for unit-tests)
            expect(eq(block.settings().staged_parameters().size(), 0u));
            block.settings().update_active_parameters();
            expect(eq(block.settings().get().size(), 11UL));
            expect(eq(block.scaling_factor, 2.f));
            expect(eq(std::get<float>(*block.settings().get("scaling_factor")), 2.f));
        };
#endif

        "empty via graph"_test = [] {
            graph testGraph;
            auto &block = testGraph.emplaceBlock<TestBlock<float>>();
            expect(eq(block.settings().get().size(), 11UL));
            expect(eq(block.scaling_factor, 1.f));
            expect(eq(std::get<float>(*block.settings().get("scaling_factor")), 1.f));
        };

        "with init parameter via graph"_test = [] {
            graph testGraph;
            auto &block = testGraph.emplaceBlock<TestBlock<float>>({ { "scaling_factor", 2.f } });
            expect(eq(block.settings().get().size(), 11UL));
            expect(eq(block.scaling_factor, 2.f));
            expect(eq(std::get<float>(*block.settings().get("scaling_factor")), 2.f));
        };
    };

    "vector-type support"_test = [] {
        graph testGraph;
        auto &block = testGraph.emplaceBlock<TestBlock<float>>();
        block.settings().update_active_parameters();
        expect(eq(block.settings().get().size(), 11UL));

        block.debug    = true;
        const auto val = block.settings().set({ { "vector_setting", std::vector{ 42.f, 2.f, 3.f } }, { "string_vector_setting", std::vector<std::string>{ "A", "B", "C" } } });
        expect(val.empty()) << "unable to stage settings";
        block.init(block.progress, block.ioThreadPool); // N.B. self-assign existing progress and thread-pool (just for unit-tests)
        expect(eq(block.vector_setting, std::vector{ 42.f, 2.f, 3.f }));
        expect(eq(block.string_vector_setting.value, std::vector<std::string>{ "A", "B", "C" }));
        expect(eq(block.update_count, 1)) << fmt::format("actual update count: {}\n", block.update_count);
    };

    "unique ID"_test = [] {
        graph       testGraph;
        const auto &block1 = testGraph.emplaceBlock<TestBlock<float>>();
        const auto &block2 = testGraph.emplaceBlock<TestBlock<float>>();
        expect(not eq(block1.unique_id, block2.unique_id)) << "unique per-type block id (size_t)";
        expect(not eq(block1.unique_name, block2.unique_name)) << "unique per-type block id (string)";

        auto merged1 = merge<"out", "in">(TestBlock<float>(), TestBlock<float>());
        auto merged2 = merge<"out", "in">(TestBlock<float>(), TestBlock<float>());
        expect(not eq(merged1.unique_id, merged2.unique_id)) << "unique per-type block id (size_t) ";
        expect(not eq(merged1.unique_name, merged2.unique_name)) << "unique per-type block id (string) ";
    };

    "run-time type-erased node setter/getter"_test = [] {
        auto progress     = std::make_shared<gr::Sequence>();
        auto ioThreadPool = std::make_shared<gr::thread_pool::BasicThreadPool>("test_pool", gr::thread_pool::TaskType::IO_BOUND, 2_UZ, std::numeric_limits<uint32_t>::max());
        //
        auto wrapped1 = BlockWrapper<TestBlock<float>>();
        wrapped1.init(progress, ioThreadPool);
        wrapped1.set_name("test_name");
        expect(eq(wrapped1.name(), "test_name"sv)) << "BlockModel wrapper name";
        expect(not wrapped1.unique_name().empty()) << "unique name";
        std::ignore                          = wrapped1.settings().set({ { "context", "a string" } });
        (wrapped1.meta_information())["key"] = "value";
        expect(eq(std::get<std::string>(wrapped1.meta_information().at("key")), "value"sv)) << "BlockModel meta-information";

        // via constructor
        auto wrapped2 = BlockWrapper<TestBlock<float>>({ { "name", "test_name" } });
        std::ignore   = wrapped2.settings().set({ { "context", "a string" } });
        wrapped2.init(progress, ioThreadPool);
        expect(eq(wrapped2.name(), "test_name"sv)) << "BlockModel wrapper name";
        expect(not wrapped2.unique_name().empty()) << "unique name";
        std::ignore                          = wrapped2.settings().set({ { "context", "a string" } });
        (wrapped2.meta_information())["key"] = "value";
        expect(eq(std::get<std::string>(wrapped2.meta_information().at("key")), "value"sv)) << "BlockModel meta-information";
    };

    "basic decimation test"_test = []() {
        graph                  testGraph;
        constexpr std::int32_t n_samples = gr::util::round_up(1'000'000, 1024);
        auto                  &src       = testGraph.emplaceBlock<Source<float>>({ { "n_samples_max", n_samples }, { "sample_rate", 1000.0f } });
        auto                  &block1    = testGraph.emplaceBlock<Decimate<float>>({ { "name", "Decimate1" }, { "denominator", std::size_t(2) } });
        auto                  &block2    = testGraph.emplaceBlock<Decimate<float>>({ { "name", "Decimate2" }, { "denominator", std::size_t(5) } });
        auto                  &sink      = testGraph.emplaceBlock<Sink<float>>();

        // check denominator
        expect(eq(block1.denominator, std::size_t(2)));
        expect(eq(block2.denominator, std::size_t(5)));

        // src -> block1 -> block2 -> sink
        expect(eq(connection_result_t::SUCCESS, testGraph.connect<"out">(src).to<"in">(block1)));
        expect(eq(connection_result_t::SUCCESS, testGraph.connect<"out">(block1).to<"in">(block2)));
        expect(eq(connection_result_t::SUCCESS, testGraph.connect<"out">(block2).to<"in">(sink)));

        gr::scheduler::simple sched{ std::move(testGraph) };
        sched.run_and_wait();

        expect(eq(src.n_samples_produced, n_samples)) << "did not produce enough output samples";
        expect(eq(sink.n_samples_consumed, n_samples / (2 * 5))) << "did not consume enough input samples";

        expect(eq(src.sample_rate, 1000.0f)) << "src matching sample_rate";
        expect(eq(block1.sample_rate, 1000.0f)) << "block1 matching sample_rate";
        expect(eq(block2.sample_rate, 500.0f)) << "block2 matching sample_rate";
        expect(eq(sink.sample_rate, 100.0f)) << "sink matching src sample_rate";
    };

    "basic store/reset settings"_test = []() {
        graph testGraph;
        auto &block = testGraph.emplaceBlock<TestBlock<float>>({ { "name", "TestName" }, { "scaling_factor", 2.f } });
        expect(block.name == "TestName");
        expect(eq(block.scaling_factor, 2.f));

        std::ignore = block.settings().set({ { "name", "TestNameAlt" }, { "scaling_factor", 42.f } });
        std::ignore = block.settings().apply_staged_parameters();
        expect(block.name == "TestNameAlt");
        expect(eq(block.scaling_factor, 42.f));

        expect(not block.resetCalled);
        block.settings().reset_defaults();
        expect(block.resetCalled);
        block.resetCalled = false;
        expect(block.name == "TestName");
        expect(eq(block.scaling_factor, 2.f));

        std::ignore = block.settings().set({ { "name", "TestNameAlt" }, { "scaling_factor", 42.f } });
        std::ignore = block.settings().apply_staged_parameters();
        expect(block.name == "TestNameAlt");
        expect(eq(block.scaling_factor, 42.f));
        block.settings().store_defaults();
        std::ignore = block.settings().set({ { "name", "TestNameAlt2" }, { "scaling_factor", 43.f } });
        std::ignore = block.settings().apply_staged_parameters();
        expect(block.name == "TestNameAlt2");
        expect(eq(block.scaling_factor, 43.f));
        block.settings().reset_defaults();
        expect(block.resetCalled);
        expect(block.name == "TestNameAlt");
        expect(eq(block.scaling_factor, 42.f));
    };
};

const boost::ut::suite AnnotationTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::setting_test;
    using namespace std::literals;

    "basic node annotations"_test = [] {
        graph             testGraph;
        TestBlock<float> &block = testGraph.emplaceBlock<TestBlock<float>>();
        expect(gr::blockDescription<TestBlock<float>>().find(std::string_view(TestBlockDoc::value)) != std::string_view::npos);
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

        // check validator
        expect(block.sample_rate.validate_and_set(1.f));
        expect(not block.sample_rate.validate_and_set(-1.f));

        constexpr auto                                             isPowerOfTwo   = [](const int &val) { return val > 0 && (val & (val - 1)) == 0; };
        Annotated<int, "power of two", Limits<0, 0, isPowerOfTwo>> needPowerOfTwo = 2;
        expect(isPowerOfTwo(4));
        expect(!isPowerOfTwo(5));
        expect(needPowerOfTwo.validate_and_set(4));
        expect(not needPowerOfTwo.validate_and_set(5));
        expect(eq(needPowerOfTwo.value, 4));

        Annotated<int, "power of two", Limits<0, 0, [](const int &val) { return (val > 0) && (val & (val - 1)) == 0; }>> needPowerOfTwoAlt = 2;
        expect(needPowerOfTwoAlt.validate_and_set(4));
        expect(not needPowerOfTwoAlt.validate_and_set(5));

        std::ignore = block.settings().set({ { "sample_rate", -1.0f } });
        std::ignore = block.settings().apply_staged_parameters(); // should print out a warning -> TODO: replace with pmt error message on msgOut port

        // fmt::print("description:\n {}", gr::node_description<TestBlock<float>>());
    };
};

const boost::ut::suite SettingsCtxTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::setting_test;

    "SettingsCtx basic"_test = [] {
        SettingsCtx a;
        SettingsCtx b;
        expect(a == b);
        a.time = std::chrono::system_clock::now();
        b.time = std::chrono::system_clock::now() + std::chrono::seconds(1);
        // chronologically sorted
        expect(a < b);
    };
};

const boost::ut::suite TransactionTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::setting_test;

    "CtxSettings"_test = [] {
        graph testGraph;
        auto &block = testGraph.emplaceBlock<TestBlock<float>>({ { "name", "TestName" }, { "scaling_factor", 2.f } });
        auto  s     = ctx_settings(block);
        auto  ctx0  = SettingsCtx(std::chrono::system_clock::now());
        std::ignore = s.set({ { "name", "TestNameAlt" }, { "scaling_factor", 42.f } }, ctx0);
        auto ctx1   = SettingsCtx(std::chrono::system_clock::now() + std::chrono::seconds(1));
        std::ignore = s.set({ { "name", "TestNameNew" }, { "scaling_factor", 43.f } }, ctx1);

        expect(eq(std::get<float>(*s.get("scaling_factor")), 43.f));       // get the latest value
        expect(eq(std::get<float>(*s.get("scaling_factor", ctx1)), 43.f)); // get same value, but over the context
        expect(eq(std::get<float>(*s.get("scaling_factor", ctx0)), 42.f)); // get value with an older timestamp
    };

    auto matchPred = [](const auto &lhs, const auto &rhs, const auto attempt) -> std::optional<bool> {
        if (attempt >= 4) {
            return std::nullopt;
        }

        constexpr std::array fields = { "BPCID", "SID", "BPID", "GID" };
        // require increasingly less fields to match for each attempt
        return std::ranges::all_of(fields | std::ranges::views::take(4 - attempt), [&](const auto &f) { return lhs.contains(f) && rhs.at(f) == lhs.at(f) && lhs.size() == 4 - attempt; });
    };

    "CtxSettings Matching"_test = [&] {
        graph      testGraph;
        auto      &block = testGraph.emplaceBlock<TestBlock<int>>({ { "scaling_factor", 42 } });
        auto       s     = ctx_settings(block, matchPred);
        const auto ctx0  = SettingsCtx(std::chrono::system_clock::now(), { { "BPCID", 1 }, { "SID", 1 }, { "BPID", 1 }, { "GID", 1 } });
        std::ignore      = s.set({ { "scaling_factor", 101 } }, ctx0);
        const auto ctx1  = SettingsCtx(std::chrono::system_clock::now(), { { "BPCID", 1 }, { "SID", 1 }, { "BPID", 1 } });
        std::ignore      = s.set({ { "scaling_factor", 102 } }, ctx1);
        const auto ctx2  = SettingsCtx(std::chrono::system_clock::now(), { { "BPCID", 1 }, { "SID", 1 } });
        std::ignore      = s.set({ { "scaling_factor", 103 } }, ctx2);
        const auto ctx3  = SettingsCtx(std::chrono::system_clock::now(), { { "BPCID", 1 } });
        std::ignore      = s.set({ { "scaling_factor", 104 } }, ctx3);

        // exact matches for contexts work
        expect(eq(std::get<int>(*s.get("scaling_factor", ctx0)), 101));
        expect(eq(std::get<int>(*s.get("scaling_factor", ctx1)), 102));
        expect(eq(std::get<int>(*s.get("scaling_factor", ctx2)), 103));
        expect(eq(std::get<int>(*s.get("scaling_factor", ctx3)), 104));

        // matching by using the custom predicate (no exact matching possible anymore)
        const auto ctx4 = SettingsCtx(std::chrono::system_clock::now(), { { "BPCID", 1 }, { "SID", 1 }, { "BPID", 1 }, { "GID", 2 } });
        expect(eq(std::get<int>(*s.get("scaling_factor", ctx4)), 102)); // no setting for 'gid=2' -> fall back to 'gid=-1'
        const auto ctx5 = SettingsCtx(std::chrono::system_clock::now(), { { "BPCID", 1 }, { "SID", 1 }, { "BPID", 2 }, { "GID", 2 } });
        expect(eq(std::get<int>(*s.get("scaling_factor", ctx5)), 103)); // no setting for 'pid=2' and 'gid=2' -> fall back to 'pid=gid=-1'

        // doesn't exist
        auto ctx6 = SettingsCtx(std::chrono::system_clock::now(), { { "BPCID", 9 }, { "SID", 9 }, { "BPID", 9 }, { "GID", 9 } });
        expect(s.get("scaling_factor", ctx6) == std::nullopt);
    };

    "CtxSettings Drop-In Settings replacement"_test = [&] {
        // the multiplexed Settings can be used as a drop-in replacement for "normal" Settings
        graph testGraph;
        auto &block = testGraph.emplaceBlock<TestBlock<float>>({ { "name", "TestName" }, { "scaling_factor", 2.f } });
        auto  s     = std::make_unique<ctx_settings<std::remove_reference<decltype(block)>::type>>(block, matchPred);
        block.setSettings(s);
        auto ctx0   = SettingsCtx(std::chrono::system_clock::now());
        std::ignore = block.settings().set({ { "name", "TestNameAlt" }, { "scaling_factor", 42.f } }, ctx0);
        expect(eq(std::get<float>(*block.settings().get("scaling_factor")), 42.f));
    };
};

int
main() { /* tests are statically executed */
}
