#include <string>

#include <boost/ut.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Buffer.hpp>
#include <gnuradio-4.0/Graph.hpp>
#ifndef __EMSCRIPTEN__
#include <gnuradio-4.0/Graph_yaml_importer.hpp>
#endif
#include <gnuradio-4.0/reflection.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/Transactions.hpp>

using namespace std::string_literals;

#if defined(__clang__) && __clang_major__ >= 16
// clang 16 does not like ut's default reporter_junit due to some issues with stream buffers and output redirection
template<>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif

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
    PortOut<T> out;
    gr::Size_t n_samples_produced = 0;
    gr::Size_t n_samples_max      = 1024;
    float      sample_rate        = 1000.0f;

    void
    settingsChanged(const property_map & /*oldSettings*/, property_map &newSettings, property_map &fwdSettings) {
        // optional init function that is called after construction and whenever settings change
        newSettings.insert_or_assign("n_samples_max", n_samples_max);
        fwdSettings.insert_or_assign("n_samples_max", n_samples_max);
    }

    [[nodiscard]] constexpr T
    processOne() noexcept {
        n_samples_produced++;
        if (n_samples_produced >= n_samples_max) {
            this->requestStop();
        }
        return T{};
    }
};

// optional shortening
template<typename T, gr::meta::fixed_string description = "", typename... Arguments>
using A = Annotated<T, description, Arguments...>;

template<typename T>
struct TestBlock : public Block<TestBlock<T>, BlockingIO<true>, SupportedTypes<float, double>> {
    using Description = Doc<R""(
some test doc documentation
)"">;
    PortIn<T>  in{};
    PortOut<T> out{};
    // parameters
    A<T, "scaling factor", Visible, Doc<"y = a * x">, Unit<"As">>                    scaling_factor = static_cast<T>(1); // N.B. unit 'As' = 'Coulomb'
    A<std::string, "context information", Visible>                                   context{};
    gr::Size_t                                                                       n_samples_max = 0;
    A<float, "sample rate", Limits<int64_t(0), std::numeric_limits<int64_t>::max()>> sample_rate   = 1.0f;
    std::vector<T>                                                                   vector_setting{ T(3), T(2), T(1) };
    A<std::vector<std::string>, "string vector">                                     string_vector_setting = {};
    int                                                                              update_count          = 0;
    bool                                                                             debug                 = true;
    bool                                                                             resetCalled           = false;
    gr::Size_t                                                                       n_samples_consumed    = 0;

    void
    settingsChanged(const property_map &oldSettings, property_map &newSettings, property_map &fwdSettings) noexcept {
        // optional function that is called whenever settings change
        update_count++;

        if (debug) {
            fmt::println("block '{}' settings changed - update_count: {}", this->name, update_count);
            utils::printChanges(oldSettings, newSettings);
            for (const auto &[key, value] : fwdSettings) {
                fmt::println(" -- forward: '{}':{}", key, value);
            }
        }
    }

    void
    reset() {
        // optional reset function
        n_samples_consumed = 0;
        resetCalled        = true;
    }

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr V
    processOne(const V &a) noexcept {
        if constexpr (gr::meta::any_simd<V>) {
            n_samples_consumed += static_cast<std::int32_t>(V::size());
        } else {
            n_samples_consumed++;
        }
        return a * scaling_factor;
    }
};

static_assert(BlockLike<TestBlock<int>>);
static_assert(BlockLike<TestBlock<float>>);
static_assert(BlockLike<TestBlock<double>>);

template<typename T, bool Average = false>
struct Decimate : public Block<Decimate<T, Average>, SupportedTypes<float, double>, ResamplingRatio<>> {
    using Description = Doc<R""(
@brief reduces sample rate by given fraction controlled by denominator
)"">;
    PortIn<T>                        in{};
    PortOut<T>                       out{};
    A<float, "sample rate", Visible> sample_rate = 1.f;

    void
    settingsChanged(const property_map & /*old_settings*/, property_map &new_settings, property_map &fwd_settings) noexcept {
        if (new_settings.contains(std::string(gr::tag::SIGNAL_RATE.shortKey())) || new_settings.contains("denominator")) {
            const float fwdSampleRate                                  = sample_rate / static_cast<float>(this->denominator);
            fwd_settings[std::string(gr::tag::SIGNAL_RATE.shortKey())] = fwdSampleRate; // TODO: handle 'gr:sample_rate' vs 'sample_rate';
            fmt::println("change sample_rate for {} --- {} / {} -> {}", this->name, sample_rate, this->denominator, fwdSampleRate);
        }
    }

    constexpr work::Status
    processBulk(std::span<const T> input, std::span<T> output) noexcept {
        assert(this->numerator == gr::Size_t(1) && "block implements only basic decimation");
        assert(this->denominator != gr::Size_t(0) && "denominator must be non-zero");

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

        return work::Status::OK;
    }
};

static_assert(BlockLike<Decimate<int>>);
static_assert(BlockLike<Decimate<float>>);
static_assert(BlockLike<Decimate<double>>);

template<typename T>
struct Sink : public Block<Sink<T>> {
    PortIn<T>  in;
    gr::Size_t n_samples_consumed = 0;
    gr::Size_t n_samples_max      = 0;
    int64_t    last_tag_position  = -1;
    float      sample_rate        = 1.0f;

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    processOne(V) noexcept {
        if constexpr (gr::meta::any_simd<V>) {
            n_samples_consumed += static_cast<gr::Size_t>(V::size());
        } else {
            n_samples_consumed++;
        }
    }
};
} // namespace gr::setting_test

ENABLE_REFLECTION_FOR_TEMPLATE(gr::setting_test::Source, out, n_samples_produced, n_samples_max, sample_rate)
ENABLE_REFLECTION_FOR_TEMPLATE(gr::setting_test::TestBlock, in, out, scaling_factor, context, n_samples_max, sample_rate, vector_setting, string_vector_setting)
ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, bool Average), (gr::setting_test::Decimate<T, Average>), in, out, sample_rate)
ENABLE_REFLECTION_FOR_TEMPLATE(gr::setting_test::Sink, in, n_samples_consumed, n_samples_max, last_tag_position, sample_rate)

const boost::ut::suite SettingsTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::setting_test;
    using namespace std::string_view_literals;

    "basic node settings tag"_test = [] {
        Graph                testGraph;
        constexpr gr::Size_t n_samples = gr::util::round_up(1'000'000, 1024);
        // define basic Sink->TestBlock->Sink flow graph
        auto &src = testGraph.emplaceBlock<Source<float>>({ { "sample_rate", 42.f }, { "n_samples_max", n_samples } });
        expect(eq(src.n_samples_max, n_samples)) << "check map constructor";
        expect(eq(src.settings().autoUpdateParameters().size(), 3UL));
        expect(eq(src.settings().autoForwardParameters().size(), 1UL)); // sample_rate
        auto &block1 = testGraph.emplaceBlock<TestBlock<float>>({ { "name", "TestBlock#1" } });
        auto &block2 = testGraph.emplaceBlock<TestBlock<float>>({ { "name", "TestBlock#2" } });
        auto &sink   = testGraph.emplaceBlock<Sink<float>>();
        expect(eq(sink.settings().autoUpdateParameters().size(), 6UL));
        expect(eq(sink.settings().autoForwardParameters().size(), 1UL)); // sample_rate

        block1.context = "Test Context";
        block1.settings().updateActiveParameters();
        expect(eq(block1.settings().autoUpdateParameters().size(), 7UL));
        expect(eq(block1.settings().autoForwardParameters().size(), 2UL));
        // need to add 'n_samples_max' to forwarding list for the block to automatically forward it
        // as the 'n_samples_max' tag is not part of the canonical 'gr::tag::DEFAULT_TAGS' list
        block1.settings().autoForwardParameters().emplace("n_samples_max");
        expect(eq(block1.settings().autoForwardParameters().size(), 3UL));
        // same check for block2
        expect(eq(block2.settings().autoForwardParameters().size(), 2UL));
        block2.settings().autoForwardParameters().emplace("n_samples_max");
        expect(eq(block2.settings().autoForwardParameters().size(), 3UL));
        sink.settings().autoForwardParameters().emplace("n_samples_max");

        expect(block1.settings().get("context").has_value());
        expect(block1.settings().get({ "context" }).has_value());
        expect(not block1.settings().get({ "test" }).has_value());

        std::vector<std::string>   keys1{ "key1", "key2", "key3" };
        std::span<std::string>     keys2{ keys1 };
        std::array<std::string, 3> keys3{ "key1", "key2", "key3" };
        expect(block1.settings().get(keys1).empty());
        expect(block1.settings().get(keys2).empty());
        expect(block1.settings().get(keys3).empty());
        expect(eq(block1.settings().get().size(), 12UL));

        // set non-existent setting
        expect(not block1.settings().changed()) << "settings not changed";
        auto ret1 = block1.settings().set({ { "unknown", "random value" } });
        expect(eq(ret1.size(), 1U)) << "setting one unknown parameter";
        expect(eq(std::get<std::string>(static_cast<property_map>(block1.meta_information).at("unknown")), "random value"sv)) << "setting one unknown parameter";

        expect(not block1.settings().changed());
        auto ret2 = block1.settings().set({ { "context", "alt context" } });
        expect(not block1.settings().stagedParameters().empty());
        expect(ret2.empty()) << "setting one known parameter";
        expect(block1.settings().changed()) << "settings changed";
        auto applyResult = block1.settings().applyStagedParameters();
        expect(eq(applyResult.forwardParameters.size(), 1u)) << "initial forward declarations";
        block1.settings().updateActiveParameters();

        // src -> block1 -> block2 -> sink
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).to<"in">(block1)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(block1).to<"in">(block2)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(block2).to<"in">(sink)));

        expect(!src.settings().autoUpdateParameters().contains("sample_rate")) << "manual setting disable auto-update";
        expect(src.settings().set({ { "sample_rate", 49000.0f } }).empty()) << "successful set returns empty map";

        auto thread_pool = std::make_shared<gr::thread_pool::BasicThreadPool>("custom pool", gr::thread_pool::CPU_BOUND, 2, 2); // use custom pool to limit number of threads for emscripten
        gr::scheduler::Simple sched{ std::move(testGraph), thread_pool };
        expect(sched.runAndWait().has_value());

        expect(eq(src.n_samples_produced, n_samples)) << "src did not produce enough output samples";
        expect(eq(static_cast<gr::Size_t>(block1.n_samples_consumed), n_samples)) << "block1 did not consume enough input samples";
        expect(eq(static_cast<gr::Size_t>(block2.n_samples_consumed), n_samples)) << "block2 did not consume enough input samples";
        expect(eq(sink.n_samples_consumed, n_samples)) << "sink did not consume enough input samples";

        for (auto &fwd : src.settings().autoUpdateParameters()) {
            fmt::println("## src auto {}", fwd);
        }
        for (auto &fwd : block1.settings().autoUpdateParameters()) {
            fmt::println("## block1 auto {}", fwd);
        }
        for (auto &fwd : block2.settings().autoUpdateParameters()) {
            fmt::println("## block2 auto {}", fwd);
        }
        for (auto &fwd : sink.settings().autoUpdateParameters()) {
            fmt::println("## sink auto {}", fwd);
        }

        expect(eq(src.n_samples_max, n_samples)) << "receive tag announcing max samples";
        expect(eq(block1.n_samples_max, n_samples)) << "receive tag announcing max samples";
        expect(eq(block2.n_samples_max, n_samples)) << "receive tag announcing max samples";
        expect(eq(sink.n_samples_max, n_samples)) << "receive tag announcing max samples";

        expect(eq(src.sample_rate, 49000.0f)) << "src matching sample_rate";
        expect(eq(block1.sample_rate, 49000.0f)) << "block1 matching sample_rate";
        expect(eq(block2.sample_rate, 49000.0f)) << "block2 matching sample_rate";
        expect(eq(sink.sample_rate, 49000.0f)) << "sink matching src sample_rate";

        // check auto-update flags
        expect(!src.settings().autoUpdateParameters().contains("sample_rate")) << "src should not retain auto-update flag (was manually set)";
        expect(block1.settings().autoUpdateParameters().contains("sample_rate")) << "block1 retained auto-update flag";
        expect(block1.settings().autoUpdateParameters().contains("n_samples_max")) << "block2 retained auto-update flag";
        expect(block1.settings().autoForwardParameters().contains("n_samples_max")) << "block2 retained auto-forward flag";
        expect(block2.settings().autoUpdateParameters().contains("sample_rate")) << "block2 retained auto-update flag";
        expect(block2.settings().autoUpdateParameters().contains("n_samples_max")) << "block2 retained auto-update flag";
        expect(block2.settings().autoForwardParameters().contains("n_samples_max")) << "block2 retained auto-forward flag";
        expect(sink.settings().autoUpdateParameters().contains("sample_rate")) << "sink retained auto-update flag";
        expect(sink.settings().autoUpdateParameters().contains("n_samples_max")) << "sink retained auto-update flag";

        fmt::println("finished test");
    };

    "constructor"_test = [] {
        "empty"_test = [] {
            auto block = TestBlock<float>();
            block.init(block.progress, block.ioThreadPool); // N.B. self-assign existing progress and thread-pool (just for unit-tests)
            expect(eq(block.settings().get().size(), 12UL));
            expect(eq(std::get<float>(*block.settings().get("scaling_factor")), 1.f));
        };

#if !defined(__clang_major__) && __clang_major__ <= 15
        "with init parameter"_test = [] {
            auto block = TestBlock<float>({ { "scaling_factor", 2.f } });
            expect(eq(block.settings().stagedParameters().size(), 1u));
            block.init(block.progress, block.ioThreadPool); // N.B. self-assign existing progress and thread-pool (just for unit-tests)
            expect(eq(block.settings().stagedParameters().size(), 0u));
            block.settings().updateActiveParameters();
            expect(eq(block.settings().get().size(), 12UL));
            expect(eq(block.scaling_factor, 2.f));
            expect(eq(std::get<float>(*block.settings().get("scaling_factor")), 2.f));
        };
#endif

        "empty via graph"_test = [] {
            Graph testGraph;
            auto &block = testGraph.emplaceBlock<TestBlock<float>>();
            expect(eq(block.settings().get().size(), 12UL));
            expect(eq(block.scaling_factor, 1.f));
            expect(eq(std::get<float>(*block.settings().get("scaling_factor")), 1.f));
        };

        "with init parameter via graph"_test = [] {
            Graph testGraph;
            auto &block = testGraph.emplaceBlock<TestBlock<float>>({ { "scaling_factor", 2.f } });
            expect(eq(block.settings().get().size(), 12UL));
            expect(eq(block.scaling_factor, 2.f));
            expect(eq(std::get<float>(*block.settings().get("scaling_factor")), 2.f));
        };
    };

    "vector-type support"_test = [] {
        Graph testGraph;
        auto &block = testGraph.emplaceBlock<TestBlock<float>>();
        block.settings().updateActiveParameters();
        expect(eq(block.settings().get().size(), 12UL));

        block.debug    = true;
        const auto val = block.settings().set({ { "vector_setting", std::vector{ 42.f, 2.f, 3.f } }, { "string_vector_setting", std::vector<std::string>{ "A", "B", "C" } } });
        expect(val.empty()) << "unable to stage settings";
        block.init(block.progress, block.ioThreadPool); // N.B. self-assign existing progress and thread-pool (just for unit-tests)
        expect(eq(block.vector_setting, std::vector{ 42.f, 2.f, 3.f }));
        expect(eq(block.string_vector_setting.value, std::vector<std::string>{ "A", "B", "C" }));
        expect(eq(block.update_count, 1)) << fmt::format("actual update count: {}\n", block.update_count);
    };

    "unique ID"_test = [] {
        Graph       testGraph;
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
        auto ioThreadPool = std::make_shared<gr::thread_pool::BasicThreadPool>("test_pool", gr::thread_pool::TaskType::IO_BOUND, 2UZ, std::numeric_limits<uint32_t>::max());
        //
        auto wrapped1 = BlockWrapper<TestBlock<float>>();
        wrapped1.init(progress, ioThreadPool);
        wrapped1.setName("test_name");
        expect(eq(wrapped1.name(), "test_name"sv)) << "BlockModel wrapper name";
        expect(not wrapped1.uniqueName().empty()) << "unique name";
        expect(wrapped1.settings().set({ { "context", "a string" } }).empty()) << "successful set returns empty map";
        (wrapped1.metaInformation())["key"] = "value";
        expect(eq(std::get<std::string>(wrapped1.metaInformation().at("key")), "value"sv)) << "BlockModel meta-information";

        // via constructor
        auto wrapped2 = BlockWrapper<TestBlock<float>>({ { "name", "test_name" } });
        expect(wrapped2.settings().set({ { "context", "a string" } }).empty()) << "successful set returns empty map";
        wrapped2.init(progress, ioThreadPool);
        expect(eq(wrapped2.name(), "test_name"sv)) << "BlockModel wrapper name";
        expect(not wrapped2.uniqueName().empty()) << "unique name";
        expect(wrapped2.settings().set({ { "context", "a string" } }).empty()) << "successful set returns empty map";
        (wrapped2.metaInformation())["key"] = "value";
        expect(eq(std::get<std::string>(wrapped2.metaInformation().at("key")), "value"sv)) << "BlockModel meta-information";
    };

    "basic decimation test"_test = []() {
        Graph                testGraph;
        constexpr gr::Size_t n_samples = gr::util::round_up(1'000'000, 1024);
        auto                &src       = testGraph.emplaceBlock<Source<float>>({ { "n_samples_max", n_samples }, { "sample_rate", 1000.0f } });
        auto                &block1    = testGraph.emplaceBlock<Decimate<float>>({ { "name", "Decimate1" }, { "denominator", gr::Size_t(2) } });
        auto                &block2    = testGraph.emplaceBlock<Decimate<float>>({ { "name", "Decimate2" }, { "denominator", gr::Size_t(5) } });
        auto                &sink      = testGraph.emplaceBlock<Sink<float>>();

        // check denominator
        expect(eq(block1.denominator, std::size_t(2)));
        expect(eq(block2.denominator, std::size_t(5)));

        // src -> block1 -> block2 -> sink
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).to<"in">(block1)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(block1).to<"in">(block2)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(block2).to<"in">(sink)));

        gr::scheduler::Simple sched{ std::move(testGraph) };
        expect(sched.runAndWait().has_value());

        expect(eq(src.n_samples_produced, n_samples)) << "did not produce enough output samples";
        expect(eq(sink.n_samples_consumed, n_samples / (2 * 5))) << "did not consume enough input samples";

        expect(eq(src.sample_rate, 1000.0f)) << "src matching sample_rate";
        expect(eq(block1.sample_rate, 1000.0f)) << "block1 matching sample_rate";
        expect(eq(block2.sample_rate, 500.0f)) << "block2 matching sample_rate";
        expect(eq(sink.sample_rate, 100.0f)) << "sink matching src sample_rate";
    };

    "basic store/reset settings"_test = []() {
        Graph testGraph;
        auto &block = testGraph.emplaceBlock<TestBlock<float>>({ { "name", "TestName" }, { "scaling_factor", 2.f } });
        expect(block.name == "TestName");
        expect(eq(block.scaling_factor, 2.f));

        expect(block.settings().set({ { "name", "TestNameAlt" }, { "scaling_factor", 42.f } }).empty()) << "successful set returns empty map";
        expect(block.settings().applyStagedParameters().forwardParameters.empty()) << "successful set returns empty map";
        expect(block.name == "TestNameAlt");
        expect(eq(block.scaling_factor, 42.f));

        expect(not block.resetCalled);
        block.settings().resetDefaults();
        expect(block.resetCalled);
        block.resetCalled = false;
        expect(block.name == "TestName");
        expect(eq(block.scaling_factor, 2.f));

        expect(block.settings().set({ { "name", "TestNameAlt" }, { "scaling_factor", 42.f } }).empty()) << "successful set returns empty map";
        expect(block.settings().applyStagedParameters().forwardParameters.empty()) << "successful set returns empty map";
        expect(block.name == "TestNameAlt");
        expect(eq(block.scaling_factor, 42.f));
        block.settings().storeDefaults();
        expect(block.settings().set({ { "name", "TestNameAlt2" }, { "scaling_factor", 43.f } }).empty()) << "successful set returns empty map";
        expect(block.settings().applyStagedParameters().forwardParameters.empty()) << "successful set returns empty map";
        expect(block.name == "TestNameAlt2");
        expect(eq(block.scaling_factor, 43.f));
        block.settings().resetDefaults();
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
        Graph testGraph;
        TestBlock<float> &block = testGraph.emplaceBlock<TestBlock<float>>();
        expect(gr::blockDescription<TestBlock<float>>().find(std::string_view(TestBlock<float>::Description::value)) != std::string_view::npos);
        expect(eq(std::get<std::string>(block.meta_information.value.at("description")), std::string(TestBlock<float>::Description::value))) << "type-erased block description";
        expect(eq(std::get<std::string>(block.meta_information.value.at("scaling_factor::description")), "scaling factor"sv));
        expect(eq(std::get<std::string>(block.meta_information.value.at("scaling_factor::documentation")), "y = a * x"sv));
        expect(eq(std::get<std::string>(block.meta_information.value.at("scaling_factor::unit")), "As"sv));
        expect(std::get<bool>(block.meta_information.value.at("scaling_factor::visible"))) << "visible being true";
        expect(block.scaling_factor.visible());
        expect(eq(block.scaling_factor.description(), std::string_view{ "scaling factor" }));
        expect(eq(block.scaling_factor.unit(), std::string_view{ "As" }));
        expect(eq(block.context.unit(), std::string_view{ "" }));
        expect(block.context.visible());
        expect(block.isBlocking());

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

        expect(block.settings().set({ { "sample_rate", -1.0f } }).empty()) << "successful set returns empty map";
        expect(!block.settings().applyStagedParameters().forwardParameters.empty()) << "successful set returns empty map";
        // should print out a warning -> TODO: replace with pmt error message on msgOut port
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
        Graph testGraph;
        auto &block = testGraph.emplaceBlock<TestBlock<float>>({ { "name", "TestName" }, { "scaling_factor", 2.f } });
        auto  s     = CtxSettings(block);
        auto  ctx0  = SettingsCtx(std::chrono::system_clock::now());
        expect(s.set({ { "name", "TestNameAlt" }, { "scaling_factor", 42.f } }, ctx0).empty()) << "successful set returns empty map";
        auto ctx1 = SettingsCtx(std::chrono::system_clock::now() + std::chrono::seconds(1));
        expect(s.set({ { "name", "TestNameNew" }, { "scaling_factor", 43.f } }, ctx1).empty()) << "successful set returns empty map";

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
        Graph      testGraph;
        auto      &block = testGraph.emplaceBlock<TestBlock<int>>({ { "scaling_factor", 42 } });
        auto       s     = CtxSettings(block, matchPred);
        const auto ctx0  = SettingsCtx(std::chrono::system_clock::now(), { { "BPCID", 1 }, { "SID", 1 }, { "BPID", 1 }, { "GID", 1 } });
        expect(s.set({ { "scaling_factor", 101 } }, ctx0).empty()) << "successful set returns empty map";
        const auto ctx1 = SettingsCtx(std::chrono::system_clock::now(), { { "BPCID", 1 }, { "SID", 1 }, { "BPID", 1 } });
        expect(s.set({ { "scaling_factor", 102 } }, ctx1).empty()) << "successful set returns empty map";
        const auto ctx2 = SettingsCtx(std::chrono::system_clock::now(), { { "BPCID", 1 }, { "SID", 1 } });
        expect(s.set({ { "scaling_factor", 103 } }, ctx2).empty()) << "successful set returns empty map";
        const auto ctx3 = SettingsCtx(std::chrono::system_clock::now(), { { "BPCID", 1 } });
        expect(s.set({ { "scaling_factor", 104 } }, ctx3).empty()) << "successful set returns empty map";

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
        Graph testGraph;
        auto &block = testGraph.emplaceBlock<TestBlock<float>>({ { "name", "TestName" }, { "scaling_factor", 2.f } });
        auto  s     = std::make_unique<CtxSettings<std::remove_reference<decltype(block)>::type>>(block, matchPred);
        block.setSettings(s);
        auto ctx0 = SettingsCtx(std::chrono::system_clock::now());
        expect(block.settings().set({ { "name", "TestNameAlt" }, { "scaling_factor", 42.f } }, ctx0).empty()) << "successful set returns empty map";
        expect(eq(std::get<float>(*block.settings().get("scaling_factor")), 42.f));
    };

#ifndef __EMSCRIPTEN__
    // TODO enable this when load_grc works in emscripten (not relying on plugins here)
    "Property auto-forwarding with GRC-loaded graph"_test = [&] {
        constexpr std::string_view grc = R"(
blocks:
  - name: source
    id: gr::setting_test::Source
    parameters:
      n_samples_max: 100
      sample_rate: 123456
  - name: test_block
    id: gr::setting_test::TestBlock
  - name: sink
    id: gr::setting_test::Sink
connections:
  - [source, 0, test_block, 0]
  - [test_block, 0, sink, 0]
)";
        BlockRegistry              registry;
        gr::registerBlock<Source, double>(registry);
        gr::registerBlock<TestBlock, double>(registry);
        gr::registerBlock<Sink, double>(registry);
        PluginLoader loader(registry, {});
        try {
            scheduler::Simple sched{ load_grc(loader, std::string(grc)) };
            expect(sched.runAndWait().has_value());
            sched.graph().forEachBlock([](auto &block) { expect(eq(std::get<float>(*block.settings().get("sample_rate")), 123456.f)) << fmt::format("sample_rate forwarded to {}", block.name()); });
        } catch (const std::string &e) {
            fmt::print(std::cerr, "GRC loading failed: {}\n", e);
            expect(false);
        }
    };
#endif
};

int
main() { /* tests are statically executed */
}
