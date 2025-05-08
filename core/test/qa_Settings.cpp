#include <string>

#include <boost/ut.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Buffer.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/Settings.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/testing/SettingsChangeRecorder.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

using namespace std::string_literals;

namespace gr::setting_test {

using gr::testing::SettingsChangeRecorder;

template<typename T>
struct Source : public Block<Source<T>> {
    PortOut<T> out;

    // settings
    gr::Size_t n_samples_max = 1024;
    float      sample_rate   = 1000.0f;

    GR_MAKE_REFLECTABLE(Source, out, n_samples_max, sample_rate);

    gr::Size_t _nSamplesProduced = 0;

    void settingsChanged(const property_map& /*oldSettings*/, property_map& newSettings, property_map& fwdSettings) {
        // optional init function that is called after construction and whenever settings change
        newSettings.insert_or_assign("n_samples_max", n_samples_max);
        fwdSettings.insert_or_assign("n_samples_max", n_samples_max);
    }

    [[nodiscard]] constexpr T processOne() noexcept {
        _nSamplesProduced++;
        if (_nSamplesProduced >= n_samples_max) {
            this->requestStop();
        }
        return T{};
    }
};

// optional shortening
template<typename T, gr::meta::fixed_string description = "", typename... Arguments>
using A = Annotated<T, description, Arguments...>;

template<typename T, bool Average = false>
struct Decimate : public Block<Decimate<T, Average>, SupportedTypes<float, double>, Resampling<>> {
    using Description = Doc<R""(
@brief reduces sample rate by given fraction controlled by input_chunk_size
)"">;
    PortIn<T>  in{};
    PortOut<T> out{};

    // settings
    A<float, "sample rate", Visible> sample_rate = 1.f;

    GR_MAKE_REFLECTABLE(Decimate, in, out, sample_rate);

    constexpr work::Status processBulk(std::span<const T>& input, std::span<T>& output) noexcept {
        assert(this->output_chunk_size == gr::Size_t(1) && "block implements only basic decimation");
        assert(this->input_chunk_size != gr::Size_t(0) && "input_chunk_size must be non-zero");

        auto outputIt = output.begin();
        if constexpr (Average) {
            for (std::size_t start = 0; start < input.size(); start += this->input_chunk_size) {
                constexpr auto chunk_begin = input.begin() + start;
                constexpr auto chunk_end   = chunk_begin + std::min(this->input_chunk_size, std::distance(chunk_begin, input.end()));
                *outputIt++                = std::reduce(chunk_begin, chunk_end, T(0)) / static_cast<T>(this->input_chunk_size);
            }
        } else {
            for (std::size_t i = 0; i < input.size(); i += this->input_chunk_size) {
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
    PortIn<T> in;

    // settings
    gr::Size_t n_samples_max = 0;
    float      sample_rate   = 1.0f;

    GR_MAKE_REFLECTABLE(Sink, in, n_samples_max, sample_rate);

    gr::Size_t _nSamplesConsumed = 0;

    [[nodiscard]] constexpr auto processOne(T) noexcept { _nSamplesConsumed++; }
};
} // namespace gr::setting_test

const boost::ut::suite SettingsTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::setting_test;
    using namespace std::string_view_literals;

    "basic node settings tag"_test = [] {
        Graph                testGraph;
        constexpr gr::Size_t n_samples = gr::util::round_up(1'000'000, 1024);
        // define basic Sink->SettingsChangeRecorder->Sink flow graph
        auto& src = testGraph.emplaceBlock<Source<float>>({{gr::tag::SAMPLE_RATE.shortKey(), 42.f}, {"n_samples_max", n_samples}});
        expect(eq(src.settings().defaultParameters().size(), 9UZ)); // 7 base + 2 derived
        expect(eq(src.settings().getNStoredParameters(), 1UZ));
        expect(eq(src.settings().getStored().value().size(), 9UZ));
        expect(eq(src.n_samples_max, n_samples)) << "check map constructor";
        expect(eq(src.sample_rate, 42.f)) << "check map constructor";
        expect(eq(src._nSamplesProduced, gr::Size_t(0))) << "default value";
        expect(eq(src.settings().getNAutoUpdateParameters(), 1UZ));
        expect(eq(src.settings().autoUpdateParameters().size(), 3UL)); // 3 base + 0 derived
        expect(eq(src.settings().autoForwardParameters().size(), gr::tag::kDefaultTags.size()));

        auto& block1 = testGraph.emplaceBlock<SettingsChangeRecorder<float>>({{"name", "SettingsChangeRecorder#1"}});
        auto& block2 = testGraph.emplaceBlock<SettingsChangeRecorder<float>>({{"name", "SettingsChangeRecorder#2"}});
        expect(eq(block1.settings().defaultParameters().size(), 13UZ)); // 7 base + 6 derived
        expect(eq(block1.settings().getNStoredParameters(), 1UZ));
        expect(eq(block1.settings().getStored().value().size(), 13UZ));
        expect(eq(block1.name, "SettingsChangeRecorder#1"s));
        expect(eq(block1.settings().getNAutoUpdateParameters(), 1UZ));
        expect(eq(block1.settings().autoUpdateParameters().size(), 8UL)); // 2 base + 6 derived
        expect(eq(block1.settings().autoForwardParameters().size(), gr::tag::kDefaultTags.size()));

        auto& sink = testGraph.emplaceBlock<Sink<float>>();
        expect(eq(sink.settings().defaultParameters().size(), 9UZ)); // 7 base + 2 derived
        expect(eq(sink.settings().getNStoredParameters(), 1UZ));
        expect(eq(sink.settings().getStored().value().size(), 9UZ));
        expect(eq(sink.settings().getNAutoUpdateParameters(), 1UZ));
        expect(eq(sink.settings().autoUpdateParameters().size(), 5UL)); // 3 base + 2 derived
        expect(eq(sink.settings().autoForwardParameters().size(), gr::tag::kDefaultTags.size()));

        // need to add 'n_samples_max' to forwarding list for the block to automatically forward it as the 'n_samples_max' tag is not part of the canonical 'gr::tag::kDefaultTags' list
        block1.settings().autoForwardParameters().emplace("n_samples_max");
        expect(eq(block1.settings().autoForwardParameters().size(), gr::tag::kDefaultTags.size() + 1UZ)); // + n_samples_max
        block2.settings().autoForwardParameters().emplace("n_samples_max");
        expect(eq(block2.settings().autoForwardParameters().size(), gr::tag::kDefaultTags.size() + 1UZ)); // + n_samples_max
        sink.settings().autoForwardParameters().emplace("n_samples_max");
        expect(eq(sink.settings().autoForwardParameters().size(), gr::tag::kDefaultTags.size() + 1UZ)); // + n_samples_max

        block1.context = "Test Context";
        expect(eq(block1.settings().activeParameters().size(), 13UL)); // 7 base + 6 derived
        expect(block1.settings().get(gr::tag::CONTEXT.shortKey()).has_value());
        expect(block1.settings().get({gr::tag::CONTEXT.shortKey()}).has_value());
        expect(not block1.settings().get({"test"}).has_value());
        expect(not eq(std::get<std::string>(block1.settings().get(gr::tag::CONTEXT.shortKey()).value()), "Test Context"s));
        block1.settings().updateActiveParameters();
        expect(eq(std::get<std::string>(block1.settings().get(gr::tag::CONTEXT.shortKey()).value()), "Test Context"s));

        std::vector<std::string>   keys1{"key1", "key2", "key3"};
        std::span<std::string>     keys2{keys1};
        std::array<std::string, 3> keys3{"key1", "key2", "key3"};
        expect(block1.settings().get(keys1).empty());
        expect(block1.settings().get(keys2).empty());
        expect(block1.settings().get(keys3).empty());
        expect(eq(block1.settings().get().size(), 13UL));

        // set non-existent setting
        expect(eq(block1.settings().getNStoredParameters(), 1UZ));
        auto ret1 = block1.settings().set({{"unknown", "random value"}});
        expect(eq(ret1.size(), 1U)) << "setting one unknown parameter";
        expect(eq(std::get<std::string>(static_cast<property_map>(block1.meta_information).at("unknown")), "random value"sv)) << "setting one unknown parameter";
        expect(eq(block1.settings().getNStoredParameters(), 1UZ));

        expect(not block1.settings().changed());
        auto ret2 = block1.settings().set({{gr::tag::CONTEXT.shortKey(), "alt context"}});
        expect(ret2.empty()) << "setting one known parameter";
        expect(block1.settings().stagedParameters().empty());          // set(...) does not change stagedParameters
        expect(not block1.settings().changed()) << "settings changed"; // set(...) does not change changed()
        std::ignore = block1.settings().activateContext();
        expect(eq(block1.settings().stagedParameters().size(), 2UZ)); // context, "name"
        expect(block1.settings().changed()) << "settings changed";
        auto applyResult = block1.settings().applyStagedParameters();
        expect(eq(applyResult.forwardParameters.size(), 1u)) << "initial forward declarations"; // context
        block1.settings().updateActiveParameters();

        // src -> block1 -> block2 -> sink
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).to<"in">(block1)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(block1).to<"in">(block2)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(block2).to<"in">(sink)));

        expect(!src.settings().autoUpdateParameters().contains(gr::tag::SAMPLE_RATE.shortKey())) << "manual setting disable auto-update";
        expect(eq(src.settings().getNStoredParameters(), 1UZ));
        expect(eq(src.settings().getNAutoUpdateParameters(), 1UZ));
        expect(src.settings().set({{gr::tag::SAMPLE_RATE.shortKey(), 49000.0f}}).empty()) << "successful set returns empty map";
        expect(eq(src.settings().getNStoredParameters(), 1UZ));     // old parameters are removed from stored
        expect(eq(src.settings().getNAutoUpdateParameters(), 1UZ)); // old parameters are removed from autoUpdate
        expect(eq(src.settings().stagedParameters().size(), 0UZ));
        expect(src.settings().activateContext() != std::nullopt);  // activateContext() fills staged parameters
        expect(eq(src.settings().stagedParameters().size(), 2UZ)); // "n_samples_max", sample_rate

        auto                  thread_pool = std::make_shared<gr::thread_pool::BasicThreadPool>("custom pool", gr::thread_pool::CPU_BOUND, 2, 2); // use custom pool to limit number of threads for emscripten
        gr::scheduler::Simple sched{std::move(testGraph), thread_pool};
        expect(sched.runAndWait().has_value());

        expect(eq(src.settings().getNStoredParameters(), 1UZ));
        expect(eq(src.settings().stagedParameters().size(), 0UZ)); // staged is cleared after applyStagedSettings is called
        expect(eq(src._nSamplesProduced, n_samples)) << "src did not produce enough output samples";
        expect(eq(static_cast<gr::Size_t>(block1._nSamplesConsumed), n_samples)) << "block1 did not consume enough input samples";
        expect(eq(static_cast<gr::Size_t>(block2._nSamplesConsumed), n_samples)) << "block2 did not consume enough input samples";
        expect(eq(sink._nSamplesConsumed, n_samples)) << "sink did not consume enough input samples";

        for (auto& fwd : src.settings().autoUpdateParameters()) {
            fmt::println("## src auto {}", fwd);
        }
        for (auto& fwd : block1.settings().autoUpdateParameters()) {
            fmt::println("## block1 auto {}", fwd);
        }
        for (auto& fwd : block2.settings().autoUpdateParameters()) {
            fmt::println("## block2 auto {}", fwd);
        }
        for (auto& fwd : sink.settings().autoUpdateParameters()) {
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
        expect(!src.settings().autoUpdateParameters().contains(gr::tag::SAMPLE_RATE.shortKey())) << "src should not retain auto-update flag (was manually set)";
        expect(block1.settings().autoUpdateParameters().contains(gr::tag::SAMPLE_RATE.shortKey())) << "block1 retained auto-update flag";
        expect(block1.settings().autoUpdateParameters().contains("n_samples_max")) << "block2 retained auto-update flag";
        expect(block1.settings().autoForwardParameters().contains("n_samples_max")) << "block2 retained auto-forward flag";
        expect(block2.settings().autoUpdateParameters().contains(gr::tag::SAMPLE_RATE.shortKey())) << "block2 retained auto-update flag";
        expect(block2.settings().autoUpdateParameters().contains("n_samples_max")) << "block2 retained auto-update flag";
        expect(block2.settings().autoForwardParameters().contains("n_samples_max")) << "block2 retained auto-forward flag";
        expect(sink.settings().autoUpdateParameters().contains(gr::tag::SAMPLE_RATE.shortKey())) << "sink retained auto-update flag";
        expect(sink.settings().autoUpdateParameters().contains("n_samples_max")) << "sink retained auto-update flag";
    };

    "constructor"_test = [] {
        "empty"_test = [] {
            auto block = SettingsChangeRecorder<float>(); // Block::init() is not called
            expect(eq(block.settings().getNStoredParameters(), 0UZ));
            expect(eq(block.settings().stagedParameters().size(), 0UZ));
            expect(eq(block.settings().getNAutoUpdateParameters(), 0UZ));
            expect(eq(block.settings().autoUpdateParameters().size(), 0UL));
            expect(eq(block.settings().autoForwardParameters().size(), gr::tag::kDefaultTags.size()));
            block.init(block.progress, block.ioThreadPool); // N.B. self-assign existing progress and thread-pool (just for unit-tests)
            expect(eq(block.settings().getNStoredParameters(), 1UZ));
            expect(eq(block.settings().getNAutoUpdateParameters(), 1UZ));
            expect(eq(block.settings().autoUpdateParameters().size(), 9UL));
            expect(block.settings().activateContext() != std::nullopt);
            expect(eq(block.settings().stagedParameters().size(), 0UZ)); // same activeCtx, no changes
            expect(eq(block.settings().get().size(), 13UL));             // all active settings
            expect(eq(std::get<float>(*block.settings().get("scaling_factor")), 1.f));
        };

        "with init parameter"_test = [] {
            auto block = SettingsChangeRecorder<float>({{"scaling_factor", 2.f}}); // Block::init() is not called
            expect(eq(block.settings().getNStoredParameters(), 0UZ));
            expect(eq(block.settings().stagedParameters().size(), 0UZ));
            expect(eq(block.settings().getNAutoUpdateParameters(), 0UZ));
            expect(eq(block.settings().autoUpdateParameters().size(), 0UL));
            expect(eq(block.settings().autoForwardParameters().size(), gr::tag::kDefaultTags.size()));
            block.init(block.progress, block.ioThreadPool); // N.B. self-assign existing progress and thread-pool (just for unit-tests)
            expect(eq(block.settings().getNStoredParameters(), 1UZ));
            expect(eq(block.settings().getNAutoUpdateParameters(), 1UZ));
            expect(eq(block.settings().autoUpdateParameters().size(), 8UL)); // no "scaling_factor"
            expect(eq(block.settings().autoUpdateParameters().contains("scaling_factor"), false));
            expect(block.settings().activateContext() != std::nullopt);
            expect(eq(block.settings().stagedParameters().size(), 0UZ)); // same activeCtx, no changes
            block.settings().updateActiveParameters();
            expect(eq(block.settings().get().size(), 13UL));
            expect(eq(block.scaling_factor, 2.f));
            expect(eq(std::get<float>(*block.settings().get("scaling_factor")), 2.f));
        };

        "empty via graph"_test = [] {
            Graph testGraph;
            auto& block = testGraph.emplaceBlock<SettingsChangeRecorder<float>>(); // Block::init() is called
            expect(eq(block.settings().getNStoredParameters(), 1UZ));              // store default parameters
            expect(eq(block.settings().getNAutoUpdateParameters(), 1UZ));
            expect(eq(block.settings().stagedParameters().size(), 0UZ));
            expect(eq(block.settings().autoUpdateParameters().size(), 9UL)); // all isWritable settings (enable reflections)
            expect(eq(block.settings().autoForwardParameters().size(), gr::tag::kDefaultTags.size()));
            expect(eq(block.settings().get().size(), 13UL));
            expect(eq(block.scaling_factor, 1.f));
            expect(eq(std::get<float>(*block.settings().get("scaling_factor")), 1.f));
        };

        "with init parameter via graph"_test = [] {
            Graph testGraph;
            auto& block = testGraph.emplaceBlock<SettingsChangeRecorder<float>>({{"scaling_factor", 2.f}});
            expect(eq(block.settings().getNStoredParameters(), 1UZ)); // store default parameters
            expect(eq(block.settings().getNAutoUpdateParameters(), 1UZ));
            expect(eq(block.settings().stagedParameters().size(), 0UZ));
            expect(eq(block.settings().autoUpdateParameters().size(), 8UL)); // "scaling_factor" removed from auto updates
            expect(eq(block.settings().autoForwardParameters().size(), gr::tag::kDefaultTags.size()));
            expect(eq(block.settings().get().size(), 13UL));
            expect(eq(block.scaling_factor, 2.f));
            expect(eq(std::get<float>(*block.settings().get("scaling_factor")), 2.f));
        };
    };

    "vector-type support"_test = [] {
        Graph testGraph;
        auto& block = testGraph.emplaceBlock<SettingsChangeRecorder<float>>();
        expect(eq(block.settings().getNStoredParameters(), 1UZ)); // store default parameters
        expect(eq(block.settings().stagedParameters().size(), 0UZ));
        block.settings().updateActiveParameters();
        expect(eq(block.settings().get().size(), 13UL));
        block._debug   = true;
        const auto val = block.settings().set({{"vector_setting", std::vector{42.f, 2.f, 3.f}}, {"string_vector_setting", std::vector<std::string>{"A", "B", "C"}}});
        expect(val.empty()) << "unable to stage settings";
        expect(eq(block.settings().getNStoredParameters(), 1UZ)); // new parameters added, but old parameters removed
        expect(eq(block.settings().stagedParameters().size(), 0UZ));
        expect(block.settings().activateContext() != std::nullopt);
        expect(eq(block.settings().stagedParameters().size(), 2UZ));                        // "vector_setting", "string_vector_setting"
        expect(eq(block.settings().applyStagedParameters().forwardParameters.size(), 0UZ)); // no autoForwardParameters

        expect(eq(block.settings().stagedParameters().size(), 0UZ)); // clear _staged after applyStagedParameters() call
        expect(eq(block.vector_setting, std::vector{42.f, 2.f, 3.f}));
        expect(eq(block.string_vector_setting.value, std::vector<std::string>{"A", "B", "C"}));
        expect(eq(block._updateCount, 1)) << fmt::format("actual update count: {}\n", block._updateCount);
        expect(eq(std::get<std::vector<float>>(*block.settings().get("vector_setting")), std::vector{42.f, 2.f, 3.f}));
        expect(eq(std::get<std::vector<std::string>>(*block.settings().get("string_vector_setting")), std::vector<std::string>{"A", "B", "C"}));
    };

    "unique ID"_test = [] {
        Graph       testGraph;
        const auto& block1 = testGraph.emplaceBlock<SettingsChangeRecorder<float>>();
        const auto& block2 = testGraph.emplaceBlock<SettingsChangeRecorder<float>>();
        expect(not eq(block1.unique_id, block2.unique_id)) << "unique per-type block id (size_t)";
        expect(not eq(block1.unique_name, block2.unique_name)) << "unique per-type block id (string)";

        auto merged1 = merge<"out", "in">(SettingsChangeRecorder<float>(), SettingsChangeRecorder<float>());
        auto merged2 = merge<"out", "in">(SettingsChangeRecorder<float>(), SettingsChangeRecorder<float>());
        expect(not eq(merged1.unique_id, merged2.unique_id)) << "unique per-type block id (size_t) ";
        expect(not eq(merged1.unique_name, merged2.unique_name)) << "unique per-type block id (string) ";
    };

    "run-time type-erased node setter/getter"_test = [] {
        auto progress     = std::make_shared<gr::Sequence>();
        auto ioThreadPool = std::make_shared<gr::thread_pool::BasicThreadPool>("test_pool", gr::thread_pool::TaskType::IO_BOUND, 2UZ, std::numeric_limits<uint32_t>::max());
        //
        auto wrapped1 = BlockWrapper<SettingsChangeRecorder<float>>();
        wrapped1.init(progress, ioThreadPool);
        expect(eq(wrapped1.settings().getNStoredParameters(), 1UZ));
        wrapped1.setName("test_name");
        expect(eq(wrapped1.name(), "test_name"sv)) << "BlockModel wrapper name";
        expect(not wrapped1.uniqueName().empty()) << "unique name";
        expect(wrapped1.settings().set({{gr::tag::CONTEXT.shortKey(), "a string"}}).empty()) << "successful set returns empty map";
        expect(eq(wrapped1.settings().getNStoredParameters(), 1UZ)); // new parameters added, but old parameters removed
        (wrapped1.metaInformation())["key"] = "value";
        expect(eq(std::get<std::string>(wrapped1.metaInformation().at("key")), "value"sv)) << "BlockModel meta-information";

        // via constructor
        auto wrapped2 = BlockWrapper<SettingsChangeRecorder<float>>({{"name", "test_name"}});
        wrapped2.init(progress, ioThreadPool);
        expect(eq(wrapped2.settings().getNStoredParameters(), 1UZ));
        expect(wrapped2.settings().set({{gr::tag::CONTEXT.shortKey(), "a string"}}).empty()) << "successful set returns empty map";
        expect(eq(wrapped2.settings().getNStoredParameters(), 1UZ)); // new parameters added, but old parameters removed
        expect(eq(wrapped2.name(), "test_name"sv)) << "BlockModel wrapper name";
        expect(not wrapped2.uniqueName().empty()) << "unique name";
        expect(wrapped2.settings().set({{gr::tag::CONTEXT.shortKey(), "a string"}}).empty()) << "successful set returns empty map";
        expect(eq(wrapped2.settings().getNStoredParameters(), 1UZ)); // new parameters added, but old parameters removed
        (wrapped2.metaInformation())["key"] = "value";
        expect(eq(std::get<std::string>(wrapped2.metaInformation().at("key")), "value"sv)) << "BlockModel meta-information";
    };

    "basic decimation test"_test = []() {
        Graph                testGraph;
        constexpr gr::Size_t n_samples = gr::util::round_up(1'000'000, 1024);
        auto&                src       = testGraph.emplaceBlock<Source<float>>({{"n_samples_max", n_samples}, {gr::tag::SAMPLE_RATE.shortKey(), 1000.0f}});
        auto&                block1    = testGraph.emplaceBlock<Decimate<float>>({{"name", "Decimate1"}, {"input_chunk_size", gr::Size_t(2)}});
        auto&                block2    = testGraph.emplaceBlock<Decimate<float>>({{"name", "Decimate2"}, {"input_chunk_size", gr::Size_t(5)}});
        auto&                sink      = testGraph.emplaceBlock<Sink<float>>();

        // check input_chunk_size
        expect(eq(block1.input_chunk_size, std::size_t(2)));
        expect(eq(block2.input_chunk_size, std::size_t(5)));

        // src -> block1 -> block2 -> sink
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).to<"in">(block1)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(block1).to<"in">(block2)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(block2).to<"in">(sink)));

        gr::scheduler::Simple sched{std::move(testGraph)};
        expect(sched.runAndWait().has_value());

        expect(eq(src._nSamplesProduced, n_samples)) << "did not produce enough output samples";
        expect(eq(sink._nSamplesConsumed, n_samples / (2 * 5))) << "did not consume enough input samples";

        expect(eq(src.sample_rate, 1000.0f)) << "src matching sample_rate";
        expect(eq(block1.sample_rate, 1000.0f)) << "block1 matching sample_rate";
        expect(eq(block2.sample_rate, 500.0f)) << "block2 matching sample_rate";
        expect(eq(sink.sample_rate, 100.0f)) << "sink matching src sample_rate";
    };

    "basic store/reset settings"_test = []() {
        Graph testGraph;
        auto& block = testGraph.emplaceBlock<SettingsChangeRecorder<float>>({{"name", "TestName"}, {"scaling_factor", 2.f}});
        expect(block.name == "TestName");
        expect(eq(block.scaling_factor, 2.f));
        expect(eq(block.settings().autoForwardParameters().size(), gr::tag::kDefaultTags.size()));
        expect(eq(block.settings().getNStoredParameters(), 1UZ));
        expect(block.settings().set({{"name", "TestNameAlt"}, {"scaling_factor", 42.f}}).empty()) << "successful set returns empty map\n";
        expect(eq(block.settings().getNStoredParameters(), 1UZ)); // new parameters added, but old parameters removed
        expect(block.settings().activateContext() != std::nullopt);
        expect(eq(block.settings().stagedParameters().size(), 2UZ));                        // "name", "scaling_factor"
        expect(eq(block.settings().applyStagedParameters().forwardParameters.size(), 0UZ)); // no autoForwardParameters
        expect(eq(block.name, "TestNameAlt"s));
        expect(eq(block.scaling_factor, 42.f));
        expect(not block._resetCalled);
        expect(eq(block.settings().defaultParameters().size(), 13UZ));
        block.settings().resetDefaults();
        expect(eq(block.settings().getNStoredParameters(), 1UZ));
        expect(block._resetCalled);
        block._resetCalled = false;
        expect(eq(block.name, "TestName"s));
        expect(eq(block.scaling_factor, 2.f));
        expect(block.settings().set({{"name", "TestNameAlt"}, {"scaling_factor", 42.f}}).empty()) << "successful set returns empty map\n";
        expect(eq(block.settings().getNStoredParameters(), 1UZ)); // new parameters added, but old parameters removed
        expect(block.settings().activateContext() != std::nullopt);
        expect(eq(block.settings().stagedParameters().size(), 2UZ));                        // "name", "scaling_factor"
        expect(eq(block.settings().applyStagedParameters().forwardParameters.size(), 0UZ)); // no autoForwardParameters
        expect(eq(block.name, "TestNameAlt"s));
        expect(eq(block.scaling_factor, 42.f));

        // test storeDefaults()
        const auto defaultParOld = block.settings().defaultParameters();
        expect(eq(defaultParOld.size(), 13UZ));
        expect(eq(std::get<std::string>(defaultParOld.at("name")), "TestName"s));
        expect(eq(std::get<float>(defaultParOld.at("scaling_factor")), 2.f));
        block.settings().storeDefaults();
        const auto defaultParNew = block.settings().defaultParameters();
        expect(eq(defaultParNew.size(), 13UZ));
        expect(eq(std::get<std::string>(defaultParNew.at("name")), "TestNameAlt"s));
        expect(eq(std::get<float>(defaultParNew.at("scaling_factor")), 42.f));
        expect(block.settings().set({{"name", "TestNameAlt2"}, {"scaling_factor", 43.f}}).empty()) << "successful set returns empty map\n";
        expect(eq(block.settings().getNStoredParameters(), 1UZ)); // new parameters added, but old parameters removed
        expect(block.settings().activateContext() != std::nullopt);
        expect(eq(block.settings().stagedParameters().size(), 2UZ));                        // "name", "scaling_factor"
        expect(eq(block.settings().applyStagedParameters().forwardParameters.size(), 0UZ)); // no autoForwardParameters
        expect(eq(block.name, "TestNameAlt2"s));
        expect(eq(block.scaling_factor, 43.f));
        block.settings().resetDefaults();
        expect(eq(block.settings().getNStoredParameters(), 1UZ));
        expect(block._resetCalled);
        expect(eq(block.name, "TestNameAlt"s));
        expect(eq(block.scaling_factor, 42.f));
    };
};

const boost::ut::suite AnnotationTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::setting_test;
    using namespace std::literals;

    "basic node annotations"_test = [] {
        Graph                          testGraph;
        SettingsChangeRecorder<float>& block = testGraph.emplaceBlock<SettingsChangeRecorder<float>>();
        expect(gr::blockDescription<SettingsChangeRecorder<float>>().find(std::string_view(SettingsChangeRecorder<float>::Description::value)) != std::string_view::npos);
        expect(eq(std::get<std::string>(block.meta_information.value.at("description")), std::string(SettingsChangeRecorder<float>::Description::value))) << "type-erased block description";
        expect(eq(std::get<std::string>(block.meta_information.value.at("scaling_factor::description")), "scaling factor"sv));
        expect(eq(std::get<std::string>(block.meta_information.value.at("scaling_factor::documentation")), "y = a * x"sv));
        expect(eq(std::get<std::string>(block.meta_information.value.at("scaling_factor::unit")), "As"sv));
        expect(std::get<bool>(block.meta_information.value.at("scaling_factor::visible"))) << "visible being true";
        expect(block.scaling_factor.visible());
        expect(eq(block.scaling_factor.description(), "scaling factor"sv));
        expect(eq(block.scaling_factor.unit(), "As"sv));
        expect(eq(block.context.unit(), ""sv));
        expect(block.context.visible());
        expect(!block.isBlocking());

        block.scaling_factor = 42.f; // test wrapper assignment operator
        expect(block.scaling_factor == 42.f) << "the answer to everything failed -- equal operator";
        expect(eq(block.scaling_factor.value, 42.f)) << "the answer to everything failed -- by value";
        expect(eq(block.scaling_factor, 42.f)) << "the answer to everything failed -- direct";

        // check validator
        expect(block.sample_rate.validate_and_set(1.f));
        expect(not block.sample_rate.validate_and_set(-1.f));

        constexpr auto isPowerOfTwo = [](const int& val) { return val > 0 && (val & (val - 1)) == 0; };

        Annotated<int, "power of two", Limits<0, 0, isPowerOfTwo>> needPowerOfTwo = 2;
        expect(isPowerOfTwo(4));
        expect(!isPowerOfTwo(5));
        expect(needPowerOfTwo.validate_and_set(4));
        expect(not needPowerOfTwo.validate_and_set(5));
        expect(eq(needPowerOfTwo.value, 4));
        Annotated<int, "power of two", Limits<0, 0, [](const int& val) { return (val > 0) && (val & (val - 1)) == 0; }>> needPowerOfTwoAlt = 2;
        expect(needPowerOfTwoAlt.validate_and_set(4));
        expect(not needPowerOfTwoAlt.validate_and_set(5));
        expect(block.settings().set({{gr::tag::SAMPLE_RATE.shortKey(), -1.0f}}).empty()) << "successful set returns empty map";
        expect(eq(block.settings().getNStoredParameters(), 1UZ)); // new parameters added, but old parameters removed
        expect(eq(std::get<float>(block.settings().getStored(gr::tag::SAMPLE_RATE.shortKey()).value()), -1.f));
        expect(block.settings().activateContext() != std::nullopt);
        expect(eq(block.settings().stagedParameters().size(), 1UZ));                        // sample_rate
        expect(eq(block.settings().applyStagedParameters().forwardParameters.size(), 1UZ)); // sample_rate
        // should print out a warning -> TODO: replace with pmt error message on msgOut port
    };

    "annotated implicit conversions"_test = [] {
        Annotated<int> annotatedInt = 10;
        int            value        = annotatedInt; // implicit conversion
        expect(value == 10);

        Annotated<std::string> annotatedString = "hello";
        std::string            str             = annotatedString; // implicit conversion
        expect(str == "hello");
    };

    "annotated forwarding member functions"_test = [] {
        Annotated<std::string> annotatedString = "hello";
        expect(annotatedString->size() == 5);    // operator-> forwarding
        expect(annotatedString[1UZ] == 'e');     // operator[]
        expect(annotatedString->at(1UZ) == 'e'); // operator->*

        annotatedString[1UZ] = 'a';
        expect(annotatedString == "hallo"); // check modified string
    };

    "const correctness"_test = [] {
        const Annotated<std::string> annotatedString = "hello";
        expect(annotatedString->size() == 5);    // const operator->
        expect(annotatedString[1UZ] == 'e');     // const operator[]
        expect(annotatedString->at(1UZ) == 'e'); // const operator->*

        // Ensure the original string is not modified
        expect(annotatedString == "hello");
    };

    "validator edge cases"_test = [] {
        constexpr auto isNonNegative = [](const int& val) { return val >= 0; };

        Annotated<int, "non-negative", Limits<0, 0, isNonNegative>> nonNegative = 0;

        expect(nonNegative.validate_and_set(0));      // Edge case: exactly 0
        expect(nonNegative.validate_and_set(100));    // Positive case
        expect(not nonNegative.validate_and_set(-1)); // Negative case

        expect(nonNegative == 100); // Ensure value is correctly set to the last valid value
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
        a.time = settings::convertTimePointToUint64Ns(std::chrono::system_clock::now());
        b.time = settings::convertTimePointToUint64Ns(std::chrono::system_clock::now() + std::chrono::seconds(1));
        // chronologically sorted
        expect(a < b);
    };
};

const boost::ut::suite CtxSettingsTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::setting_test;

    "CtxSettings Time"_test = [] {
        Graph testGraph;
        auto& block    = testGraph.emplaceBlock<SettingsChangeRecorder<float>>({{"name", "TestName0"}, {"scaling_factor", 0.f}});
        auto  settings = CtxSettings(block);
        expect(block.settings().applyStagedParameters().forwardParameters.empty());
        block.settings().storeDefaults();
        const auto timeNow = std::chrono::system_clock::now();
        // Store t = 0, t = 2, t = 4
        // Test t = -1, t = 0, t = 1, t = 2, t = 3, t = 4, t = 5, t = nullopt
        auto ctx0 = SettingsCtx(settings::convertTimePointToUint64Ns(timeNow));
        expect(settings.set({{"name", "TestName10"}, {"scaling_factor", 10.f}}, ctx0).empty()) << "successful set returns empty map";
        auto ctx2 = SettingsCtx(settings::convertTimePointToUint64Ns(timeNow + std::chrono::seconds(2)));
        expect(settings.set({{"name", "TestName12"}, {"scaling_factor", 12.f}}, ctx2).empty()) << "successful set returns empty map";
        auto ctx4 = SettingsCtx(settings::convertTimePointToUint64Ns(timeNow + std::chrono::seconds(4)));
        expect(settings.set({{"name", "TestName14"}, {"scaling_factor", 14.f}}, ctx4).empty()) << "successful set returns empty map";
        expect(eq(settings.getNStoredParameters(), 3UZ));
        auto ctxM1   = SettingsCtx(settings::convertTimePointToUint64Ns(timeNow - std::chrono::seconds(1)));
        auto ctx1    = SettingsCtx(settings::convertTimePointToUint64Ns(timeNow + std::chrono::seconds(1)));
        auto ctx3    = SettingsCtx(settings::convertTimePointToUint64Ns(timeNow + std::chrono::seconds(3)));
        auto ctx5    = SettingsCtx(settings::convertTimePointToUint64Ns(timeNow + std::chrono::seconds(5)));
        auto ctxNull = SettingsCtx();
        // one key
        expect(eq(std::get<float>(settings.getStored("scaling_factor").value()), 10.f));          // return as ctx.time = std::chrono::system_clock::now()
        expect(settings.getStored("scaling_factor", ctxM1) == std::nullopt);                      // return std::nullopt, all settings ctx times are in the future
        expect(eq(std::get<float>(settings.getStored("scaling_factor", ctx0).value()), 10.f));    // return exact
        expect(eq(std::get<float>(settings.getStored("scaling_factor", ctx1).value()), 10.f));    // return previous
        expect(eq(std::get<float>(settings.getStored("scaling_factor", ctx2).value()), 12.f));    // return exact
        expect(eq(std::get<float>(settings.getStored("scaling_factor", ctx3).value()), 12.f));    // return previous
        expect(eq(std::get<float>(settings.getStored("scaling_factor", ctx4).value()), 14.f));    // return exact
        expect(eq(std::get<float>(settings.getStored("scaling_factor", ctx5).value()), 14.f));    // return latest
        expect(eq(std::get<float>(settings.getStored("scaling_factor", ctxNull).value()), 10.f)); // return as ctx.time = std::chrono::system_clock::now()
        // several keys
        std::vector<std::string> parameterKeys = {"scaling_factor", "name"};

        const property_map params = settings.getStored(parameterKeys).value(); // return as ctx.time = std::chrono::system_clock::now()
        expect(eq(std::get<float>(params.at("scaling_factor")), 10.f));
        expect(eq(std::get<std::string>(params.at("name")), "TestName10"s));
        const property_map paramsAll = settings.getStored().value(); // test API without parameters, should return all keys
        expect(eq(paramsAll.size(), 2UZ));

        expect(settings.getStored(parameterKeys, ctxM1) == std::nullopt); // return std::nullopt, all settings ctx times are in the future

        const property_map params0 = settings.getStored(parameterKeys, ctx0).value(); // return exact
        expect(eq(std::get<float>(params0.at("scaling_factor")), 10.f));
        expect(eq(std::get<std::string>(params0.at("name")), "TestName10"s));

        const property_map params1 = settings.getStored(parameterKeys, ctx1).value(); // return previous
        expect(eq(std::get<float>(params1.at("scaling_factor")), 10.f));
        expect(eq(std::get<std::string>(params1.at("name")), "TestName10"s));

        const property_map params2 = settings.getStored(parameterKeys, ctx2).value(); // return exact
        expect(eq(std::get<float>(params2.at("scaling_factor")), 12.f));
        expect(eq(std::get<std::string>(params2.at("name")), "TestName12"s));

        const property_map params3 = settings.getStored(parameterKeys, ctx3).value(); // return previous
        expect(eq(std::get<float>(params3.at("scaling_factor")), 12.f));
        expect(eq(std::get<std::string>(params3.at("name")), "TestName12"s));

        const property_map params4 = settings.getStored(parameterKeys, ctx4).value(); // return exact
        expect(eq(std::get<float>(params4.at("scaling_factor")), 14.f));
        expect(eq(std::get<std::string>(params4.at("name")), "TestName14"s));

        const property_map params5 = settings.getStored(parameterKeys, ctx5).value(); // return latest
        expect(eq(std::get<float>(params5.at("scaling_factor")), 14.f));
        expect(eq(std::get<std::string>(params5.at("name")), "TestName14"s));

        const property_map paramsNull = settings.getStored(parameterKeys, ctxNull).value(); // return as ctx.time = std::chrono::system_clock::now()
        expect(eq(std::get<float>(paramsNull.at("scaling_factor")), 10.f));
        expect(eq(std::get<std::string>(paramsNull.at("name")), "TestName10"s));
    };
#ifdef __EMSCRIPTEN__
    "CtxSettings Resolve Duplicate Timestamp"_test = [] {
        Graph               testGraph;
        auto&               block     = testGraph.emplaceBlock<SettingsChangeRecorder<float>>({{"name", "TestName0"}, {"scaling_factor", 0.f}});
        auto                settings  = CtxSettings(block);
        const std::uint64_t timeNowNs = settings::convertTimePointToUint64Ns(std::chrono::system_clock::now());
        auto                ctx0      = SettingsCtx(timeNowNs);

        expect(settings.set({{"name", "TestName10"}, {"scaling_factor", 10.f}}, ctx0).empty());
        expect(eq(settings.getNStoredParameters(), 1UZ));
        expect(settings.set({{"name", "TestName11"}, {"scaling_factor", 11.f}}, ctx0).empty());
        expect(eq(settings.getNStoredParameters(), 1UZ)); // remove old parameters
        expect(settings.set({{"name", "TestName12"}, {"scaling_factor", 12.f}}, ctx0).empty());
        expect(eq(settings.getNStoredParameters(), 1UZ));

        const auto& stored = settings.getStoredAll();
        expect(stored.contains("")); // empty string is default context
        const auto& vec = stored.at("");
        expect(eq(vec.size(), 1UZ));
        expect(eq(vec[0].first.time, timeNowNs + 2));

        expect(eq(std::get<float>(settings.getStored("scaling_factor").value()), 12.f));
    };
#endif
    "CtxSettings Expired Parameters"_test = [] {
        Graph testGraph;
        auto& block    = testGraph.emplaceBlock<SettingsChangeRecorder<float>>({{"scaling_factor", 0.f}});
        auto  settings = CtxSettings(block);
        expect(block.settings().applyStagedParameters().forwardParameters.empty());
        block.settings().storeDefaults();
        const auto timeNow = std::chrono::system_clock::now();

        expect(eq(settings.expiry_time, std::numeric_limits<std::uint64_t>::max()));

        // now - 20
        auto ctxM20 = SettingsCtx(settings::convertTimePointToUint64Ns(timeNow - std::chrono::seconds(20)));
        expect(settings.set({{"scaling_factor", -20.f}}, ctxM20).empty());
        expect(eq(settings.getNStoredParameters(), 1UZ));
        expect(eq(std::get<float>(settings.getStored("scaling_factor").value()), -20.f));

        // now - 10
        auto ctxM10 = SettingsCtx(settings::convertTimePointToUint64Ns(timeNow - std::chrono::seconds(10)));
        expect(settings.set({{"scaling_factor", -10.f}}, ctxM10).empty());
        expect(eq(settings.getNStoredParameters(), 1UZ)); // ctxM20 should be outdated and removed
        expect(eq(std::get<float>(settings.getStored("scaling_factor").value()), -10.f));

        // now + 10
        auto ctx10 = SettingsCtx(settings::convertTimePointToUint64Ns(timeNow + std::chrono::seconds(10)));
        expect(settings.set({{"scaling_factor", 10.f}}, ctx10).empty());
        expect(eq(settings.getNStoredParameters(), 2UZ)); // ctxM10 and ctx10
        expect(eq(std::get<float>(settings.getStored("scaling_factor").value()), -10.f));

        // now - 5
        auto ctxM5 = SettingsCtx(settings::convertTimePointToUint64Ns(timeNow - std::chrono::seconds(5)));
        expect(settings.set({{"scaling_factor", -5.f}}, ctxM5).empty());
        expect(eq(settings.getNStoredParameters(), 2UZ)); // ctxM10 should be outdated and removed
        expect(eq(std::get<float>(settings.getStored("scaling_factor").value()), -5.f));

        settings.expiry_time = 2'000'000'000; // expiry_time is in ns

        // now + 5
        auto ctx5 = SettingsCtx(settings::convertTimePointToUint64Ns(timeNow + std::chrono::seconds(5)));
        expect(settings.set({{"scaling_factor", 5.f}}, ctx5).empty());
        expect(eq(settings.getNStoredParameters(), 2UZ));             // ctxM5 is expired and should be removed
        expect(settings.getStored("scaling_factor") == std::nullopt); // n parameters are available, ctx5 and ctx10 are in the future

        // now - 3
        auto ctxM3 = SettingsCtx(settings::convertTimePointToUint64Ns(timeNow - std::chrono::seconds(3)));
        expect(settings.set({{"scaling_factor", -3.f}}, ctxM3).empty());
        expect(eq(settings.getNStoredParameters(), 2UZ));             // ctxM3 is expired and should be immediately removed
        expect(settings.getStored("scaling_factor") == std::nullopt); // n parameters are available, ctx5 and ctx10 are in the future

        // now - 1
        auto ctxM1 = SettingsCtx(settings::convertTimePointToUint64Ns(timeNow - std::chrono::seconds(1)));
        expect(settings.set({{"scaling_factor", -1.f}}, ctxM1).empty());
        expect(eq(settings.getNStoredParameters(), 3UZ)); // ctxM1 is not expired and should be stored
        expect(eq(std::get<float>(settings.getStored("scaling_factor").value()), -1.f));
    };

    auto matchPred = [](const auto& table, const auto& search, const auto attempt) -> std::optional<bool> {
        if (std::holds_alternative<std::string>(table) && std::holds_alternative<std::string>(search)) {
            const auto tableString  = std::get<std::string>(table);
            const auto searchString = std::get<std::string>(search);

            if (!searchString.starts_with("FAIR.SELECTOR.")) {
                return std::nullopt;
            }

            if (attempt >= searchString.length()) {
                return std::nullopt;
            }
            auto [it1, it2] = std::ranges::mismatch(searchString, tableString);
            if (std::distance(searchString.begin(), it1) == static_cast<std::ptrdiff_t>(searchString.length() - attempt) && std::distance(searchString.begin(), it1) == static_cast<std::ptrdiff_t>(tableString.length())) {
                return true;
            }
        }
        return false;
    };

    "CtxSettings Matching"_test = [&] {
        Graph      testGraph;
        auto&      block    = testGraph.emplaceBlock<SettingsChangeRecorder<int>>({{"scaling_factor", 42}});
        auto       settings = CtxSettings(block, matchPred);
        const auto timeNow  = std::chrono::system_clock::now();
        const auto ctx0     = SettingsCtx(settings::convertTimePointToUint64Ns(timeNow), "FAIR.SELECTOR.C=1:S=1:P=1");
        expect(settings.set({{"scaling_factor", 101}}, ctx0).empty()) << "successful set returns empty map";
        const auto ctx1 = SettingsCtx(settings::convertTimePointToUint64Ns(timeNow), "FAIR.SELECTOR.C=1:S=1");
        expect(settings.set({{"scaling_factor", 102}}, ctx1).empty()) << "successful set returns empty map";
        const auto ctx2 = SettingsCtx(settings::convertTimePointToUint64Ns(timeNow), "FAIR.SELECTOR.C=1");
        expect(settings.set({{"scaling_factor", 103}}, ctx2).empty()) << "successful set returns empty map";

        // exact matches
        expect(eq(std::get<int>(settings.getStored("scaling_factor", ctx0).value()), 101));
        expect(eq(std::get<int>(settings.getStored("scaling_factor", ctx1).value()), 102));
        expect(eq(std::get<int>(settings.getStored("scaling_factor", ctx2).value()), 103));

        // matching by using the custom predicate (no exact matching possible anymore)
        const auto ctx3 = SettingsCtx(settings::convertTimePointToUint64Ns(timeNow), "FAIR.SELECTOR.C=1:S=1:P=2");
        expect(eq(std::get<int>(settings.getStored("scaling_factor", ctx3).value()), 102)); // no setting for 'P=2' -> fall back to "FAIR.SELECTOR.C=1:S=1"
        const auto ctx4 = SettingsCtx(settings::convertTimePointToUint64Ns(timeNow), "FAIR.SELECTOR.C=1:S=2:P=2");
        expect(eq(std::get<int>(settings.getStored("scaling_factor", ctx4).value()), 103)); // no setting for 'S=2' and 'P=2' -> fall back to "FAIR.SELECTOR.C=1"

        // doesn't exist
        auto ctx5 = SettingsCtx(settings::convertTimePointToUint64Ns(timeNow), "FAIR.SELECTOR.C=2:S=2:P=2");
        expect(settings.getStored("scaling_factor", ctx5) == std::nullopt);
    };

    "CtxSettings Drop-In Settings replacement"_test = [&] {
        // the multiplexed Settings can be used as a drop-in replacement for "normal" Settings
        Graph testGraph;
        auto& block = testGraph.emplaceBlock<SettingsChangeRecorder<float>>({{"name", "TestName"}, {"scaling_factor", 2.f}});
        auto  s     = CtxSettings<std::remove_reference<decltype(block)>::type>(block, matchPred);
        block.setSettings(s);
        auto ctx0 = SettingsCtx(settings::convertTimePointToUint64Ns(std::chrono::system_clock::now()));
        expect(block.settings().set({{"name", "TestNameAlt"}, {"scaling_factor", 42.f}}, ctx0).empty()) << "successful set returns empty map";
        expect(eq(std::get<float>(block.settings().getStored("scaling_factor").value()), 42.f)); // TODO:
    };

    "CtxSettings autoUpdateParameters"_test = [&] {
        using namespace gr::testing;

        gr::Size_t         n_samples      = 20;
        bool               verboseConsole = true;
        Graph              testGraph;
        const property_map srcParameter = {{"n_samples_max", n_samples}, {"name", "TagSource"}, {"verbose_console", verboseConsole}};
        auto&              src          = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>(srcParameter);
        const auto         timeNow      = std::chrono::system_clock::now();

        for (std::size_t i = 0; i < 10; i++) {
            src._tags.push_back(gr::Tag(i, {{gr::tag::SAMPLE_RATE.shortKey(), static_cast<float>(i)}}));
        }
        // this sample_rates should not be applied
        src._tags.push_back({15, {{gr::tag::SAMPLE_RATE.shortKey(), 15.f}, {std::string(gr::tag::TRIGGER_TIME.shortKey()), settings::convertTimePointToUint64Ns(timeNow + std::chrono::seconds(20))}, //
                                     {std::string(gr::tag::TRIGGER_NAME.shortKey()), "name20"}, {std::string(gr::tag::TRIGGER_OFFSET.shortKey()), 0.f}}});
        src._tags.push_back({18, {{gr::tag::SAMPLE_RATE.shortKey(), 18.f}, {std::string(gr::tag::TRIGGER_TIME.shortKey()), settings::convertTimePointToUint64Ns(timeNow + std::chrono::seconds(10))}, //
                                     {std::string(gr::tag::TRIGGER_NAME.shortKey()), "name10"}, {std::string(gr::tag::TRIGGER_OFFSET.shortKey()), 0.f}}});

        auto& monitorBulk = testGraph.emplaceBlock<TagMonitor<float, ProcessFunction::USE_PROCESS_BULK>>({{"name", "TagMonitorBulk"}, {"n_samples_expected", n_samples}, {"verbose_console", verboseConsole}});
        auto& monitorOne  = testGraph.emplaceBlock<TagMonitor<float, ProcessFunction::USE_PROCESS_ONE>>({{"name", "TagMonitorOne"}, {"n_samples_expected", n_samples}, {"verbose_console", verboseConsole}});
        auto& sinkBulk    = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"name", "TagSinkBulk"}, {"n_samples_expected", n_samples}, {"verbose_console", verboseConsole}});
        auto& sinkOne     = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"name", "TagSinkOne"}, {"n_samples_expected", n_samples}, {"verbose_console", verboseConsole}});

        //                                  -> sinkOne
        // src -> monitorBulk -> monitorOne -> sinkBulk
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).to<"in">(monitorBulk)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(monitorBulk).to<"in">(monitorOne)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(monitorOne).to<"in">(sinkBulk)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(monitorOne).to<"in">(sinkOne)));

        scheduler::Simple sched{std::move(testGraph)};
        expect(sched.runAndWait().has_value());

        expect(eq(src._nSamplesProduced, n_samples)) << src.name.value;
        expect(eq(monitorBulk._nSamplesProduced, n_samples)) << monitorBulk.name.value;
        expect(eq(monitorOne._nSamplesProduced, n_samples)) << monitorOne.name.value;
        expect(eq(sinkBulk._nSamplesProduced, n_samples)) << sinkBulk.name.value;
        expect(eq(sinkOne._nSamplesProduced, n_samples)) << sinkOne.name.value;

        expect(eq(src.sample_rate, 1000.0f)) << src.name.value; // default value (set in the class)
        expect(eq(monitorBulk.sample_rate, 18.0f)) << monitorBulk.name.value;
        expect(eq(monitorOne.sample_rate, 18.0f)) << monitorOne.name.value;
        expect(eq(sinkBulk.sample_rate, 18.0f)) << sinkBulk.name.value;
        expect(eq(sinkOne.sample_rate, 18.0f)) << sinkOne.name.value;

        // The parameters that are changed via Tag are not stored
        expect(eq(src.settings().getNStoredParameters(), 1UZ)) << src.name.value;
        expect(eq(monitorBulk.settings().getNStoredParameters(), 1UZ)) << monitorBulk.name.value;
        expect(eq(monitorOne.settings().getNStoredParameters(), 1UZ)) << monitorOne.name.value;
        expect(eq(sinkBulk.settings().getNStoredParameters(), 1UZ)) << sinkBulk.name.value;
        expect(eq(sinkOne.settings().getNStoredParameters(), 1UZ)) << sinkOne.name.value;

        expect(eq(src._tags.size(), 12UZ)) << src.name.value;
        expect(eq(monitorBulk._tags.size(), 12UZ)) << monitorBulk.name.value;
        expect(eq(monitorOne._tags.size(), 12UZ)) << monitorOne.name.value;
        expect(eq(sinkBulk._tags.size(), 12UZ)) << sinkBulk.name.value;
        expect(eq(sinkOne._tags.size(), 12UZ)) << sinkOne.name.value;

        auto testStored = [&](BlockLike auto& block) {
            const auto& stored = block.settings().getStoredAll();
            expect(stored.contains("")) << block.name.value; // empty string is default context
            const auto& vec = stored.at("");
            expect(eq(vec.size(), 1UZ)) << block.name.value;            // no stored parameters were added via Tag
            expect(eq(vec[0].second.size(), 13UZ)) << block.name.value; // always store all parameters

            expect(eq(std::get<float>(vec[0].second.at(gr::tag::SAMPLE_RATE.shortKey())), 1000.f)); // Parameters changed via Tag are not changed in the storedParameters

            const auto& autoUpdate = block.settings().autoUpdateParameters();
            expect(eq(autoUpdate.size(), 6UZ)) << block.name.value;
            expect(eq(autoUpdate.contains("name"), false)) << block.name.value;
            expect(eq(autoUpdate.contains("n_samples_expected"), false)) << block.name.value;
            expect(eq(autoUpdate.contains("verbose_console"), false)) << block.name.value;
        };
        testStored(monitorBulk);
        testStored(monitorOne);
        testStored(sinkBulk);
        testStored(sinkOne);
    };

    "CtxSettings supported context types"_test = [&] {
        Graph      testGraph;
        auto&      block    = testGraph.emplaceBlock<SettingsChangeRecorder<int>>({{"scaling_factor", 1}});
        auto       settings = CtxSettings(block);
        const auto timeNow  = settings::convertTimePointToUint64Ns(std::chrono::system_clock::now());

        const auto ctxStr = SettingsCtx(timeNow, "String context"); // OK: string
        expect(settings.set({{"scaling_factor", 1}}, ctxStr).empty()) << "successful set returns empty map";
        const auto ctxInt = SettingsCtx(timeNow, 1); // OK: int
        expect(settings.set({{"scaling_factor", 2}}, ctxInt).empty()) << "successful set returns empty map";

        auto runType = [&]<typename TCtx>() {
            expect(throws([&] {
                const auto ctx = SettingsCtx(timeNow, static_cast<TCtx>(1));
                std::ignore    = settings.set({{"scaling_factor", 3}}, ctx);
            }));
        };

        runType.template operator()<float>();
        runType.template operator()<double>();
        runType.template operator()<std::size_t>();
    };

    // TODO enable this when load_grc works in emscripten (not relying on plugins here)
#ifndef NOPLUGINS
    "Property auto-forwarding with GRC-loaded graph"_test = [&] {
        constexpr std::string_view grc = R"(
blocks:
  - name: source
    id: gr::setting_test::Source<float64>
    parameters:
      n_samples_max: !!uint32 100
      sample_rate: !!float32 123456
  - name: test_block
    id: gr::testing::SettingsChangeRecorder<float64>
  - name: sink
    id: gr::setting_test::Sink<float64>
connections:
  - [source, 0, test_block, 0]
  - [test_block, 0, sink, 0]
)";
        BlockRegistry              registry;
        gr::registerBlock<Source, double>(registry);
        gr::registerBlock<SettingsChangeRecorder, double>(registry);
        gr::registerBlock<Sink, double>(registry);
        PluginLoader loader(registry, {});
        try {
            scheduler::Simple sched{loadGrc(loader, std::string(grc))};
            expect(sched.runAndWait().has_value());
            sched.graph().forEachBlock([](auto& block) { expect(eq(std::get<float>(*block.settings().get(gr::tag::SAMPLE_RATE.shortKey())), 123456.f)) << fmt::format("sample_rate forwarded to {}", block.name()); });
        } catch (const std::string& e) {
            expect(false) << fmt::format("GRC loading failed: {}\n", e);
        }
    };
#endif
};

int main() { /* tests are statically executed */ }
