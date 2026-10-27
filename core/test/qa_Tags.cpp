#include <boost/ut.hpp>

#include <algorithm>
#include <format>
#include <iterator>
#include <memory_resource>
#include <ranges>
#include <string>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Buffer.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/MemoryAllocators.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>

#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>
#include <gnuradio-4.0/testing/Sleep.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
template<>
struct std::formatter<gr::Tag> {
    template<typename ParseContext>
    constexpr auto parse(ParseContext& ctx) {
        return ctx.begin();
    }

    template<typename FormatContext>
    constexpr auto format(const gr::Tag& tag, FormatContext& ctx) const {
        return std::format_to(ctx.out(), "  {}->{{ {} }}\n", tag.index, tag.map);
    }
};

template<typename T>
struct RealignTagsToChunks : gr::Block<RealignTagsToChunks<T>> {
    using Description = gr::Doc<R""(A block that forwards samples and tags, moving tags onto the first sample of the current
or next chunk, whichever is closer. Also adds an "offset" key to the tag map signifying how much it was moved.)"">;
    // gr::PortIn<T, Doc<"In">, RequiredSamples<1024, 1024, true>> inPort;
    gr::PortIn<T, gr::Doc<"In">, gr::RequiredSamples<24, 24, false>> inPort;
    // gr::PortIn<T, Doc<"In">> inPort;
    gr::PortOut<T> outPort;

    GR_MAKE_REFLECTABLE(RealignTagsToChunks, inPort, outPort);

    double sampling_rate = 1.0;

    gr::work::Status processBulk(gr::InputSpanLike auto& inSamples, gr::OutputSpanLike auto& outSamples) {
        std::copy(inSamples.begin(), inSamples.end(), outSamples.begin());
        std::size_t tagsForwarded = 0;
        for (const auto& tagView : inSamples.rawTags()) {
            if (tagView.index < (inSamples.streamIndex + (inSamples.size() + 1) / 2)) {
                gr::property_map m{tagView.map}; // materialise to owning: the loop mutates the map
                m.insert_or_assign("offset", sampling_rate * static_cast<double>(tagView.index - inSamples.streamIndex));
                outSamples.publishTag(m, 0);
                tagsForwarded++;
            } else {
                break;
            }
        }
        if (inSamples.consumeRawTags(tagsForwarded)) {
            return gr::work::Status::OK;
        } else {
            return gr::work::Status::ERROR;
        }
    }
};

template<typename T>
struct DecimatorBackward : gr::Block<DecimatorBackward<T>, gr::Resampling<1UZ, 1UZ, false>, gr::BackwardTagPropagation> {
    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    gr::Size_t decim{1};

    GR_MAKE_REFLECTABLE(DecimatorBackward, in, out, decim);

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) { this->input_chunk_size = decim; }

    [[nodiscard]] gr::work::Status processBulk(std::span<const T> input, std::span<T> output) noexcept {
        assert(output.size() >= input.size() / decim);

        std::size_t out_sample_idx = 0;
        for (std::size_t i = 0; i < input.size(); ++i) {
            if (i % decim == 0) {
                output[out_sample_idx++] = input[i];
            }
        }
        return gr::work::Status::OK;
    }
};

template<typename T>
struct DecimatorForward : gr::Block<DecimatorForward<T>, gr::Resampling<1UZ, 1UZ, false>> {
    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    gr::Size_t decim{1};

    GR_MAKE_REFLECTABLE(DecimatorForward, in, out, decim);

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) { this->input_chunk_size = decim; }

    [[nodiscard]] gr::work::Status processBulk(std::span<const T> input, std::span<T> output) noexcept {
        assert(output.size() >= input.size() / decim);

        std::size_t out_sample_idx = 0;
        for (std::size_t i = 0; i < input.size(); ++i) {
            if (i % decim == 0) {
                output[out_sample_idx++] = input[i];
            }
        }
        return gr::work::Status::OK;
    }
};

template<typename T>
struct AutoForwardParametersBlock : public gr::Block<AutoForwardParametersBlock<T>> {
    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    float not_auto_forward_parameter = 0.f; // this parameter should be set but not forwarded

    // all gr::tag::kDefaultTags parameters except "reset_default", "store_default", "end_of_stream"
    float            sample_rate       = 0.f;
    std::pmr::string signal_name       = "";
    std::pmr::string signal_quantity   = "";
    std::pmr::string signal_unit       = "";
    float            signal_min        = 0.f;
    float            signal_max        = 0.f;
    gr::Size_t       n_dropped_samples = gr::Size_t(0);
    std::pmr::string trigger_name      = "";
    std::uint64_t    trigger_time      = 0;
    float            trigger_offset    = 0.f;
    gr::property_map trigger_meta_info = gr::property_map{};
    std::pmr::string context           = "";
    std::uint64_t    ctx_time          = 0;

    GR_MAKE_REFLECTABLE(AutoForwardParametersBlock, in, out, not_auto_forward_parameter, sample_rate, signal_name, signal_quantity, signal_unit, //
        signal_min, signal_max, n_dropped_samples, trigger_name, trigger_time, trigger_offset, trigger_meta_info, context, ctx_time);

    [[nodiscard]] constexpr auto processOne(T) noexcept { return T(0); }
};

static_assert(gr::HasProcessBulkFunction<RealignTagsToChunks<float>>);

namespace gr::testing {
static_assert(HasProcessOneFunction<TagSource<int, ProcessFunction::USE_PROCESS_ONE>>);
static_assert(not HasProcessBulkFunction<TagSource<int, ProcessFunction::USE_PROCESS_ONE>>);
static_assert(HasRequiredProcessFunction<TagSource<int, ProcessFunction::USE_PROCESS_ONE>>);
static_assert(not HasProcessOneFunction<TagSource<int, ProcessFunction::USE_PROCESS_BULK>>);
static_assert(HasProcessBulkFunction<TagSource<int, ProcessFunction::USE_PROCESS_BULK>>);
static_assert(HasRequiredProcessFunction<TagSource<int, ProcessFunction::USE_PROCESS_BULK>>);

static_assert(HasProcessOneFunction<TagMonitor<int, ProcessFunction::USE_PROCESS_ONE>>);
static_assert(not HasProcessOneFunction<TagMonitor<int, ProcessFunction::USE_PROCESS_BULK>>);
static_assert(not HasProcessBulkFunction<TagMonitor<int, ProcessFunction::USE_PROCESS_ONE>>);
static_assert(HasProcessBulkFunction<TagMonitor<int, ProcessFunction::USE_PROCESS_BULK>>);
static_assert(HasRequiredProcessFunction<TagMonitor<int, ProcessFunction::USE_PROCESS_ONE>>);
static_assert(HasRequiredProcessFunction<TagMonitor<int, ProcessFunction::USE_PROCESS_BULK>>);

static_assert(HasRequiredProcessFunction<TagSink<int, ProcessFunction::USE_PROCESS_ONE>>);
static_assert(HasRequiredProcessFunction<TagSink<int, ProcessFunction::USE_PROCESS_BULK>>);
} // namespace gr::testing

const boost::ut::suite<"TagTests"> _TagTests = [] {
    using namespace gr;

    static_assert(std::is_trivially_copyable_v<Tag>, "Tag is a trivially-copyable, non-owning device/USM by-value transport form");
    static_assert(gr::refl::data_member_count<Tag> == 2, "index and map being declared");
    static_assert(gr::refl::data_member_name<Tag, 0> == "index", "class field index is public API");
    static_assert(gr::refl::data_member_name<Tag, 1> == "map", "class field map is public API");

    using namespace boost::ut;

    "the estimated frame rate, bandwidth and band edges round-trip through their canonical id"_test = [] {
        namespace k = gr::pmt::keys;
        static_assert(k::idOf<"gr:est_frame_rate"> == 0x0022);
        property_map map;
        map.insert_or_assign(tag::EST_FRAME_RATE.key(), 2.5f);
        map.insert_or_assign(tag::BANDWIDTH.key(), 200e3);
        map.insert_or_assign(tag::FREQUENCY_MIN.key(), 99.9e6);
        map.insert_or_assign(tag::FREQUENCY_MAX.key(), 100.1e6);
        expect(map.find_value(tag::EST_FRAME_RATE.key()).value() == 2.5f);
        expect(map.find_value(tag::BANDWIDTH.key()).value() == 200e3);
        expect(map.find_value(tag::FREQUENCY_MIN.key()).value() == 99.9e6);
        expect(map.find_value(tag::FREQUENCY_MAX.key()).value() == 100.1e6);
        expect(eq(k::lookupShortId(tag::FREQUENCY_MIN.shortKey()), std::uint16_t{0x0024}));
        expect(eq(k::lookupShortId(tag::BANDWIDTH.shortKey()), std::uint16_t{0x0023}));
        expect(eq(k::lookupShortId(tag::EST_FRAME_RATE.shortKey()), std::uint16_t{0x0022}));
        expect(eq(k::lookupShortId(tag::FREQUENCY_MAX.shortKey()), std::uint16_t{0x0025}));
        expect(eq(std::ranges::count(tag::kDefaultTags, "bandwidth"sv), 1));
    };
    "a setting alias carries the type, unit and description of its DefaultTag"_test = [] {
        using namespace std::string_view_literals;
        static_assert(std::is_same_v<gr::SampleRate::value_type, float>);
        static_assert(std::is_same_v<gr::Frequency::value_type, double>);
        static_assert(std::is_same_v<gr::NumChannels::value_type, gr::Size_t>);
        static_assert(gr::SampleRate::unit() == "Hz"sv && gr::SampleRate::description() == "sample_rate"sv);
        static_assert(gr::SampleRate::documentation() == tag::SAMPLE_RATE.description() && gr::SampleRate::description() == tag::SAMPLE_RATE.shortKey() && gr::SampleRate::unit() == tag::SAMPLE_RATE.unit());
        static_assert(gr::SignalName::unit().empty() && gr::SignalName::description() == "signal_name"sv && gr::SignalName::documentation() == "signal name"sv);
        static_assert(std::is_same_v<gr::SignalName::value_type, std::pmr::string>);
        static_assert(gr::SampleRate::visible() && gr::FrameRate::visible() && gr::Frequency::visible());
        static_assert(!gr::EstSampleRate::visible());
        static_assert(gr::SampleRate::documentation().size() == std::string_view("sampling rate fₛ: samples per second on this edge").size(), "fixed_string counts the UTF-8 bytes");
        static_assert(gr::SampleRate::documentation().contains("fₛ") && std::string_view(tag::TRIGGER_TIME_ERROR.description()).contains("σ") && std::string_view(tag::ACQUISITION_FLAGS.description()).contains("→"));

        gr::SampleRate sampleRate = 1000.f;
        expect(sampleRate.validate_and_set(0.f)) << "zero stays valid: not yet known";
        expect(sampleRate.validate_and_set(1000.f));
        expect(!sampleRate.validate_and_set(-1.f));
        expect(eq(static_cast<float>(sampleRate), 1000.f)) << "a rejected value leaves the setting unchanged";
        expect(sampleRate.validate_and_set(48000.f));
        expect(eq(static_cast<float>(sampleRate), 48000.f));
    };

    "DefaultTags"_test = [] {
        using namespace std::string_view_literals;
        property_map m; // the map ops below exercise the DefaultTag fluent API on an owning property_map (Tag::map is a non-owning view)

        // Note: DefaultTag's std::string conversion yields the prefixed `key()` ("gr:sample_rate"),
        // not the shortKey ("sample_rate"). The typed-fluent `tag::SAMPLE_RATE(value)` returns
        // a pair using the full prefixed key, so callers go through the prefixed key everywhere.
        m.insert_or_assign(tag::SAMPLE_RATE.key(), 3.0f);
        m.insert_or_assign(tag::SAMPLE_RATE(4.0f)); // pair-shaped overload accepts the typed-fluent directly
        // m.insert_or_assign(tag::SAMPLE_RATE(5.0)); // type-mismatch -> won't compile
        expect(m.find_value(tag::SAMPLE_RATE.key()).value() == 4.0f);

        const property_map singleEntry = tag::SAMPLE_RATE(42.0f);
        expect(eq(singleEntry.size(), 1UZ));
        expect(singleEntry.find_value(tag::SAMPLE_RATE).value() == 42.0f);
        std::vector<std::pair<std::size_t, property_map>> tagList = {{0UZ, {{tag::CONTEXT, "1"}}}, {100UZ, tag::CONTEXT(std::string{"2"})}};
        expect(eq(tagList.size(), 2UZ));
        expect(tagList[0].second.contains(tag::CONTEXT));
        expect(tagList[1].second.contains(tag::CONTEXT));
        expect(tag::SAMPLE_RATE.shortKey() == "sample_rate"sv);
        expect(tag::SAMPLE_RATE.key() == std::string{GR_TAG_PREFIX}.append("sample_rate"));
        expect(tag::settingsKey(tag::SAMPLE_RATE.key()) == tag::SAMPLE_RATE.shortKey());
        expect(tag::settingsKey(tag::SAMPLE_RATE.shortKey()) == tag::SAMPLE_RATE.shortKey());

        // map.at throws on miss (std::map parity); use find()/contains() for exception-free probes.
        expect(m.contains(tag::SAMPLE_RATE.key()));
        expect(not m.contains(tag::SIGNAL_NAME.key()));

        // test other tag on key definition only
        static_assert(tag::SIGNAL_UNIT.shortKey() == "signal_unit"sv);
        static_assert(tag::SIGNAL_MIN.shortKey() == "signal_min"sv);
        static_assert(tag::SIGNAL_MAX.shortKey() == "signal_max"sv);
        static_assert(tag::TRIGGER_NAME.shortKey() == "trigger_name"sv);
        static_assert(tag::TRIGGER_TIME.shortKey() == "trigger_time"sv);
        static_assert(tag::TRIGGER_OFFSET.shortKey() == "trigger_offset"sv);

        // test other tag on key definition only
        static_assert(tag::SIGNAL_UNIT.key() == "gr:signal_unit"sv);
        static_assert(tag::SIGNAL_MIN.key() == "gr:signal_min"sv);
        static_assert(tag::SIGNAL_MAX.key() == "gr:signal_max"sv);
        static_assert(tag::TRIGGER_NAME.key() == "gr:trigger_name"sv);
        static_assert(tag::TRIGGER_TIME.key() == "gr:trigger_time"sv);
        static_assert(tag::TRIGGER_OFFSET.key() == "gr:trigger_offset"sv);
        static_assert(tag::USER_DATA.shortKey() == "user_data"sv);
        static_assert(tag::USER_DATA.key() == "gr:user_data"sv);

        using namespace std::string_literals;
        using namespace std::string_view_literals;
        static_assert(tag::SIGNAL_UNIT == "gr:signal_unit"s); // DefaultTag compares as its canonical gr:-prefixed key
        static_assert("gr:signal_unit"s == tag::SIGNAL_UNIT);

        static_assert("gr:signal_unit"sv == tag::SIGNAL_UNIT);
        static_assert(tag::SIGNAL_UNIT == "gr:signal_unit"sv);
        static_assert(tag::USER_DATA == "gr:user_data"sv);
        static_assert("gr:user_data"sv == tag::USER_DATA);

        static_assert(tag::SIGNAL_UNIT == "gr:signal_unit");
        static_assert("gr:signal_unit" == tag::SIGNAL_UNIT);

        // alt definition -> eventually needed for SigMF compatibility
        using namespace gr::tag;
        static_assert(SIGNAL_UNIT == "gr:signal_unit"sv);
        static_assert("gr:signal_unit" == tag::SIGNAL_UNIT);
    };
};

const boost::ut::suite<"TagPassThroughForwarding"> _tagPassThroughForwarding = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;
    using namespace gr::tag;

    // a tag whose keys are all auto-forwarded, flowing through a settings-free block (no value to substitute,
    // no key to drop), must forward as a verbatim wire-blob copy — not a key-by-key property_map rebuild.
    "settings-free block forwards an all-auto-forward tag unchanged (pass-through)"_test = [] {
        const gr::Size_t                          nSamples = 100;
        Graph                                     g;
        const std::vector<gr::testing::OwningTag> emitted = {gr::testing::OwningTag(1UZ, {{SAMPLE_RATE.key(), 48000.f}, {SIGNAL_NAME.key(), "ch0"}, {CONTEXT.key(), "ctx-42"}})};
        auto&                                     src     = g.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"name", "src"}, {"n_samples_max", nSamples}});
        src._tags                                         = emitted;
        auto& copy                                        = g.emplaceBlock<gr::testing::Copy<float>>(); // no writable settings → classifyForward can never substitute
        auto& sink                                        = g.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"name", "sink"}});
        expect(g.connect<"out", "in">(src, copy).has_value());
        expect(g.connect<"out", "in">(copy, sink).has_value());

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(g)).has_value());
        expect(sched.runAndWait().has_value());

        expect(eq(sink._tags.size(), 1UZ)) << "the tag reached the sink through the forwarding block";
        expect(sink._tags[0].map == emitted[0].map) << "pass-through preserves the tag content exactly";
    };

    // the same path under load must not allocate per tag: a rebuild would heap-build a property_map for every
    // forwarded tag (~several allocs each), pass-through is one memcpy into the pre-grown ring slot.
    "pass-through forwarding does not allocate per tag (memcpy, not rebuild)"_test = [] {
        auto allocsForRun = [](gr::Size_t nSamples) {
            // PMR lifetime contract: a resource installed via set_default_resource() must outlive everything
            // allocated from it. Buffers materialised during runAndWait() bind the live default and free through it
            // at teardown, so defaultCounter is declared first (destroyed last, after `sched`); it is installed as
            // the default only around the run, so allocCount measures run-time draws, not construction.
            gr::allocator::pmr::CountingResource defaultCounter;

            Graph g;
            auto& src   = g.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"name", "src"}, {"n_samples_max", nSamples}, {"repeat_tags", true}});
            src._tags   = {gr::testing::OwningTag(1UZ, {{SAMPLE_RATE.key(), 48000.f}, {SIGNAL_NAME.key(), "ch0"}})};
            auto& copy1 = g.emplaceBlock<gr::testing::Copy<float>>();
            auto& copy2 = g.emplaceBlock<gr::testing::Copy<float>>(); // terminal, settings-free, records nothing
            expect(g.connect<"out", "in">(src, copy1).has_value());
            expect(g.connect<"out", "in">(copy1, copy2).has_value());
            gr::scheduler::Simple<> sched;
            expect(sched.exchange(std::move(g)).has_value());

            auto* prev = std::pmr::set_default_resource(&defaultCounter);
            expect(sched.runAndWait().has_value());
            std::pmr::set_default_resource(prev);
            return defaultCounter.allocCount;
        };
        // 10x the tags → 10x the rebuilds, but the same warm-up. The per-tag forward cost is the *difference*;
        // for a memcpy pass-through it is ~0, for a per-tag rebuild it would be ~5 × 9 × nBase allocs.
        const std::size_t base = allocsForRun(2000U);
        const std::size_t ten  = allocsForRun(20000U);
        expect(lt(ten - base, 2000UZ)) << std::format("per-tag forward allocations detected: {} extra over 18000 more tags (base={}, 10x={})", ten - base, base, ten);
    };

    "a key without gr: is forwarded like any other"_test = [] {
        const gr::Size_t                          nSamples = 100;
        Graph                                     g;
        const std::vector<gr::testing::OwningTag> emitted = {gr::testing::OwningTag(1UZ, {{"custom_key", std::int32_t{7}}, {std::string(gr::tag::TRIGGER_NAME.key()), "one"}})};
        auto&                                     src     = g.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"name", "src"}, {"n_samples_max", nSamples}});
        src._tags                                         = emitted;
        auto& copy                                        = g.emplaceBlock<gr::testing::Copy<float>>();
        auto& sink                                        = g.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"name", "sink"}});
        expect(g.connect<"out", "in">(src, copy).has_value());
        expect(g.connect<"out", "in">(copy, sink).has_value());

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(g)).has_value());
        expect(sched.runAndWait().has_value());

        expect(eq(sink._tags.size(), 1UZ));
        expect(sink._tags[0].map == emitted[0].map) << "the whole tag goes on";
    };
};

const boost::ut::suite<"TagPropagation"> _TagPropagation = [] {
    using namespace std::string_literals;
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;

    "settings changed at construction go downstream as gr: tags for the standard keys only"_test = [&] {
        using namespace gr::testing;
        using namespace gr::tag;
        const gr::Size_t nSamples = 10;
        Graph            testGraph;

        // "reset_default", "store_default", "end_of_stream" are not included because they have special meaning
        const property_map srcParametersOnlyAutoForward = {{SAMPLE_RATE.shortKey(), 42.f},    //
            {SIGNAL_NAME.shortKey(), "SIGNAL_NAME_42"},                                       //
            {SIGNAL_QUANTITY.shortKey(), "SIGNAL_QUANTITY_42"},                               //
            {SIGNAL_UNIT.shortKey(), "SIGNAL_UNIT_42"},                                       //
            {SIGNAL_MIN.shortKey(), 42.f},                                                    //
            {SIGNAL_MAX.shortKey(), 42.f},                                                    //
            {N_DROPPED_SAMPLES.shortKey(), gr::Size_t(42)},                                   //
            {TRIGGER_NAME.shortKey(), "TRIGGER_NAME_42"},                                     //
            {TRIGGER_TIME.shortKey(), std::uint64_t(42)},                                     //
            {TRIGGER_OFFSET.shortKey(), 42.f},                                                //
            {TRIGGER_META_INFO.shortKey(), property_map{{"TRIGGER_META_INFO_KEY_42", 42.f}}}, //
            {CONTEXT.shortKey(), "CONTEXT_42"},                                               //
            {CONTEXT_TIME.shortKey(), std::uint64_t(42)}};

        property_map srcParametersForwardedAsTags; // canonical settings are emitted downstream as gr:-prefixed tags
        for (const auto& [key, value] : srcParametersOnlyAutoForward) {
            srcParametersForwardedAsTags.insert_or_assign(std::string(gr::GR_TAG_PREFIX.view()) + std::string(key), value);
        }

        property_map srcParameter = srcParametersOnlyAutoForward;
        srcParameter.insert({"not_auto_forward_parameter", 42.f});
        //
        auto& src              = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"name", "TagSource"}, {"n_samples_max", nSamples}});
        auto& autoForwardBlock = testGraph.emplaceBlock<AutoForwardParametersBlock<float>>(srcParameter);
        auto& monitor          = testGraph.emplaceBlock<TagMonitor<float, ProcessFunction::USE_PROCESS_BULK>>({{"name", "TagMonitor"}});
        auto& sink             = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"name", "TagSink"}});

        expect(testGraph.connect<"out", "in">(src, autoForwardBlock).has_value());
        expect(testGraph.connect<"out", "in">(autoForwardBlock, monitor).has_value());
        expect(testGraph.connect<"out", "in">(monitor, sink).has_value());

        gr::scheduler::Simple<> sched;
        if (auto ret = sched.exchange(std::move(testGraph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        expect(sched.runAndWait().has_value());

        expect(eq(src._nSamplesProduced, nSamples));
        expect(eq(monitor._nSamplesProduced, nSamples));
        expect(eq(sink._nSamplesProduced, nSamples));

        expect(eq(src.sample_rate, 1000.0f)); // default value (set in the class)
        expect(eq(autoForwardBlock.sample_rate, 42.0f));
        expect(eq(monitor.sample_rate, 42.0f));
        expect(eq(sink.sample_rate, 42.0f));

        expect(eq(monitor._tags.size(), 1UZ));
        expect(eq(sink._tags.size(), 1UZ));

        expect(eq(monitor._tags[0].map.size(), srcParametersForwardedAsTags.size()));
        expect(eq(sink._tags[0].map.size(), srcParametersForwardedAsTags.size()));

        expect(monitor._tags[0].map == srcParametersForwardedAsTags);
        map_diff_report(monitor._tags[0].map, srcParametersForwardedAsTags, "monitor._tags", "srcParametersForwardedAsTags");

        expect(sink._tags[0].map == srcParametersForwardedAsTags);
        map_diff_report(sink._tags[0].map, srcParametersForwardedAsTags, "sink._tags", "srcParametersForwardedAsTags");
    };

    auto runTest = []<auto srcType>(bool verbose = true) {
        using namespace gr::testing;
        using namespace gr::tag;

        const gr::Size_t nSamples = 100;
        Graph            testGraph;

        // "reset_default", "store_default", "end_of_stream" are not included because they have special meaning
        const std::vector<gr::testing::OwningTag> tagsOnlyAutoForward = {gr::testing::OwningTag(1UZ, {{SAMPLE_RATE.key(), 42.f}}), gr::testing::OwningTag(2UZ, {{SIGNAL_NAME.key(), "SIGNAL_NAME_42"}}), //
            gr::testing::OwningTag(3UZ, {{SIGNAL_QUANTITY.key(), "SIGNAL_QUANTITY_42"}}),                                                                                                                //
            gr::testing::OwningTag(4UZ, {{SIGNAL_UNIT.key(), "SIGNAL_UNIT_42"}}),                                                                                                                        //
            gr::testing::OwningTag(5UZ, {{SIGNAL_MIN.key(), 42.f}}),                                                                                                                                     //
            gr::testing::OwningTag(6UZ, {{SIGNAL_MAX.key(), 42.f}}),                                                                                                                                     //
            gr::testing::OwningTag(7UZ, {{N_DROPPED_SAMPLES.key(), gr::Size_t(42)}}),                                                                                                                    //
            gr::testing::OwningTag(8UZ, {{TRIGGER_NAME.key(), "TRIGGER_NAME_42"}}),                                                                                                                      //
            gr::testing::OwningTag(9UZ, {{TRIGGER_TIME.key(), std::uint64_t(42)}}),                                                                                                                      //
            gr::testing::OwningTag(10UZ, {{TRIGGER_OFFSET.key(), 42.f}}),                                                                                                                                //
            gr::testing::OwningTag(11UZ, {{TRIGGER_META_INFO.key(), property_map{{"TRIGGER_META_INFO_KEY_42", 42.f}}}}),                                                                                 //
            gr::testing::OwningTag(12UZ, {{CONTEXT.key(), "CONTEXT_42"}}),                                                                                                                               //
            gr::testing::OwningTag(13UZ, {{CONTEXT_TIME.key(), std::uint64_t(42)}})};

        std::vector<gr::testing::OwningTag> tags = tagsOnlyAutoForward;
        tags.push_back(gr::testing::OwningTag(14UZ, {{"not_auto_forward_parameter", 42.f}}));

        auto& src              = testGraph.emplaceBlock<TagSource<float, srcType>>({{"name", "TagSource"}, {"n_samples_max", nSamples}, {"verbose_console", true && verbose}});
        src._tags              = tags;
        auto& monitorOne       = testGraph.emplaceBlock<TagMonitor<float, ProcessFunction::USE_PROCESS_ONE>>({{"name", "TagMonitorOne"}, {"verbose_console", true && verbose}});
        auto& monitorBulk1     = testGraph.emplaceBlock<TagMonitor<float, ProcessFunction::USE_PROCESS_BULK>>({{"name", "TagMonitorBulk1"}, {"verbose_console", true && verbose}});
        auto& autoForwardBlock = testGraph.emplaceBlock<AutoForwardParametersBlock<float>>({{"name", "AutoForwardParametersBlock"}});
        auto& monitorBulk2     = testGraph.emplaceBlock<TagMonitor<float, ProcessFunction::USE_PROCESS_BULK>>({{"name", "TagMonitorBulk2"}, {"verbose_console", true && verbose}});
        auto& sinkOne          = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"name", "TagSinkOne"}, {"verbose_console", true && verbose}});
        auto& sinkBulk         = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"name", "TagSinkBulk"}, {"verbose_console", true && verbose}});

        /** every block forwards every tag; autoForwardBlock applies the keys naming its settings -- `not_auto_forward_parameter` included -- and forwards them with its values */
        // src -> monitorOne -> monitorBulk1 -> autoForwardBlock -> monitorBulk2 ┬─> sinkOne
        //                                                                       └─> sinkOne
        expect(testGraph.connect<"out", "in">(src, monitorOne).has_value());
        expect(testGraph.connect<"out", "in">(monitorOne, monitorBulk1).has_value());
        expect(testGraph.connect<"out", "in">(monitorBulk1, autoForwardBlock).has_value());
        expect(testGraph.connect<"out", "in">(autoForwardBlock, monitorBulk2).has_value());
        expect(testGraph.connect<"out", "in">(monitorBulk2, sinkOne).has_value());
        expect(testGraph.connect<"out", "in">(monitorBulk2, sinkBulk).has_value());

        gr::scheduler::Simple<> sched;
        if (auto ret = sched.exchange(std::move(testGraph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        expect(sched.runAndWait().has_value());

        expect(eq(src._nSamplesProduced, nSamples));
        expect(eq(monitorOne._nSamplesProduced, nSamples));
        expect(eq(monitorBulk1._nSamplesProduced, nSamples));
        expect(eq(monitorBulk2._nSamplesProduced, nSamples));
        expect(eq(sinkOne._nSamplesProduced, nSamples));
        expect(eq(sinkBulk._nSamplesProduced, nSamples));

        expect(!monitorOne.log_samples || eq(monitorOne._samples.size(), nSamples));
        expect(!monitorBulk1.log_samples || eq(monitorBulk1._samples.size(), nSamples));
        expect(!monitorBulk2.log_samples || eq(monitorBulk2._samples.size(), nSamples));
        expect(!sinkOne.log_samples || eq(sinkOne._samples.size(), nSamples));
        expect(!sinkBulk.log_samples || eq(sinkBulk._samples.size(), nSamples));

        expect(eq(src.sample_rate, 1000.0f)); // default value
        expect(eq(monitorOne.sample_rate, 42.0f));
        expect(eq(monitorBulk1.sample_rate, 42.0f));
        expect(eq(monitorBulk2.sample_rate, 42.0f));
        expect(eq(sinkOne.sample_rate, 42.0f));
        expect(eq(sinkBulk.sample_rate, 42.0f));

        expect(eq(monitorOne.signal_name, "SIGNAL_NAME_42"_spmr));
        expect(eq(monitorBulk1.signal_name, "SIGNAL_NAME_42"_spmr));
        expect(eq(monitorBulk2.signal_name, "SIGNAL_NAME_42"_spmr));
        expect(eq(sinkBulk.signal_name, "SIGNAL_NAME_42"_spmr));
        expect(eq(sinkOne.signal_name, "SIGNAL_NAME_42"_spmr));

        expect(eq(autoForwardBlock.not_auto_forward_parameter, 42.f)) << "a bare key naming a downstream setting reconfigures it";
        expect(eq(autoForwardBlock.sample_rate, 42.f));
        expect(eq(autoForwardBlock.signal_name, "SIGNAL_NAME_42"_spmr));
        expect(eq(autoForwardBlock.signal_quantity, "SIGNAL_QUANTITY_42"_spmr));
        expect(eq(autoForwardBlock.signal_unit, "SIGNAL_UNIT_42"_spmr));
        expect(eq(autoForwardBlock.signal_min, 42.f));
        expect(eq(autoForwardBlock.signal_max, 42.f));
        expect(eq(autoForwardBlock.n_dropped_samples, gr::Size_t(42)));
        expect(eq(autoForwardBlock.trigger_name, "TRIGGER_NAME_42"_spmr));
        expect(eq(autoForwardBlock.trigger_time, uint64_t(42)));
        expect(eq(autoForwardBlock.trigger_offset, 42.f));
        expect(eq(autoForwardBlock.trigger_meta_info.size(), 1UZ));
        expect(eq(autoForwardBlock.context, "CONTEXT_42"_spmr));
        expect(eq(autoForwardBlock.ctx_time, std::uint64_t(42)));

        expect(equal_tag_lists(monitorOne._tags, tags)); // all tags from src
        expect(equal_tag_lists(monitorBulk1._tags, tags));
        expect(equal_tag_lists(monitorBulk2._tags, tags));
        expect(equal_tag_lists(sinkOne._tags, tags));
        expect(equal_tag_lists(sinkBulk._tags, tags));
    };

    "TagPropagation forwards every tag from TagSource<float, USE_PROCESS_BULK>"_test = [&] { runTest.template operator()<ProcessFunction::USE_PROCESS_BULK>(true); };
    "TagPropagation forwards every tag from TagSource<float, USE_PROCESS_ONE>"_test  = [&] { runTest.template operator()<ProcessFunction::USE_PROCESS_ONE>(true); };

    "CustomTagHandling"_test = []() {
        gr::Size_t         n_samples = 1024;
        Graph              testGraph;
        const property_map srcParameter = {{"n_samples_max", n_samples}, {"name", "TagSource"}, {gr::tag::SIGNAL_NAME.shortKey(), "tagStream"}, {"verbose_console", true}};
        auto&              src          = testGraph.emplaceBlock<TagSource<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>(srcParameter);
        src._tags                       = {
            {0, gr::property_map{{"key", "value@0"}, {"key0", "value@0"}}},          //
            {1, gr::property_map{{"key", "value@1"}, {"key1", "value@1"}}},          //
            {100, gr::property_map{{"key", "value@100"}, {"key2", "value@100"}}},    //
            {150, gr::property_map{{"key", "value@150"}, {"key3", "value@150"}}},    //
            {1000, gr::property_map{{"key", "value@1000"}, {"key4", "value@1000"}}}, //
            {1001, gr::property_map{{"key", "value@1001"}, {"key5", "value@1001"}}}, //
            {1002, gr::property_map{{"key", "value@1002"}, {"key6", "value@1002"}}}, //
            {1023, gr::property_map{{"key", "value@1023"}, {"key7", "value@1023"}}}  //
        };
        expect(eq("tagStream"_spmr, src.signal_name)) << "src signal_name -> needed for setting-via-tag forwarding";
        auto& realign = testGraph.emplaceBlock<RealignTagsToChunks<float>>();
        auto& sink    = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"name", "TagSinkN"}, {"n_samples_expected", n_samples}, {"verbose_console", true}});

        // [ TagSource ] -> [ tag realign block ] -> [ TagSink ]
        expect(testGraph.connect<"out", "inPort">(src, realign).has_value());
        expect(testGraph.connect<"outPort", "in">(realign, sink).has_value());

        gr::scheduler::Simple<> sched;
        if (auto ret = sched.exchange(std::move(testGraph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        expect(sched.runAndWait().has_value());

        expect(eq(src._nSamplesProduced, n_samples)) << "src did not produce enough output samples";
        expect(eq(sink._nSamplesProduced, 1008U)) << "sinkOne did not consume enough input samples"; // default policy is to drop epilogue samples
        expect(ge(sink._tags.size(), 3UZ));                                                          // at least the runtime tags (init-time forwarding may add more)
    };

    auto runPolicyTest = []<typename TDecimator>(const std::vector<gr::testing::OwningTag>& expectedTags) {
        gr::Size_t nSamples = 45;
        gr::Size_t decim    = 10;

        Graph testGraph;
        auto& src = testGraph.emplaceBlock<TagSource<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", nSamples}, {"verbose_console", true}});
        src._tags = {
            {0, gr::property_map{{"key", "value@0"}, {"key0", "value@0"}}},     //
            {4, gr::property_map{{"key", "value@4"}, {"key4", "value@4"}}},     //
            {5, gr::property_map{{"key", "value@5"}, {"key5", "value@5"}}},     //
            {15, gr::property_map{{"key", "value@15"}, {"key15", "value@15"}}}, //
            {20, gr::property_map{{"key", "value@20"}, {"key20", "value@20"}}}, //
            {25, gr::property_map{{"key", "value@25"}, {"key25", "value@25"}}}, //
            {35, gr::property_map{{"key", "value@35"}, {"key35", "value@35"}}}  //
        };

        auto& decimator = testGraph.emplaceBlock<TDecimator>({{"decim", decim}});
        auto& sink      = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", true}});

        expect(testGraph.connect<"out", "in">(src, decimator).has_value());
        expect(testGraph.connect<"out", "in">(decimator, sink).has_value());

        gr::scheduler::Simple<> sched;
        if (auto ret = sched.exchange(std::move(testGraph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        expect(sched.runAndWait().has_value());

        expect(eq(src._nSamplesProduced, nSamples));
        expect(eq(sink._nSamplesProduced, 4U));

        expect(equal_tag_lists(sink._tags, expectedTags));
    };

    "Tag propagation with decimation - Forward policy"_test = [&runPolicyTest]() {
        // individual tags forwarded per input position — unconsumed tags carry to next output chunk
        std::vector<gr::testing::OwningTag> expectedTags = std::vector<gr::testing::OwningTag>{//
            {0, gr::property_map{{"key", "value@0"}, {"key0", "value@0"}}},                    //
            {1, gr::property_map{{"key", "value@4"}, {"key4", "value@4"}}},                    //
            {1, gr::property_map{{"key", "value@5"}, {"key5", "value@5"}}},                    //
            {2, gr::property_map{{"key", "value@15"}, {"key15", "value@15"}}},                 //
            {2, gr::property_map{{"key", "value@20"}, {"key20", "value@20"}}},                 //
            {3, gr::property_map{{"key", "value@25"}, {"key25", "value@25"}}}};
        runPolicyTest.template              operator()<DecimatorForward<float>>(expectedTags);
    };

    "Tag propagation with decimation - Backward policy"_test = [&runPolicyTest]() {
        // backward: all tags in each input chunk mapped to output position 0
        std::vector<gr::testing::OwningTag> expectedTags = std::vector<gr::testing::OwningTag>{//
            {0, gr::property_map{{"key", "value@0"}, {"key0", "value@0"}}},                    //
            {0, gr::property_map{{"key", "value@4"}, {"key4", "value@4"}}},                    //
            {0, gr::property_map{{"key", "value@5"}, {"key5", "value@5"}}},                    //
            {1, gr::property_map{{"key", "value@15"}, {"key15", "value@15"}}},                 //
            {2, gr::property_map{{"key", "value@20"}, {"key20", "value@20"}}},                 //
            {2, gr::property_map{{"key", "value@25"}, {"key25", "value@25"}}},                 //
            {3, gr::property_map{{"key", "value@35"}, {"key35", "value@35"}}}};
        runPolicyTest.template              operator()<DecimatorBackward<float>>(expectedTags);
    };
};

template<typename T>
struct AsyncEndOfStreamReader : gr::Block<AsyncEndOfStreamReader<T>> {
    gr::PortIn<T, gr::Async> in;
    gr::PortOut<T>           out;
    GR_MAKE_REFLECTABLE(AsyncEndOfStreamReader, in, out);

    std::size_t _endOfStreamSeen = 0UZ;

    gr::work::Status processBulk(gr::InputSpanLike auto& input, gr::OutputSpanLike auto& output) {
        if (gr::samples_to_eos_tag(in).has_value()) {
            ++_endOfStreamSeen;
        }
        const std::size_t n = std::min(input.size(), output.size());
        std::copy_n(input.begin(), n, output.begin());
        std::ignore = input.consume(n);
        output.publish(n);
        return gr::work::Status::OK;
    }
};

template<typename T>
struct StridedPicker : gr::Block<StridedPicker<T>, gr::Resampling<2UZ, 1UZ, true>, gr::Stride<4UZ, true>> {
    gr::PortIn<T>  in;
    gr::PortOut<T> out;
    GR_MAKE_REFLECTABLE(StridedPicker, in, out);

    gr::work::Status processBulk(gr::InputSpanLike auto& input, gr::OutputSpanLike auto& output) {
        for (std::size_t w = 0UZ; w < output.size(); ++w) {
            output[w] = input[w * 4UZ];
        }
        return gr::work::Status::OK;
    }
};

const boost::ut::suite<"StrideSkippedTags"> _strideSkippedTags = [] {
    using namespace boost::ut;
    using namespace gr::testing;

    "a tag on a sample a stride skips goes onto the next output instead of being lost"_test = [] {
        /** windows of 2 every 4: [0,1] [4,5] [8,9] [12,13]; 2, 3, 6, 7, 10, 11, 14, 15 are skipped */
        gr::Graph graph;
        auto&     source = graph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(16)}, {"mark_tag", false}, {"verbose_console", false}});
        source._tags     = {{3UZ, {{std::string(gr::tag::TRIGGER_NAME.key()), "skipped 3"}}}, {9UZ, {{std::string(gr::tag::TRIGGER_NAME.key()), "inside 9"}}}, {13UZ, {{std::string(gr::tag::TRIGGER_NAME.key()), "inside 13"}}}};
        auto& picker     = graph.emplaceBlock<StridedPicker<float>>();
        auto& sink       = graph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"log_samples", true}, {"verbose_console", false}});
        expect(graph.connect<"out", "in">(source, picker).has_value());
        expect(graph.connect<"out", "in">(picker, sink).has_value());
        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(graph)).has_value());
        expect(sched.runAndWait().has_value());
        expect(std::ranges::equal(sink._samples, std::vector<float>{0.f, 4.f, 8.f, 12.f})) << "a tag must not shift the window grid";
        std::vector<std::pair<std::size_t, std::string>> arrived;
        for (const auto& received : sink._tags) {
            if (received.map.contains(std::string(gr::tag::TRIGGER_NAME.key()))) {
                arrived.emplace_back(received.index, received.map.template value_or<std::string>(std::string(gr::tag::TRIGGER_NAME.key()), std::string{}));
            }
        }
        expect(arrived == std::vector<std::pair<std::size_t, std::string>>{{1UZ, "skipped 3"}, {3UZ, "inside 9"}}) << "3 is skipped: next output 1; 9 carries forward past the skipped 10, 11 to output 3; 13 has no output left" << std::format(" -- arrived {}", arrived);
    };
};

const boost::ut::suite<"EndOfStreamLookup"> _endOfStreamLookup = [] {
    using namespace boost::ut;
    using namespace gr::testing;

    "looking for the end of stream inside processBulk leaves the span's tags to be released"_test = [] {
        /** 4096 tags: more than the tag ring holds unreleased */
        constexpr gr::Size_t  kSamples   = 1U << 17U;
        constexpr std::size_t kTagPeriod = 64UZ;
        gr::Graph             graph;
        auto&                 source = graph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", kSamples}, {"mark_tag", false}, {"verbose_console", false}});
        for (std::size_t i = 0UZ; i < kSamples; i += kTagPeriod) {
            source._tags.push_back({i, {{std::string(gr::tag::TRIGGER_NAME.key()), "t"}}});
            source._tags.push_back({i + kTagPeriod - 1UZ, {{"local_key", 1}}});
        }
        auto& reader = graph.emplaceBlock<AsyncEndOfStreamReader<float>>();
        auto& sink   = graph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"log_samples", false}, {"verbose_console", false}});
        expect(graph.connect<"out", "in">(source, reader).has_value());
        expect(graph.connect<"out", "in">(reader, sink).has_value());
        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(graph)).has_value());
        expect(sched.runAndWait().has_value());
        expect(eq(sink._nSamplesProduced, kSamples)) << "the stream ended -- its end-of-stream tag was not dropped from a full tag ring -- and with its last sample";
        expect(ge(reader._endOfStreamSeen, 1UZ));
    };
};

const boost::ut::suite<"RepeatedTags"> _RepeatedTags = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::tag;
    using namespace gr::testing;

    auto runTest = []<auto srcType>(bool verbose = true) {
        gr::Size_t         n_samples = 30U;
        Graph              testGraph;
        const property_map srcParameter = {{"n_samples_max", n_samples}, {"name", "TagSource"}, {"verbose_console", true && verbose}, {"repeat_tags", true}};
        auto&              src          = testGraph.emplaceBlock<TagSource<float, srcType>>(srcParameter);
        src._tags                       = {
            {2, {{SAMPLE_RATE.key(), 2.f}}}, //
            {3, {{SAMPLE_RATE.key(), 3.f}}}, //
            {5, {{SAMPLE_RATE.key(), 5.f}}}, //
            {8, {{SAMPLE_RATE.key(), 8.f}}}  //
        };

        auto& monitorOne = testGraph.emplaceBlock<TagMonitor<float, ProcessFunction::USE_PROCESS_ONE>>({{"name", "TagMonitorOne"}, {"n_samples_expected", n_samples}, {"verbose_console", false && verbose}});
        auto& sinkOne    = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"name", "TagSinkOne"}, {"n_samples_expected", n_samples}, {"verbose_console", false && verbose}});

        // src -> monitorOne -> sinkOne
        expect(testGraph.connect<"out", "in">(src, monitorOne).has_value());
        expect(testGraph.connect<"out", "in">(monitorOne, sinkOne).has_value());

        gr::scheduler::Simple<> sched;
        if (auto ret = sched.exchange(std::move(testGraph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        expect(sched.runAndWait().has_value());

        expect(eq(src._tags.size(), 4UZ));
        expect(eq(monitorOne._tags.size(), 13UZ));
        expect(eq(sinkOne._tags.size(), 13UZ));
        for (std::size_t i = 0; i < monitorOne._tags.size(); i++) {
            expect(monitorOne._tags[i].map.find_value(SAMPLE_RATE.key()).value() == src._tags[i % src._tags.size()].map.find_value(SAMPLE_RATE.key()).value());
            expect(sinkOne._tags[i].map.find_value(SAMPLE_RATE.key()).value() == src._tags[i % src._tags.size()].map.find_value(SAMPLE_RATE.key()).value());
        }
    };

    "TagSource<float, USE_PROCESS_BULK>"_test = [&runTest] { runTest.template operator()<ProcessFunction::USE_PROCESS_BULK>(true); };

    "TagSource<float, USE_PROCESS_ONE>"_test = [&runTest] { runTest.template operator()<ProcessFunction::USE_PROCESS_ONE>(true); };
};

template<typename T>
struct ForwardPropBlock : gr::Block<ForwardPropBlock<T>, gr::ForwardTagPropagation, gr::Resampling<10UZ, 10UZ, true>> {
    using Description = gr::Doc<"fixed-chunk block with forward tag propagation (chunk=10)">;
    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    GR_MAKE_REFLECTABLE(ForwardPropBlock, in, out);

    gr::work::Status processBulk(std::span<const T> input, std::span<T> output) noexcept {
        std::ranges::copy(input, output.begin());
        return gr::work::Status::OK;
    }
};

template<typename T>
struct MergePropBlock : gr::Block<MergePropBlock<T>, gr::MergeTagPropagation> {
    using Description = gr::Doc<"block with merge tag propagation (legacy mode)">;
    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    float sample_rate = 1000.f;

    GR_MAKE_REFLECTABLE(MergePropBlock, in, out, sample_rate);

    [[nodiscard]] constexpr auto processOne(T x) const noexcept { return x; }
};

template<typename T>
struct NoForwardBlock : gr::Block<NoForwardBlock<T>, gr::NoTagPropagation> {
    using Description = gr::Doc<"block that suppresses default tag forwarding">;
    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    float sample_rate = 1000.f;

    GR_MAKE_REFLECTABLE(NoForwardBlock, in, out, sample_rate);

    [[nodiscard]] constexpr auto processOne(T x) const noexcept { return x; }
};

template<typename T>
struct DelayedForwardBlock : gr::Block<DelayedForwardBlock<T>, gr::NoTagPropagation> {
    using Description                   = gr::Doc<"passes samples through and forwards each tag kDelay samples later">;
    static constexpr std::size_t kDelay = 3UZ;
    gr::PortIn<T>                in;
    gr::PortOut<T>               out;

    float sample_rate = 1000.f;

    GR_MAKE_REFLECTABLE(DelayedForwardBlock, in, out, sample_rate);

    gr::DelayedTags _pending;

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& input, gr::OutputSpanLike auto& output) {
        std::ranges::copy(input, output.begin());
        this->forwardDelayedTags(_pending, input, std::tie(output), output.size(), kDelay);
        return gr::work::Status::OK;
    }
};

template<typename T>
struct ProcessOnePublisher : gr::Block<ProcessOnePublisher<T>> {
    using Description = gr::Doc<"processOne block that publishes tags at selected samples">;
    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    GR_MAKE_REFLECTABLE(ProcessOnePublisher, in, out);

    std::size_t              _nSamples = 0;
    std::vector<std::size_t> publishAtSamples{};

    [[nodiscard]] T processOne(T x) noexcept {
        if (std::ranges::find(publishAtSamples, _nSamples) != publishAtSamples.end()) {
            const auto       key = std::format("published_at_{}", _nSamples);
            gr::property_map tag;
            tag.insert_or_assign(key, static_cast<std::uint64_t>(_nSamples));
            this->publishTag(tag);
        }
        _nSamples++;
        return x;
    }
};

template<typename T>
struct MergePropTwoInput : gr::Block<MergePropTwoInput<T>, gr::MergeTagPropagation> {
    using Description = gr::Doc<"two-input block with merge tag propagation">;
    gr::PortIn<T>  in0;
    gr::PortIn<T>  in1;
    gr::PortOut<T> out;

    GR_MAKE_REFLECTABLE(MergePropTwoInput, in0, in1, out);

    [[nodiscard]] gr::work::Status processBulk(std::span<const T> a, std::span<const T> b, std::span<T> o) noexcept {
        for (std::size_t i = 0; i < o.size(); ++i) {
            o[i] = a[i] + b[i];
        }
        return gr::work::Status::OK;
    }
};

template<typename T>
struct TwoInputAdder : gr::Block<TwoInputAdder<T>> {
    using Description = gr::Doc<"adds two inputs — for multi-port dedup testing">;
    gr::PortIn<T>  in0;
    gr::PortIn<T>  in1;
    gr::PortOut<T> out;

    GR_MAKE_REFLECTABLE(TwoInputAdder, in0, in1, out);

    [[nodiscard]] gr::work::Status processBulk(std::span<const T> a, std::span<const T> b, std::span<T> o) noexcept {
        for (std::size_t i = 0; i < o.size(); ++i) {
            o[i] = a[i] + b[i];
        }
        return gr::work::Status::OK;
    }
};

const boost::ut::suite<"CustomForwardTests"> _CustomForwardTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;

    "NoTagPropagation suppresses auto-forwarding"_test = [] {
        Graph testGraph;
        auto& src  = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(30)}, {"verbose_console", false}});
        src._tags  = {{0, tag::SAMPLE_RATE(42.f)}};
        auto& nofw = testGraph.emplaceBlock<NoForwardBlock<float>>();
        auto& sink = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", false}});

        expect(testGraph.connect<"out", "in">(src, nofw).has_value());
        expect(testGraph.connect<"out", "in">(nofw, sink).has_value());

        scheduler::Simple<> sched;
        expect(sched.exchange(std::move(testGraph)).has_value());
        expect(sched.runAndWait().has_value());

        expect(eq(sink._tags.size(), 0UZ)) << "no tags forwarded with NoTagPropagation";
        expect(eq(nofw.sample_rate, 42.f)) << "settings still auto-updated from input tags";
    };

    "delayed forwarding moves a tag delay samples on, filtered and substituted as by default"_test = [] {
        Graph testGraph;
        auto& src     = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(30)}, {"verbose_console", false}});
        src._tags     = {{0, tag::SAMPLE_RATE(42.f)}, {2, {{std::string(tag::TRIGGER_NAME.key()), "moved"}, {"local_key", 1.f}}}, {28, {{std::string(tag::TRIGGER_NAME.key()), "late"}}}};
        auto& delayed = testGraph.emplaceBlock<DelayedForwardBlock<float>>({{"sample_rate", 500.f}});
        auto& sink    = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", false}});
        expect(testGraph.connect<"out", "in">(src, delayed).has_value());
        expect(testGraph.connect<"out", "in">(delayed, sink).has_value());
        scheduler::Simple<> sched;
        expect(sched.exchange(std::move(testGraph)).has_value());
        expect(sched.runAndWait().has_value());

        const auto tagAt = [&sink](std::size_t index, std::string_view key) { return std::ranges::find_if(sink._tags, [&](const auto& tag) { return tag.index == index && tag.map.contains(std::string(key)); }); };
        expect(tagAt(5UZ, tag::TRIGGER_NAME.key()) != sink._tags.end()) << "the tag on input 2 lands on output 2 + 3";
        expect(tagAt(5UZ, "local_key") != sink._tags.end()) << "a key without gr: goes on with the rest of its tag";
        const auto rate = tagAt(3UZ, tag::SAMPLE_RATE.key());
        expect(fatal(rate != sink._tags.end())) << "the sample_rate tag on input 0 lands on output 3";
        expect(eq(rate->map.template value_or<float>(std::string(tag::SAMPLE_RATE.key()), 0.f), 500.f)) << "a key naming a block setting carries the block's value";
        expect(std::ranges::none_of(sink._tags, [](const auto& tag) { return tag.index >= 30UZ; })) << "a tag within the delay of the end has no sample left to mark";
    };

    "processOne publishTag with multiple tags"_test = [] {
        Graph testGraph;
        auto& src            = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(30)}, {"verbose_console", false}});
        auto& pub            = testGraph.emplaceBlock<ProcessOnePublisher<float>>();
        pub.publishAtSamples = {3UZ, 7UZ, 11UZ};
        auto& sink           = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", false}});

        expect(testGraph.connect<"out", "in">(src, pub).has_value());
        expect(testGraph.connect<"out", "in">(pub, sink).has_value());

        scheduler::Simple<> sched;
        expect(sched.exchange(std::move(testGraph)).has_value());
        expect(sched.runAndWait().has_value());

        for (const auto index : pub.publishAtSamples) {
            const auto expectedKey = std::format("published_at_{}", index);
            const auto found       = std::ranges::find_if(sink._tags, [&expectedKey](const auto& tag) { return tag.map.contains(expectedKey); });

            expect(found != sink._tags.end()) << std::format("processOne tag at sample {} must be forwarded", index);
            if (found != sink._tags.end()) {
                expect(eq(found->index, index)) << std::format("tag must be at sample {}", index);

                auto                     forbiddenKeys = pub.publishAtSamples | std::views::filter([index](auto i) { return i != index; }) | std::views::transform([](auto i) { return std::format("published_at_{}", i); });
                std::vector<std::string> staleKeys;
                std::ranges::copy_if(forbiddenKeys, std::back_inserter(staleKeys), [&found](const auto& key) { return found->map.contains(key); });

                expect(staleKeys.empty()) << std::format("tag at sample {} contains stale keys: {}", index, gr::join(staleKeys, ", "));
            }
        }
    };

    "processOne mergedInputTag receives tag and clears flag"_test = [] {
        Graph testGraph;
        auto& src     = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(10)}, {"verbose_console", false}});
        src._tags     = {{3, tag::SIGNAL_NAME("test_signal"s)}};
        auto& monitor = testGraph.emplaceBlock<TagMonitor<float, ProcessFunction::USE_PROCESS_ONE>>({{"verbose_console", false}});
        auto& sink    = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", false}});

        expect(testGraph.connect<"out", "in">(src, monitor).has_value());
        expect(testGraph.connect<"out", "in">(monitor, sink).has_value());

        scheduler::Simple<> sched;
        expect(sched.exchange(std::move(testGraph)).has_value());
        expect(sched.runAndWait().has_value());

        expect(eq(monitor._tags.size(), 1UZ)) << "exactly one tag received";
        expect(eq(monitor._tags[0].index, 3UZ)) << "tag at correct position";
        expect(eq(monitor.signal_name, "test_signal"_spmr)) << "settings auto-updated";
    };

    "init-time settings forwarded as tag"_test = [] {
        Graph testGraph;
        auto& src  = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(10)}, {"verbose_console", false}});
        auto& gain = testGraph.emplaceBlock<AutoForwardParametersBlock<float>>({{"sample_rate", 48000.f}, {"signal_name", "init_test"}});
        auto& sink = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", false}});

        expect(testGraph.connect<"out", "in">(src, gain).has_value());
        expect(testGraph.connect<"out", "in">(gain, sink).has_value());

        scheduler::Simple<> sched;
        expect(sched.exchange(std::move(testGraph)).has_value());
        expect(sched.runAndWait().has_value());

        expect(ge(sink._tags.size(), 1UZ)) << "init-time forward tag expected";
        bool hasSampleRate = false;
        for (const auto& tag : sink._tags) {
            if (auto it = tag.map.find(tag::SAMPLE_RATE.key()); it != tag.map.end()) {
                expect(*(*it).second.get_if<float>() == 48000.f);
                hasSampleRate = true;
            }
        }
        expect(hasSampleRate) << "sample_rate from init must be forwarded";
    };

    "forward policy carries unconsumed tags to next chunk"_test = [] {
        // tags at 0, 4, 5 with processOne monitor: tag at 0 consumed immediately,
        // tags at 4, 5 carry forward and appear in subsequent chunks at position 0
        Graph testGraph;
        auto& src     = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(10)}, {"verbose_console", false}});
        src._tags     = {{0, {{"key", "at_0"}}}, {4, {{"key", "at_4"}}}, {5, {{"key", "at_5"}}}};
        auto& monitor = testGraph.emplaceBlock<TagMonitor<float, ProcessFunction::USE_PROCESS_ONE>>({{"verbose_console", false}});
        auto& sink    = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", false}});

        expect(testGraph.connect<"out", "in">(src, monitor).has_value());
        expect(testGraph.connect<"out", "in">(monitor, sink).has_value());

        scheduler::Simple<> sched;
        expect(sched.exchange(std::move(testGraph)).has_value());
        expect(sched.runAndWait().has_value());

        // monitor (processOne) sees each tag at its position via limitByFirstTag
        expect(eq(monitor._tags.size(), 3UZ)) << "all 3 tags received by processOne monitor";
        expect(eq(monitor._tags[0].index, 0UZ));
        expect(eq(monitor._tags[1].index, 4UZ));
        expect(eq(monitor._tags[2].index, 5UZ));
    };

    "multiple tags on same sample are all visible"_test = [] {
        Graph testGraph;
        auto& src = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(10)}, {"verbose_console", false}});
        src._tags = {
            {3, {{"key_a", "first_tag"}}}, {3, {{"key_b", "second_tag"}}} // same sample index, different tag
        };
        auto& sink = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", false}});

        expect(testGraph.connect<"out", "in">(src, sink).has_value());

        scheduler::Simple<> sched;
        expect(sched.exchange(std::move(testGraph)).has_value());
        expect(sched.runAndWait().has_value());

        // both tags at sample 3 must be visible — no merging
        std::size_t tagsAt3 = 0;
        for (const auto& tag : sink._tags) {
            if (tag.index == 3) {
                tagsAt3++;
            }
        }
        expect(ge(tagsAt3, 2UZ)) << "both tags at sample 3 must be preserved (no implicit merge)";
    };

    "a processOne block sees several tags on one sample as one tag, the later winning a shared key"_test = [] {
        Graph testGraph;
        auto& src  = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(10)}, {"verbose_console", false}});
        src._tags  = {{3, {{"key_a", "first"}, {"shared", "first"}}}, {3, {{"key_b", "second"}, {"shared", "second"}}}};
        auto& sink = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"verbose_console", false}});
        expect(testGraph.connect<"out", "in">(src, sink).has_value());

        scheduler::Simple<> sched;
        expect(sched.exchange(std::move(testGraph)).has_value());
        expect(sched.runAndWait().has_value());

        std::vector<property_map> at3;
        for (const auto& tag : sink._tags) {
            if (tag.index == 3) {
                at3.push_back(tag.map);
            }
        }
        expect(eq(at3.size(), 1UZ)) << fatal << "one tag per sample in processOne, none lost";
        expect(eq(at3[0].value_or<std::string>("key_a", std::string()), std::string("first")));
        expect(eq(at3[0].value_or<std::string>("key_b", std::string()), std::string("second")));
        expect(eq(at3[0].value_or<std::string>("shared", std::string()), std::string("second"))) << "the later tag wins a key both carry";
    };

    "processOne handles multi-sample chunk with tag at sample 0 only"_test = [] {
        // after limitByFirstTag removal: processOne processes multi-sample chunks
        // tag at position 0 is visible via mergedInputTag() for the first sample only
        Graph testGraph;
        auto& src     = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(30)}, {"verbose_console", false}});
        src._tags     = {{0, tag::SIGNAL_NAME("test"s)}}; // single tag at start
        auto& monitor = testGraph.emplaceBlock<TagMonitor<float, ProcessFunction::USE_PROCESS_ONE>>({{"verbose_console", false}});
        auto& sink    = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", false}});

        expect(testGraph.connect<"out", "in">(src, monitor).has_value());
        expect(testGraph.connect<"out", "in">(monitor, sink).has_value());

        scheduler::Simple<> sched;
        expect(sched.exchange(std::move(testGraph)).has_value());
        expect(sched.runAndWait().has_value());

        expect(eq(monitor._tags.size(), 1UZ)) << "exactly one tag received (not duplicated across samples)";
        expect(eq(monitor._tags[0].index, 0UZ)) << "tag at sample 0";
        expect(eq(monitor._nSamplesProduced, gr::Size_t(30))) << "all samples processed";
        expect(eq(monitor.signal_name, "test"_spmr)) << "settings auto-updated";
    };

    "value substitution in forwarding replaces with block-current value"_test = [] {
        Graph testGraph;
        auto& src  = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(10)}, {"verbose_console", false}});
        src._tags  = {{0, tag::SAMPLE_RATE(42.f)}};
        auto& fwd  = testGraph.emplaceBlock<AutoForwardParametersBlock<float>>(); // default sample_rate, auto-updateable
        auto& sink = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", false}});

        expect(testGraph.connect<"out", "in">(src, fwd).has_value());
        expect(testGraph.connect<"out", "in">(fwd, sink).has_value());

        scheduler::Simple<> sched;
        expect(sched.exchange(std::move(testGraph)).has_value());
        expect(sched.runAndWait().has_value());

        // sample_rate auto-updated from input tag, then forwarded with block-current value
        expect(eq(fwd.sample_rate, 42.f)) << "block setting auto-updated from input tag";
        expect(eq(sink.sample_rate, 42.f)) << "downstream setting updated from forwarded tag";
    };

    "multi-input dedup removes identical tags from fan-out"_test = [] {
        // src → in0 of adder, src → in1 of adder → both ports see the same tag → dedup
        Graph testGraph;
        auto& src  = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(10)}, {"verbose_console", false}});
        src._tags  = {{0, {{tag::SAMPLE_RATE, 42.f}, {"unique_key", "A"}}}};
        auto& add  = testGraph.emplaceBlock<TwoInputAdder<float>>();
        auto& sink = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", false}});

        expect(testGraph.connect<"out", "in0">(src, add).has_value());
        expect(testGraph.connect<"out", "in1">(src, add).has_value());
        expect(testGraph.connect<"out", "in">(add, sink).has_value());

        scheduler::Simple<> sched;
        expect(sched.exchange(std::move(testGraph)).has_value());
        expect(sched.runAndWait().has_value());

        // the identical tag from both ports should be deduped → forwarded once
        std::size_t srTags = 0;
        for (const auto& tag : sink._tags) {
            if (tag.map.contains(tag::SAMPLE_RATE.key())) {
                srTags++;
            }
        }
        expect(eq(srTags, 1UZ)) << "identical tag from fan-out forwarded once (deduped)";
        expect(eq(sink.sample_rate, 42.f)) << "settings auto-updated";
    };

    "ForwardTagPropagation does not break chunks at tags"_test = [] {
        // verify that ForwardTagPropagation doesn't break chunks at tag positions
        // (tags within the chunk carry forward instead of splitting the chunk)
        Graph testGraph;
        auto& src  = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(30)}, {"verbose_console", false}});
        src._tags  = {{0, tag::SAMPLE_RATE(1.f)}, {5, tag::SAMPLE_RATE(5.f)}};
        auto& fwd  = testGraph.emplaceBlock<ForwardPropBlock<float>>();
        auto& sink = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", false}});

        expect(testGraph.connect<"out", "in">(src, fwd).has_value());
        expect(testGraph.connect<"out", "in">(fwd, sink).has_value());

        scheduler::Simple<> sched;
        expect(sched.exchange(std::move(testGraph)).has_value());
        expect(sched.runAndWait().has_value());

        expect(eq(sink._nSamplesProduced, gr::Size_t(30))) << "all samples processed in fixed chunks";
        expect(ge(sink._tags.size(), 1UZ)) << "tag at position 0 forwarded";
    };

    "MergeTagPropagation merges all tags into one"_test = [] {
        Graph testGraph;
        auto& src   = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(10)}, {"verbose_console", false}});
        src._tags   = {{0, tag::SAMPLE_RATE(42.f)}, {0, tag::SIGNAL_NAME("merged_test"s)}};
        auto& merge = testGraph.emplaceBlock<MergePropBlock<float>>();
        auto& sink  = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", false}});

        expect(testGraph.connect<"out", "in">(src, merge).has_value());
        expect(testGraph.connect<"out", "in">(merge, sink).has_value());

        scheduler::Simple<> sched;
        expect(sched.exchange(std::move(testGraph)).has_value());
        expect(sched.runAndWait().has_value());

        // MergeTagPropagation merges both input tags (sample_rate + signal_name) into one output tag
        // check that the sink received merged content rather than 2 separate tags
        bool foundMerged = false;
        for (const auto& tag : sink._tags) {
            if (tag.map.contains(tag::SAMPLE_RATE.key()) && tag.map.contains(tag::SIGNAL_NAME.key())) {
                foundMerged = true;
            }
        }
        expect(foundMerged) << "merged tag must contain both sample_rate and signal_name";
    };

    "MergeTagPropagation overlapping keys use last value"_test = [] {
        Graph testGraph;
        auto& src0  = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(10)}, {"verbose_console", false}});
        auto& src1  = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(10)}, {"verbose_console", false}});
        src0._tags  = {{0, tag::SIGNAL_NAME("first"s)}, {0, tag::SIGNAL_NAME("second"s)}};
        src1._tags  = {{0, tag::SIGNAL_NAME("third"s)}, {0, tag::SIGNAL_NAME("fourth"s)}};
        auto& merge = testGraph.emplaceBlock<MergePropTwoInput<float>>();
        auto& sink  = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", false}});

        expect(testGraph.connect<"out", "in0">(src0, merge).has_value());
        expect(testGraph.connect<"out", "in1">(src1, merge).has_value());
        expect(testGraph.connect<"out", "in">(merge, sink).has_value());

        scheduler::Simple<> sched;
        expect(sched.exchange(std::move(testGraph)).has_value());
        expect(sched.runAndWait().has_value());

        const auto signalNameKey = tag::SIGNAL_NAME.key();
        const auto found         = std::ranges::find_if(sink._tags, [&signalNameKey](const auto& tag) { return tag.map.contains(signalNameKey); });

        expect(found != sink._tags.end());
        if (found != sink._tags.end()) {
            expect(eq(found->index, 0UZ));
            expect(eq(found->map.value_or<std::string>(signalNameKey, std::string{}), std::string{"fourth"})) << "last overlapping key wins";
        }
    };

    "MergeTagPropagation with multi-input deduplicates and merges"_test = [] {
        // fan-out: same source → both inputs of merge block
        // identical tags from fan-out should be deduped, then merged into one output tag
        Graph testGraph;
        auto& src   = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(10)}, {"verbose_console", false}});
        src._tags   = {{0, {tag::SAMPLE_RATE(42.f), tag::SIGNAL_NAME("multi_merge"s)}}};
        auto& merge = testGraph.emplaceBlock<MergePropTwoInput<float>>();
        auto& sink  = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", false}});

        expect(testGraph.connect<"out", "in0">(src, merge).has_value());
        expect(testGraph.connect<"out", "in1">(src, merge).has_value());
        expect(testGraph.connect<"out", "in">(merge, sink).has_value());

        scheduler::Simple<> sched;
        expect(sched.exchange(std::move(testGraph)).has_value());
        expect(sched.runAndWait().has_value());

        // fan-out dedup + merge: identical tags from both ports → ONE merged output tag
        bool foundMerged = false;
        for (const auto& tag : sink._tags) {
            if (tag.map.contains(tag::SAMPLE_RATE.key()) && tag.map.contains(tag::SIGNAL_NAME.key())) {
                foundMerged = true;
            }
        }
        expect(foundMerged) << "merged tag with both keys from deduped fan-out";
    };

    "ForwardTagPropagation carry-forward tag arrives in next chunk"_test = [] {
        // tag at position 5 within a 10-sample chunk should carry forward to the next chunk
        // use a longer stream (50 samples) to ensure multiple chunks
        Graph testGraph;
        auto& src  = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(50)}, {"verbose_console", false}});
        src._tags  = {{0, tag::SAMPLE_RATE(1.f)}, {5, tag::SAMPLE_RATE(5.f)}};
        auto& fwd  = testGraph.emplaceBlock<ForwardPropBlock<float>>();
        auto& sink = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", false}});

        expect(testGraph.connect<"out", "in">(src, fwd).has_value());
        expect(testGraph.connect<"out", "in">(fwd, sink).has_value());

        scheduler::Simple<> sched;
        expect(sched.exchange(std::move(testGraph)).has_value());
        expect(sched.runAndWait().has_value());

        expect(eq(sink._nSamplesProduced, gr::Size_t(50))) << "all samples processed";
        expect(ge(sink._tags.size(), 1UZ)) << "at least tag at position 0 forwarded";
    };
};

const boost::ut::suite<"SettingsTagInteraction"> _SettingsTagInteraction = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;

    "autoUpdate does not clear staged params from prior tag"_test = [] {
        // regression: a non-matching tag after a matching tag wiped _stagedParameters
        // graph: src(sample_rate=42, custom_key=X) → sink — two tags at position 0
        Graph testGraph;
        auto& src = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(10)}, {"verbose_console", false}});
        src._tags = {
            {0, tag::SAMPLE_RATE(42.f)},                    // matching: auto-updates sample_rate
            {0, {{"custom_nonexistent_key", "some_value"}}} // non-matching: must NOT wipe staged sample_rate
        };
        auto& sink = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", false}});

        expect(testGraph.connect<"out", "in">(src, sink).has_value());

        scheduler::Simple<> sched;
        expect(sched.exchange(std::move(testGraph)).has_value());
        expect(sched.runAndWait().has_value());

        expect(eq(sink.sample_rate, 42.f)) << "sample_rate must be auto-updated despite non-matching tag following";
    };

    "init-time forward params survive no-data early return"_test = [] {
        // regression: pendingForwardParams lost when resampledIn==0 (block has no input data yet)
        // graph: src(sample_rate=42) → delay → sink
        // delay has no data in first work call → pendingForwardParams must be re-staged
        Graph testGraph;
        auto& src   = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(100)}, {"sample_rate", 42.f}, {"verbose_console", false}});
        auto& delay = testGraph.emplaceBlock<testing::Sleep<float>>({{"delay_ms", 0.f}});
        auto& sink  = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", false}});

        expect(testGraph.connect<"out", "in">(src, delay).has_value());
        expect(testGraph.connect<"out", "in">(delay, sink).has_value());

        scheduler::Simple<> sched;
        expect(sched.exchange(std::move(testGraph)).has_value());
        expect(sched.runAndWait().has_value());

        expect(eq(sink.sample_rate, 42.f)) << "sample_rate must propagate through delay despite initial no-data work calls";
    };

    "settings auto-update with mixed metadata and trigger tags"_test = [] {
        // regression: trigger tags with context key caused activateContext to overwrite auto-updated values
        // simplified version: two tags at same position — one with sample_rate, one with trigger keys
        Graph testGraph;
        auto& src  = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(30)}, {"verbose_console", false}});
        src._tags  = {{0, {tag::SAMPLE_RATE(42.f), tag::SIGNAL_NAME("test"s)}}, {0, {tag::TRIGGER_NAME("evt"s), tag::TRIGGER_TIME(std::uint64_t(123))}}};
        auto& sink = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", false}});

        expect(testGraph.connect<"out", "in">(src, sink).has_value());

        scheduler::Simple<> sched;
        expect(sched.exchange(std::move(testGraph)).has_value());
        expect(sched.runAndWait().has_value());

        expect(eq(sink.sample_rate, 42.f)) << "sample_rate auto-updated despite trigger tag at same position";
        expect(eq(sink.signal_name, "test"_spmr)) << "signal_name auto-updated";
    };
};

/** every fourth sample: a decimator resamples the signal, a framer only packs it (`gr::Framing`) */
template<typename T, typename... TFraming>
struct EveryFourth : gr::Block<EveryFourth<T, TFraming...>, gr::Resampling<4UZ, 1UZ, true>, TFraming...> {
    gr::PortIn<T>  in;
    gr::PortOut<T> out;
    GR_MAKE_REFLECTABLE(EveryFourth, in, out);

    [[nodiscard]] gr::work::Status processBulk(std::span<const T> input, std::span<T> output) noexcept {
        for (std::size_t k = 0UZ; k < output.size(); ++k) {
            output[k] = input[4UZ * k];
        }
        return gr::work::Status::OK;
    }
};

/** each sample four times, unpacking what `EveryFourth<T, gr::Framing>` packed */
template<typename T>
struct Unframe : gr::Block<Unframe<T>, gr::Resampling<1UZ, 4UZ, true>, gr::Framing> {
    gr::PortIn<T>  in;
    gr::PortOut<T> out;
    GR_MAKE_REFLECTABLE(Unframe, in, out);

    [[nodiscard]] gr::work::Status processBulk(std::span<const T> input, std::span<T> output) noexcept {
        for (std::size_t k = 0UZ; k < output.size(); ++k) {
            output[k] = input[k / 4UZ];
        }
        return gr::work::Status::OK;
    }
};

using BulkSource = gr::testing::TagSource<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>;
using BulkSink   = gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>;

struct SourceAndSink {
    BulkSource& source;
    BulkSink&   sink;
};

[[nodiscard]] SourceAndSink emplaceSourceAndSink(gr::testing::GraphFixture<>& fixture, std::vector<gr::testing::OwningTag> tags) {
    auto& source = fixture.emplace<BulkSource>({{"n_samples_max", gr::Size_t(64)}, {"mark_tag", false}});
    source._tags = std::move(tags);
    return {source, fixture.emplace<BulkSink>({{"log_tags", true}, {"log_samples", false}})};
}

const boost::ut::suite<"FrameRate"> _frameRate = [] {
    using namespace boost::ut;
    using namespace gr;

    const auto rates = []<typename... TChain>(std::tuple<TChain...>, const pmt::Value& inputRate = pmt::Value(1000.f), bool* sampleRateIsFloat = nullptr) {
        gr::testing::GraphFixture<> fixture;
        auto [source, sink] = emplaceSourceAndSink(fixture, {gr::testing::OwningTag(0UZ, {{std::string(tag::SAMPLE_RATE.key()), inputRate}})});
        const auto chain    = [&]<typename TBlock>() -> auto& { return fixture.template emplace<TBlock>(); };
        auto& first         = chain.template operator()<std::tuple_element_t<0, std::tuple<TChain...>>>();
        expect(fixture.connect<"out", "in">(source, first).has_value());
        if constexpr (sizeof...(TChain) == 1UZ) {
            expect(fixture.connect<"out", "in">(first, sink).has_value());
        } else {
            auto& second = chain.template operator()<std::tuple_element_t<1, std::tuple<TChain...>>>();
            expect(fixture.connect<"out", "in">(first, second).has_value());
            expect(fixture.connect<"out", "in">(second, sink).has_value());
        }
        expect(fixture.run().has_value());
        const auto rateTag = std::ranges::find_if(sink._tags, [](const auto& t) { return t.map.contains(tag::SAMPLE_RATE.key()); });
        expect(rateTag != sink._tags.end()) << "a sample_rate arrives";
        const float sampleRate = rateTag == sink._tags.end() ? 0.f : rateTag->map.template value_or<float>(tag::SAMPLE_RATE.key(), 0.f);
        if (sampleRateIsFloat != nullptr && rateTag != sink._tags.end()) {
            *sampleRateIsFloat = rateTag->map.find(tag::SAMPLE_RATE.key())->second.template get_if<float>() != nullptr;
        }
        const float frameRate = rateTag == sink._tags.end() ? 0.f : rateTag->map.template value_or<float>(tag::FRAME_RATE.key(), 0.f);
        return std::pair{sampleRate, frameRate};
    };

    "a framer keeps the sample rate and adds the frame rate by its chunk ratio"_test = [&rates] {
        const auto [sampleRate, frameRate] = rates(std::tuple<EveryFourth<float, gr::Framing>>{});
        expect(eq(sampleRate, 1000.f));
        expect(eq(frameRate, 250.f));
    };

    "a decimator rescales the sample rate and adds no frame rate"_test = [&rates] {
        const auto [sampleRate, frameRate] = rates(std::tuple<EveryFourth<float>>{});
        expect(eq(sampleRate, 250.f));
        expect(eq(frameRate, 0.f)) << "absent";
    };

    "a decimator rescales a sample rate of any numeric type and forwards it as float"_test = [&rates] {
        for (const pmt::Value& inputRate : {pmt::Value(1000.), pmt::Value(std::int32_t{1000}), pmt::Value(std::uint64_t{1000})}) {
            bool isFloat                       = false;
            const auto [sampleRate, frameRate] = rates(std::tuple<EveryFourth<float>>{}, inputRate, &isFloat);
            expect(eq(sampleRate, 250.f));
            expect(isFloat) << "the rescaled rate is the canonical float";
            expect(eq(frameRate, 0.f)) << "absent";
        }
    };

    "a sample rate that is not a number passes a decimator unchanged"_test = [&rates] {
        const auto [sampleRate, frameRate] = rates(std::tuple<EveryFourth<float>>{}, pmt::Value("fast"));
        expect(eq(sampleRate, 0.f)) << "a string is no float; nothing is rescaled";
        expect(eq(frameRate, 0.f));
    };

    "framing and unframing again give the stream back: the sample rate, no frame rate"_test = [&rates] {
        const auto [sampleRate, frameRate] = rates(std::tuple<EveryFourth<float, gr::Framing>, Unframe<float>>{});
        expect(eq(sampleRate, 1000.f));
        expect(eq(frameRate, 0.f)) << "a frame rate equal to the sample rate is dropped";
    };
};

const boost::ut::suite<"AxisKeys"> _axisKeys = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace std::string_literals;

    "a tensor's axes, per dimension, pass a block as they came"_test = [] {
        const property_map          axes{{std::string(tag::AXIS_ORIGIN.key()), Tensor<double>{-1.5, 0.0}}, {std::string(tag::AXIS_STEP.key()), Tensor<double>{0.25, 2.0}}, //
                     {std::string(tag::AXIS_PERIODIC.key()), Tensor<bool>(gr::data_from, {false, true})},                                                                  //
                     {std::string(tag::AXIS_UNIT.key()), Tensor<pmt::Value>{pmt::Value{"rad"}, pmt::Value{"rad"}}},                                                        //
                     {std::string(tag::AXIS_NAME.key()), Tensor<pmt::Value>{pmt::Value{"elevation"}, pmt::Value{"azimuth"}}}};
        gr::testing::GraphFixture<> fixture;
        auto [source, sink] = emplaceSourceAndSink(fixture, {gr::testing::OwningTag(0UZ, axes)});
        auto& block         = fixture.emplace<DecimatorForward<float>>();
        expect(fixture.connect<"out", "in">(source, block).has_value());
        expect(fixture.connect<"out", "in">(block, sink).has_value());
        expect(fixture.run().has_value());
        const auto axesTag = std::ranges::find_if(sink._tags, [](const auto& t) { return t.map.contains(tag::AXIS_STEP.key()); });
        expect(axesTag != sink._tags.end()) << "the axes arrive";
        if (axesTag == sink._tags.end()) {
            return;
        }
        const auto& map  = axesTag->map;
        const auto  step = map.template value_or<Tensor<double>>(tag::AXIS_STEP.key(), Tensor<double>{});
        const auto  wrap = map.template value_or<Tensor<bool>>(tag::AXIS_PERIODIC.key(), Tensor<bool>{});
        const auto  name = map.template value_or<Tensor<pmt::Value>>(tag::AXIS_NAME.key(), Tensor<pmt::Value>{});
        expect(map.template value_or<Tensor<double>>(tag::AXIS_ORIGIN.key(), Tensor<double>{}) == Tensor<double>{-1.5, 0.0});
        expect(step == Tensor<double>{0.25, 2.0});
        expect(wrap.size() == 2UZ && !wrap[0] && wrap[1]);
        expect(name.size() == 2UZ && name[1].value_or(std::string{}) == "azimuth"s);
        expect(map.template value_or<Tensor<pmt::Value>>(tag::AXIS_UNIT.key(), Tensor<pmt::Value>{}).size() == 2UZ);
    };
};

template<typename T>
struct StridedEqualChunks : gr::Block<StridedEqualChunks<T>, gr::Resampling<8UZ, 8UZ, true>, gr::Stride<4UZ, true>> {
    gr::PortIn<T>  in;
    gr::PortOut<T> out;
    GR_MAKE_REFLECTABLE(StridedEqualChunks, in, out);

    [[nodiscard]] gr::work::Status processBulk(std::span<const T> input, std::span<T> output) noexcept {
        std::copy_n(input.begin(), output.size(), output.begin());
        return gr::work::Status::OK;
    }
};

template<typename T>
struct UnstridedEqualChunks : gr::Block<UnstridedEqualChunks<T>, gr::Resampling<8UZ, 8UZ, true>> {
    gr::PortIn<T>  in;
    gr::PortOut<T> out;
    GR_MAKE_REFLECTABLE(UnstridedEqualChunks, in, out);

    [[nodiscard]] gr::work::Status processBulk(std::span<const T> input, std::span<T> output) noexcept {
        std::copy_n(input.begin(), output.size(), output.begin());
        return gr::work::Status::OK;
    }
};

const boost::ut::suite<"HopRule"> _hopRule = [] {
    using namespace boost::ut;
    using namespace gr;

    const auto forwardedSampleRate = []<typename TBlock>() {
        gr::testing::GraphFixture<> fixture;
        auto [source, sink] = emplaceSourceAndSink(fixture, {gr::testing::OwningTag(0UZ, {{std::string(tag::SAMPLE_RATE.key()), 1000.f}})});
        auto& block         = fixture.template emplace<TBlock>();
        expect(fixture.connect<"out", "in">(source, block).has_value());
        expect(fixture.connect<"out", "in">(block, sink).has_value());
        expect(fixture.run().has_value());
        const auto rateTag = std::ranges::find_if(sink._tags, [](const auto& t) { return t.map.contains(tag::SAMPLE_RATE.key()); });
        expect(rateTag != sink._tags.end()) << "a sample_rate arrives";
        return rateTag == sink._tags.end() ? 0.f : rateTag->map.template value_or<float>(tag::SAMPLE_RATE.key(), 0.f);
    };

    "a strided block with equal chunks of 8 and stride 4 advances 4 samples per 8 outputs: the bare sample_rate doubles"_test = [&forwardedSampleRate] { expect(eq(forwardedSampleRate.template operator()<StridedEqualChunks<float>>(), 2000.f)) << "8 / 4 * 1000"; };

    "an unstrided block with equal chunks passes the sample_rate unchanged"_test = [&forwardedSampleRate] { expect(eq(forwardedSampleRate.template operator()<UnstridedEqualChunks<float>>(), 1000.f)); };
};

const boost::ut::suite<"WrongTypedSettingTags"> _wrongTypedSettingTags = [] {
    using namespace boost::ut;
    using namespace gr;

    struct Outcome {
        float sampleRateApplied{0.f};
        float sampleRateDownstream{0.f};
        bool  everyForwardedRateIsFloat{true};
    };
    const auto run = [](pmt::Value sampleRateInTag) {
        gr::testing::GraphFixture<> fixture;
        auto [source, sink] = emplaceSourceAndSink(fixture, {gr::testing::OwningTag(0UZ, {{std::string(tag::SAMPLE_RATE.key()), 1000.f}}), gr::testing::OwningTag(1UZ, {{std::string(tag::SAMPLE_RATE.key()), std::move(sampleRateInTag)}})});
        auto& block         = fixture.emplace<AutoForwardParametersBlock<float>>();
        expect(fixture.connect<"out", "in">(source, block).has_value());
        expect(fixture.connect<"out", "in">(block, sink).has_value());
        const auto ran = fixture.run();
        expect(ran.has_value()) << (ran ? std::string{} : std::format("{}", ran.error()));
        Outcome outcome{.sampleRateApplied = block.sample_rate, .sampleRateDownstream = sink.sample_rate};
        for (const auto& received : sink._tags) {
            if (const auto rate = received.map.find(tag::SAMPLE_RATE.key()); rate != received.map.end()) {
                outcome.everyForwardedRateIsFloat = outcome.everyForwardedRateIsFloat && (*rate).second.template holds<float>();
            }
        }
        return outcome;
    };

    "a double 48000 for the float sample_rate is applied as 48000.f and forwarded as a float"_test = [&run] {
        const Outcome outcome = run(pmt::Value(48000.0));
        expect(eq(outcome.sampleRateApplied, 48000.f));
        expect(eq(outcome.sampleRateDownstream, 48000.f));
        expect(outcome.everyForwardedRateIsFloat);
    };

    "an int 44100 for the float sample_rate converts to 44100.f"_test = [&run] {
        const Outcome outcome = run(pmt::Value(44100));
        expect(eq(outcome.sampleRateApplied, 44100.f));
        expect(eq(outcome.sampleRateDownstream, 44100.f));
        expect(outcome.everyForwardedRateIsFloat);
    };

    "a string for the float sample_rate is dropped and the old 1000.f is forwarded"_test = [&run] {
        const Outcome outcome = run(pmt::Value("fast"));
        expect(eq(outcome.sampleRateApplied, 1000.f));
        expect(eq(outcome.sampleRateDownstream, 1000.f));
        expect(outcome.everyForwardedRateIsFloat);
    };
};

const boost::ut::suite<"EndOfStreamMatcher"> _endOfStreamMatcher = [] {
    using namespace boost::ut;
    using namespace gr;

    const auto tagWithInlineEndOfStream = [](bool endOfStream, std::vector<std::byte>& storage) {
        const gr::property_map canonical{{std::string(tag::END_OF_STREAM.key()), endOfStream}};
        const auto             blob = canonical.blob();
        storage.assign(blob.begin(), blob.end());
        auto* entry = reinterpret_cast<gr::pmt::PackedEntry*>(storage.data() + sizeof(gr::pmt::Header));
        expect(eq(entry->keyId, gr::pmt::keys::idOf<"gr:end_of_stream">)) << "stored by its canonical id before the patch";
        const std::string_view key = tag::END_OF_STREAM.key();
        entry->keyId               = gr::pmt::keys::kInlineKeyId;
        entry->inlineKey[0]        = static_cast<char>(key.size());
        std::copy(key.begin(), key.end(), entry->inlineKey + 1);
        return gr::makeStoredTag(0UZ, storage);
    };

    "a foreign blob storing gr:end_of_stream inline as a string ends the stream"_test = [&] {
        std::vector<std::byte> storage;
        const gr::Tag          foreign = tagWithInlineEndOfStream(true, storage);
        expect(eq(foreign.map.keys().begin().keyId(), gr::pmt::keys::kInlineKeyId));
        expect(gr::detail::defaultEOSTagMatcher(foreign, 0UZ));
        expect(!gr::detail::defaultEOSTagMatcher(foreign, 1UZ)) << "a tag before the read position is not seen";
    };

    "a foreign blob storing gr:end_of_stream inline as false does not end the stream"_test = [&] {
        std::vector<std::byte> storage;
        expect(!gr::detail::defaultEOSTagMatcher(tagWithInlineEndOfStream(false, storage), 0UZ));
    };

    "the canonical end_of_stream still ends the stream and any other key does not"_test = [] {
        const gr::property_map canonical{{std::string(tag::END_OF_STREAM.key()), true}};
        const gr::property_map other{{std::string(tag::TRIGGER_NAME.key()), "t"}};
        expect(gr::detail::defaultEOSTagMatcher(gr::makeStoredTag(0UZ, canonical.blob()), 0UZ));
        expect(!gr::detail::defaultEOSTagMatcher(gr::makeStoredTag(0UZ, other.blob()), 0UZ));
    };
};

int main() { /* tests are statically executed */ }
