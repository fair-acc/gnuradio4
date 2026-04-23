#include <boost/ut.hpp>

#include <algorithm>
#include <format>
#include <iterator>
#include <ranges>
#include <string>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Buffer.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>

#include <gnuradio-4.0/testing/Delay.hpp>
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
        for (gr::Tag tag : inSamples.rawTags) {
            if (tag.index < (inSamples.streamIndex + (inSamples.size() + 1) / 2)) {
                tag.insert_or_assign("offset", sampling_rate * static_cast<double>(tag.index - inSamples.streamIndex));
                outSamples.publishTag(tag.map, 0);
                tagsForwarded++;
            } else {
                break;
            }
        }
        if (inSamples.rawTags.consume(tagsForwarded)) {
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
    std::string      signal_name       = "";
    std::string      signal_quantity   = "";
    std::string      signal_unit       = "";
    float            signal_min        = 0.f;
    float            signal_max        = 0.f;
    gr::Size_t       n_dropped_samples = gr::Size_t(0);
    std::string      trigger_name      = "";
    std::uint64_t    trigger_time      = 0;
    float            trigger_offset    = 0.f;
    gr::property_map trigger_meta_info = {};
    std::string      context           = "";
    std::uint64_t    time              = 0;

    GR_MAKE_REFLECTABLE(AutoForwardParametersBlock, in, out, not_auto_forward_parameter, sample_rate, signal_name, signal_quantity, signal_unit, //
        signal_min, signal_max, n_dropped_samples, trigger_name, trigger_time, trigger_offset, trigger_meta_info, context, time);

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

    static_assert(sizeof(Tag) % 64 == 0, "needs to meet L1 cache size");
    static_assert(gr::refl::data_member_count<Tag> == 2, "index and map being declared");
    static_assert(gr::refl::data_member_name<Tag, 0> == "index", "class field index is public API");
    static_assert(gr::refl::data_member_name<Tag, 1> == "map", "class field map is public API");

    using namespace boost::ut;

    "DefaultTags"_test = [] {
        using namespace std::string_view_literals;
        Tag testTag{};

        testTag.insert_or_assign(tag::SAMPLE_RATE, 3.0f);
        testTag.insert_or_assign(tag::SAMPLE_RATE(4.0f));
        // testTag.insert_or_assign(tag::SAMPLE_RATE(5.0)); // type-mismatch -> won't compile
        expect(testTag.at(tag::SAMPLE_RATE) == 4.0f);
        expect(tag::SAMPLE_RATE.shortKey() == "sample_rate"sv);
        expect(tag::SAMPLE_RATE.key() == std::string{GR_TAG_PREFIX}.append("sample_rate"));

        expect(testTag.get(tag::SAMPLE_RATE).has_value());
        static_assert(!std::is_const_v<decltype(testTag.get(tag::SAMPLE_RATE).value())>);
        expect(not testTag.get(tag::SIGNAL_NAME).has_value());

        static_assert(std::is_same_v<decltype(tag::SAMPLE_RATE), decltype(tag::SIGNAL_RATE)>);
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

        using namespace std::string_literals;
        using namespace std::string_view_literals;
        static_assert(tag::SIGNAL_UNIT == "signal_unit"s);
        static_assert("signal_unit"s == tag::SIGNAL_UNIT);

        static_assert("signal_unit"sv == tag::SIGNAL_UNIT);
        static_assert(tag::SIGNAL_UNIT == "signal_unit"sv);

        static_assert(tag::SIGNAL_UNIT == "signal_unit");
        static_assert("signal_unit" == tag::SIGNAL_UNIT);

        // alt definition -> eventually needed for SigMF compatibility
        using namespace gr::tag;
        static_assert(SIGNAL_UNIT == "gr:signal_unit"sv);
        static_assert("gr:signal_unit" == tag::SIGNAL_UNIT);
    };
};

const boost::ut::suite<"TagPropagation"> _TagPropagation = [] {
    using namespace std::string_literals;
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;

    "TagPropagation autoForwardParameters constructor"_test = [&] {
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

        expect(eq(monitor._tags[0].map.size(), srcParametersOnlyAutoForward.size()));
        expect(eq(sink._tags[0].map.size(), srcParametersOnlyAutoForward.size()));

        expect(monitor._tags[0].map == srcParametersOnlyAutoForward);
        map_diff_report(monitor._tags[0].map, srcParametersOnlyAutoForward, "monitor._tags", "srcParameter");

        expect(sink._tags[0].map == srcParametersOnlyAutoForward);
        map_diff_report(sink._tags[0].map, srcParametersOnlyAutoForward, "sink._tags", "srcParameter");
    };

    auto runTest = []<auto srcType>(bool verbose = true) {
        using namespace gr::testing;
        using namespace gr::tag;

        const gr::Size_t nSamples = 100;
        Graph            testGraph;

        // "reset_default", "store_default", "end_of_stream" are not included because they have special meaning
        const std::vector<Tag> tagsOnlyAutoForward = {gr::Tag(1UZ, {{SAMPLE_RATE.shortKey(), 42.f}}),          //
            gr::Tag(2UZ, {{SIGNAL_NAME.shortKey(), "SIGNAL_NAME_42"}}),                                        //
            gr::Tag(3UZ, {{SIGNAL_QUANTITY.shortKey(), "SIGNAL_QUANTITY_42"}}),                                //
            gr::Tag(4UZ, {{SIGNAL_UNIT.shortKey(), "SIGNAL_UNIT_42"}}),                                        //
            gr::Tag(5UZ, {{SIGNAL_MIN.shortKey(), 42.f}}),                                                     //
            gr::Tag(6UZ, {{SIGNAL_MAX.shortKey(), 42.f}}),                                                     //
            gr::Tag(7UZ, {{N_DROPPED_SAMPLES.shortKey(), gr::Size_t(42)}}),                                    //
            gr::Tag(8UZ, {{TRIGGER_NAME.shortKey(), "TRIGGER_NAME_42"}}),                                      //
            gr::Tag(9UZ, {{TRIGGER_TIME.shortKey(), std::uint64_t(42)}}),                                      //
            gr::Tag(10UZ, {{TRIGGER_OFFSET.shortKey(), 42.f}}),                                                //
            gr::Tag(11UZ, {{TRIGGER_META_INFO.shortKey(), property_map{{"TRIGGER_META_INFO_KEY_42", 42.f}}}}), //
            gr::Tag(12UZ, {{CONTEXT.shortKey(), "CONTEXT_42"}}),                                               //
            gr::Tag(13UZ, {{CONTEXT_TIME.shortKey(), std::uint64_t(42)}})};

        std::vector<Tag> tags = tagsOnlyAutoForward;
        tags.push_back(gr::Tag(14UZ, {{"not_auto_forward_parameter", 42.f}}));

        auto& src              = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"name", "TagSource"}, {"n_samples_max", nSamples}, {"verbose_console", true && verbose}});
        src._tags              = tags;
        auto& monitorOne       = testGraph.emplaceBlock<TagMonitor<float, ProcessFunction::USE_PROCESS_ONE>>({{"name", "TagMonitorOne"}, {"verbose_console", true && verbose}});
        auto& monitorBulk1     = testGraph.emplaceBlock<TagMonitor<float, ProcessFunction::USE_PROCESS_BULK>>({{"name", "TagMonitorBulk1"}, {"verbose_console", true && verbose}});
        auto& autoForwardBlock = testGraph.emplaceBlock<AutoForwardParametersBlock<float>>({{"name", "AutoForwardParametersBlock"}});
        auto& monitorBulk2     = testGraph.emplaceBlock<TagMonitor<float, ProcessFunction::USE_PROCESS_BULK>>({{"name", "TagMonitorBulk2"}, {"verbose_console", true && verbose}});
        auto& sinkOne          = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"name", "TagSinkOne"}, {"verbose_console", true && verbose}});
        auto& sinkBulk         = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"name", "TagSinkBulk"}, {"verbose_console", true && verbose}});

        // - monitorOne receives *all* Tags from src, but only publishes autoForward Tags.
        // - monitorBulk1 receives 'autoForward' Tags and republishes them.
        // - autoForwardBlock then receives the 'autoForward' Tags and performs applyStagedSettings and republish applied/forward settings, `not_auto_forward_parameter` is not changed
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

        expect(eq(monitorOne.signal_name, "SIGNAL_NAME_42"s));
        expect(eq(monitorBulk1.signal_name, "SIGNAL_NAME_42"s));
        expect(eq(monitorBulk2.signal_name, "SIGNAL_NAME_42"s));
        expect(eq(sinkBulk.signal_name, "SIGNAL_NAME_42"s));
        expect(eq(sinkOne.signal_name, "SIGNAL_NAME_42"s));

        expect(eq(autoForwardBlock.not_auto_forward_parameter, 0.f)); // default value, this parameter is not forwarded
        expect(eq(autoForwardBlock.sample_rate, 42.f));
        expect(eq(autoForwardBlock.signal_name, "SIGNAL_NAME_42"s));
        expect(eq(autoForwardBlock.signal_quantity, "SIGNAL_QUANTITY_42"s));
        expect(eq(autoForwardBlock.signal_unit, "SIGNAL_UNIT_42"s));
        expect(eq(autoForwardBlock.signal_min, 42.f));
        expect(eq(autoForwardBlock.signal_max, 42.f));
        expect(eq(autoForwardBlock.n_dropped_samples, gr::Size_t(42)));
        expect(eq(autoForwardBlock.trigger_name, "TRIGGER_NAME_42"s));
        expect(eq(autoForwardBlock.trigger_time, uint64_t(42)));
        expect(eq(autoForwardBlock.trigger_offset, 42.f));
        expect(eq(autoForwardBlock.trigger_meta_info.size(), 1UZ));
        expect(eq(autoForwardBlock.context, "CONTEXT_42"s));
        expect(eq(autoForwardBlock.time, std::uint64_t(42)));

        expect(equal_tag_lists(monitorOne._tags, tags)); // all tags from src
        expect(equal_tag_lists(monitorBulk1._tags, tagsOnlyAutoForward));
        expect(equal_tag_lists(monitorBulk2._tags, tagsOnlyAutoForward));
        expect(equal_tag_lists(sinkOne._tags, tagsOnlyAutoForward));
        expect(equal_tag_lists(sinkBulk._tags, tagsOnlyAutoForward));
    };

    "TagPropagation autoForwardParameters tags from TagSource<float, USE_PROCESS_BULK>"_test = [&] { runTest.template operator()<ProcessFunction::USE_PROCESS_BULK>(true); };
    "TagPropagation autoForwardParameters tags from TagSource<float, USE_PROCESS_ONE>"_test  = [&] { runTest.template operator()<ProcessFunction::USE_PROCESS_BULK>(true); };

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
        expect(eq("tagStream"s, src.signal_name)) << "src signal_name -> needed for setting-via-tag forwarding";
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

    auto runPolicyTest = []<typename TDecimator>(const std::vector<Tag>& expectedTags) {
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

        auto&                    decimator             = testGraph.emplaceBlock<TDecimator>({{"decim", decim}});
        std::vector<std::string> customAutoForwardKeys = {"key", "key0", "key4", "key5", "key15", "key20", "key25", "key35"};
        decimator.settings().autoForwardParameters().insert(customAutoForwardKeys.begin(), customAutoForwardKeys.end());
        auto& sink = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", true}});

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
        std::vector<Tag>       expectedTags = std::vector<Tag>{                      //
            {0, gr::property_map{{"key", "value@0"}, {"key0", "value@0"}}},    //
            {1, gr::property_map{{"key", "value@4"}, {"key4", "value@4"}}},    //
            {1, gr::property_map{{"key", "value@5"}, {"key5", "value@5"}}},    //
            {2, gr::property_map{{"key", "value@15"}, {"key15", "value@15"}}}, //
            {2, gr::property_map{{"key", "value@20"}, {"key20", "value@20"}}}, //
            {3, gr::property_map{{"key", "value@25"}, {"key25", "value@25"}}}};
        runPolicyTest.template operator()<DecimatorForward<float>>(expectedTags);
    };

    "Tag propagation with decimation - Backward policy"_test = [&runPolicyTest]() {
        // backward: all tags in each input chunk mapped to output position 0
        std::vector<Tag>       expectedTags = std::vector<Tag>{                      //
            {0, gr::property_map{{"key", "value@0"}, {"key0", "value@0"}}},    //
            {0, gr::property_map{{"key", "value@4"}, {"key4", "value@4"}}},    //
            {0, gr::property_map{{"key", "value@5"}, {"key5", "value@5"}}},    //
            {1, gr::property_map{{"key", "value@15"}, {"key15", "value@15"}}}, //
            {2, gr::property_map{{"key", "value@20"}, {"key20", "value@20"}}}, //
            {2, gr::property_map{{"key", "value@25"}, {"key25", "value@25"}}}, //
            {3, gr::property_map{{"key", "value@35"}, {"key35", "value@35"}}}};
        runPolicyTest.template operator()<DecimatorBackward<float>>(expectedTags);
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
            {2, {{SAMPLE_RATE.shortKey(), 2.f}}}, //
            {3, {{SAMPLE_RATE.shortKey(), 3.f}}}, //
            {5, {{SAMPLE_RATE.shortKey(), 5.f}}}, //
            {8, {{SAMPLE_RATE.shortKey(), 8.f}}}  //
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
            expect(monitorOne._tags[i].map.at(SAMPLE_RATE.shortKey()) == src._tags[i % src._tags.size()].map.at(SAMPLE_RATE.shortKey()));
            expect(sinkOne._tags[i].map.at(SAMPLE_RATE.shortKey()) == src._tags[i % src._tags.size()].map.at(SAMPLE_RATE.shortKey()));
        }
    };

    "TagSource<float, USE_PROCESS_BULK>"_test = [&runTest] { runTest.template operator()<ProcessFunction::USE_PROCESS_BULK>(true); };

    "TagSource<float, USE_PROCESS_ONE>"_test = [&runTest] { runTest.template operator()<ProcessFunction::USE_PROCESS_ONE>(true); };
};

// -- test blocks for custom forwardTags and NoTagPropagation --

template<typename T>
struct CustomForwardTagsBlock : gr::Block<CustomForwardTagsBlock<T>> {
    using Description = gr::Doc<"block with user-defined forwardTags()">;
    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    float sample_rate = 1000.f;

    GR_MAKE_REFLECTABLE(CustomForwardTagsBlock, in, out, sample_rate);

    bool forwardTagsCalled = false;

    template<typename TInputSpans, typename TOutputSpans>
    void forwardTags(TInputSpans& inputSpans, TOutputSpans& outputSpans, std::size_t /*processedIn*/) {
        forwardTagsCalled = true;
        gr::for_each_reader_span(
            [&](auto& in_) {
                if (!in_.isSync || !in_.isConnected) {
                    return;
                }
                for (const auto& [relIndex, tagMapRef] : in_.tags()) {
                    gr::property_map forwarded = tagMapRef.get();
                    forwarded.insert_or_assign("custom_added", "yes");
                    gr::for_each_writer_span([&](auto& out_) { out_.publishTag(forwarded, 0); }, outputSpans);
                }
            },
            inputSpans);
    }

    [[nodiscard]] constexpr auto processOne(T x) const noexcept { return x; }
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
            tag.insert_or_assign(std::pmr::string(key.data(), key.size()), static_cast<std::uint64_t>(_nSamples));
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
struct BackwardCustomForward : gr::Block<BackwardCustomForward<T>, gr::BackwardTagPropagation> {
    using Description = gr::Doc<"backward policy block with custom forwardTags override">;
    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    GR_MAKE_REFLECTABLE(BackwardCustomForward, in, out);

    bool forwardTagsCalled = false;

    template<typename TInputSpans, typename TOutputSpans>
    void forwardTags(TInputSpans& inputSpans, TOutputSpans& outputSpans, std::size_t /*processedIn*/) {
        forwardTagsCalled = true;
        gr::for_each_reader_span(
            [&](auto& in_) {
                if (!in_.isSync || !in_.isConnected) {
                    return;
                }
                for (const auto& [relIndex, tagMapRef] : in_.tags()) {
                    gr::property_map fwd = tagMapRef.get();
                    fwd.insert_or_assign("custom_override", "yes");
                    gr::for_each_writer_span([&](auto& out_) { out_.publishTag(fwd, 0); }, outputSpans);
                }
            },
            inputSpans);
    }

    [[nodiscard]] constexpr auto processOne(T x) const noexcept { return x; }
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

    "custom forwardTags override is called"_test = [] {
        Graph testGraph;
        auto& src    = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(30)}, {"verbose_console", false}});
        src._tags    = {{0, {{tag::SAMPLE_RATE.shortKey(), 42.f}}}};
        auto& custom = testGraph.emplaceBlock<CustomForwardTagsBlock<float>>();
        auto& sink   = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", false}});

        expect(testGraph.connect<"out", "in">(src, custom).has_value());
        expect(testGraph.connect<"out", "in">(custom, sink).has_value());

        scheduler::Simple<> sched;
        expect(sched.exchange(std::move(testGraph)).has_value());
        expect(sched.runAndWait().has_value());

        expect(custom.forwardTagsCalled) << "forwardTags() must be called by workInternal";
        expect(ge(sink._tags.size(), 1UZ)) << "at least one tag forwarded";
        expect(sink._tags[0].map.contains("custom_added")) << "custom key must be present in forwarded tag";
    };

    "NoTagPropagation suppresses auto-forwarding"_test = [] {
        Graph testGraph;
        auto& src  = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(30)}, {"verbose_console", false}});
        src._tags  = {{0, {{tag::SAMPLE_RATE.shortKey(), 42.f}}}};
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
            const auto found       = std::ranges::find_if(sink._tags, [&expectedKey](const gr::Tag& tag) { return tag.map.contains(expectedKey); });

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
        src._tags     = {{3, {{tag::SIGNAL_NAME.shortKey(), "test_signal"}}}};
        auto& monitor = testGraph.emplaceBlock<TagMonitor<float, ProcessFunction::USE_PROCESS_ONE>>({{"verbose_console", false}});
        auto& sink    = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", false}});

        expect(testGraph.connect<"out", "in">(src, monitor).has_value());
        expect(testGraph.connect<"out", "in">(monitor, sink).has_value());

        scheduler::Simple<> sched;
        expect(sched.exchange(std::move(testGraph)).has_value());
        expect(sched.runAndWait().has_value());

        expect(eq(monitor._tags.size(), 1UZ)) << "exactly one tag received";
        expect(eq(monitor._tags[0].index, 3UZ)) << "tag at correct position";
        expect(eq(monitor.signal_name, "test_signal"s)) << "settings auto-updated";
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
            if (auto it = tag.map.find(tag::SAMPLE_RATE.shortKey()); it != tag.map.end()) {
                expect(*it->second.get_if<float>() == 48000.f);
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

    "processOne handles multi-sample chunk with tag at sample 0 only"_test = [] {
        // after limitByFirstTag removal: processOne processes multi-sample chunks
        // tag at position 0 is visible via mergedInputTag() for the first sample only
        Graph testGraph;
        auto& src     = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(30)}, {"verbose_console", false}});
        src._tags     = {{0, {{tag::SIGNAL_NAME.shortKey(), "test"}}}}; // single tag at start
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
        expect(eq(monitor.signal_name, "test"s)) << "settings auto-updated";
    };

    "value substitution in forwarding replaces with block-current value"_test = [] {
        Graph testGraph;
        auto& src  = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(10)}, {"verbose_console", false}});
        src._tags  = {{0, {{tag::SAMPLE_RATE.shortKey(), 42.f}}}};
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
        src._tags  = {{0, {{tag::SAMPLE_RATE.shortKey(), 42.f}, {"unique_key", "A"}}}};
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
            if (tag.map.contains(tag::SAMPLE_RATE.shortKey())) {
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
        src._tags  = {{0, {{tag::SAMPLE_RATE.shortKey(), 1.f}}}, {5, {{tag::SAMPLE_RATE.shortKey(), 5.f}}}};
        auto& fwd  = testGraph.emplaceBlock<ForwardPropBlock<float>>();
        auto& sink = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", false}});

        expect(testGraph.connect<"out", "in">(src, fwd).has_value());
        expect(testGraph.connect<"out", "in">(fwd, sink).has_value());

        scheduler::Simple<> sched;
        expect(sched.exchange(std::move(testGraph)).has_value());
        expect(sched.runAndWait().has_value());

        expect(eq(sink._nSamplesProduced, gr::Size_t(30))) << "all samples processed in fixed chunks";
        expect(ge(sink._tags.size(), 1UZ)) << "tag at position 0 forwarded";
        // tag at 5 auto-updates fwd's settings even though it's mid-chunk (via applyInputTagsFromPorts)
        // the forwarded value is based on the block's current state after auto-update
        expect(ge(sink._tags.size(), 1UZ));
    };

    "MergeTagPropagation merges all tags into one"_test = [] {
        Graph testGraph;
        auto& src   = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(10)}, {"verbose_console", false}});
        src._tags   = {{0, {{tag::SAMPLE_RATE.shortKey(), 42.f}}}, {0, {{tag::SIGNAL_NAME.shortKey(), "merged_test"}}}};
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
            if (tag.map.contains(tag::SAMPLE_RATE.shortKey()) && tag.map.contains(tag::SIGNAL_NAME.shortKey())) {
                foundMerged = true;
            }
        }
        expect(foundMerged) << "merged tag must contain both sample_rate and signal_name";
    };

    "MergeTagPropagation overlapping keys use last value"_test = [] {
        Graph testGraph;
        auto& src0  = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(10)}, {"verbose_console", false}});
        auto& src1  = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(10)}, {"verbose_console", false}});
        src0._tags  = {{0, {{tag::SIGNAL_NAME.shortKey(), "first"}}}, {0, {{tag::SIGNAL_NAME.shortKey(), "second"}}}};
        src1._tags  = {{0, {{tag::SIGNAL_NAME.shortKey(), "third"}}}, {0, {{tag::SIGNAL_NAME.shortKey(), "fourth"}}}};
        auto& merge = testGraph.emplaceBlock<MergePropTwoInput<float>>();
        auto& sink  = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", false}});

        expect(testGraph.connect<"out", "in0">(src0, merge).has_value());
        expect(testGraph.connect<"out", "in1">(src1, merge).has_value());
        expect(testGraph.connect<"out", "in">(merge, sink).has_value());

        scheduler::Simple<> sched;
        expect(sched.exchange(std::move(testGraph)).has_value());
        expect(sched.runAndWait().has_value());

        const auto signalNameKey = tag::SIGNAL_NAME.shortKey();
        const auto found         = std::ranges::find_if(sink._tags, [&signalNameKey](const gr::Tag& tag) { return tag.map.contains(signalNameKey); });

        expect(found != sink._tags.end());
        if (found != sink._tags.end()) {
            expect(eq(found->index, 0UZ));
            expect(eq(found->map.at(signalNameKey).value_or(std::string{}), std::string{"fourth"})) << "last overlapping key wins";
        }
    };

    "MergeTagPropagation with multi-input deduplicates and merges"_test = [] {
        // fan-out: same source → both inputs of merge block
        // identical tags from fan-out should be deduped, then merged into one output tag
        Graph testGraph;
        auto& src   = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(10)}, {"verbose_console", false}});
        src._tags   = {{0, {{tag::SAMPLE_RATE.shortKey(), 42.f}, {tag::SIGNAL_NAME.shortKey(), "multi_merge"}}}};
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
            if (tag.map.contains(tag::SAMPLE_RATE.shortKey()) && tag.map.contains(tag::SIGNAL_NAME.shortKey())) {
                foundMerged = true;
            }
        }
        expect(foundMerged) << "merged tag with both keys from deduped fan-out";
    };

    "forwardTags override takes precedence over BackwardTagPropagation"_test = [] {
        // block has BackwardTagPropagation CRTP tag BUT provides forwardTags() override
        // the override should fire, NOT the default backward logic
        Graph testGraph;
        auto& src    = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(10)}, {"verbose_console", false}});
        src._tags    = {{0, {{tag::SAMPLE_RATE.shortKey(), 42.f}}}};
        auto& custom = testGraph.emplaceBlock<BackwardCustomForward<float>>();
        auto& sink   = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", false}});

        expect(testGraph.connect<"out", "in">(src, custom).has_value());
        expect(testGraph.connect<"out", "in">(custom, sink).has_value());

        scheduler::Simple<> sched;
        expect(sched.exchange(std::move(testGraph)).has_value());
        expect(sched.runAndWait().has_value());

        expect(custom.forwardTagsCalled) << "custom forwardTags() must be called despite BackwardTagPropagation CRTP tag";
        bool hasCustomKey = false;
        for (const auto& tag : sink._tags) {
            if (tag.map.contains("custom_override")) {
                hasCustomKey = true;
            }
        }
        expect(hasCustomKey) << "custom_override key must be present (override, not default backward)";
    };

    "ForwardTagPropagation carry-forward tag arrives in next chunk"_test = [] {
        // tag at position 5 within a 10-sample chunk should carry forward to the next chunk
        // use a longer stream (50 samples) to ensure multiple chunks
        Graph testGraph;
        auto& src  = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(50)}, {"verbose_console", false}});
        src._tags  = {{0, {{tag::SAMPLE_RATE.shortKey(), 1.f}}}, {5, {{tag::SAMPLE_RATE.shortKey(), 5.f}}}};
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
            {0, {{tag::SAMPLE_RATE.shortKey(), 42.f}}},     // matching: auto-updates sample_rate
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
        auto& delay = testGraph.emplaceBlock<testing::Delay<float>>({{"delay_ms", 0.f}});
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
        src._tags  = {{0, {{tag::SAMPLE_RATE.shortKey(), 42.f}, {tag::SIGNAL_NAME.shortKey(), "test"}}}, {0, {{tag::TRIGGER_NAME.shortKey(), "evt"}, {tag::TRIGGER_TIME.shortKey(), std::uint64_t(123)}}}};
        auto& sink = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", false}});

        expect(testGraph.connect<"out", "in">(src, sink).has_value());

        scheduler::Simple<> sched;
        expect(sched.exchange(std::move(testGraph)).has_value());
        expect(sched.runAndWait().has_value());

        expect(eq(sink.sample_rate, 42.f)) << "sample_rate auto-updated despite trigger tag at same position";
        expect(eq(sink.signal_name, "test"s)) << "signal_name auto-updated";
    };
};

int main() { /* tests are statically executed */ }
