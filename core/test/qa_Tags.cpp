#include <boost/ut.hpp>

#include <format>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Buffer.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>

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

    // TODO: References are required here because InputSpan and OutputSpan have internal states
    // (e.g., tagsPublished) that are unique to each instance. Copying these objects without proper
    // state management can lead to incorrect behavior when the publishTag() method is called.
    // This issue must be resolved in future updates, see https://github.com/fair-acc/gnuradio4/issues/439
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
struct DecimatorBackward : gr::Block<DecimatorBackward<T>, gr::Resampling<1UZ, 1UZ, false>, gr::BackwardTagForwarding> {
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

const boost::ut::suite TagTests = [] {
    using namespace gr;

    static_assert(sizeof(Tag) % 64 == 0, "needs to meet L1 cache size");
    static_assert(gr::refl::data_member_count<Tag> == 2, "index and map being declared");
    static_assert(gr::refl::data_member_name<Tag, 0> == "index", "class field index is public API");
    static_assert(gr::refl::data_member_name<Tag, 1> == "map", "class field map is public API");

    using namespace boost::ut;

    "DefaultTags"_test = [] {
        using namespace std::string_view_literals;
        Tag testTag{};

        testTag.insert_or_assign(tag::SAMPLE_RATE, pmtv::pmt(3.0f));
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

const boost::ut::suite TagPropagation = [] {
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
            {TRIGGER_TIME.shortKey(), uint64_t(42)},                                          //
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

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).to<"in">(autoForwardBlock)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(autoForwardBlock).to<"in">(monitor)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(monitor).to<"in">(sink)));

        scheduler::Simple sched{std::move(testGraph)};
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
            gr::Tag(9UZ, {{TRIGGER_TIME.shortKey(), uint64_t(42)}}),                                           //
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
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).to<"in">(monitorOne)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(monitorOne).to<"in">(monitorBulk1)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(monitorBulk1).to<"in">(autoForwardBlock)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(autoForwardBlock).to<"in">(monitorBulk2)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(monitorBulk2).to<"in">(sinkOne)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(monitorBulk2).to<"in">(sinkBulk)));

        scheduler::Simple sched{std::move(testGraph)};
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
            {0, {{"key", "value@0"}, {"key0", "value@0"}}},          //
            {1, {{"key", "value@1"}, {"key1", "value@1"}}},          //
            {100, {{"key", "value@100"}, {"key2", "value@100"}}},    //
            {150, {{"key", "value@150"}, {"key3", "value@150"}}},    //
            {1000, {{"key", "value@1000"}, {"key4", "value@1000"}}}, //
            {1001, {{"key", "value@1001"}, {"key5", "value@1001"}}}, //
            {1002, {{"key", "value@1002"}, {"key6", "value@1002"}}}, //
            {1023, {{"key", "value@1023"}, {"key7", "value@1023"}}}  //
        };
        expect(eq("tagStream"s, src.signal_name)) << "src signal_name -> needed for setting-via-tag forwarding";
        auto& realign = testGraph.emplaceBlock<RealignTagsToChunks<float>>();
        auto& sink    = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"name", "TagSinkN"}, {"n_samples_expected", n_samples}, {"verbose_console", true}});

        // [ TagSource ] -> [ tag realign block ] -> [ TagSink ]
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).template to<"inPort">(realign)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"outPort">(realign).to<"in">(sink)));

        scheduler::Simple sched{std::move(testGraph)};
        expect(sched.runAndWait().has_value());

        expect(eq(src._nSamplesProduced, n_samples)) << "src did not produce enough output samples";
        expect(eq(sink._nSamplesProduced, 1008U)) << "sinkOne did not consume enough input samples"; // default policy is to drop epilogue samples
        expect(eq(sink._tags.size(), 3UZ));                                                          // default policy is to drop epilogue samples
    };

    auto runPolicyTest = []<typename TDecimator>(const std::vector<Tag>& expectedTags) {
        gr::Size_t nSamples = 45;
        gr::Size_t decim    = 10;

        Graph testGraph;
        auto& src = testGraph.emplaceBlock<TagSource<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", nSamples}, {"verbose_console", true}});
        src._tags = {
            {0, {{"key", "value@0"}, {"key0", "value@0"}}},     //
            {4, {{"key", "value@4"}, {"key4", "value@4"}}},     //
            {5, {{"key", "value@5"}, {"key5", "value@5"}}},     //
            {15, {{"key", "value@15"}, {"key15", "value@15"}}}, //
            {20, {{"key", "value@20"}, {"key20", "value@20"}}}, //
            {25, {{"key", "value@25"}, {"key25", "value@25"}}}, //
            {35, {{"key", "value@35"}, {"key35", "value@35"}}}  //
        };

        auto&                    decimator             = testGraph.emplaceBlock<TDecimator>({{"decim", decim}});
        std::vector<std::string> customAutoForwardKeys = {"key", "key0", "key4", "key5", "key15", "key20", "key25", "key35"};
        decimator.settings().autoForwardParameters().insert(customAutoForwardKeys.begin(), customAutoForwardKeys.end());
        auto& sink = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", true}});

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).template to<"in">(decimator)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(decimator).template to<"in">(sink)));

        scheduler::Simple sched{std::move(testGraph)};
        expect(sched.runAndWait().has_value());

        expect(eq(src._nSamplesProduced, nSamples));
        expect(eq(sink._nSamplesProduced, 4U));
        expect(eq(sink._tags.size(), 4UZ));

        expect(equal_tag_lists(sink._tags, expectedTags));
    };

    "Tag propagation with decimation - Forward policy"_test = [&runPolicyTest]() {
        std::vector<Tag>       expectedTags = std::vector<Tag>{                             //
            {0, {{"key", "value@0"}, {"key0", "value@0"}}},                           //
            {1, {{"key", "value@5"}, {"key4", "value@4"}, {"key5", "value@5"}}},      //
            {2, {{"key", "value@20"}, {"key15", "value@15"}, {"key20", "value@20"}}}, //
            {3, {{"key", "value@25"}, {"key25", "value@25"}}}};
        runPolicyTest.template operator()<DecimatorForward<float>>(expectedTags);
    };

    "Tag propagation with decimation - Backward policy"_test = [&runPolicyTest]() {
        std::vector<Tag>       expectedTags = std::vector<Tag>{                                             //
            {0, {{"key", "value@5"}, {"key0", "value@0"}, {"key4", "value@4"}, {"key5", "value@5"}}}, //
            {1, {{"key", "value@15"}, {"key15", "value@15"}}},                                        //
            {2, {{"key", "value@25"}, {"key20", "value@20"}, {"key25", "value@25"}}},                 //
            {3, {{"key", "value@35"}, {"key35", "value@35"}}}};
        runPolicyTest.template operator()<DecimatorBackward<float>>(expectedTags);
    };
};

const boost::ut::suite RepeatedTags = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::tag;
    using namespace gr::testing;

    auto runTest = []<auto srcType>(bool verbose = true) {
        gr::Size_t         n_samples = 30U;
        Graph              testGraph;
        const property_map srcParameter = {{"n_samples_max", n_samples}, {"name", "TagSource"}, {"verbose_console", true && verbose}, {"repeat_tags", true}};
        auto&              src          = testGraph.emplaceBlock<TagSource<float, srcType>>(srcParameter);
        src._tags                       = {{2, {{SAMPLE_RATE.shortKey(), 2.f}}}, {3, {{SAMPLE_RATE.shortKey(), 3.f}}}, {5, {{SAMPLE_RATE.shortKey(), 5.f}}}, {8, {{SAMPLE_RATE.shortKey(), 8.f}}}};

        auto& monitorOne = testGraph.emplaceBlock<TagMonitor<float, ProcessFunction::USE_PROCESS_ONE>>({{"name", "TagMonitorOne"}, {"n_samples_expected", n_samples}, {"verbose_console", false && verbose}});
        auto& sinkOne    = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"name", "TagSinkOne"}, {"n_samples_expected", n_samples}, {"verbose_console", false && verbose}});

        // src -> monitorOne -> sinkOne
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).template to<"in">(monitorOne)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(monitorOne).to<"in">(sinkOne)));

        scheduler::Simple sched{std::move(testGraph)};
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

int main() { /* tests are statically executed */ }
