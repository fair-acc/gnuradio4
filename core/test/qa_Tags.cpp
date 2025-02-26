#include <boost/ut.hpp>

#include <fmt/format.h>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Buffer.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>

#include <gnuradio-4.0/testing/TagMonitors.hpp>

template<>
struct fmt::formatter<gr::Tag> {
    template<typename ParseContext>
    constexpr auto parse(ParseContext& ctx) {
        return ctx.begin();
    }

    template<typename FormatContext>
    constexpr auto format(const gr::Tag& tag, FormatContext& ctx) const {
        return fmt::format_to(ctx.out(), "  {}->{{ {} }}\n", tag.index, tag.map);
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

    auto runTest = []<auto srcType>(bool verbose = true) {
        gr::Size_t         n_samples = 1024;
        Graph              testGraph;
        const property_map srcParameter = {{"n_samples_max", n_samples}, {"name", "TagSource"}, {gr::tag::SIGNAL_NAME.shortKey(), "tagStream"}, {"verbose_console", true && verbose}};
        auto&              src          = testGraph.emplaceBlock<TagSource<float, srcType>>(srcParameter);
        src._tags                       = {
            // TODO: allow parameter settings to include maps?!?
            {0, {{"key", "value@0"}}},       //
            {1, {{"key", "value@1"}}},       //
            {100, {{"key", "value@100"}}},   //
            {150, {{"key", "value@150"}}},   //
            {1000, {{"key", "value@1000"}}}, //
            {1001, {{"key", "value@1001"}}}, //
            {1002, {{"key", "value@1002"}}}, //
            {1023, {{"key", "value@1023"}}}  //
        };
        expect(eq("tagStream"s, src.signal_name)) << "src signal_name -> needed for setting-via-tag forwarding";

        auto& monitorBulk = testGraph.emplaceBlock<TagMonitor<float, ProcessFunction::USE_PROCESS_BULK>>({{"name", "TagMonitorBulk"}, {"n_samples_expected", n_samples}, {"verbose_console", true && verbose}});
        auto& monitorOne  = testGraph.emplaceBlock<TagMonitor<float, ProcessFunction::USE_PROCESS_ONE>>({{"name", "TagMonitorOne"}, {"n_samples_expected", n_samples}, {"verbose_console", false && verbose}});
        auto& sinkBulk    = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"name", "TagSinkN"}, {"n_samples_expected", n_samples}, {"verbose_console", true && verbose}});
        auto& sinkOne     = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"name", "TagSinkOne"}, {"n_samples_expected", n_samples}, {"verbose_console", true && verbose}});

        // src ─> monitorBulk ─> monitorOne ┬─> sinkBulk
        //                                  └─> sinkOne
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).template to<"in">(monitorBulk)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(monitorBulk).to<"in">(monitorOne)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(monitorOne).to<"in">(sinkBulk)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(monitorOne).to<"in">(sinkOne)));

        scheduler::Simple sched{std::move(testGraph)};
        expect(sched.runAndWait().has_value());

        // settings forwarding
        expect(eq("tagStream"s, src.signal_name)) << "src signal_name -> needed for setting-via-tag forwarding";
        expect(eq(src.signal_name, monitorBulk.signal_name)) << "monitorBulk signal_name";
        expect(eq(src.signal_name, monitorOne.signal_name)) << "monitorOne signal_name";
        expect(eq(src.signal_name, sinkBulk.signal_name)) << "sinkBulk signal_name";
        expect(eq(src.signal_name, sinkOne.signal_name)) << "sinkOne signal_name";

        expect(eq(src._nSamplesProduced, n_samples)) << "src did not produce enough output samples";
        expect(eq(monitorBulk._nSamplesProduced, n_samples)) << "monitorBulk did not consume enough input samples";
        expect(eq(monitorOne._nSamplesProduced, n_samples)) << "monitorOne did not consume enough input samples";
        expect(eq(sinkBulk._nSamplesProduced, n_samples)) << "sinkBulk did not consume enough input samples";
        expect(eq(sinkOne._nSamplesProduced, n_samples)) << "sinkOne did not consume enough input samples";

        expect(!monitorBulk.log_samples || eq(monitorBulk._samples.size(), n_samples)) << "monitorBulk did not log enough input samples";
        expect(!monitorOne.log_samples || eq(monitorOne._samples.size(), n_samples)) << "monitorOne did not log enough input samples";
        expect(!sinkBulk.log_samples || eq(sinkBulk._samples.size(), n_samples)) << "sinkBulk did not log enough input samples";
        expect(!sinkOne.log_samples || eq(sinkOne._samples.size(), n_samples)) << "sinkOne did not log enough input samples";

        const std::vector<std::string> ignoreKeys = {gr::tag::SIGNAL_RATE.shortKey(), gr::tag::SIGNAL_NAME.shortKey()};
        expect(equal_tag_lists(src._tags, monitorBulk._tags, ignoreKeys)) << "monitorBulk did not receive the required tags";
        expect(equal_tag_lists(src._tags, monitorOne._tags, ignoreKeys)) << "monitorOne did not receive the required tags";
        expect(equal_tag_lists(src._tags, sinkBulk._tags, ignoreKeys)) << "sinkBulk did not receive the required tags";
        expect(equal_tag_lists(src._tags, sinkOne._tags, ignoreKeys)) << "sinkOne did not receive the required tags";
    };

    "TagSource<float, USE_PROCESS_BULK>"_test = [&runTest] { runTest.template operator()<ProcessFunction::USE_PROCESS_BULK>(true); };

    "TagSource<float, USE_PROCESS_ONE>"_test = [&runTest] { runTest.template operator()<ProcessFunction::USE_PROCESS_ONE>(true); };

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
};

const boost::ut::suite RepeatedTags = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;

    auto runTest = []<auto srcType>(bool verbose = true) {
        gr::Size_t         n_samples = 30U;
        Graph              testGraph;
        const property_map srcParameter = {{"n_samples_max", n_samples}, {"name", "TagSource"}, {"verbose_console", true && verbose}, {"repeat_tags", true}};
        auto&              src          = testGraph.emplaceBlock<TagSource<float, srcType>>(srcParameter);
        src._tags                       = {{2, {{"key", "value@2"}}}, {3, {{"key", "value@3"}}}, {5, {{"key", "value@5"}}}, {8, {{"key", "value@8"}}}};

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
            expect(monitorOne._tags[i].map.at("key") == src._tags[i % src._tags.size()].map.at("key"));
            expect(sinkOne._tags[i].map.at("key") == src._tags[i % src._tags.size()].map.at("key"));
        }
    };

    "TagSource<float, USE_PROCESS_BULK>"_test = [&runTest] { runTest.template operator()<ProcessFunction::USE_PROCESS_BULK>(true); };

    "TagSource<float, USE_PROCESS_ONE>"_test = [&runTest] { runTest.template operator()<ProcessFunction::USE_PROCESS_ONE>(true); };
};

int main() { /* tests are statically executed */ }
