#include <boost/ut.hpp>

#include <fmt/format.h>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Buffer.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/reflection.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/Tag.hpp>

#include <gnuradio-4.0/testing/TagMonitors.hpp>

#if defined(__clang__) && __clang_major__ >= 15
// clang 16 does not like ut's default reporter_junit due to some issues with stream buffers and output redirection
template<>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif

const boost::ut::suite TagTests = [] {
    using namespace boost::ut;
    using namespace gr;

    "TagReflection"_test = [] {
        static_assert(sizeof(Tag) % 64 == 0, "needs to meet L1 cache size");
        static_assert(refl::descriptor::type_descriptor<gr::Tag>::name == "gr::Tag");
        static_assert(refl::member_list<Tag>::size == 2, "index and map being declared");
        static_assert(refl::trait::get_t<0, refl::member_list<Tag>>::name == "index", "class field index is public API");
        static_assert(refl::trait::get_t<1, refl::member_list<Tag>>::name == "map", "class field map is public API");
    };

    "DefaultTags"_test = [] {
        Tag testTag{};

        testTag.insert_or_assign(tag::SAMPLE_RATE, pmtv::pmt(3.0f));
        testTag.insert_or_assign(tag::SAMPLE_RATE(4.0f));
        // testTag.insert_or_assign(tag::SAMPLE_RATE(5.0)); // type-mismatch -> won't compile
        expect(testTag.at(tag::SAMPLE_RATE) == 4.0f);
        expect(tag::SAMPLE_RATE.shortKey() == "sample_rate");
        expect(tag::SAMPLE_RATE.key() == std::string{ GR_TAG_PREFIX }.append("sample_rate"));

        expect(testTag.get(tag::SAMPLE_RATE).has_value());
        static_assert(!std::is_const_v<decltype(testTag.get(tag::SAMPLE_RATE).value())>);
        expect(not testTag.get(tag::SIGNAL_NAME).has_value());

        static_assert(std::is_same_v<decltype(tag::SAMPLE_RATE), decltype(tag::SIGNAL_RATE)>);
        // test other tag on key definition only
        static_assert(tag::SIGNAL_UNIT.shortKey() == "signal_unit");
        static_assert(tag::SIGNAL_MIN.shortKey() == "signal_min");
        static_assert(tag::SIGNAL_MAX.shortKey() == "signal_max");
        static_assert(tag::TRIGGER_NAME.shortKey() == "trigger_name");
        static_assert(tag::TRIGGER_TIME.shortKey() == "trigger_time");
        static_assert(tag::TRIGGER_OFFSET.shortKey() == "trigger_offset");

        // test other tag on key definition only
        static_assert(tag::SIGNAL_UNIT.key() == "gr:signal_unit");
        static_assert(tag::SIGNAL_MIN.key() == "gr:signal_min");
        static_assert(tag::SIGNAL_MAX.key() == "gr:signal_max");
        static_assert(tag::TRIGGER_NAME.key() == "gr:trigger_name");
        static_assert(tag::TRIGGER_TIME.key() == "gr:trigger_time");
        static_assert(tag::TRIGGER_OFFSET.key() == "gr:trigger_offset");

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
        const property_map srcParameter = { { "n_samples_max", n_samples }, { "name", "TagSource" }, { "signal_name", "tagStream" }, { "verbose_console", true && verbose } };
        auto              &src          = testGraph.emplaceBlock<TagSource<float, srcType>>(srcParameter);
        src.tags                        = {
            // TODO: allow parameter settings to include maps?!?
            { 0, { { "key", "value@0" } } },       //
            { 1, { { "key", "value@1" } } },       //
            { 100, { { "key", "value@100" } } },   //
            { 150, { { "key", "value@150" } } },   //
            { 1000, { { "key", "value@1000" } } }, //
            { 1001, { { "key", "value@1001" } } }, //
            { 1002, { { "key", "value@1002" } } }, //
            { 1023, { { "key", "value@1023" } } }  //
        };
        expect(eq("tagStream"s, src.signal_name)) << "src signal_name -> needed for setting-via-tag forwarding";

        auto &monitorBulk = testGraph.emplaceBlock<TagMonitor<float, ProcessFunction::USE_PROCESS_BULK>>(
                { { "name", "TagMonitorBulk" }, { "n_samples_expected", n_samples }, { "verbose_console", true && verbose } });
        auto &monitorOne = testGraph.emplaceBlock<TagMonitor<float, ProcessFunction::USE_PROCESS_ONE>>(
                { { "name", "TagMonitorOne" }, { "n_samples_expected", n_samples }, { "verbose_console", false && verbose } });
        auto &monitorOneSIMD = testGraph.emplaceBlock<TagMonitor<float, ProcessFunction::USE_PROCESS_ONE_SIMD>>(
                { { "name", "TagMonitorOneSIMD" }, { "n_samples_expected", n_samples }, { "verbose_console", false && verbose } });
        auto &sinkBulk = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>(
                { { "name", "TagSinkN" }, { "n_samples_expected", n_samples }, { "verbose_console", true && verbose } });
        auto &sinkOne = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>(
                { { "name", "TagSinkOne" }, { "n_samples_expected", n_samples }, { "verbose_console", true && verbose } });

        // src ─> monitorBulk ─> monitorOne ─> monitorOneSIMD ┬─> sinkBulk
        //                                                    └─> sinkOne
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).template to<"in">(monitorBulk)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(monitorBulk).to<"in">(monitorOne)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(monitorOne).to<"in">(monitorOneSIMD)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(monitorOneSIMD).to<"in">(sinkBulk)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(monitorOneSIMD).to<"in">(sinkOne)));

        scheduler::Simple sched{ std::move(testGraph) };
        sched.runAndWait();

        // settings forwarding
        expect(eq("tagStream"s, src.signal_name)) << "src signal_name -> needed for setting-via-tag forwarding";
        expect(eq(src.signal_name, monitorBulk.signal_name)) << "monitorBulk signal_name";
        expect(eq(src.signal_name, monitorOne.signal_name)) << "monitorOne signal_name";
        expect(eq(src.signal_name, monitorOneSIMD.signal_name)) << "monitorOneSIMD signal_name";
        expect(eq(src.signal_name, sinkBulk.signal_name)) << "sinkBulk signal_name";
        expect(eq(src.signal_name, sinkOne.signal_name)) << "sinkOne signal_name";

        expect(eq(src.n_samples_produced, n_samples)) << "src did not produce enough output samples";
        expect(eq(monitorBulk.n_samples_produced, n_samples)) << "monitorBulk did not consume enough input samples";
        expect(eq(monitorOne.n_samples_produced, n_samples)) << "monitorOne did not consume enough input samples";
        expect(eq(monitorOneSIMD.n_samples_produced, n_samples)) << "monitorOneSIMD did not consume enough input samples";
        expect(eq(sinkBulk.n_samples_produced, n_samples)) << "sinkBulk did not consume enough input samples";
        expect(eq(sinkOne.n_samples_produced, n_samples)) << "sinkOne did not consume enough input samples";

        expect(!monitorBulk.log_samples || eq(monitorBulk.samples.size(), n_samples)) << "monitorBulk did not log enough input samples";
        expect(!monitorOne.log_samples || eq(monitorOne.samples.size(), n_samples)) << "monitorOne did not log enough input samples";
        expect(!monitorOneSIMD.log_samples || eq(monitorOneSIMD.samples.size(), n_samples)) << "monitorOneSIMD did not log enough input samples";
        expect(!sinkBulk.log_samples || eq(sinkBulk.samples.size(), n_samples)) << "sinkBulk did not log enough input samples";
        expect(!sinkOne.log_samples || eq(sinkOne.samples.size(), n_samples)) << "sinkOne did not log enough input samples";

        expect(equal_tag_lists(src.tags, monitorBulk.tags, "signal_name"s)) << "monitorBulk did not receive the required tags";
        expect(equal_tag_lists(src.tags, monitorOne.tags, "signal_name"s)) << "monitorOne did not receive the required tags";
        expect(equal_tag_lists(src.tags, monitorOneSIMD.tags, "signal_name"s)) << "monitorOneSIMD did not receive the required tags";
        expect(equal_tag_lists(src.tags, sinkBulk.tags, "signal_name"s)) << "sinkBulk did not receive the required tags";
        expect(equal_tag_lists(src.tags, sinkOne.tags, "signal_name"s)) << "sinkOne did not receive the required tags";
    };

    "TagSource<float, USE_PROCESS_BULK>"_test = [&runTest] { runTest.template operator()<ProcessFunction::USE_PROCESS_BULK>(true); };

    "TagSource<float, USE_PROCESS_ONE>"_test = [&runTest] { runTest.template operator()<ProcessFunction::USE_PROCESS_ONE>(true); };
};

int
main() { /* tests are statically executed */
}
