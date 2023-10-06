#include <boost/ut.hpp>

#include <fmt/format.h>

#include <gnuradio-4.0/buffer.hpp>
#include <gnuradio-4.0/graph.hpp>
#include <gnuradio-4.0/node.hpp>
#include <gnuradio-4.0/reflection.hpp>
#include <gnuradio-4.0/scheduler.hpp>
#include <gnuradio-4.0/tag.hpp>

#include <gnuradio-4.0/testing/tag_monitors.hpp>

#if defined(__clang__) && __clang_major__ >= 15
// clang 16 does not like ut's default reporter_junit due to some issues with stream buffers and output redirection
template<>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif

const boost::ut::suite TagTests = [] {
    using namespace boost::ut;
    using namespace gr;

    "TagReflection"_test = [] {
        static_assert(sizeof(tag_t) % 64 == 0, "needs to meet L1 cache size");
        static_assert(refl::descriptor::type_descriptor<gr::tag_t>::name == "gr::tag_t");
        static_assert(refl::member_list<tag_t>::size == 2, "index and map being declared");
        static_assert(refl::trait::get_t<0, refl::member_list<tag_t>>::name == "index", "class field index is public API");
        static_assert(refl::trait::get_t<1, refl::member_list<tag_t>>::name == "map", "class field map is public API");
    };

    "DefaultTags"_test = [] {
        tag_t testTag{};

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
    };
};

const boost::ut::suite TagPropagation = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;

    "tag_source"_test = [] {
        std::int64_t n_samples = 1024;
        graph        flow_graph;
        auto        &src = flow_graph.make_node<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({ { "n_samples_max", n_samples }, { "name", "TagSource" } });
        src.tags         = {
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
        auto &monitor1 = flow_graph.make_node<TagMonitor<float, ProcessFunction::USE_PROCESS_BULK>>({ { "name", "TagMonitor1" } });
        auto &monitor2 = flow_graph.make_node<TagMonitor<float, ProcessFunction::USE_PROCESS_ONE>>({ { "name", "TagMonitor2" } });
        auto &sink1    = flow_graph.make_node<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({ { "name", "TagSink1" } });
        auto &sink2    = flow_graph.make_node<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({ { "name", "TagSink2" } });
        expect(eq(connection_result_t::SUCCESS, flow_graph.connect<"out">(src).to<"in">(monitor1)));
        expect(eq(connection_result_t::SUCCESS, flow_graph.connect<"out">(monitor1).to<"in">(monitor2)));
        expect(eq(connection_result_t::SUCCESS, flow_graph.connect<"out">(monitor2).to<"in">(sink1)));
        expect(eq(connection_result_t::SUCCESS, flow_graph.connect<"out">(monitor2).to<"in">(sink2)));

        scheduler::simple sched{ std::move(flow_graph) };
        sched.run_and_wait();

        expect(eq(src.n_samples_produced, n_samples)) << "src did not produce enough output samples";
        expect(eq(monitor1.n_samples_produced, n_samples)) << "monitor1 did not consume enough input samples";
        expect(eq(monitor2.n_samples_produced, n_samples)) << "monitor2 did not consume enough input samples";
        expect(eq(sink1.n_samples_produced, n_samples)) << "sink1 did not consume enough input samples";
        expect(eq(sink2.n_samples_produced, n_samples)) << "sink2 did not consume enough input samples";

        expect(equal_tag_lists(src.tags, monitor1.tags)) << "monitor1 did not receive the required tags";
        expect(equal_tag_lists(src.tags, monitor2.tags)) << "monitor2 did not receive the required tags";
        expect(equal_tag_lists(src.tags, sink1.tags)) << "sink1 did not receive the required tags";
        expect(equal_tag_lists(src.tags, sink2.tags)) << "sink1 did not receive the required tags";
    };
};

int
main() { /* tests are statically executed */
}
