#include <boost/ut.hpp>

#include <graph.hpp>
#include <node.hpp>
#include <scheduler.hpp>
#include <tag.hpp>

#include "blocklib/core/sources/clock_source.hpp"
#include "blocklib/core/unit-test/tag_monitors.hpp"

#include <ctime>

#if defined(__clang__) && __clang_major__ >= 15
// clang 16 does not like ut's default reporter_junit due to some issues with stream buffers and output redirection
template<>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif

const boost::ut::suite TagTests = [] {
    using namespace boost::ut;
    using namespace fair::graph;
    using namespace fair::graph::sources;
    using namespace fair::graph::tag_test;

    "source_test"_test = [] {
        constexpr std::uint64_t n_samples   = 1900;
        constexpr float         sample_rate = 2000.f;
        graph                   flow_graph;
        auto                   &src = flow_graph.make_node<fair::graph::sources::ClockSource<float>>({ { "sample_rate", sample_rate }, { "n_samples_max", n_samples }, { "name", "TagSource" } });
        src.tags                    = {
            { 0, { { "key", "value@0" } } },       //
            { 1, { { "key", "value@1" } } },       //
            { 100, { { "key", "value@100" } } },   //
            { 150, { { "key", "value@150" } } },   //
            { 1000, { { "key", "value@1000" } } }, //
            { 1001, { { "key", "value@1001" } } }, //
            { 1002, { { "key", "value@1002" } } }, //
            { 1023, { { "key", "value@1023" } } }  //
        };
        auto &sink1 = flow_graph.make_node<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({ { "name", "TagSink1" } });
        auto &sink2 = flow_graph.make_node<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({ { "name", "TagSink2" } });
        expect(eq(connection_result_t::SUCCESS, flow_graph.connect<"out">(src).to<"in">(sink1)));
        expect(eq(connection_result_t::SUCCESS, flow_graph.connect<"out">(src).to<"in">(sink2)));

        scheduler::simple sched{ std::move(flow_graph) };
        sched.run_and_wait();

        expect(eq(src.n_samples_produced, n_samples)) << "src did not produce enough output samples";
        expect(eq(static_cast<std::uint64_t>(sink1.n_samples_produced), n_samples)) << "sink1 did not consume enough input samples";
        expect(eq(static_cast<std::uint64_t>(sink1.n_samples_produced), n_samples)) << "sink2 did not consume enough input samples";
        // expect(approx(sink1.effective_sample_rate(), sample_rate, 500.f)) << "effective sample rate does not match"
        // expect(approx(sink2.effective_sample_rate(), sample_rate, 500.f)) << "effective sample rate does not match"
        fmt::print("sink1: effective {} vs. expected {} sample rate [Hz]\n", sink1.effective_sample_rate(), sample_rate);
        fmt::print("sink2: effective {} vs. expected {} sample rate [Hz]\n", sink2.effective_sample_rate(), sample_rate);

        expect(equal_tag_lists(src.tags, sink1.tags)) << "sink1 did not receive the required tags";
        expect(equal_tag_lists(src.tags, sink2.tags)) << "sink1 did not receive the required tags";
    };
};

int
main() { /* not needed for UT */
}
