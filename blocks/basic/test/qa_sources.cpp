#include <boost/ut.hpp>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/Tag.hpp>

#include <gnuradio-4.0/basic/clock_source.hpp>
#include <gnuradio-4.0/testing/tag_monitors.hpp>

#if defined(__clang__) && __clang_major__ >= 15
// clang 16 does not like ut's default reporter_junit due to some issues with stream buffers and output redirection
template<>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif

const boost::ut::suite TagTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::basic;
    using namespace gr::testing;

    "source_test"_test = [] {
        constexpr bool          useIoThreadPool = true; // true: scheduler/graph-provided thread, false: use user-provided call-back or thread
        constexpr std::uint32_t n_samples       = 1900;
        constexpr float         sample_rate     = 2000.f;
        Graph                   testGraph;
        auto &src = testGraph.emplaceBlock<gr::basic::ClockSource<float, useIoThreadPool>>({ { "sample_rate", sample_rate }, { "n_samples_max", n_samples }, { "name", "ClockSource" } });
        src.tags  = {
            { 0, { { "key", "value@0" } } },       //
            { 1, { { "key", "value@1" } } },       //
            { 100, { { "key", "value@100" } } },   //
            { 150, { { "key", "value@150" } } },   //
            { 1000, { { "key", "value@1000" } } }, //
            { 1001, { { "key", "value@1001" } } }, //
            { 1002, { { "key", "value@1002" } } }, //
            { 1023, { { "key", "value@1023" } } }  //
        };
        auto &sink1 = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({ { "name", "TagSink1" } });
        auto &sink2 = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({ { "name", "TagSink2" } });
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).to<"in">(sink1)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).to<"in">(sink2)));

        scheduler::Simple sched{ std::move(testGraph) };
        if constexpr (!useIoThreadPool) {
            src.tryStartThread();
        }
        sched.runAndWait();
        if constexpr (!useIoThreadPool) {
            src.stopThread();
        }

        expect(eq(src.n_samples_max, n_samples)) << "src did not accept require max input samples";
        expect(eq(src.n_samples_produced, n_samples)) << "src did not produce enough output samples";
        expect(eq(static_cast<std::uint32_t>(sink1.n_samples_produced), n_samples)) << fmt::format("sink1 did not consume enough input samples ({} vs. {})", sink1.n_samples_produced, n_samples);
        expect(eq(static_cast<std::uint32_t>(sink2.n_samples_produced), n_samples)) << fmt::format("sink2 did not consume enough input samples ({} vs. {})", sink2.n_samples_produced, n_samples);

        if (std::getenv("DISABLE_SENSITIVE_TESTS") != nullptr) {
            expect(approx(sink1.effective_sample_rate(), sample_rate, 500.f))
                    << fmt::format("sink1: effective sample rate {} vs {} +- {} does not match", sink1.effective_sample_rate(), sample_rate, 500.f);
            expect(approx(sink2.effective_sample_rate(), sample_rate, 500.f))
                    << fmt::format("sink2: effective sample rate {} vs {} +- {} does not match", sink1.effective_sample_rate(), sample_rate, 500.f);
        }

        fmt::print("sink1 (USE_PROCESS_ONE): effective {} vs. expected {} sample rate [Hz]\n", sink1.effective_sample_rate(), sample_rate);
        fmt::print("sink2 (USE_PROCESS_BULK): effective {} vs. expected {} sample rate [Hz]\n", sink2.effective_sample_rate(), sample_rate);

        // TODO: last decimator/interpolator + stride addition seems to break the limiting the input samples to the min of available vs. n_samples-until next tags
        // expect(equal_tag_lists(src.tags, sink1.tags)) << "sink1 (USE_PROCESS_ONE) did not receive the required tags";
        // expect(equal_tag_lists(src.tags, sink2.tags)) << "sink2 (USE_PROCESS_BULK) did not receive the required tags";
    };
};

int
main() { /* not needed for UT */
}
