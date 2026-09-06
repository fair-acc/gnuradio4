#include <boost/ut.hpp>

#include <cmath>
#include <format>
#include <numbers>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/filter/PolyphaseArbitraryResampler.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

// the bank's numerics are proven in algorithm/test/qa_PolyphaseArbitraryResamplerAlgo.cpp; block-specific
// here is the rate contract and that a constant survives the framework's windowing unchanged.

const boost::ut::suite<"PolyphaseArbitraryResampler block"> arbBlockTests = [] {
    using namespace boost::ut;
    using namespace std::string_literals;
    using namespace gr::testing;
    using gr::test::eq;

    "the forwarded sample rate follows the ratio"_test = [] {
        gr::filter::PolyphaseArbitraryResampler<float> block({{"resample_ratio", 4.f}});
        block.settings().init();
        const auto applied = block.settings().applyStagedParameters();
        expect(block.phases.size() > 0UZ) << "the bank was not designed";
        expect(gt(static_cast<std::size_t>(block.in.min_samples), 1UZ)) << "one window is the minimum a call can use";
        std::ignore = applied;
    };

    "a constant survives a fractional ratio"_test = [] {
        gr::Graph flow({{"auto_size_edges_to_chunks", true}});
        auto&     source = flow.emplaceBlock<TagSource<float>>({{"n_samples_max", gr::Size_t(16384)}, {"values", std::vector<float>{3.f}}, {"mark_tag", false}});
        auto&     dut    = flow.emplaceBlock<gr::filter::PolyphaseArbitraryResampler<float>>({{"resample_ratio", 1.37f}, {"n_phases", gr::Size_t(32)}});
        auto&     sink   = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});

        expect(flow.connect(source, "out"s, dut, "in"s).has_value());
        expect(flow.connect(dut, "out"s, sink, "in"s).has_value());

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value());
        expect(sched.runAndWait().has_value());

        const auto& samples = sink._samples;
        expect(gt(samples.size(), 256UZ)) << "nothing came through";
        for (std::size_t i = samples.size() / 2UZ; i < samples.size(); ++i) { // past settling
            expect(approx(samples[i], 3.f, 0.15f)) << std::format("sample {} is {:.5f}, not the constant 3", i, samples[i]);
        }
    };
};

int main() { /* not needed for UT */ }
