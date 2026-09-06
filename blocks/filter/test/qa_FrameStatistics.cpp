#include <boost/ut.hpp>

#include <algorithm>
#include <cmath>
#include <format>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/filter/FrameStatistics.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace {
using namespace gr::testing;

[[nodiscard]] std::vector<float> runRms(gr::Size_t frameSize, gr::Size_t nSamples) {
    gr::Graph flow({{"auto_size_edges_to_chunks", true}});
    auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", nSamples}, {"mark_tag", false}});
    auto&     dut    = flow.emplaceBlock<gr::filter::RootMeanSquare<float>>({{"frame_size", frameSize}});
    auto&     sink   = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});

    boost::ut::expect(flow.connect<"out", "in">(source, dut).has_value());
    boost::ut::expect(flow.connect<"out", "in">(dut, sink).has_value());
    gr::scheduler::Simple<> sched;
    boost::ut::expect(sched.exchange(std::move(flow)).has_value());
    boost::ut::expect(sched.runAndWait().has_value());

    std::vector<float> figures(sink._samples.size());
    for (std::size_t i = 0UZ; i < figures.size(); ++i) {
        figures[i] = sink._samples[i];
    }
    return figures;
}
} // namespace

const boost::ut::suite<"RootMeanSquare"> _rms = [] {
    using namespace boost::ut;

    "one figure per frame, and the figure is the root mean square"_test = [] {
        for (const gr::Size_t frame : {16U, 64U, 1024U}) {
            constexpr gr::Size_t     kSamples = 4096U;
            const std::vector<float> got      = runRms(frame, kSamples);
            expect(eq(got.size(), static_cast<std::size_t>(kSamples / frame))) << std::format("frame {} left samples behind", frame);

            bool matches = true;
            for (std::size_t f = 0UZ; f < got.size(); ++f) {
                double sumOfSquares = 0.0; // the source ramps, so frame f covers [f*N, (f+1)*N)
                for (std::size_t i = 0UZ; i < frame; ++i) {
                    const double sample = static_cast<double>(f * frame + i);
                    sumOfSquares += sample * sample;
                }
                matches = matches && std::abs(static_cast<double>(got[f]) - std::sqrt(sumOfSquares / frame)) < 1e-2;
            }
            expect(matches) << std::format("frame {} does not match the closed form", frame);
        }
    };

    "a constant comes out at its own level"_test = [] {
        // frame of one: the root mean square of a single sample is its magnitude
        const std::vector<float> got = runRms(1U, 64U);
        expect(eq(got.size(), 64UZ));
        bool magnitudes = true;
        for (std::size_t n = 0UZ; n < got.size(); ++n) {
            magnitudes = magnitudes && std::abs(got[n] - static_cast<float>(n)) < 1e-2f;
        }
        expect(magnitudes) << "with one sample per frame the RMS is that sample";
    };
    // CLAUDE.md section 8 requires tag propagation coverage: a windowed block must carry a tag through and rescale
    // the sample rate by the ratio it declares, or downstream timing silently drifts
    "a tag survives the block and the sample rate follows the declared ratio"_test = [] {
        constexpr float kInputRate = 48000.f;

        gr::Graph flow({{"auto_size_edges_to_chunks", true}});
        auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(4096)}, {"sample_rate", kInputRate}, {"mark_tag", false}});
        source._tags     = {{512UZ, {{"key", "mid_stream"}}}};
        auto& dut        = flow.emplaceBlock<gr::filter::RootMeanSquare<float>>({{"frame_size", gr::Size_t(64)}});
        dut.settings().autoForwardParameters().insert("key"); // only `gr:`-prefixed keys cross a block boundary unaided
        auto& sink = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", false}});

        expect(flow.connect<"out", "in">(source, dut).has_value());
        expect(flow.connect<"out", "in">(dut, sink).has_value());

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value());
        expect(sched.runAndWait().has_value());

        expect(ge(sink._tags.size(), 1UZ)) << "the tag must reach the sink rather than being dropped by the block";
        expect(std::ranges::any_of(sink._tags, [](const auto& emitted) { return emitted.map.contains("key"); })) << "and carry its payload through unchanged, alongside the settings tags the source emits";

        const float expectedRate = kInputRate * static_cast<float>(dut.output_chunk_size) / static_cast<float>(dut.input_chunk_size);
        expect(std::abs(sink.sample_rate - expectedRate) < 1.f) << std::format("sink saw {} Hz, the declared ratio gives {} Hz", sink.sample_rate, expectedRate);
    };
};

int main() { /* tests run from the suite */ }
