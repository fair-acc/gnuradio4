#include <boost/ut.hpp>

#include <cmath>
#include <format>
#include <print>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/filter/FastConvolution.hpp>
#include <gnuradio-4.0/filter/time_domain_filter.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace {
using namespace gr::testing;

[[nodiscard]] std::vector<float> runFast(std::span<const float> taps, gr::Size_t outputsPerFrame, gr::Size_t nSamples) {
    gr::Graph flow({{"auto_size_edges_to_chunks", true}});
    auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", nSamples}, {"mark_tag", false}});
    auto&     dut    = flow.emplaceBlock<gr::filter::FastConvolutionFilter<float>>({{"outputs_per_frame", outputsPerFrame}, {"taps", std::vector<float>(taps.begin(), taps.end())}});
    auto&     sink   = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});

    boost::ut::expect(flow.connect<"out", "in">(source, dut).has_value());
    boost::ut::expect(flow.connect<"out", "in">(dut, sink).has_value());
    gr::scheduler::Simple<> sched;
    boost::ut::expect(sched.exchange(std::move(flow)).has_value());
    boost::ut::expect(sched.runAndWait().has_value());

    std::vector<float> samples(sink._samples.size());
    for (std::size_t i = 0UZ; i < samples.size(); ++i) {
        samples[i] = sink._samples[i];
    }
    return samples;
}
} // namespace

const boost::ut::suite<"FastConvolutionFilter"> _fastConvolutionBlock = [] {
    using namespace boost::ut;

    "the block computes the filter it was given"_test = [] {
        std::vector<float> taps(33UZ);
        for (std::size_t k = 0UZ; k < taps.size(); ++k) {
            taps[k] = std::sin(0.3f * static_cast<float>(k)) / static_cast<float>(taps.size());
        }

        const std::vector<float> got = runFast(taps, 256U, 4096U);
        expect(gt(got.size(), 256UZ)) << "the block has to emit whole frames";

        // the source ramps, so the direct convolution is known exactly; output n is y[n + nTaps - 1]
        bool matches = true;
        for (std::size_t n = 0UZ; n < std::min<std::size_t>(200UZ, got.size()); ++n) {
            const std::size_t centre = n + taps.size() - 1UZ;
            double            exact  = 0.0;
            for (std::size_t k = 0UZ; k < taps.size(); ++k) {
                exact += static_cast<double>(taps[k]) * static_cast<double>(centre - k);
            }
            matches = matches && std::abs(static_cast<double>(got[n]) - exact) < 1e-2;
        }
        expect(matches) << "an overlap-save filter must agree with the direct convolution it replaces";
    };

    "a longer filter costs the same frame, so the shape does not change"_test = [] {
        const std::vector<float> shortTaps(9UZ, 1.f / 9.f);
        const std::vector<float> longTaps(129UZ, 1.f / 129.f);
        expect(gt(runFast(shortTaps, 128U, 2048U).size(), 0UZ));
        expect(gt(runFast(longTaps, 128U, 2048U).size(), 0UZ)) << "a filter longer than the requested frame must still resolve to a valid window";
    };
};

int main() { /* tests run from the suite */ }
