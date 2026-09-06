#include <boost/ut.hpp>

#include <cmath>
#include <format>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/filter/DriftResampler.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace {
using namespace gr::testing;

[[nodiscard]] std::vector<float> runDrift(float ratio, gr::Size_t nSamples) {
    gr::Graph flow;
    auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", nSamples}, {"mark_tag", false}});
    auto&     dut    = flow.emplaceBlock<gr::filter::DriftResampler<float>>({{"ratio", ratio}});
    auto&     sink   = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});

    boost::ut::expect(flow.connect<"out", "in">(source, dut).has_value());
    boost::ut::expect(flow.connect<"out", "in">(dut, sink).has_value());
    gr::scheduler::Simple<> sched;
    boost::ut::expect(sched.exchange(std::move(flow)).has_value());
    std::ignore = sched.runAndWait();

    std::vector<float> samples(sink._samples.size());
    for (std::size_t i = 0UZ; i < samples.size(); ++i) {
        samples[i] = sink._samples[i];
    }
    return samples;
}
} // namespace

const boost::ut::suite<"DriftResampler"> _driftResampler = [] {
    using namespace boost::ut;

    "the output count follows the ratio, whatever it is"_test = [] {
        constexpr gr::Size_t kSamples = 8192U;
        for (const float ratio : {0.5f, 0.997f, 1.f, 1.003f, 1.5f}) {
            const std::vector<float> got      = runDrift(ratio, kSamples);
            const double             observed = static_cast<double>(got.size()) / static_cast<double>(kSamples);
            expect(std::abs(observed - static_cast<double>(ratio)) < 0.02) << std::format("ratio {} gave {} samples from {}", ratio, got.size(), kSamples);
        }
    };

    "a ramp survives an irrational ratio as a ramp"_test = [] {
        // the source ramps by one per input sample, so the output must ramp by 1/ratio per output sample
        constexpr float          kRatio = 1.003f;
        const std::vector<float> got    = runDrift(kRatio, 8192U);
        expect(gt(got.size(), 256UZ));

        const std::size_t first = 64UZ;
        const std::size_t last  = got.size() - 64UZ;
        const double      slope = static_cast<double>(got[last] - got[first]) / static_cast<double>(last - first);
        expect(std::abs(slope - 1.0 / static_cast<double>(kRatio)) < 1e-3) << std::format("average slope {} is not the inverse of the ratio", slope);
    };

    "an unchanged rate is a pass-through"_test = [] {
        const std::vector<float> got = runDrift(1.f, 4096U);
        expect(gt(got.size(), 1024UZ));
        bool identity = true;
        for (std::size_t n = 0UZ; n < std::min<std::size_t>(512UZ, got.size()); ++n) {
            identity = identity && std::abs(got[n] - static_cast<float>(n)) < 1e-2f;
        }
        expect(identity) << "at a ratio of one the interpolator has nothing to do";
    };
};

int main() { /* tests run from the suite */ }
