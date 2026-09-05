#include <boost/ut.hpp>

#include <cmath>
#include <format>
#include <print>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/device/SyclRuntime.hpp>
#include <gnuradio-4.0/filter/RationalResampler.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace {
using namespace gr::testing;

struct RunResult {
    std::vector<float>   samples;
    gr::Size_t           inputChunk  = 0U;
    gr::Size_t           outputChunk = 0U;
    gr::Size_t           hop         = 0U;
    gr::lifecycle::State state       = gr::lifecycle::State::IDLE;
};

[[nodiscard]] RunResult runResampler(std::string_view domain, gr::Size_t interpolation, gr::Size_t decimation, gr::Size_t nSamples) {
    gr::Graph flow;
    flow.autoSizeEdgesToChunks = true;

    auto& source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", nSamples}, {"mark_tag", false}});
    auto& dut    = flow.emplaceBlock<gr::filter::RationalResampler<float>>({{"gr:compute_domain", std::string(domain)}, //
           {"interpolation", interpolation}, {"decimation", decimation}, {"n_taps", gr::Size_t(24)}});
    auto& sink   = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});

    boost::ut::expect(flow.connect<"out", "in">(source, dut).has_value());
    boost::ut::expect(flow.connect<"out", "in">(dut, sink).has_value());

    gr::scheduler::Simple<> sched;
    boost::ut::expect(sched.exchange(std::move(flow)).has_value());
#if __cpp_exceptions
    try {
        std::ignore = sched.runAndWait();
    } catch (...) { // a refused block ends the run in ERROR, which the caller asserts on
    }
#else
    std::ignore = sched.runAndWait();
#endif

    RunResult result{.samples = std::vector<float>(sink._samples.size()), .inputChunk = dut.input_chunk_size, .outputChunk = dut.output_chunk_size, .hop = dut.stride, .state = dut.state()};
    for (std::size_t i = 0UZ; i < result.samples.size(); ++i) {
        result.samples[i] = sink._samples[i];
    }
    return result;
}
} // namespace

int main() {
    using namespace boost::ut;

    std::ignore = gr::device::registerSyclRuntime();

    "the declared window is one hop of decimation in, interpolation out"_test = [] {
        const RunResult run = runResampler("host", 3U, 2U, 512U);
        expect(eq(run.outputChunk, gr::Size_t(3))) << "one window must produce exactly `interpolation` samples";
        expect(eq(run.hop, gr::Size_t(2))) << "and advance by `decimation`, or the phases drift";
        expect(gt(run.inputChunk, run.hop)) << "the window has to reach back over the filter history";
    };

    "the output rate follows interpolation/decimation"_test = [] {
        constexpr gr::Size_t kSamples = 4096U;
        for (const auto [L, M] : std::vector<std::pair<gr::Size_t, gr::Size_t>>{{1U, 1U}, {3U, 2U}, {2U, 3U}, {5U, 4U}}) {
            const RunResult run      = runResampler("host", L, M, kSamples);
            const double    ratio    = static_cast<double>(run.samples.size()) / static_cast<double>(kSamples);
            const double    expected = static_cast<double>(L) / static_cast<double>(M);
            expect(std::abs(ratio - expected) < 0.05) << std::format("L/M = {}/{} produced {} samples from {}", L, M, run.samples.size(), kSamples);
        }
    };

    "a ramp is converted at the rate it was asked for"_test = [] {
        const RunResult run = runResampler("host", 3U, 2U, 2048U);
        expect(gt(run.samples.size(), 256UZ));

        // the source ramps by one per input sample, so consecutive outputs must advance by decimation/interpolation
        const std::size_t first = 64UZ;
        const std::size_t last  = run.samples.size() - 64UZ;
        const float       mean  = (run.samples[last] - run.samples[first]) / static_cast<float>(last - first);
        expect(std::abs(mean - 2.f / 3.f) < 1e-3f) << std::format("average slope {} is not decimation/interpolation", mean);

        float worst = 0.f; // individual steps ripple: each phase is a different fractional delay, none of them ideal
        for (std::size_t n = first; n + 1UZ < last; ++n) {
            worst = std::max(worst, std::abs((run.samples[n + 1UZ] - run.samples[n]) - 2.f / 3.f));
        }
        expect(lt(worst, 0.1f)) << std::format("worst step deviates by {}, far more than a 24-tap prototype should", worst);
    };

    "every served device returns what the host returns"_test = [] {
        const RunResult host = runResampler("host", 3U, 2U, 2048U);
        for (std::string_view domain : {"host:sycl", "gpu:sycl"}) {
            if (gr::device::DeviceContextRegistry::instance().tryResolve(domain) == nullptr) {
                continue;
            }
            const RunResult onDevice = runResampler(domain, 3U, 2U, 2048U);
            expect(eq(onDevice.samples.size(), host.samples.size())) << std::format("'{}' produced a different number of samples", domain);
            bool matches = onDevice.samples.size() == host.samples.size();
            for (std::size_t i = 0UZ; matches && i < host.samples.size(); ++i) {
                matches = std::abs(onDevice.samples[i] - host.samples[i]) < 1e-3f;
            }
            expect(matches) << std::format("'{}' must return what the same block returns on the host", domain);
        }
    };

    return 0;
}
