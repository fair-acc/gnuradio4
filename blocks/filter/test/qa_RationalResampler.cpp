#include <boost/ut.hpp>

#include <algorithm>
#include <cmath>
#include <format>
#include <optional>
#include <print>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/device/SyclRuntime.hpp>
#include <gnuradio-4.0/filter/RationalResampler.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/testing/DeviceExpectation.hpp>
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

struct Retune {
    std::size_t atSample      = 0UZ;
    gr::Size_t  interpolation = 0U;
    gr::Size_t  decimation    = 0U;
};

[[nodiscard]] RunResult runResampler(std::string_view domain, gr::Size_t interpolation, gr::Size_t decimation, gr::Size_t nSamples, std::optional<Retune> retune = std::nullopt) {
    gr::Graph flow({{"auto_size_edges_to_chunks", true}});

    auto& source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", nSamples}, {"mark_tag", false}});
    // a setting named in the constructor is no longer auto-updated from tags, so a retuned run has to be driven
    // entirely by tags -- including its initial ratio
    gr::property_map initialSettings{{"gr:compute_domain", std::string(domain)}, {"n_taps", gr::Size_t(24)}};
    if (!retune.has_value()) {
        initialSettings["interpolation"] = interpolation;
        initialSettings["decimation"]    = decimation;
    }
    auto& dut  = flow.emplaceBlock<gr::filter::RationalResampler<float>>(initialSettings);
    auto& sink = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});

    if (retune.has_value()) {
        source._tags.emplace_back(0UZ, gr::property_map{{"interpolation", interpolation}, {"decimation", decimation}});
        source._tags.emplace_back(retune->atSample, gr::property_map{{"interpolation", retune->interpolation}, {"decimation", retune->decimation}});
    }

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
    boost::ut::expect(!gr::device::registerSyclRuntime() || gr::device::hostSyclIsServed()) //
        << "a build with a SYCL backend must serve 'host:sycl'; without it every device case below skips and asserts nothing";

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
            if (!gr::device::DeviceContextRegistry::instance().isServedExactly(domain)) {
                boost::ut::expect(!gr::testing::deviceDomainRequired(domain)) << "GR4_REQUIRE_DEVICE names this domain, so the lane must exercise it rather than skip";
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

    // the phase length must follow from the reflected settings alone, so a block that was retuned into a ratio must
    // declare the same window as one constructed with it. Host-only: a settings tag does not reach a device mirror
    // at all today, which is a separate defect and the reason this cannot yet be asserted across domains.
    "a retuned block declares the same window as one built with those settings"_test = [] {
        constexpr Retune kRetune{.atSample = 1024UZ, .interpolation = 2U, .decimation = 3U};

        const RunResult retuned = runResampler("host", 3U, 2U, 4096U, kRetune);
        const RunResult direct  = runResampler("host", 2U, 3U, 4096U);

        expect(eq(retuned.outputChunk, direct.outputChunk)) << "one window must still produce `interpolation` samples";
        expect(eq(retuned.hop, direct.hop)) << "and advance by `decimation`";
        expect(eq(retuned.inputChunk, direct.inputChunk)) << "the window reaches back over the phase length, which the retune has to re-derive";
    };

    // CLAUDE.md section 8 requires tag propagation coverage: a windowed block must carry a tag through and rescale
    // the sample rate by the ratio it declares, or downstream timing silently drifts
    "a tag survives the block and the sample rate follows the declared ratio"_test = [] {
        constexpr float kInputRate = 48000.f;

        gr::Graph flow({{"auto_size_edges_to_chunks", true}});
        auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(4096)}, {"sample_rate", kInputRate}, {"mark_tag", false}});
        source._tags     = {{512UZ, {{"key", "mid_stream"}}}};
        auto& dut        = flow.emplaceBlock<gr::filter::RationalResampler<float>>({{"interpolation", gr::Size_t(3)}, {"decimation", gr::Size_t(2)}, {"n_taps", gr::Size_t(24)}});
        dut.settings().autoForwardParameters().insert("key"); // only `gr:`-prefixed keys cross a block boundary unaided
        auto& sink = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", false}});

        expect(flow.connect<"out", "in">(source, dut).has_value());
        expect(flow.connect<"out", "in">(dut, sink).has_value());

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value());
        expect(sched.runAndWait().has_value());

        expect(ge(sink._tags.size(), 1UZ)) << "the tag must reach the sink rather than being dropped by the block";
        expect(std::ranges::any_of(sink._tags, [](const auto& emitted) { return emitted.map.contains("key"); })) << "and carry its payload through unchanged, alongside the settings tags the source emits";

        // the PHYSICAL rate, not the framework's own formula: deriving the expectation from the chunk sizes makes
        // this assertion a tautology that stays green whatever the block reports
        const float expectedRate = kInputRate * static_cast<float>(dut.interpolation) / static_cast<float>(dut.decimation);
        expect(std::abs(sink.sample_rate - expectedRate) < 1.f) << std::format("sink saw {} Hz, interpolation/decimation gives {} Hz", sink.sample_rate, expectedRate);
    };

    return 0;
}
