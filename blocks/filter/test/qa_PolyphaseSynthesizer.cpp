#include <boost/ut.hpp>

#include <complex>
#include <format>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/device/SyclRuntime.hpp>
#include <gnuradio-4.0/filter/PolyphaseChannelizer.hpp>
#include <gnuradio-4.0/filter/PolyphaseSynthesizer.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

// the bank's numerics are proven in algorithm/test/qa_PolyphaseSynthesizerAlgo.cpp. Block-specific here: the
// input port collection following `n_channels`, the rate contract, and that analysis feeding synthesis
// reconstructs the band through a real graph.

namespace {
using C = std::complex<float>;

/// analysis into synthesis through a real graph; returns the mean magnitude past settling, which for a unit
/// tone is the round trip's gain
[[nodiscard]] float roundTripGain(std::string_view domain) {
    using namespace boost::ut;
    using namespace std::string_literals;
    using namespace gr::testing;
    constexpr gr::Size_t kChannels = 4U;

    gr::Graph flow({{"auto_size_edges_to_chunks", true}});

    // {1, i, -1, -i} is exactly a tone at the centre of channel 1 for a 4-channel bank
    std::vector<C> tone{C{1.f, 0.f}, C{0.f, 1.f}, C{-1.f, 0.f}, C{0.f, -1.f}};
    auto&          source = flow.emplaceBlock<TagSource<C>>({{"n_samples_max", gr::Size_t(16384)}, {"values", tone}, {"mark_tag", false}});
    auto&          split  = flow.emplaceBlock<gr::filter::PolyphaseChannelizer<C>>({{"n_channels", kChannels}, {"n_taps", gr::Size_t(32)}, {"outputs_per_frame", gr::Size_t(32)}, {"gr:compute_domain", std::string(domain)}});
    auto&          merge  = flow.emplaceBlock<gr::filter::PolyphaseSynthesizer<C>>({{"n_channels", kChannels}, {"n_taps", gr::Size_t(32)}, {"outputs_per_frame", gr::Size_t(32)}, {"gr:compute_domain", std::string(domain)}});
    auto&          sink   = flow.emplaceBlock<TagSink<C, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});

    expect(flow.connect(source, "out"s, split, "in"s).has_value());
    for (gr::Size_t k = 0U; k < kChannels; ++k) {
        expect(flow.connect(split, "out#"s + std::to_string(k), merge, "in#"s + std::to_string(k)).has_value());
    }
    expect(flow.connect(merge, "out"s, sink, "in"s).has_value());

    gr::scheduler::Simple<> sched;
    expect(sched.exchange(std::move(flow)).has_value());
    expect(sched.runAndWait().has_value());

    const auto& samples = sink._samples;
    expect(gt(samples.size(), 64UZ)) << std::format("the round trip on '{}' produced nothing", domain);
    if (samples.size() < 2UZ) {
        return 0.f;
    }
    float             mean = 0.f;
    const std::size_t from = samples.size() / 2UZ;
    for (std::size_t i = from; i < samples.size(); ++i) {
        mean += std::abs(samples[i]);
    }
    return mean / static_cast<float>(samples.size() - from);
}

/// the cases run from main() rather than a namespace-scope `boost::ut::suite`: a suite executes from the
/// runner's destructor, after ComputeRegistry's function-local static is gone, and a case that resolves a
/// compute domain then walks a freed map
void hostCases() {
    using namespace boost::ut;
    using gr::test::eq;

    "the input port collection follows n_channels"_test = [] {
        gr::filter::PolyphaseSynthesizer<C> block({{"n_channels", gr::Size_t(8)}, {"n_taps", gr::Size_t(64)}});
        block.settings().init();
        std::ignore = block.settings().applyStagedParameters();

        expect(eq(block.in.size(), 8UZ)) << "one channel port per channel";
        expect(gt(block.phaseLength(), 1UZ)) << "a single tap per arm is a bare commutator, not a filterbank";
    };

    "the rate contract multiplies the channel rate by n_channels"_test = [] {
        gr::filter::PolyphaseSynthesizer<C> block({{"n_channels", gr::Size_t(4)}, {"n_taps", gr::Size_t(32)}, {"outputs_per_frame", gr::Size_t(16)}});
        block.settings().init();
        std::ignore = block.settings().applyStagedParameters();

        expect(eq(static_cast<std::size_t>(block.output_chunk_size), 64UZ)) << "16 sets of 4 channels each";
        expect(eq(static_cast<std::size_t>(block.stride), 16UZ)) << "each channel stream advances one sample per output set";
        expect(gt(static_cast<std::size_t>(block.input_chunk_size), 16UZ)) << "the arms need history beyond the sets produced";
    };

    "analysis then synthesis reconstructs the band"_test = [] {
        // unity, not merely non-zero: the 1/nChannels folded into the synthesis twiddles is what makes the
        // analysis-synthesis pair a unity round trip, and only an absolute check can catch a factor of M there
        const float gain = roundTripGain("host");
        expect(approx(gain, 1.f, 0.05f)) << std::format("the round trip has gain {:.5f}, not unity", gain);
    };
}

} // namespace

int main() {
    using namespace boost::ut;

    hostCases();

    const bool syclAvailable = gr::device::registerSyclRuntime();
    expect(!syclAvailable || gr::device::hostSyclIsServed()) //
        << "a build with a SYCL backend must serve 'host:sycl'; without it every device case below skips and asserts nothing";
    if (!syclAvailable) {
        return 0;
    }

    // both banks carry a processBulk(ctx, ...) hatch, so this drives the whole analysis-synthesis pair on the
    // device and asserts the round trip still closes at unity there
    for (const auto* domain : {"host:sycl", "gpu:sycl"}) {
        const float gain = roundTripGain(domain);
        expect(approx(gain, 1.f, 0.05f)) << std::format("the round trip on '{}' has gain {:.5f}, not unity", domain, gain);
    }
}
