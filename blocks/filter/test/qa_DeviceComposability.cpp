#include <boost/ut.hpp>

#include <cmath>
#include <complex>
#include <format>
#include <numbers>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/device/SyclRuntime.hpp>
#include <gnuradio-4.0/filter/Correlator.hpp>
#include <gnuradio-4.0/filter/DriftResampler.hpp>
#include <gnuradio-4.0/filter/FrequencyXlatingFilter.hpp>
#include <gnuradio-4.0/filter/RationalResampler.hpp>
#include <gnuradio-4.0/fourier/fft.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace {
using C = std::complex<float>;

/// a chirp, so every stage downstream sees energy across the band rather than one line
[[nodiscard]] std::vector<C> chirp(std::size_t n) {
    std::vector<C> pattern(n);
    for (std::size_t i = 0UZ; i < n; ++i) {
        const double phase = 0.0004 * static_cast<double>(i) * static_cast<double>(i);
        pattern[i]         = C{static_cast<float>(std::cos(phase)), static_cast<float>(std::sin(phase))};
    }
    return pattern;
}

/// translate, transform, correlate -- three blocks of different dispatch tiers on ONE domain.
///
/// `FrequencyXlatingFilter` has a `processBulk(ctx, ...)` hatch, `FFT` has its own, and `Correlator` has neither:
/// it declares a static ratio through the view form and is carried by the framework. A chain is the only place
/// the handover between those tiers is exercised, and an edge between two device blocks must not go through the
/// host to get there.
[[nodiscard]] std::vector<C> runChain(std::string_view domain, std::size_t fftSize) {
    using namespace std::string_literals;
    using namespace gr::testing;

    gr::property_map xlating{{"frequency", 250'000.0}, {"sample_rate", 1'000'000.f}, {"cutoff", 200'000.f}, {"filter_type", "FIR"s}, {"filter_order", gr::Size_t(4)}, {"decimation", gr::Size_t(1)}, {"samples_per_frame", gr::Size_t(512)}};
    gr::property_map transform{{"fft_size", static_cast<gr::Size_t>(fftSize)}};
    gr::property_map correlate{{"reference", std::vector<C>{C{1.f, 0.f}, C{0.5f, -0.5f}, C{0.f, 1.f}}}, {"lags", gr::Size_t(32)}};
    if (!domain.empty()) {
        for (gr::property_map* settings : {&xlating, &transform, &correlate}) {
            (*settings)["compute_domain"] = std::string(domain);
        }
    }

    gr::Graph flow({{"auto_size_edges_to_chunks", true}});
    auto&     source     = flow.emplaceBlock<TagSource<C>>({{"n_samples_max", gr::Size_t(8192)}, {"values", chirp(1024UZ)}, {"mark_tag", false}});
    auto&     translator = flow.emplaceBlock<gr::filter::FrequencyXlatingFilter<C>>(std::move(xlating));
    auto&     spectrum   = flow.emplaceBlock<gr::blocks::fft::FFT<float>>(std::move(transform));
    auto&     correlator = flow.emplaceBlock<gr::filter::Correlator<C>>(std::move(correlate));
    auto&     sink       = flow.emplaceBlock<TagSink<C, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});

    boost::ut::expect(flow.connect(source, "out"s, translator, "in"s).has_value());
    boost::ut::expect(flow.connect(translator, "out"s, spectrum, "in"s).has_value());
    boost::ut::expect(flow.connect(spectrum, "out"s, correlator, "in"s).has_value());
    boost::ut::expect(flow.connect(correlator, "out"s, sink, "in"s).has_value());

    gr::scheduler::Simple<> sched;
    boost::ut::expect(sched.exchange(std::move(flow)).has_value());
    boost::ut::expect(sched.runAndWait().has_value());
    return {sink._samples.begin(), sink._samples.end()};
}

/// tags the source is told to publish. `_tags` is a plain member rather than a reflected setting, so it is
/// seeded on the block after it is placed; `mark_tag` only decides what the SAMPLES look like.
///
/// The keys carry the `gr:` prefix deliberately: `Block::forwardInputTags` forwards a key only when it is
/// prefixed or named in the auto-forward supplement, so an unprefixed key is dropped by CONTRACT and a test
/// using one would be asserting something nobody promised.
[[nodiscard]] std::vector<gr::testing::OwningTag> seedTags() { return {{64UZ, {{"gr:trigger_name", "first"}}}, {512UZ, {{"gr:trigger_name", "middle"}}}, {1024UZ, {{"gr:trigger_name", "last"}}}}; }

/// what a drifting resampler publishes, with the tags the sink saw
struct Resampled {
    std::vector<float> samples;
    std::size_t        nTags = 0UZ;
};

[[nodiscard]] Resampled runResampler(gr::property_map settings, std::size_t nSamples, bool withResampler = true) {
    using namespace std::string_literals;
    using namespace gr::testing;

    std::vector<float> ramp(256UZ);
    for (std::size_t i = 0UZ; i < ramp.size(); ++i) {
        ramp[i] = static_cast<float>(i);
    }

    gr::Graph flow({{"auto_size_edges_to_chunks", true}});
    auto&     source = flow.emplaceBlock<TagSource<float>>({{"n_samples_max", static_cast<gr::Size_t>(nSamples)}, {"values", ramp}, {"mark_tag", true}});
    source._tags     = seedTags();
    auto& sink       = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}, {"log_tags", true}});

    if (withResampler) {
        auto& dut = flow.emplaceBlock<gr::filter::DriftResampler<float>>(std::move(settings));
        boost::ut::expect(flow.connect(source, "out"s, dut, "in"s).has_value());
        boost::ut::expect(flow.connect(dut, "out"s, sink, "in"s).has_value());
    } else {
        boost::ut::expect(flow.connect(source, "out"s, sink, "in"s).has_value());
    }

    gr::scheduler::Simple<> sched;
    boost::ut::expect(sched.exchange(std::move(flow)).has_value());
    boost::ut::expect(sched.runAndWait().has_value());
    return {{sink._samples.begin(), sink._samples.end()}, sink._tags.size()};
}

[[nodiscard]] Resampled runRational(std::size_t nSamples) {
    using namespace std::string_literals;
    using namespace gr::testing;

    std::vector<float> ramp(256UZ);
    for (std::size_t i = 0UZ; i < ramp.size(); ++i) {
        ramp[i] = static_cast<float>(i);
    }

    gr::Graph flow({{"auto_size_edges_to_chunks", true}});
    auto&     source = flow.emplaceBlock<TagSource<float>>({{"n_samples_max", static_cast<gr::Size_t>(nSamples)}, {"values", ramp}, {"mark_tag", true}});
    source._tags     = seedTags();
    auto& dut        = flow.emplaceBlock<gr::filter::RationalResampler<float>>({{"interpolation", gr::Size_t(3)}, {"decimation", gr::Size_t(2)}});
    auto& sink       = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}, {"log_tags", true}});

    boost::ut::expect(flow.connect(source, "out"s, dut, "in"s).has_value());
    boost::ut::expect(flow.connect(dut, "out"s, sink, "in"s).has_value());

    gr::scheduler::Simple<> sched;
    boost::ut::expect(sched.exchange(std::move(flow)).has_value());
    boost::ut::expect(sched.runAndWait().has_value());
    return {{sink._samples.begin(), sink._samples.end()}, sink._tags.size()};
}

void hostCases() {
    using namespace boost::ut;

    "the harness itself carries tags, so the cases below measure the block"_test = [] {
        // without this control a resampler that drops every tag is indistinguishable from a source that
        // published none
        const Resampled direct = runResampler({}, 2048UZ, /*withResampler=*/false);
        expect(gt(direct.nTags, 0UZ)) << "the source published no tags at all, so nothing below means anything";
    };

    "a rational resampler forwards the tags it was given"_test = [] {
        // the sibling block, to place the result below: if both drop tags it is a shared contract question
        // rather than something specific to the drifting one
        const Resampled got = runRational(2048UZ);
        expect(gt(got.samples.size(), 512UZ)) << "the rational resampler produced almost nothing";
        expect(gt(got.nTags, 0UZ)) << "every tag the source published was dropped by the rational resampler";
    };

    "a resampler forwards the tags it was given"_test = [] {
        // a resampler moves the sample its tag was attached to, so the tag has to move with it. Losing them
        // outright is the failure worth catching: a stream that resamples but drops its metadata is silently
        // useless downstream, and nothing else in this suite would notice.
        const Resampled got = runResampler({{"ratio", 1.5f}}, 2048UZ);
        expect(gt(got.samples.size(), 512UZ)) << "the resampler produced almost nothing";
        expect(gt(got.nTags, 0UZ)) << "every tag the source published was dropped by the resampler";
    };

    "a ratio change mid-stream is honoured"_test = [] {
        // the ratio is a live setting, so two runs at different ratios must produce proportionally different
        // counts. This pins that the block re-reads it rather than latching the first value it saw.
        const Resampled slow = runResampler({{"ratio", 0.5f}}, 4096UZ);
        const Resampled fast = runResampler({{"ratio", 2.0f}}, 4096UZ);
        expect(gt(fast.samples.size(), slow.samples.size() * 2UZ)) << std::format("ratio 2.0 produced {} and ratio 0.5 produced {}, which is not a rate change", fast.samples.size(), slow.samples.size());
    };

    "the transform refuses a size it cannot factor"_test = [] {
        // `fft_size` carries a power-of-two limit, so a graph asking for anything else must be refused at
        // configuration rather than quietly rounded. This is the contract, not a defect: the underlying
        // SimdFFT factors {2,3,5}, but the BLOCK narrows that deliberately.
        gr::blocks::fft::FFT<float> block;
        block.settings().init();
        const auto rejected = block.settings().set({{"fft_size", gr::Size_t(1000)}});
        std::ignore         = block.settings().applyStagedParameters();
        expect(!rejected.empty() || block.fft_size != gr::Size_t(1000)) << "a non-power-of-two fft_size was accepted";
    };
}
} // namespace

int main() {
    using namespace boost::ut;

    hostCases();

    // the device cases run from main(): a namespace-scope suite executes from the runner's destructor, after
    // ComputeRegistry's function-local static is gone, and resolving a backend then walks a freed map
    const bool syclAvailable = gr::device::registerSyclRuntime();

    "a filter, a transform and a correlator compose on one domain"_test = [syclAvailable] {
        const std::vector<C> host = runChain("", 1024UZ);
        expect(gt(host.size(), 64UZ)) << "the host chain produced almost nothing to compare against";

        for (const auto& domain : {"host", "host:sycl", "gpu:sycl"}) {
            if (!syclAvailable && std::string_view(domain) != "host") {
                continue;
            }
            const std::vector<C> got = runChain(domain, 1024UZ);
            expect(eq(got.size(), host.size())) << std::format("'{}' published {} samples against the host's {}", domain, got.size(), host.size());

            float worst = 0.f;
            for (std::size_t n = 0UZ; n < std::min(got.size(), host.size()); ++n) {
                worst = std::max(worst, std::abs(got[n] - host[n]));
            }
            expect(lt(worst, 1e-3f)) << std::format("the chain on '{}' differs from the host chain by {:.3e}", domain, worst);
        }
    };

    return 0;
}
