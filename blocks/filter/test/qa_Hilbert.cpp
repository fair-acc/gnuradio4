#include <boost/ut.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <format>
#include <numbers>
#include <ranges>
#include <span>
#include <string>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/filter/Hilbert.hpp>
#include <gnuradio-4.0/test/DeviceTestHelper.hpp>
#include <gnuradio-4.0/testing/StreamProbes.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace {
using namespace boost::ut;
using gr::testing::kTriggerKey;

constexpr std::size_t kTaps  = 65UZ;
constexpr std::size_t kDelay = kTaps / 2UZ;

[[nodiscard]] std::vector<float> tone(std::size_t n, float cyclesPerSample, float amplitude = 1.f) {
    std::vector<float> samples(n);
    for (std::size_t i = 0UZ; i < n; ++i) {
        samples[i] = amplitude * std::cos(2.f * std::numbers::pi_v<float> * cyclesPerSample * static_cast<float>(i));
    }
    return samples;
}

template<typename TBlock, typename... TOut>
void callInChunks(TBlock& block, const std::vector<float>& samples, std::size_t perCall, std::vector<TOut>&... outputs) {
    (outputs.resize(samples.size()), ...);
    for (std::size_t done = 0UZ; done < samples.size(); done += perCall) {
        const std::size_t now = std::min(perCall, samples.size() - done);
        expect(gr::work::Status::OK == block.processBulk(std::span(samples).subspan(done, now), std::span(outputs).subspan(done, now)...));
    }
}

[[nodiscard]] std::vector<std::complex<float>> analytic(const std::vector<float>& samples, std::size_t perCall) {
    gr::filter::HilbertTransform<float> block({{"n_taps", static_cast<gr::Size_t>(kTaps)}});
    block.init(block.progress);
    std::vector<std::complex<float>> output;
    callInChunks(block, samples, perCall, output);
    return output;
}

const auto definitions = [] {
    "the design is antisymmetric, zero at even offsets, odd in length, and 2/(pi k) windowed"_test = [] {
        const auto taps = gr::filter::fir::designHilbert<double>(64UZ);
        expect(eq(taps.size(), kTaps)) << "an even length rounds up";
        std::vector<double> window(kTaps);
        gr::algorithm::window::create(std::span<double>(window), gr::algorithm::window::Type::BlackmanHarris);
        for (std::size_t n = 0UZ; n < taps.size(); ++n) {
            const auto   k     = static_cast<std::ptrdiff_t>(n) - static_cast<std::ptrdiff_t>(kDelay);
            const double ideal = k % 2 == 0 ? 0. : 2. / (std::numbers::pi * static_cast<double>(k));
            expect(std::abs(taps[n] - ideal * window[n]) < 1e-15) << "tap " << n;
            expect(std::abs(taps[n] + taps[taps.size() - 1UZ - n]) < 1e-15) << "antisymmetric at " << n;
        }
    };

    "a tone comes out in quadrature at its own amplitude, after the filter delay"_test = [] {
        const auto input  = tone(2048UZ, 0.0625f, 0.8f);
        const auto output = analytic(input, 1024UZ);
        float      worst  = 0.f;
        for (std::size_t n = kTaps; n < output.size(); ++n) {
            const std::complex<float> expected = std::polar(0.8f, 2.f * std::numbers::pi_v<float> * 0.0625f * static_cast<float>(n - kDelay));
            worst                              = std::max(worst, std::abs(output[n] - expected));
        }
        expect(lt(worst, 1e-3f)) << std::format("worst deviation from x + jH(x) is {}", worst);
    };

    "the stream may be cut anywhere"_test = [] {
        const auto input = tone(1000UZ, 0.11f);
        const auto whole = analytic(input, 1000UZ);
        for (const std::size_t perCall : {1UZ, 7UZ, 64UZ, 333UZ}) {
            const auto cut = analytic(input, perCall);
            expect(std::ranges::equal(whole, cut, [](auto a, auto b) { return std::abs(a - b) < 1e-5f; })) << perCall << " samples per call";
        }
    };

    "a tag marks the same sample after the filter: it arrives (n_taps - 1) / 2 later, or not at all within that of the end"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       source = fixture.emplace<gr::testing::ProbeSource<float>>({{"values", tone(300UZ, 0.1f)}, {"n_samples_max", gr::Size_t(300)}, {"mark_tag", false}});
        source._tags                       = {{0UZ, {{kTriggerKey, "0"}, {"local_key", 0}}}, {100UZ, {{kTriggerKey, "100"}}}, {130UZ, {{kTriggerKey, "130"}}}, {299UZ - kDelay, {{kTriggerKey, "266"}}}, {299UZ, {{kTriggerKey, "299"}}}};
        auto& envelope                     = fixture.emplace<gr::filter::HilbertEnvelope<float>>({{"n_taps", static_cast<gr::Size_t>(kTaps)}});
        auto& sink                         = fixture.emplace<gr::testing::ProbeSink<float>>({{"log_samples", true}, {"log_tags", true}});
        expect(fixture.connect<"out", "in">(source, envelope).has_value());
        expect(fixture.connect<"out", "in">(envelope, sink).has_value());
        expect(fixture.run().has_value());
        const auto isMarked = [](const auto& tag) { return tag.map.contains(kTriggerKey); };
        const auto arrived  = sink._tags | std::views::filter(isMarked) | std::views::transform([](const auto& tag) { return tag.index; }) | std::ranges::to<std::vector<std::size_t>>();
        expect(arrived == std::vector<std::size_t>{kDelay, 100UZ + kDelay, 130UZ + kDelay, 299UZ}) << std::format("tags arrived at {}", arrived);
        expect(std::ranges::any_of(sink._tags, [](const auto& tag) { return tag.index == kDelay && tag.map.contains("local_key"); })) << "a key without gr: moves with its tag";
    };

    "a new length redesigns the filter and empties its lead-in"_test = [] {
        gr::filter::HilbertTransform<float> block({{"n_taps", gr::Size_t(31)}});
        block.init(block.progress);
        std::vector<std::complex<float>> output;
        callInChunks(block, tone(200UZ, 0.1f), 100UZ, output);
        std::ignore = block.settings().setStaged({{"n_taps", static_cast<gr::Size_t>(kTaps)}});
        std::ignore = block.settings().applyStagedParameters();
        expect(eq(block._taps.size(), kTaps));
        expect(std::ranges::all_of(block._leadIn, [](float sample) { return sample == 0.f; })) << "nothing of the earlier stream is convolved into the next";
    };

    "reset forgets the stream so far: silence after it comes out as silence"_test = [] {
        gr::filter::HilbertEnvelope<float> block({{"n_taps", static_cast<gr::Size_t>(kTaps)}});
        block.init(block.progress);
        std::vector<float> magnitude;
        callInChunks(block, tone(200UZ, 0.1f), 100UZ, magnitude);
        block.reset();
        callInChunks(block, std::vector<float>(100UZ, 0.f), 100UZ, magnitude);
        expect(std::ranges::all_of(magnitude, [](float sample) { return sample == 0.f; })) << "nothing of the earlier tone is convolved after the reset";
    };

    "the two-port and envelope forms are the parts and magnitude of the analytic signal"_test = [] {
        std::vector<float> modulated = tone(2048UZ, 0.125f);
        for (std::size_t n = 0UZ; n < modulated.size(); ++n) {
            modulated[n] *= 1.f + 0.5f * std::sin(2.f * std::numbers::pi_v<float> * static_cast<float>(n) / 512.f);
        }
        const auto whole = analytic(modulated, 1024UZ);

        gr::filter::HilbertTransformRealImag<float> parts({{"n_taps", static_cast<gr::Size_t>(kTaps)}});
        parts.init(parts.progress);
        std::vector<float> re;
        std::vector<float> im;
        callInChunks(parts, modulated, 1024UZ, re, im);

        gr::filter::HilbertEnvelope<float> envelope({{"n_taps", static_cast<gr::Size_t>(kTaps)}});
        envelope.init(envelope.progress);
        std::vector<float> magnitude;
        callInChunks(envelope, modulated, 1024UZ, magnitude);

        const auto  allSamples   = std::views::iota(0UZ, modulated.size());
        const auto  afterLeadIn  = std::views::iota(kTaps, modulated.size());
        const float worstParts   = std::ranges::max(allSamples | std::views::transform([&](std::size_t n) { return std::abs(std::complex(re[n], im[n]) - whole[n]); }));
        const float worstMag     = std::ranges::max(allSamples | std::views::transform([&](std::size_t n) { return std::abs(magnitude[n] - std::abs(whole[n])); }));
        const auto  modulationAt = [](std::size_t n) { return 1.f + 0.5f * std::sin(2.f * std::numbers::pi_v<float> * static_cast<float>(n - kDelay) / 512.f); };
        const float worstAm      = std::ranges::max(afterLeadIn | std::views::transform([&](std::size_t n) { return std::abs(magnitude[n] - modulationAt(n)); }));
        expect(lt(worstParts, 1e-6f));
        expect(lt(worstMag, 1e-6f));
        expect(lt(worstAm, 5e-3f)) << std::format("the envelope follows the modulation to {}", worstAm);
    };
};

const auto domains = [] {
    using gr::testing::operator""_domain_test;

    "the analytic signal is the same on every served domain"_domain_test = [](auto& fixture, std::string_view domain) {
        const std::string placed = std::string(domain) + "!";
        const auto        input  = tone(4096UZ, 0.037f, 2.f);
        const auto        whole  = analytic(input, 1024UZ);

        auto& source       = fixture.template emplace<gr::testing::ProbeSource<float>>({{"values", input}, {"n_samples_max", static_cast<gr::Size_t>(input.size())}, {"mark_tag", false}});
        auto& hilbert      = fixture.template emplace<gr::filter::HilbertTransform<float>>({{"n_taps", static_cast<gr::Size_t>(kTaps)}, {"gr:compute_domain", placed}});
        auto& parts        = fixture.template emplace<gr::filter::HilbertTransformRealImag<float>>({{"n_taps", static_cast<gr::Size_t>(kTaps)}, {"gr:compute_domain", placed}});
        auto& envelope     = fixture.template emplace<gr::filter::HilbertEnvelope<float>>({{"n_taps", static_cast<gr::Size_t>(kTaps)}, {"gr:compute_domain", placed}});
        auto& complexSink  = fixture.template emplace<gr::testing::ProbeSink<std::complex<float>>>({{"log_samples", true}});
        auto& realSink     = fixture.template emplace<gr::testing::ProbeSink<float>>({{"log_samples", true}});
        auto& imagSink     = fixture.template emplace<gr::testing::ProbeSink<float>>({{"log_samples", true}});
        auto& envelopeSink = fixture.template emplace<gr::testing::ProbeSink<float>>({{"log_samples", true}});
        expect(fixture.template connect<"out", "in">(source, hilbert).has_value());
        expect(fixture.template connect<"out", "in">(source, parts).has_value());
        expect(fixture.template connect<"out", "in">(source, envelope).has_value());
        expect(fixture.template connect<"out", "in">(hilbert, complexSink).has_value());
        expect(fixture.template connect<"real", "in">(parts, realSink).has_value());
        expect(fixture.template connect<"imag", "in">(parts, imagSink).has_value());
        expect(fixture.template connect<"out", "in">(envelope, envelopeSink).has_value());
        const auto ran = fixture.run();
        expect(ran.has_value()) << domain << (ran.has_value() ? std::string{} : ran.error().message);

        expect(fatal(eq(complexSink._samples.size(), whole.size()))) << domain;
        expect(fatal(eq(realSink._samples.size(), whole.size()))) << domain;
        expect(fatal(eq(imagSink._samples.size(), whole.size()))) << domain;
        expect(fatal(eq(envelopeSink._samples.size(), whole.size()))) << domain;
        const auto deviationAt = [&](std::size_t n) {
            const std::complex<float> fromParts(realSink._samples[n], imagSink._samples[n]);
            return std::max({std::abs(complexSink._samples[n] - whole[n]), std::abs(fromParts - whole[n]), std::abs(envelopeSink._samples[n] - std::abs(whole[n]))});
        };
        const float worst = std::ranges::max(std::views::iota(0UZ, whole.size()) | std::views::transform(deviationAt));
        expect(lt(worst, 1e-4f)) << domain;
    } | gr::testing::overGraphs(gr::testing::kAllDomains);
};

} // namespace

int main() {
    "Hilbert transform"_test                 = definitions;
    "Hilbert transform on every domain"_test = domains;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
