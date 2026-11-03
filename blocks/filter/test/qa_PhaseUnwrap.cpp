#include <boost/ut.hpp>

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <numbers>
#include <ranges>
#include <span>
#include <string>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/filter/PhaseUnwrap.hpp>
#include <gnuradio-4.0/test/DeviceTestHelper.hpp>
#include <gnuradio-4.0/testing/StreamProbes.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace {
using namespace boost::ut;
using gr::filter::UnwrapAxis;

[[nodiscard]] float wrapped(float phase) { return std::atan2(std::sin(phase), std::cos(phase)); }

[[nodiscard]] std::vector<float> phases(std::size_t frameSize, std::size_t nFrames, bool wrap) {
    std::vector<float> samples(frameSize * nFrames);
    for (std::size_t f = 0UZ; f < nFrames; ++f) {
        for (std::size_t p = 0UZ; p < frameSize; ++p) {
            const float phase          = (0.3f + 2.5f * static_cast<float>(p) / static_cast<float>(frameSize)) * static_cast<float>(f) + 0.9f * static_cast<float>(p) - 1.f;
            samples[f * frameSize + p] = wrap ? wrapped(phase) : phase;
        }
    }
    return samples;
}

[[nodiscard]] std::vector<float> unwrapped(const std::vector<float>& samples, std::size_t frameSize, UnwrapAxis axis, std::size_t framesPerCall) {
    gr::filter::PhaseUnwrap<float> block({{"frame_size", static_cast<gr::Size_t>(frameSize)}, {"axis", std::string(gr::meta::enumName(axis).value_or(""))}});
    block.init(block.progress);
    std::vector<float> output(samples.size());
    for (std::size_t done = 0UZ; done < samples.size(); done += framesPerCall * frameSize) {
        const std::size_t now = std::min(framesPerCall * frameSize, samples.size() - done);
        expect(gr::work::Status::OK == block.processBulk(std::span<const float>(samples).subspan(done, now), std::span(output).subspan(done, now)));
    }
    return output;
}

const auto definitions = [] {
    "a wrapped phase ramp comes back as the ramp, however the stream is cut"_test = [] {
        const auto truth = phases(1UZ, 500UZ, false);
        for (const std::size_t perCall : {1UZ, 7UZ, 500UZ}) {
            const auto  got       = unwrapped(phases(1UZ, 500UZ, true), 1UZ, UnwrapAxis::frames, perCall);
            const auto  deviation = [&](std::size_t i) { return std::abs(got[i] - truth[i] - (got[0] - truth[0])); };
            const float worst     = std::ranges::max(std::views::iota(0UZ, got.size()) | std::views::transform(deviation));
            expect(lt(worst, 1e-3f)) << perCall << " samples per call";
        }
    };

    "across frames, every position keeps its own count of turns"_test = [] {
        const auto  truth     = phases(6UZ, 200UZ, false);
        const auto  got       = unwrapped(phases(6UZ, 200UZ, true), 6UZ, UnwrapAxis::frames, 3UZ);
        const auto  deviation = [&](std::size_t i) { return std::abs((got[i] - got[i % 6UZ]) - (truth[i] - truth[i % 6UZ])); };
        const float worst     = std::ranges::max(std::views::iota(0UZ, got.size()) | std::views::transform(deviation));
        expect(lt(worst, 1e-3f));
    };

    "along each frame, each frame is unwrapped on its own: nothing carries to the next"_test = [] {
        const auto samples = phases(16UZ, 9UZ, true);
        const auto got     = unwrapped(samples, 16UZ, UnwrapAxis::positions, 2UZ);
        for (std::size_t f = 0UZ; f < 9UZ; ++f) {
            std::vector<float> expected(samples.begin() + static_cast<std::ptrdiff_t>(f * 16UZ), samples.begin() + static_cast<std::ptrdiff_t>((f + 1UZ) * 16UZ));
            gr::algorithm::fft::unwrapPhase(std::span<float>(expected));
            expect(std::ranges::equal(expected, std::span(got).subspan(f * 16UZ, 16UZ))) << "frame " << f;
        }
    };

    "reset forgets the turns counted so far"_test = [] {
        gr::filter::PhaseUnwrap<float> block;
        block.init(block.progress);
        block.start();
        std::vector<float>       unwrappedOut(2UZ);
        const std::vector<float> wrapsOnce{3.f, -3.f};
        expect(gr::work::Status::OK == block.processBulk(std::span<const float>(wrapsOnce), std::span(unwrappedOut)));
        expect(std::abs(unwrappedOut[1] - (-3.f + 2.f * std::numbers::pi_v<float>)) < 1e-5f);
        block.reset();
        const std::vector<float> fresh{-3.f, -3.f};
        expect(gr::work::Status::OK == block.processBulk(std::span<const float>(fresh), std::span(unwrappedOut)));
        expect(unwrappedOut == fresh) << "a fresh start takes its first phase as it is";
    };

    "a new axis restarts the count"_test = [] {
        gr::filter::PhaseUnwrap<float> block({{"frame_size", gr::Size_t(2)}});
        block.init(block.progress);
        std::vector<float>       unwrappedOut(4UZ);
        const std::vector<float> rising{3.f, 0.f, -3.f, 0.f};
        expect(gr::work::Status::OK == block.processBulk(std::span<const float>(rising), std::span(unwrappedOut)));
        std::ignore = block.settings().setStaged({{"axis", std::string("positions")}});
        std::ignore = block.settings().applyStagedParameters();
        const std::vector<float> flat{1.f, 1.f, 1.f, 1.f};
        expect(gr::work::Status::OK == block.processBulk(std::span<const float>(flat), std::span(unwrappedOut)));
        expect(unwrappedOut == flat);
    };

    "a tag stays on the sample it marks"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       source = fixture.emplace<gr::testing::ProbeSource<float>>({{"values", phases(4UZ, 20UZ, true)}, {"n_samples_max", gr::Size_t(80)}, {"mark_tag", false}});
        source._tags                       = {{12UZ, {{"key", "marker"}}}};
        auto& unwrap                       = fixture.emplace<gr::filter::PhaseUnwrap<float>>({{"frame_size", gr::Size_t(4)}});
        auto& sink                         = fixture.emplace<gr::testing::ProbeSink<float>>({{"log_samples", true}});
        expect(fixture.connect<"out", "in">(source, unwrap).has_value());
        expect(fixture.connect<"out", "in">(unwrap, sink).has_value());
        expect(fixture.run().has_value());
        expect(std::ranges::any_of(sink._tags, [](const auto& tag) { return tag.index == 12UZ && tag.map.contains("key"); }));
    };

    "a non-finite phase passes without corrupting what follows"_test = [] {
        auto samples     = phases(1UZ, 100UZ, true);
        samples[40]      = std::numeric_limits<float>::quiet_NaN();
        const auto got   = unwrapped(samples, 1UZ, UnwrapAxis::frames, 100UZ);
        const auto clean = unwrapped(phases(1UZ, 100UZ, true), 1UZ, UnwrapAxis::frames, 100UZ);
        expect(std::isnan(got[40]));
        expect(std::abs(got[60] - clean[60]) < 2.f * std::numbers::pi_v<float> + 1e-3f) << "at most the one turn the gap hides";
    };
};

const auto domains = [] {
    using gr::testing::operator""_domain_test;

    "phases unwrap the same on every served domain"_domain_test = [](auto& fixture, std::string_view domain) {
        const auto samples        = phases(64UZ, 50UZ, true);
        auto&      source         = fixture.template emplace<gr::testing::ProbeSource<float>>({{"values", samples}, {"n_samples_max", static_cast<gr::Size_t>(samples.size())}, {"mark_tag", false}});
        const auto unwrappedAlong = [&](UnwrapAxis axis) -> auto& {
            auto& unwrap = fixture.template emplace<gr::filter::PhaseUnwrap<float>>({{"frame_size", gr::Size_t(64)}, {"axis", std::string(gr::meta::enumName(axis).value_or(""))}, {"gr:compute_domain", std::string(domain) + "!"}});
            auto& sink   = fixture.template emplace<gr::testing::ProbeSink<float>>({{"log_samples", true}});
            expect(fixture.template connect<"out", "in">(source, unwrap).has_value());
            expect(fixture.template connect<"out", "in">(unwrap, sink).has_value());
            return sink;
        };
        auto&      acrossFrames   = unwrappedAlong(UnwrapAxis::frames);
        auto&      alongPositions = unwrappedAlong(UnwrapAxis::positions);
        const auto ran            = fixture.run();
        expect(ran.has_value()) << domain << (ran.has_value() ? std::string{} : ran.error().message);

        for (const auto& [sink, axis] : {std::pair{&acrossFrames, UnwrapAxis::frames}, std::pair{&alongPositions, UnwrapAxis::positions}}) {
            const auto expected = unwrapped(samples, 64UZ, axis, 50UZ);
            expect(fatal(eq(sink->_samples.size(), expected.size()))) << domain;
            const auto  deviation = [&](std::size_t i) { return std::abs(sink->_samples[i] - expected[i]); };
            const float worst     = std::ranges::max(std::views::iota(0UZ, expected.size()) | std::views::transform(deviation));
            expect(lt(worst, 1e-4f)) << domain << " " << gr::meta::enumName(axis).value_or("");
        }
    } | gr::testing::overGraphs(gr::testing::kAllDomains);
};

} // namespace

int main() {
    "PhaseUnwrap"_test                 = definitions;
    "PhaseUnwrap on every domain"_test = domains;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
