#include <boost/ut.hpp>

#include <algorithm>
#include <numbers>
#include <numeric>
#include <ranges>
#include <string>
#include <vector>

#include <gnuradio-4.0/filter/Delay.hpp>
#include <gnuradio-4.0/filter/FractionalDelay.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/test/DeviceTestHelper.hpp>
#include <gnuradio-4.0/testing/StreamProbes.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace {
using namespace boost::ut;
using gr::filter::Delay;
using gr::testing::kTriggerKey;
using gr::testing::ramp;
using gr::testing::tagAt;

using Delayed = gr::testing::ProbedChain<Delay<float>>;

const auto definitions = [] {
    "out[i] = in[i - delay], zeros before; the last delay samples stay held"_test = [] {
        Delayed delayed(ramp(10UZ, 10.f), {{"delay", gr::Size_t(3)}});
        expect(delayed.run() == std::vector<float>{0.f, 0.f, 0.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f});
    };

    "a delay of zero passes the stream unchanged"_test = [] {
        Delayed delayed(ramp(5UZ, 10.f), {{"delay", gr::Size_t(0)}});
        expect(delayed.run() == ramp(5UZ, 10.f));
    };

    "raising the delay inserts zeros where it is set"_test = [] {
        Delayed delayed(ramp(10UZ, 10.f), {});
        delayed.source->_tags = {{0UZ, {{"delay", gr::Size_t(2)}}}, {5UZ, {{"delay", gr::Size_t(4)}}}};
        expect(delayed.run() == std::vector<float>{0.f, 0.f, 10.f, 11.f, 12.f, 0.f, 0.f, 13.f, 14.f, 15.f});
    };

    "lowering the delay drops the samples due next, and their tags"_test = [] {
        Delayed delayed(ramp(10UZ, 10.f), {});
        delayed.source->_tags = {{0UZ, {{"delay", gr::Size_t(4)}}}, {3UZ, {{kTriggerKey, "dropped"}}}, {6UZ, {{"delay", gr::Size_t(1)}}}};
        expect(delayed.run() == std::vector<float>{0.f, 0.f, 0.f, 0.f, 10.f, 11.f, 15.f, 16.f, 17.f, 18.f}) << "12, 13 and 14 were due next";
        expect(std::ranges::none_of(delayed.sink->_tags, [](const auto& tag) { return tag.map.contains(kTriggerKey); })) << "the tag on 13 went with it";
    };

    "a tag due after a lowered delay lands that much earlier, one due after a raised delay that much later"_test = [] {
        Delayed lowered(ramp(10UZ, 10.f), {});
        lowered.source->_tags = {{0UZ, {{"delay", gr::Size_t(4)}}}, {5UZ, {{kTriggerKey, "kept"}}}, {6UZ, {{"delay", gr::Size_t(1)}}}};
        std::ignore           = lowered.run();
        expect(tagAt(*lowered.sink, 6UZ)) << "input 5 was due at 9, three dropped samples move it to 6, where 15 now is";
        Delayed raised(ramp(10UZ, 10.f), {});
        raised.source->_tags = {{0UZ, {{"delay", gr::Size_t(2)}}}, {4UZ, {{kTriggerKey, "kept"}}}, {5UZ, {{"delay", gr::Size_t(4)}}}};
        std::ignore          = raised.run();
        expect(tagAt(*raised.sink, 8UZ)) << "input 4 was due at 6, two inserted zeros move it to 8, where 14 now is";
    };

    "a delay longer than the stream emits only zeros"_test = [] {
        Delayed delayed(ramp(5UZ, 10.f), {{"delay", gr::Size_t(10)}});
        expect(delayed.run() == std::vector<float>(5UZ, 0.f));
    };

    "a settings tag goes downstream too, carrying the block's value"_test = [] {
        Delayed delayed(ramp(10UZ, 10.f), {});
        delayed.source->_tags = {{0UZ, {{"delay", gr::Size_t(2)}}}};
        std::ignore           = delayed.run();
        expect(std::ranges::any_of(delayed.sink->_tags, [](const auto& tag) { return tag.map.template value_or<gr::Size_t>("delay", gr::Size_t(0)) == gr::Size_t(2); })) << "every key goes on; a setting's key carries the value the block applied";
    };

    "a tag moves with its sample"_test = [] {
        Delayed delayed(ramp(10UZ, 10.f), {{"delay", gr::Size_t(3)}});
        delayed.source->_tags = {{2UZ, {{kTriggerKey, "moved"}}}};
        std::ignore           = delayed.run();
        expect(tagAt(*delayed.sink, 5UZ)) << "input 2 is output 2 + 3";
    };
};

const auto domains = [] {
    using gr::testing::operator""_domain_test;

    "a stream is delayed the same on every served domain"_domain_test = [](auto& fixture, std::string_view domain) {
        const auto values = ramp(4096UZ, 1.f);
        auto&      source = fixture.template emplace<gr::testing::ProbeSource<float>>({{"values", values}, {"n_samples_max", gr::Size_t(4096)}, {"mark_tag", false}});
        auto&      delay  = fixture.template emplace<Delay<float>>({{"delay", gr::Size_t(37)}, {"gr:compute_domain", std::string(domain) + "!"}});
        auto&      sink   = fixture.template emplace<gr::testing::ProbeSink<float>>({{"log_samples", true}});
        expect(fixture.template connect<"out", "in">(source, delay).has_value());
        expect(fixture.template connect<"out", "in">(delay, sink).has_value());
        const auto ran = fixture.run();
        expect(ran.has_value()) << domain << (ran.has_value() ? std::string{} : ran.error().message);
        expect(fatal(eq(sink._samples.size(), values.size()))) << domain;
        const auto matchesClosedForm = [&](std::size_t i) { return sink._samples[i] == (i < 37UZ ? 0.f : values[i - 37UZ]); };
        expect(std::ranges::all_of(std::views::iota(0UZ, values.size()), matchesClosedForm)) << domain << ": out[i] = in[i - 37]";
    } | gr::testing::overGraphs(gr::testing::kAllDomains);
};

const auto fractionalDefinitions = [] {
    using gr::testing::operator""_domain_test;

    "causal Lagrange interpolation reproduces a cubic at a half sample"_test = [] {
        const std::array<double, 4UZ> values{27.0, 8.0, 1.0, 0.0};
        expect(eq(gr::filter::lagrangeDelay(values, 0.5), 15.625));
        expect(eq(gr::filter::lagrangeDelay(values, 0.0), 27.0));
        expect(eq(gr::filter::lagrangeDelay(values, 1.0), 8.0));
    };

    "a half sample impulse has the causal four coefficient response and delayed tag"_domain_test = [](auto& fixture, std::string_view domain) {
        auto& source = fixture.template emplace<gr::testing::TagSource<double>>({{"n_samples_max", gr::Size_t{8U}}, {"values", gr::Tensor<double>(gr::data_from, {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0})}});
        source._tags = {{0U, {{"impulse", true}}}};
        auto& delay  = fixture.template emplace<gr::filter::FractionalDelay<double>>({{"delay", 0.5}, {"gr:compute_domain", domain == "host" ? std::string(domain) : std::string(domain) + "!"}});
        auto& sink   = fixture.template emplace<gr::testing::TagSink<double, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        expect(fixture.template connect<"out", "in">(source, delay).has_value());
        expect(fixture.template connect<"out", "in">(delay, sink).has_value());
        expect(fixture.run().has_value());
        expect(eq(sink._samples, std::vector<double>{0.3125, 0.9375, -0.3125, 0.0625, 0.0, 0.0, 0.0, 0.0}));
        expect(std::ranges::any_of(sink._tags, [](const auto& entry) { return entry.index == 1U && entry.map.contains("impulse"); }));
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "a slow sine is delayed by the fractional delay, a phase of -2 pi f d / fs"_test = [] {
        constexpr std::size_t kSamples = 256UZ;
        constexpr double      kCycles  = 0.01; // per sample
        constexpr double      kDelay   = 2.3;
        const auto            sine     = [](double n) { return std::sin(2.0 * std::numbers::pi * kCycles * n); };
        std::vector<double>   values(kSamples);
        for (std::size_t n = 0UZ; n < kSamples; ++n) {
            values[n] = sine(static_cast<double>(n));
        }
        gr::testing::ProbedChain<gr::filter::FractionalDelay<double>, double> delayed(values, {{"delay", kDelay}});
        const auto                                                            output = delayed.run();
        expect(fatal(eq(output.size(), kSamples)));
        for (std::size_t n = 8UZ; n < kSamples; ++n) {
            expect(approx(output[n], sine(static_cast<double>(n) - kDelay), 1.e-5)) << n;
        }
    };

    "a ramp comes out exactly delayed: the group delay equals the fractional delay"_test = [] {
        constexpr double    kDelay = 2.3;
        std::vector<double> ramp(64UZ);
        std::iota(ramp.begin(), ramp.end(), 0.0);
        gr::testing::ProbedChain<gr::filter::FractionalDelay<double>, double> delayed(ramp, {{"delay", kDelay}});
        const auto                                                            output = delayed.run();
        expect(fatal(eq(output.size(), ramp.size())));
        for (std::size_t n = 6UZ; n < output.size(); ++n) { // past the zero history the four taps reach
            expect(approx(output[n], static_cast<double>(n) - kDelay, 1e-12)) << n;
        }
    };

    "raising the fractional delay keeps the history and repeats the last samples"_test = [] {
        gr::filter::FractionalDelay<double> block;
        gr::PortOut<double>                 upstream;
        gr::PortIn<double>                  downstream;
        expect(upstream.connect(block.in).has_value());
        expect(block.out.connect(downstream).has_value());
        block.delay = 1.0;
        block.start();
        const auto feed = [&](const std::vector<double>& values) {
            const std::size_t n = values.size();
            {
                auto written = upstream.tryReserve<gr::SpanReleasePolicy::ProcessAll>(n);
                std::ranges::copy(values, written.begin());
                written.publish(n);
            }
            {
                auto input  = block.in.get<gr::SpanReleasePolicy::ProcessAll>(n);
                auto output = block.out.tryReserve<gr::SpanReleasePolicy::ProcessAll>(n);
                expect(block.processBulk(input, output) == gr::work::Status::OK);
            }
            auto delayed = downstream.get<gr::SpanReleasePolicy::ProcessAll>(n);
            return std::vector<double>(delayed.begin(), delayed.end());
        };
        expect(eq(feed({1.0, 2.0, 3.0, 4.0}), std::vector<double>{0.0, 1.0, 2.0, 3.0}));
        block.delay = 2.0;
        block.settingsChanged({{"delay", 1.0}}, {{"delay", 2.0}});
        expect(eq(feed({5.0, 6.0}), std::vector<double>{3.0, 4.0})) << "out[n] = in[n - 2] across the change";
    };
};

} // namespace

int main() {
    "Delay"_test                 = definitions;
    "Delay on every domain"_test = domains;
    "FractionalDelay"_test       = fractionalDefinitions;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
