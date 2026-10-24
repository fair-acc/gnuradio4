#include <boost/ut.hpp>

#include <algorithm>
#include <complex>
#include <ranges>
#include <string>
#include <vector>

#include <gnuradio-4.0/basic/Repeat.hpp>
#include <gnuradio-4.0/filter/time_domain_filter.hpp>
#include <gnuradio-4.0/test/DeviceTestHelper.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/StreamProbes.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace {
using namespace boost::ut;
using gr::basic::Repeat;
using gr::testing::kProbeRate;
using gr::testing::kRateKey;
using gr::testing::kTriggerKey;
using gr::testing::ramp;
using gr::testing::tagAt;

using Repeated = gr::testing::ProbedChain<Repeat<float>>;

const auto definitions = [] {
    "every sample comes out n times, and the sink sees n times the rate"_test = [] {
        Repeated repeated(ramp(4UZ), {{"n", gr::Size_t(3)}});
        expect(repeated.run() == std::vector<float>{0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 2.f, 2.f, 2.f, 3.f, 3.f, 3.f});
        expect(eq(repeated.sink->sample_rate, 3.f * kProbeRate));
    };

    "a tag lands on the first copy of its sample"_test = [] {
        Repeated repeated(ramp(6UZ), {{"n", gr::Size_t(3)}});
        repeated.source->_tags = {{2UZ, {{kTriggerKey, "here"}}}};
        std::ignore            = repeated.run();
        expect(tagAt(*repeated.sink, 6UZ)) << "input sample 2 is output 6";
    };

    "a new n set on a sample takes effect there and announces the new rate"_test = [] {
        Repeated repeated(ramp(6UZ), {});
        repeated.source->_tags = {{0UZ, {{"n", gr::Size_t(2)}}}, {3UZ, {{"n", gr::Size_t(3)}}}};
        expect(repeated.run() == std::vector<float>{0.f, 0.f, 1.f, 1.f, 2.f, 2.f, 3.f, 3.f, 3.f, 4.f, 4.f, 4.f, 5.f, 5.f, 5.f});
        expect(eq(repeated.sink->sample_rate, 3.f * kProbeRate)) << "the last announced rate is the new one";
    };

    "a new n meeting an output sized for the old one still announces its rate"_test = [] {
        Repeated repeated(ramp(4UZ), {});
        repeated.source->_tags = {{1UZ, {{"n", gr::Size_t(4)}}}, {2UZ, {{kTriggerKey, "split"}}}};
        std::ignore            = repeated.run();
        expect(eq(repeated.sink->sample_rate, 4.f * kProbeRate)) << "the one-sample chunk at the change emits nothing, the next carries the announcement";
    };

    "the announcement lands on the first copy of the sample that changed n"_test = [] {
        Repeated repeated(ramp(6UZ), {});
        repeated.source->_tags = {{0UZ, {{"n", gr::Size_t(2)}}}, {3UZ, {{"n", gr::Size_t(3)}}}};
        std::ignore            = repeated.run();
        const auto announced   = std::ranges::find_if(repeated.sink->_tags, [](const auto& tag) { return tag.index == 6UZ && tag.map.contains(kRateKey); });
        expect(fatal(announced != repeated.sink->_tags.end())) << "input 3 is output 6 at n = 2";
        expect(eq(announced->map.template value_or<float>(kRateKey, 0.f), 3.f * kProbeRate));
    };

    "repeating n times and keeping one in n is the identity"_test = [] {
        gr::testing::GraphFixture<> fixture;
        const auto                  values = ramp(64UZ);
        auto&                       source = fixture.emplace<gr::testing::ProbeSource<float>>({{"values", values}, {"n_samples_max", gr::Size_t(64)}, {"mark_tag", false}});
        auto&                       repeat = fixture.emplace<Repeat<float>>({{"n", gr::Size_t(4)}});
        auto&                       keep   = fixture.emplace<gr::filter::Decimator<float>>({{"decim", gr::Size_t(4)}});
        auto&                       sink   = fixture.emplace<gr::testing::ProbeSink<float>>({{"log_samples", true}});
        expect(fixture.connect<"out", "in">(source, repeat).has_value());
        expect(fixture.connect<"out", "in">(repeat, keep).has_value());
        expect(fixture.connect<"out", "in">(keep, sink).has_value());
        expect(fixture.run().has_value());
        expect(std::vector<float>(sink._samples.begin(), sink._samples.end()) == values);
    };
};

const auto domains = [] {
    using gr::testing::operator""_domain_test;

    "samples repeat the same on every served domain"_domain_test = [](auto& fixture, std::string_view domain) {
        const auto values = ramp(256UZ);
        auto&      source = fixture.template emplace<gr::testing::ProbeSource<float>>({{"values", values}, {"n_samples_max", gr::Size_t(256)}, {"mark_tag", false}});
        auto&      repeat = fixture.template emplace<Repeat<float>>({{"n", gr::Size_t(5)}, {"gr:compute_domain", std::string(domain) + "!"}});
        auto&      sink   = fixture.template emplace<gr::testing::ProbeSink<float>>({{"log_samples", true}});
        expect(fixture.template connect<"out", "in">(source, repeat).has_value());
        expect(fixture.template connect<"out", "in">(repeat, sink).has_value());
        const auto ran = fixture.run();
        expect(ran.has_value()) << domain << (ran.has_value() ? std::string{} : ran.error().message);
        expect(fatal(eq(sink._samples.size(), 5UZ * values.size()))) << domain;
        const auto matchesClosedForm = [&](std::size_t i) { return sink._samples[i] == values[i / 5UZ]; };
        expect(std::ranges::all_of(std::views::iota(0UZ, sink._samples.size()), matchesClosedForm)) << domain << ": out[i] = in[i / n]";
    } | gr::testing::overGraphs(gr::testing::kAllDomains);
};

} // namespace

int main() {
    "Repeat"_test                 = definitions;
    "Repeat on every domain"_test = domains;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
