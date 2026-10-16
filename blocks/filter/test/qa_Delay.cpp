#include <boost/ut.hpp>

#include <algorithm>
#include <ranges>
#include <string>
#include <vector>

#include <gnuradio-4.0/filter/Delay.hpp>
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

} // namespace

int main() {
    "Delay"_test                 = definitions;
    "Delay on every domain"_test = domains;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
