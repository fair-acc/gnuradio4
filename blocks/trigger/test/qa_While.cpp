#include <boost/ut.hpp>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <string>
#include <vector>

#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/trigger/While.hpp>

#include "TriggerTest.hpp"

namespace {
using namespace gr::blocks::trigger;
using gr::trigger_test::eventNamed;
using gr::trigger_test::EventScript;
using gr::trigger_test::EventTap;

constexpr std::uint64_t kT0 = 1'000'000'000U; /// ns, sample 0 at 1 kHz
constexpr std::uint64_t kMs = 1'000'000U;

struct Outcome {
    std::vector<std::uint64_t> passed;
    gr::Size_t                 rejected = 0U;
    gr::Size_t                 unknown  = 0U;
    gr::Size_t                 gaps     = 0U;
};

[[nodiscard]] Outcome qualify(std::vector<float> line, std::vector<gr::property_map> events, gr::property_map settings, std::vector<gr::testing::OwningTag> moreTags = {}) {
    gr::testing::GraphFixture fixture;
    settings[std::string("sample_rate")] = 1000.f;
    auto& source                         = fixture.emplace<gr::testing::TagSource<float>>({{"values", line}, {"n_samples_max", static_cast<gr::Size_t>(line.size())}, {"mark_tag", false}});
    source._tags.push_back({0UZ, gr::property_map{{std::string(gr::tag::TRIGGER_TIME.key()), kT0}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}});
    std::ranges::copy(moreTags, std::back_inserter(source._tags));
    auto& script         = fixture.emplace<EventScript>();
    script._events       = std::move(events);
    script._stopWhenDone = moreTags.empty(); // a script that stops the run would end it before the samples after a gap
    auto& gate           = fixture.emplace<While<float>>(std::move(settings));
    auto& sink           = fixture.emplace<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
    auto& tap            = fixture.emplace<EventTap>();
    boost::ut::expect(fixture.connect<"out", "in">(source, gate).has_value());
    boost::ut::expect(fixture.connect<"out", "in">(gate, sink).has_value());
    boost::ut::expect(fixture.connect<"evtOut", "evtIn">(script, gate).has_value());
    boost::ut::expect(fixture.connect<"evtOut", "evtIn">(gate, tap).has_value());
    boost::ut::expect(fixture.run().has_value());
    return {tap.times(), gate.n_rejected.value, gate.n_unknown.value, gate.n_gaps.value};
}
} // namespace

const boost::ut::suite<"While (spec §6)"> _while = [] {
    using namespace boost::ut;
    // high from the crossing at 2.5 samples (0 → 5) to the one at 5.5 (5 → 0)
    const std::vector<float> line{0.f, 0.f, 0.f, 5.f, 5.f, 5.f, 0.f, 0.f};

    "an event passes when the line was high at its trigger_time"_test = [&] {
        const auto found = qualify(line, {eventNamed("e", kT0 + 1U * kMs), eventNamed("e", kT0 + 3U * kMs), eventNamed("e", kT0 + 5U * kMs), eventNamed("e", kT0 + 7U * kMs)}, {{"threshold", 2.5f}});
        expect(found.passed == std::vector<std::uint64_t>{kT0 + 3U * kMs, kT0 + 5U * kMs});
        expect(eq(found.rejected, 2U)) << "1 ms and 7 ms saw the line low";
    };

    "the level is read at the crossing, not at the sample after it"_test = [&] {
        const auto found = qualify(line, {eventNamed("e", kT0 + 2'400'000U), eventNamed("e", kT0 + 2'600'000U)}, {{"threshold", 2.5f}});
        expect(found.passed == std::vector<std::uint64_t>{kT0 + 2'600'000U}) << "2.4 ms is before the 2.5 ms crossing, 2.6 ms after it";
    };

    "level low passes the complement"_test = [&] {
        const auto found = qualify(line, {eventNamed("e", kT0 + 1U * kMs), eventNamed("e", kT0 + 3U * kMs)}, {{"threshold", 2.5f}, {"level", std::string("low")}});
        expect(found.passed == std::vector<std::uint64_t>{kT0 + 1U * kMs});
    };

    "reference start reads the line at the leading edge of a pulse decided after the line fell"_test = [&] {
        // the line is high 2.5..5.5 ms; the pulse began at 3 ms and was decided at 7 ms, when the line was low again
        gr::property_map decided                               = eventNamed("e", kT0 + 7U * kMs);
        decided[std::string(gr::tag::TRIGGER_META_INFO.key())] = gr::property_map{
            {std::string(gr::tag::TRIGGER_NAME.key()), std::string("rising")},
            {std::string(gr::tag::TRIGGER_TIME.key()), kT0 + 3U * kMs},
            {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f},
        };
        const auto byDecision = qualify(line, {decided}, {{"threshold", 2.5f}});
        expect(byDecision.passed.empty() && eq(byDecision.rejected, 1U)) << "at 7 ms the line was low";
        const auto byStart = qualify(line, {decided}, {{"threshold", 2.5f}, {"reference", std::string("start")}});
        expect(byStart.passed == std::vector<std::uint64_t>{kT0 + 7U * kMs}) << "at 3 ms it was high; the event keeps its decision time";
        const auto fallback = qualify(line, {eventNamed("e", kT0 + 3U * kMs)}, {{"threshold", 2.5f}, {"reference", std::string("start")}});
        expect(fallback.passed == std::vector<std::uint64_t>{kT0 + 3U * kMs}) << "no start tuple: the decision time";
    };

    "an undated event, or one before the line's history, is not known and does not pass"_test = [&] {
        const auto found = qualify(line, {gr::trigger_test::undatedEvent("e"), eventNamed("e", kT0 - kMs)}, {{"threshold", 2.5f}});
        expect(found.passed.empty());
        expect(eq(found.unknown, 2U));
    };

    "inside the hysteresis band at the start the line is unknown, not low"_test = [] {
        // 2.5 lies inside 2..3, so until the line reaches 3 at sample 2 nothing about it is known
        const auto found = qualify({2.5f, 2.5f, 5.f, 5.f}, {eventNamed("e", kT0), eventNamed("e", kT0 + 3U * kMs)}, {{"threshold", 2.5f}, {"hysteresis", 1.f}, {"level", std::string("low")}});
        expect(found.passed.empty()) << "neither event saw a low line";
        expect(eq(found.unknown, 1U)) << "the first saw no known level";
        expect(eq(found.rejected, 1U)) << "the second saw it high";
    };

    "a gap leaves the dropped samples unknown, and the line is read again from the first sample after it"_test = [&] {
        // samples 0..3 at 0..3 ms, then 6 ms were dropped: samples 4.. are dated 10 ms on, all of them high
        const gr::property_map redated{
            {std::string(gr::tag::TRIGGER_TIME.key()), kT0 + 10U * kMs},
            {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f},
            {std::string(gr::tag::N_DROPPED_SAMPLES.key()), gr::Size_t{6U}},
        };
        const std::vector<float>            high(8UZ, 5.f);
        const std::vector<gr::property_map> events{eventNamed("e", kT0 + 2U * kMs), eventNamed("e", kT0 + 6U * kMs), eventNamed("e", kT0 + 11U * kMs)};
        const auto                          found = qualify(high, events, {{"threshold", 2.5f}}, {{4UZ, redated}});
        expect(found.passed == std::vector<std::uint64_t>{kT0 + 2U * kMs, kT0 + 11U * kMs}) << "before the gap and after it the line is high";
        expect(eq(found.unknown, 1U)) << "6 ms lies in the dropped stretch: neither high nor low";
        expect(eq(found.gaps, 1U));
        const auto control = qualify(high, events, {{"threshold", 2.5f}}, {{4UZ, gr::property_map{{std::string(gr::tag::TRIGGER_TIME.key()), kT0 + 10U * kMs}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}}});
        expect(eq(control.passed.size(), 3UZ)) << "the same re-dating without dropped samples keeps the line high throughout";
        expect(eq(control.gaps, 0U));
    };
};

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
