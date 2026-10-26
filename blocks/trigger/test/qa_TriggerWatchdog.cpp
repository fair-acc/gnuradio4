#include <boost/ut.hpp>

#include <optional>
#include <string>
#include <vector>

#include "Marble.hpp"
#include <gnuradio-4.0/test/EventMarbles.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/trigger/Gate.hpp>
#include <gnuradio-4.0/trigger/TriggerWatchdog.hpp>

#include "TriggerTest.hpp"

using namespace gr::blocks::trigger;
using gr::trigger_test::EventTap;

const boost::ut::suite<"TriggerWatchdog"> _watchdog = [] {
    using namespace boost::ut;

    const gr::property_map values{{"a", 1.0f}, {"b", 2.0f}, {"c", 3.0f}, {"d", 4.0f}, {"e", 5.0f}, {"f", 6.0f}};
    const gr::property_map tags{{"T", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("start")}, {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1U}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}}};

    "an outage is reported once, and so is the recovery"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     source    = fixture.emplace<MarbleSource<float>>({{"script", std::string("T:a b c d T:e f |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&                     watchdog  = fixture.emplace<TriggerWatchdog<float>>({{"filter", std::string("start")}, {"timeout_samples", 2U}});
        auto&                     sink      = fixture.emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        auto&                     collector = fixture.emplace<EventTap>();
        expect(fixture.connect<"out", "in">(source, watchdog).has_value());
        expect(fixture.connect<"out", "in">(watchdog, sink).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(watchdog, collector).has_value());
        expect(fixture.run().has_value());

        expect(eq(sink._samples.size(), 6UZ)) << "the stream passes through untouched";
        expect(eq(watchdog.n_timeouts.value, 1U)) << "one outage, reported once however long it lasts";
        const std::vector<std::string> names = collector.names();
        expect(eq(names.size(), 2UZ)) << "one timeout event and one recovery event";
        if (names.size() == 2UZ) {
            expect(eq(names[0], std::string("timeout")));
            expect(eq(names[1], std::string("recovered")));
        }
    };

    "an outage closes a gate downstream, and the recovery opens it again"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     source    = fixture.emplace<MarbleSource<float>>({{"script", std::string("T:a b c d T:e f |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&                     watchdog  = fixture.emplace<TriggerWatchdog<float>>({{"filter", std::string("start")}, {"timeout_samples", 2U}, {"timeout_action", std::string("close")}});
        auto&                     collector = fixture.emplace<EventTap>();
        expect(fixture.connect<"out", "in">(source, watchdog).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(watchdog, collector).has_value());
        expect(fixture.run().has_value());

        const std::vector<std::string> names = collector.names();
        expect(eq(names.size(), 4UZ)) << "each report is followed by what the gate should do";
        if (names.size() == 4UZ) {
            expect(eq(names[0], std::string("timeout")));
            expect(eq(names[1], std::string("close"))) << "the action names what the outage does to the gate";
            expect(eq(names[2], std::string("recovered")));
            expect(eq(names[3], std::string("open"))) << "and the recovery does the opposite";
        }
    };

    "the action can be the other way round, for a chain that must run when the trigger is lost"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     source    = fixture.emplace<MarbleSource<float>>({{"script", std::string("T:a b c d T:e f |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&                     watchdog  = fixture.emplace<TriggerWatchdog<float>>({{"filter", std::string("start")}, {"timeout_samples", 2U}, {"timeout_action", std::string("open")}});
        auto&                     collector = fixture.emplace<EventTap>();
        expect(fixture.connect<"out", "in">(source, watchdog).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(watchdog, collector).has_value());
        expect(fixture.run().has_value());

        const std::vector<std::string> names = collector.names();
        expect(eq(names.size(), 4UZ));
        if (names.size() == 4UZ) {
            expect(eq(names[1], std::string("open")));
            expect(eq(names[3], std::string("close")));
        }
    };

    "report says nothing to a gate, which is the default"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     source    = fixture.emplace<MarbleSource<float>>({{"script", std::string("T:a b c d T:e f |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&                     watchdog  = fixture.emplace<TriggerWatchdog<float>>({{"filter", std::string("start")}, {"timeout_samples", 2U}});
        auto&                     collector = fixture.emplace<EventTap>();
        expect(fixture.connect<"out", "in">(source, watchdog).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(watchdog, collector).has_value());
        expect(fixture.run().has_value());

        expect(eq(collector.names().size(), 2UZ)) << "the two reports and nothing else";
    };

    "an outage closing a real gate stops the stream, drawn"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     source    = fixture.emplace<MarbleSource<float>>({{"script", std::string("T:a b c d T:e f |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&                     watchdog  = fixture.emplace<TriggerWatchdog<float>>({{"filter", std::string("start")}, {"timeout_samples", 2U}, {"timeout_action", std::string("close")}});
        auto&                     gate      = fixture.emplace<Gate<float>>({{"mode", std::string("toggle")}, {"open_filter", std::string("open")}, {"close_filter", std::string("close")}, {"initial_state", true}});
        auto&                     sink      = fixture.emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        auto&                     collector = fixture.emplace<EventTap>();
        expect(fixture.connect<"out", "in">(source, watchdog).has_value());
        expect(fixture.connect<"out", "in">(watchdog, gate).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(watchdog, gate).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(watchdog, collector).has_value());
        expect(fixture.connect<"out", "in">(gate, sink).has_value());
        expect(fixture.run().has_value());

        gr::testing::MarbleDiagram diagram{"TriggerWatchdog: an outage closes the gate, the recovery opens it"};
        diagram.unit = "sample";
        diagram.row("in").at(0U, "start").at(4U, "start").completes();
        diagram.condition("TriggerWatchdog(timeout = 2 samples, action = close) -> Gate(toggle)");
        const std::vector<std::string> names   = collector.names();
        auto&                          reports = diagram.row("evtOut");
        for (std::size_t i = 0UZ; i < names.size(); ++i) {
            reports.at(i < 2UZ ? 2U : 4U, names[i]);
        }
        reports.completes();
        diagram.print();
        std::println("the samples the gate let through: '{}'", sink.script());

        expect(ge(names.size(), 2UZ)) << "the outage is reported whether or not a gate acts on it";
    };

    "anchored, the watchdog reports an absence once, only after its anchor and only without a match (spec §6 Absent)"_test = [&] {
        const gr::property_map anchored{{"T", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("start")}, {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1U}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}}, {"A", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("arm")}, {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1U}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}}, {"G", gr::property_map{{std::string(gr::tag::N_DROPPED_SAMPLES.key()), gr::Size_t{5U}}}}};
        const auto             watch = [&](std::string script) {
            gr::testing::GraphFixture fixture;
            auto&                     source   = fixture.emplace<MarbleSource<float>>({{"script", std::move(script)}, {"sample_values", values}, {"sample_tags", anchored}});
            auto&                     watchdog = fixture.emplace<TriggerWatchdog<float>>({{"filter", std::string("start")}, {"arm_filter", std::string("arm")}, {"timeout_samples", 2U}});
            auto&                     sink     = fixture.emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", anchored}});
            auto&                     tap      = fixture.emplace<EventTap>();
            expect(fixture.connect<"out", "in">(source, watchdog).has_value());
            expect(fixture.connect<"out", "in">(watchdog, sink).has_value());
            expect(fixture.connect<"evtOut", "evtIn">(watchdog, tap).has_value());
            expect(fixture.run().has_value());
            return std::pair{watchdog.n_timeouts.value, watchdog.n_inconclusive.value};
        };
        expect(watch("a b c d e f |") == std::pair{0U, 0U}) << "no anchor, no window";
        expect(watch("a A:b c d e f |") == std::pair{1U, 0U}) << "anchored at b, nothing by d: one absence, however long the quiet lasts";
        expect(watch("A:a T:b c d e f |") == std::pair{0U, 0U}) << "the match closes the window";
        expect(watch("A:a G:b c d e f |") == std::pair{0U, 1U}) << "dropped samples are no proof of absence";
    };

    "a disabled watchdog reports nothing"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     source    = fixture.emplace<MarbleSource<float>>({{"script", std::string("a b c d |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&                     watchdog  = fixture.emplace<TriggerWatchdog<float>>({{"filter", std::string("start")}});
        auto&                     sink      = fixture.emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        auto&                     collector = fixture.emplace<EventTap>();
        expect(fixture.connect<"out", "in">(source, watchdog).has_value());
        expect(fixture.connect<"out", "in">(watchdog, sink).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(watchdog, collector).has_value());
        expect(fixture.run().has_value());

        expect(eq(watchdog.n_timeouts.value, 0U)) << "both limits are zero, so the watchdog is unarmed";
        expect(collector.names().empty());
    };

    "a seconds limit with no rate to convert it stays unarmed and says so"_test = [&] {
        const gr::property_map untimedTags{{"T", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("start")}}}};

        gr::testing::GraphFixture fixture;
        auto&                     source    = fixture.emplace<MarbleSource<float>>({{"script", std::string("T:a b c d e f |")}, {"sample_values", values}, {"sample_tags", untimedTags}});
        auto&                     watchdog  = fixture.emplace<TriggerWatchdog<float>>({{"filter", std::string("start")}, {"timeout_seconds", 1.f}});
        auto&                     sink      = fixture.emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", untimedTags}});
        auto&                     collector = fixture.emplace<EventTap>();
        expect(fixture.connect<"out", "in">(source, watchdog).has_value());
        expect(fixture.connect<"out", "in">(watchdog, sink).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(watchdog, collector).has_value());
        gr::trigger_test::ErrorMessages sent{fixture.scheduler()};
        expect(fixture.run().has_value());

        expect(eq(watchdog.n_timeouts.value, 0U)) << "an unconvertible limit must not expire";
        expect(eq(watchdog.n_invalid_triggers.value, 1U)) << "and the trigger that carried no time is counted";
        expect(eq(sent.countContaining("timeout_seconds needs a sample_rate"), 1UZ)) << "and must not stay silent either";
        expect(collector.names().empty()) << "an error is no event";
    };

    "a seconds limit needs the nominal rate, which timed tags alone do not give"_test = [&] {
        const auto timed = [](std::uint64_t at, std::optional<float> rate) {
            gr::property_map anchorTag{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("start")}, {std::string(gr::tag::TRIGGER_TIME.key()), at}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}};
            if (rate) {
                anchorTag[std::string("sample_rate")] = *rate;
            }
            return anchorTag;
        };
        const auto run = [&](std::optional<float> rate) {
            const gr::property_map    timedTags{{"T", timed(1'000'000'000U, rate)}, {"U", timed(1'002'000'000U, rate)}};
            gr::testing::GraphFixture fixture;
            auto&                     source    = fixture.emplace<MarbleSource<float>>({{"script", std::string("T:a b U:c d e f |")}, {"sample_values", values}, {"sample_tags", timedTags}});
            auto&                     watchdog  = fixture.emplace<TriggerWatchdog<float>>({{"filter", std::string("start")}, {"timeout_seconds", 0.002f}});
            auto&                     sink      = fixture.emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", timedTags}});
            auto&                     collector = fixture.emplace<EventTap>();
            expect(fixture.connect<"out", "in">(source, watchdog).has_value());
            expect(fixture.connect<"out", "in">(watchdog, sink).has_value());
            expect(fixture.connect<"evtOut", "evtIn">(watchdog, collector).has_value());
            gr::trigger_test::ErrorMessages sent{fixture.scheduler()};
            expect(fixture.run().has_value());
            return std::tuple{watchdog.n_timeouts.value, collector.names(), sent.countContaining("needs a sample_rate")};
        };

        const auto [withoutRate, unarmed, nUnarmed] = run(std::nullopt);
        expect(eq(withoutRate, 0U)) << "two anchors no longer imply a rate";
        expect(eq(nUnarmed, 1UZ)) << "so the limit reports that it cannot arm";
        expect(unarmed.empty());

        const auto [withRate, expired, nRateErrors] = run(1000.f);
        expect(eq(nRateErrors, 0UZ));
        expect(eq(withRate, 1U)) << "with the stream's sample_rate the limit expires";
        expect(eq(expired.empty() ? std::string() : expired.front(), std::string("timeout")));
    };
};

namespace {
constexpr std::uint64_t kStart = 1'000'000'000U;
constexpr std::uint64_t kMs    = 1'000'000U;

struct Absences {
    std::vector<std::uint64_t> at;
    gr::Size_t                 absences = 0U;
    gr::Size_t                 matches  = 0U;
    gr::Size_t                 pending  = 0U;
};

/** an event-only graph: the scripted events are the only clock; time is what their trigger_time says */
[[nodiscard]] Absences watchEvents(std::vector<gr::property_map> events, gr::property_map settings) {
    gr::testing::GraphFixture fixture;
    auto&                     script = fixture.emplace<gr::trigger_test::EventScript>();
    script._events                   = std::move(events);
    auto& watchdog                   = fixture.emplace<EventWatchdog>(std::move(settings));
    auto& tap                        = fixture.emplace<EventTap>();
    boost::ut::expect(fixture.connect<"evtOut", "evtIn">(script, watchdog).has_value());
    boost::ut::expect(fixture.connect<"evtOut", "evtIn">(watchdog, tap).has_value());
    boost::ut::expect(fixture.run().has_value());
    return {tap.times(), watchdog.n_absences.value, watchdog.n_matches.value, watchdog.n_pending.value};
}

[[nodiscard]] gr::property_map absentIn(double windowMs, double latenessMs = 0.) { return gr::property_map{{"arm_filter", std::string("arm")}, {"filter", std::string("edge")}, {"window", windowMs * 1e-3}, {"max_lateness", latenessMs * 1e-3}}; }
} // namespace

const boost::ut::suite<"EventWatchdog (spec §6 Absent on events)"> _absent = [] {
    using namespace boost::ut;
    using gr::trigger_test::eventNamed;

    "a window that no event ended is reported at its deadline once a later event shows time has passed"_test = [] {
        const auto found = watchEvents({eventNamed("arm", kStart), eventNamed("heartbeat", kStart + 30U * kMs)}, absentIn(20.));
        expect(found.at == std::vector<std::uint64_t>{kStart + 20U * kMs}) << "dated at A + window, not at the heartbeat";
        expect(eq(found.absences, 1U));
        expect(eq(found.pending, 0U));
    };

    "the event inside the window ends it: no absence"_test = [] {
        const auto found = watchEvents(
            {
                eventNamed("arm", kStart),
                eventNamed("edge", kStart + 10U * kMs),
                eventNamed("heartbeat", kStart + 30U * kMs),
            },
            absentIn(20.));
        expect(found.at.empty());
        expect(eq(found.matches, 1U));
    };

    "the event exactly at the deadline is too late"_test = [] {
        const auto found = watchEvents({eventNamed("arm", kStart), eventNamed("edge", kStart + 20U * kMs)}, absentIn(20.));
        expect(found.at == std::vector<std::uint64_t>{kStart + 20U * kMs});
        expect(eq(found.matches, 0U));
    };

    "time passes only as events show it: with none after the window opened there is no verdict"_test = [] {
        const auto found = watchEvents({eventNamed("arm", kStart)}, absentIn(20.));
        expect(found.at.empty());
        expect(eq(found.pending, 1U)) << "the open window is counted";
    };

    "the verdict waits for max_lateness: the watermark less the lateness has to reach the deadline"_test = [] {
        const std::vector<gr::property_map> early{eventNamed("arm", kStart), eventNamed("heartbeat", kStart + 30U * kMs)};
        expect(watchEvents(early, absentIn(20., 20.)).at.empty()) << "30 ms less 20 ms lateness is 10 ms, short of the 20 ms deadline";
        const std::vector<gr::property_map> late{eventNamed("arm", kStart), eventNamed("heartbeat", kStart + 50U * kMs)};
        expect(watchEvents(late, absentIn(20., 20.)).at == std::vector<std::uint64_t>{kStart + 20U * kMs}) << "50 ms less 20 ms reaches it";
    };

    "an event that arrives late within max_lateness still ends the window"_test = [] {
        const auto found = watchEvents(
            {
                eventNamed("arm", kStart),
                eventNamed("heartbeat", kStart + 30U * kMs),
                eventNamed("edge", kStart + 10U * kMs),
                eventNamed("heartbeat", kStart + 60U * kMs),
            },
            absentIn(20., 25.));
        expect(found.at.empty()) << "the edge was in time by its own trigger_time";
        expect(eq(found.matches, 1U));
    };

    "an armed window opens once: a second arm inside it does not move the deadline"_test = [] {
        const auto found = watchEvents(
            {
                eventNamed("arm", kStart),
                eventNamed("arm", kStart + 15U * kMs),
                eventNamed("heartbeat", kStart + 30U * kMs),
            },
            absentIn(20.));
        expect(found.at == std::vector<std::uint64_t>{kStart + 20U * kMs});
    };

    "without an arm filter the watchdog watches from the first event: one report per outage, ended by the next match"_test = [] {
        const std::vector<gr::property_map> events{eventNamed("edge", kStart), eventNamed("edge", kStart + 10U * kMs), eventNamed("heartbeat", kStart + 40U * kMs), eventNamed("heartbeat", kStart + 80U * kMs), //
            eventNamed("edge", kStart + 90U * kMs), eventNamed("heartbeat", kStart + 120U * kMs)};
        const auto                          found = watchEvents(events, {{"filter", std::string("edge")}, {"window", 20e-3}});
        expect(found.at == std::vector<std::uint64_t>{kStart + 30U * kMs, kStart + 110U * kMs}) << "10 ms + 20 ms, quiet through 80 ms; then 90 ms + 20 ms";
        expect(eq(found.absences, 2U));
    };

    "a window of no time is refused and the previous one stays"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     watchdog = fixture.emplace<EventWatchdog>({{"filter", std::string("edge")}, {"window", 0.5}});
        watchdog.settings().init();
        std::ignore = watchdog.settings().applyStagedParameters();
        expect(watchdog.settings().set({{"window", 0.}}).empty());
        std::ignore = watchdog.settings().activateContext();
        std::ignore = watchdog.settings().applyStagedParameters();
        expect(eq(watchdog.window.value, 0.5)) << "a window of no time would report every window absent";
    };
};

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
