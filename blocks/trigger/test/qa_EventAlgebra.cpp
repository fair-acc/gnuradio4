#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <string>
#include <vector>

#include <gnuradio-4.0/basic/ClockSource.hpp>
#include <gnuradio-4.0/test/EventMarbles.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/trigger/EventFilter.hpp>
#include <gnuradio-4.0/trigger/EventReduce.hpp>
#include <gnuradio-4.0/trigger/StateLogic.hpp>
#include <gnuradio-4.0/trigger/TagToEvent.hpp>
#include <gnuradio-4.0/trigger/While.hpp>

#include "Marble.hpp"
#include "TriggerTest.hpp"

/** nanoseconds per millisecond: one sample at 1 kHz */
constexpr std::uint64_t kMs = 1'000'000U;

namespace qa_event_filter {
using namespace gr::blocks::trigger;
using gr::trigger_test::eventNamed;
using gr::trigger_test::EventScript;
using gr::trigger_test::EventTap;

struct Outcome {
    std::vector<std::string> names;
    gr::Size_t               passed  = 0U;
    gr::Size_t               blocked = 0U;
};

[[nodiscard]] Outcome sift(std::vector<gr::property_map> events, gr::property_map settings) {
    gr::testing::GraphFixture fixture;
    auto&                     script = fixture.emplace<EventScript>();
    script._events                   = std::move(events);
    auto& sieve                      = fixture.emplace<EventFilter>(std::move(settings));
    auto& tap                        = fixture.emplace<EventTap>();

    boost::ut::expect(fixture.connect<"evtOut", "evtIn">(script, sieve).has_value());
    boost::ut::expect(fixture.connect<"evtOut", "evtIn">(sieve, tap).has_value());
    boost::ut::expect(fixture.run().has_value());

    return Outcome{.names = tap.names(), .passed = sieve.n_passed.value, .blocked = sieve.n_blocked.value};
}

const boost::ut::suite<"EventFilter"> _eventFilter = [] {
    using namespace boost::ut;

    "only the events the filter names get through"_test = [] {
        const auto found = sift({eventNamed("edge", 100U), eventNamed("noise", 200U), eventNamed("edge", 300U)}, {{"filter", std::string("edge")}});

        expect(eq(found.names.size(), 2UZ));
        expect(eq(found.passed, 2U));
        expect(eq(found.blocked, 1U));
        expect(std::ranges::all_of(found.names, [](const std::string& name) { return name == "edge"; }));
    };

    "an empty filter passes everything"_test = [] {
        const auto found = sift({eventNamed("edge", 100U), eventNamed("noise", 200U)}, {});

        expect(eq(found.names.size(), 2UZ)) << "no filter is not the same as a filter that matches nothing";
        expect(eq(found.blocked, 0U));
    };

    "a leading '!' drops what it names and a '*' accepts a family of names"_test = [] {
        const std::vector<gr::property_map> events{eventNamed("edge_rise", 100U), eventNamed("noise", 200U), eventNamed("edge_fall", 300U)};
        const auto                          quiet = sift(events, {{"filter", std::string("!noise")}});
        expect(quiet.names == std::vector<std::string>{"edge_rise", "edge_fall"});
        expect(eq(quiet.blocked, 1U));
        const auto edges = sift(events, {{"filter", std::string("edge_*")}});
        expect(edges.names == std::vector<std::string>{"edge_rise", "edge_fall"});
    };

    "an interval filter passes its start and its stop"_test = [] {
        const auto found = sift({eventNamed("start", 100U), eventNamed("noise", 200U), eventNamed("stop", 300U)}, {{"filter", std::string("[start, stop]")}});
        expect(found.names == std::vector<std::string>{"start", "stop"});
    };

    "a filter that names nothing in the stream blocks all of it"_test = [] {
        const auto found = sift({eventNamed("edge", 100U), eventNamed("edge", 200U)}, {{"filter", std::string("something_else")}});

        expect(eq(found.names.size(), 0UZ));
        expect(eq(found.blocked, 2U));
    };

    "what the filter did, drawn"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     script = fixture.emplace<EventScript>();
        script._events                   = {eventNamed("edge", 100U), eventNamed("noise", 200U), eventNamed("edge", 300U), eventNamed("edge", 400U)};
        auto& sieve                      = fixture.emplace<EventFilter>({{"filter", std::string("edge")}});
        auto& tap                        = fixture.emplace<EventTap>();
        expect(fixture.connect<"evtOut", "evtIn">(script, sieve).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(sieve, tap).has_value());
        expect(fixture.run().has_value());

        gr::testing::MarbleDiagram diagram{"EventFilter: edge accepted, noise blocked"};
        auto&                      in = diagram.row("evtIn");
        for (const auto& [at, name] : std::vector<std::pair<std::uint64_t, std::string>>{{100U, "edge"}, {200U, "noise"}, {300U, "edge"}, {400U, "edge"}}) {
            in.at(at, name);
        }
        in.completes();
        diagram.condition("EventFilter(filter = \"edge\")");
        auto& out = diagram.row("evtOut");
        for (const auto& [at, name] : tap.dated()) {
            out.at(at, name);
        }
        out.completes();
        diagram.print();

        expect(eq(tap.countOf("edge"), 3UZ));
    };
};
} // namespace qa_event_filter

namespace qa_count {
using namespace gr::blocks::trigger;
using gr::trigger_test::eventNamed;
using gr::trigger_test::EventScript;
using gr::trigger_test::EventTap;
using gr::trigger_test::undatedEvent;

/** events one millisecond apart, so the rate is exactly 1 kHz and can be asserted rather than approximated loosely */
[[nodiscard]] std::vector<gr::property_map> ticks(std::size_t count, std::string name = "edge") {
    std::vector<gr::property_map> events;
    for (std::size_t i = 0UZ; i < count; ++i) {
        events.push_back(eventNamed(name, 1'000'000'000U + static_cast<std::uint64_t>(i) * 1'000'000U));
    }
    return events;
}

const boost::ut::suite<"Count"> _count = [] {
    using namespace boost::ut;

    "every nth match is reported, and carries the running total"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     script = fixture.emplace<EventScript>();
        script._events                   = ticks(10UZ);
        auto& counter                    = fixture.emplace<Count>({{"filter", std::string("edge")}, {"mode", std::string("every")}, {"n", 3U}});
        auto& tap                        = fixture.emplace<EventTap>();
        expect(fixture.connect<"evtOut", "evtIn">(script, counter).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(counter, tap).has_value());
        expect(fixture.run().has_value());

        expect(eq(counter.n_matches.value, 10U));
        expect(eq(tap.size(), 3UZ)) << "the third, sixth and ninth of ten";
        if (tap.size() == 3UZ) {
            expect(eq(tap.metaOf(0).value_or<std::uint64_t>(std::string_view{"count"}, 0U), 3U)) << "the report says which occurrence it was";
            expect(eq(tap.metaOf(2).value_or<std::uint64_t>(std::string_view{"count"}, 0U), 9U));
            expect(!gr::trigger_test::carriesSource(tap._events[0])) << "the report names no source";
            expect(!tap._events[0].contains("count")) << "and count is not at the root";
        }
    };

    "nth reports exactly once, however long the stream runs"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     script = fixture.emplace<EventScript>();
        script._events                   = ticks(10UZ);
        auto& counter                    = fixture.emplace<Count>({{"mode", std::string("nth")}, {"n", 4U}});
        auto& tap                        = fixture.emplace<EventTap>();
        expect(fixture.connect<"evtOut", "evtIn">(script, counter).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(counter, tap).has_value());
        expect(fixture.run().has_value());

        expect(eq(tap.size(), 1UZ));
        expect(eq(counter.n_matches.value, 10U)) << "it keeps counting after it has reported";
    };

    "the rate comes from the events' own times, not from a clock of its own"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     script = fixture.emplace<EventScript>();
        script._events                   = ticks(11UZ);
        auto& counter                    = fixture.emplace<Count>({{"n", 100U}});
        auto& tap                        = fixture.emplace<EventTap>();
        expect(fixture.connect<"evtOut", "evtIn">(script, counter).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(counter, tap).has_value());
        expect(fixture.run().has_value());

        expect(eq(counter.n_matches.value, 11U));
        expect(approx(counter.rate.value, 1000., 1e-6)) << "eleven events, ten milliseconds apart in total";
        expect(eq(tap.size(), 0UZ)) << "a prescale larger than the stream reports nothing";
    };

    "an undated event is counted but cannot inform the rate"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     script = fixture.emplace<EventScript>();
        script._events                   = {undatedEvent("edge"), undatedEvent("edge")};
        auto& counter                    = fixture.emplace<Count>({{"n", 1U}});
        auto& tap                        = fixture.emplace<EventTap>();
        expect(fixture.connect<"evtOut", "evtIn">(script, counter).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(counter, tap).has_value());
        expect(fixture.run().has_value());

        expect(eq(counter.n_matches.value, 2U));
        expect(eq(counter.n_undated.value, 2U));
        expect(approx(counter.rate.value, 0., 1e-12)) << "a rate from undated events would be invented";
    };

    "a prescale of zero is refused"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     counter = fixture.emplace<Count>({{"n", 5U}});
        counter.settings().init();
        std::ignore = counter.settings().applyStagedParameters();
        expect(eq(counter.n.value, 5U));

        expect(counter.settings().set({{"n", gr::Size_t(0)}}).empty());
        std::ignore = counter.settings().activateContext();
        std::ignore = counter.settings().applyStagedParameters();
        expect(eq(counter.n.value, 5U)) << "a prescale of no events has no meaning, so the previous one stays";
    };

    "what the prescaler let through, drawn"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     script = fixture.emplace<EventScript>();
        script._events                   = ticks(9UZ);
        auto& counter                    = fixture.emplace<Count>({{"n", 3U}, {"trigger_name", std::string("every3rd")}});
        auto& tap                        = fixture.emplace<EventTap>();
        expect(fixture.connect<"evtOut", "evtIn">(script, counter).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(counter, tap).has_value());
        expect(fixture.run().has_value());

        gr::testing::MarbleDiagram diagram{"Count: one event in three, with the rate the stream itself gives"};
        auto&                      in = diagram.row("evtIn");
        for (const gr::property_map& event : script._events) {
            in.at(*gr::property_map_view{event}.get_if<std::uint64_t>(std::string_view{gr::tag::TRIGGER_TIME.key()}), "edge");
        }
        in.completes();
        diagram.condition(std::format("Count(every 3rd), rate {:.0f} Hz", counter.rate.value));
        auto& out = diagram.row("evtOut");
        for (const std::uint64_t at : tap.times()) {
            out.at(at, "every3rd");
        }
        out.completes();
        diagram.print();

        expect(eq(tap.size(), 3UZ));
    };

    "a reset event starts the count over and is not itself counted"_test = [] {
        const auto reports = [](std::string resetFilter) {
            std::vector<gr::property_map> events{eventNamed("edge", 1'000'000'000U), eventNamed("edge", 1'000'001'000U), eventNamed("restart", 1'000'002'000U), //
                eventNamed("edge", 1'000'003'000U), eventNamed("edge", 1'000'004'000U), eventNamed("edge", 1'000'005'000U)};
            gr::testing::GraphFixture     fixture;
            auto&                         script = fixture.emplace<EventScript>();
            script._events                       = std::move(events);
            auto& counter                        = fixture.emplace<Count>({{"n", 3U}, {"reset_filter", std::move(resetFilter)}});
            auto& tap                            = fixture.emplace<EventTap>();
            boost::ut::expect(fixture.connect<"evtOut", "evtIn">(script, counter).has_value());
            boost::ut::expect(fixture.connect<"evtOut", "evtIn">(counter, tap).has_value());
            boost::ut::expect(fixture.run().has_value());
            return std::pair{tap.times(), counter.n_resets.value};
        };
        const auto [withReset, nResets] = reports("restart");
        expect(withReset == std::vector<std::uint64_t>{1'000'005'000U}) << "two edges, the reset, then three: one report, at the third edge after it";
        expect(eq(nResets, 1U));
        const auto [withoutReset, noResets] = reports("");
        expect(withoutReset == std::vector<std::uint64_t>{1'000'002'000U, 1'000'005'000U}) << "uncounted by a reset, the restart is the third event and the sixth is the last edge";
        expect(eq(noResets, 0U));
    };

    "events held back for max_lateness when the run ends are counted"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     script = fixture.emplace<EventScript>();
        script._events                   = ticks(4UZ);
        auto& counter                    = fixture.emplace<Count>({{"n", 2U}, {"max_lateness", 1.}});
        auto& tap                        = fixture.emplace<EventTap>();
        expect(fixture.connect<"evtOut", "evtIn">(script, counter).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(counter, tap).has_value());
        expect(fixture.run().has_value());

        expect(eq(counter.n_pending.value, 4U)) << "the watermark never passed any of them by a second";
        expect(eq(tap.size(), 0UZ));
    };

    "within reports n matches inside the interval, a tumbling window opening at the first"_test = [] {
        const auto reports = [](std::vector<std::uint64_t> atMs, double intervalMs, std::string window) {
            gr::testing::GraphFixture fixture;
            auto&                     script = fixture.emplace<EventScript>();
            for (const std::uint64_t ms : atMs) {
                script._events.push_back(eventNamed("edge", 1'000'000'000U + ms * 1'000'000U));
            }
            auto& counter = fixture.emplace<Count>({{"filter", std::string("edge")}, {"mode", std::string("within")}, {"n", 3U}, {"interval", intervalMs * 1e-3}, {"window", std::move(window)}});
            auto& tap     = fixture.emplace<EventTap>();
            boost::ut::expect(fixture.connect<"evtOut", "evtIn">(script, counter).has_value());
            boost::ut::expect(fixture.connect<"evtOut", "evtIn">(counter, tap).has_value());
            boost::ut::expect(fixture.run().has_value());
            return tap.times();
        };
        // 0,1,2 ms and 3,4,5 ms each fit 2.5 ms; consumed once, so the second three start afresh
        expect(reports({0U, 1U, 2U, 3U, 4U, 5U}, 2.5, "tumbling") == std::vector<std::uint64_t>{1'002'000'000U, 1'005'000'000U});
        expect(reports({0U, 1U, 2U, 3U, 4U, 5U}, 1.5, "tumbling").empty()) << "no three fit 1.5 ms";
        // 0,2,3,4 ms within 2 ms: a tumbling window from 0 ends before 3, a sliding one holds 2,3,4
        expect(reports({0U, 2U, 3U, 4U}, 2., "tumbling").empty());
        expect(reports({0U, 2U, 3U, 4U}, 2., "sliding") == std::vector<std::uint64_t>{1'004'000'000U});
    };
};
} // namespace qa_count

namespace qa_then {
using gr::trigger_test::kStreamStart;
using namespace gr::blocks::trigger;
using gr::trigger_test::eventNamed;
using gr::trigger_test::EventScript;
using gr::trigger_test::EventTap;

struct Outcome {
    std::vector<std::pair<std::uint64_t, std::string>> dated;
    gr::Size_t                                         sequences = 0U;
    gr::Size_t                                         timeouts  = 0U;
    gr::Size_t                                         disarmed  = 0U;
    gr::Size_t                                         ignored   = 0U;
    gr::Size_t                                         pending   = 0U;
    std::vector<gr::property_map>                      events;
    std::string                                        blockName; /// the Then block's `name`, the default trigger_name
};

[[nodiscard]] Outcome follow(std::vector<gr::property_map> events, gr::property_map settings) {
    gr::testing::GraphFixture fixture;
    auto&                     script = fixture.emplace<EventScript>();
    script._events                   = std::move(events);
    auto& then                       = fixture.emplace<Then>(std::move(settings));
    auto& tap                        = fixture.emplace<EventTap>();

    boost::ut::expect(fixture.connect<"evtOut", "evtIn">(script, then).has_value());
    boost::ut::expect(fixture.connect<"evtOut", "evtIn">(then, tap).has_value());
    boost::ut::expect(fixture.run().has_value());

    return Outcome{
        .dated     = tap.dated(),
        .sequences = then.n_sequences.value,
        .timeouts  = then.n_timeouts.value,
        .disarmed  = then.n_disarmed.value,
        .ignored   = then.n_ignored.value,
        .pending   = then.n_pending.value,
        .events    = tap._events,
        .blockName = std::string(then.name.value),
    };
}

/** A = "arm", B = "edge", cancelled by "disarm"; delays in ms */
[[nodiscard]] gr::property_map waitFor(double minMs, double maxMs, std::string bounds = "closed", std::string onRepeat = "ignore") {
    return gr::property_map{{"arm_filter", std::string("arm")}, {"trigger_filter", std::string("edge")}, {"disarm_filter", std::string("disarm")}, //
        {"min_delay", minMs * 1e-3}, {"max_delay", maxMs * 1e-3}, {"bounds", std::move(bounds)}, {"on_repeat", std::move(onRepeat)}};
}

/** only the timeouts of A = "arm" after B = "edge" go out; delays in ms */
[[nodiscard]] gr::property_map watchFor(double maxMs, double latenessMs = 0.) { return gr::property_map{{"arm_filter", std::string("arm")}, {"trigger_filter", std::string("edge")}, {"max_delay", maxMs * 1e-3}, {"max_lateness", latenessMs * 1e-3}, {"report", std::string("timeout")}}; }

/** a delayed decision as a pulse-width trigger reports it: dated when decided, the leading edge a start tuple in `trigger_meta_info` */
[[nodiscard]] gr::property_map decidedAfterStart(std::string name, std::uint64_t decidedNs, std::uint64_t startNs) {
    gr::property_map event                               = eventNamed(std::move(name), decidedNs);
    event[std::string(gr::tag::TRIGGER_META_INFO.key())] = gr::property_map{
        {std::string(gr::tag::TRIGGER_NAME.key()), std::string("rising")},
        {std::string(gr::tag::TRIGGER_TIME.key()), startNs},
        {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f},
    };
    return event;
}
/** a clock dated at kStreamStart tags `tickAtMs` (the first tag arms, `arm`), a bridge turns them into events, and Then waits for an "edge" that never comes */
[[nodiscard]] Outcome timedOutBy(std::vector<std::uint64_t> tickAtMs, gr::property_map settings) {
    gr::Tensor<std::uint64_t> tagTimes;
    std::vector<std::string>  tagValues;
    for (const std::uint64_t at : tickAtMs) {
        tagTimes.push_back(at * kMs);
        tagValues.push_back(tagTimes.size() == 1UZ ? "arm/" : "tick/");
    }
    gr::testing::GraphFixture fixture;
    auto&                     clock  = fixture.emplace<gr::basic::ClockSource<std::uint8_t>>({
        {"sample_rate", 1000.f},
        {"n_samples_max", gr::Size_t{100U}},
        {"start_time", kStreamStart},
        {"tag_times", std::move(tagTimes)},
        {"tag_values", std::move(tagValues)},
    });
    auto&                     bridge = fixture.emplace<TagToEvent<std::uint8_t>>();
    auto&                     sink   = fixture.emplace<gr::testing::TagSink<std::uint8_t, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
    auto&                     then   = fixture.emplace<Then>(std::move(settings));
    auto&                     tap    = fixture.emplace<EventTap>();
    boost::ut::expect(fixture.connect<"out", "in">(clock, bridge).has_value());
    boost::ut::expect(fixture.connect<"out", "in">(bridge, sink).has_value());
    boost::ut::expect(fixture.connect<"evtOut", "evtIn">(bridge, then).has_value());
    boost::ut::expect(fixture.connect<"evtOut", "evtIn">(then, tap).has_value());
    boost::ut::expect(fixture.run().has_value());
    return Outcome{
        .dated     = tap.dated(),
        .sequences = then.n_sequences.value,
        .timeouts  = then.n_timeouts.value,
        .disarmed  = then.n_disarmed.value,
        .ignored   = then.n_ignored.value,
        .pending   = then.n_pending.value,
        .events    = tap._events,
        .blockName = std::string(then.name.value),
    };
}

const boost::ut::suite<"Then"> _then = [] {
    using namespace boost::ut;

    "reference start dates B by the leading edge of a pulse decided long after it"_test = [] {
        // A at 0; B a pulse from 3 ms, decided at 12 ms: 12 ms after A by decision, 3 ms by its start tuple
        const std::vector<gr::property_map> events{eventNamed("arm", kStreamStart), decidedAfterStart("edge", kStreamStart + 12U * kMs, kStreamStart + 3U * kMs)};
        auto                                byStart = waitFor(2., 5.);
        byStart["reference"]                        = std::string("start");
        const auto late                             = follow(events, waitFor(2., 5.));
        expect(eq(late.sequences, 0U)) << "decided 12 ms after A, beyond max_delay 5 ms";
        const auto found = follow(events, byStart);
        expect(eq(found.sequences, 1U)) << "3 ms from A, inside [2, 5] ms";
        expect(found.dated == std::vector<std::pair<std::uint64_t, std::string>>{{kStreamStart + 12U * kMs, found.blockName}}) << "still dated by its decision";
    };

    "reference start without a start tuple falls back to the decision time"_test = [] {
        auto settings         = waitFor(2., 20.);
        settings["reference"] = std::string("start");
        const auto found      = follow({eventNamed("arm", kStreamStart), eventNamed("edge", kStreamStart + 5U * kMs)}, settings);
        expect(eq(found.sequences, 1U));
    };

    "B within [min, max] after A is reported, dated as B"_test = [] {
        const auto found = follow({eventNamed("arm", kStreamStart), eventNamed("edge", kStreamStart + 5U * kMs)}, waitFor(2., 20.));
        expect(eq(found.sequences, 1U));
        expect(found.dated == std::vector<std::pair<std::uint64_t, std::string>>{{kStreamStart + 5U * kMs, found.blockName}});
    };

    "the report is decided at B and holds A's own trigger in trigger_meta_info, from which the delay follows"_test = [] {
        const auto found = follow({eventNamed("arm", kStreamStart), eventNamed("edge", kStreamStart + 5U * kMs)}, waitFor(2., 20.));
        expect(eq(found.events.size(), 1UZ)) << fatal;
        const gr::property_map_view report{found.events[0]};
        const auto                  meta = report.get_if<gr::property_map_view>(gr::tag::TRIGGER_META_INFO.key());
        expect(meta.has_value()) << fatal;
        const auto decided = report.get_if<std::uint64_t>(gr::tag::TRIGGER_TIME.key());
        const auto armedAt = meta->get_if<std::uint64_t>(gr::tag::TRIGGER_TIME.key());
        const auto armedBy = meta->get_if<std::string_view>(gr::tag::TRIGGER_NAME.key());
        expect(decided != nullptr and *decided == kStreamStart + 5U * kMs) << "B's time";
        expect(armedAt != nullptr and *armedAt == kStreamStart) << "A's time";
        expect(armedBy.has_value() and *armedBy == "arm") << "A's own name";
        expect(decided != nullptr and armedAt != nullptr and *decided - *armedAt == 5U * kMs) << "the delay is root time minus meta time: 5 ms";
        expect(!meta->contains(std::string_view{"armed_at"}) and !meta->contains(std::string_view{"elapsed"})) << "no stored duration";
        expect(!gr::trigger_test::carriesSource(found.events[0]));
    };

    "B before A, or earlier than min_delay, is ignored"_test = [] {
        const auto found = follow({eventNamed("edge", kStreamStart), eventNamed("arm", kStreamStart + kMs), eventNamed("edge", kStreamStart + 2U * kMs), eventNamed("edge", kStreamStart + 4U * kMs)}, waitFor(2., 20.));
        expect(eq(found.ignored, 2U)) << "the B before A, and the B 1 ms after it";
        expect(eq(found.sequences, 1U)) << "the B 3 ms after A";
    };

    "one B per A: a second B needs a second A"_test = [] {
        const auto found = follow({eventNamed("arm", kStreamStart), eventNamed("edge", kStreamStart + 3U * kMs), eventNamed("edge", kStreamStart + 4U * kMs)}, waitFor(0., 20.));
        expect(eq(found.sequences, 1U));
        expect(eq(found.ignored, 1U));
    };

    "closed bounds accept B exactly at min and max delay; open bounds do not"_test = [] {
        const std::vector<gr::property_map> atMin{eventNamed("arm", kStreamStart), eventNamed("edge", kStreamStart + 2U * kMs)};
        const std::vector<gr::property_map> atMax{eventNamed("arm", kStreamStart), eventNamed("edge", kStreamStart + 20U * kMs)};
        expect(eq(follow(atMin, waitFor(2., 20.)).sequences, 1U));
        expect(eq(follow(atMax, waitFor(2., 20.)).sequences, 1U));
        expect(eq(follow(atMin, waitFor(2., 20., "open")).sequences, 0U));
        expect(eq(follow(atMax, waitFor(2., 20., "open")).sequences, 0U));
        expect(eq(follow(atMin, waitFor(2., 20., "maxOpen")).sequences, 1U)) << "only the upper end open";
    };

    "by default the delay range is [start, deadline): B at the deadline is late, B at the minimum is on time"_test = [] {
        auto settings = waitFor(2., 20.);
        settings.erase("bounds");
        const std::vector<gr::property_map> atMin{eventNamed("arm", kStreamStart), eventNamed("edge", kStreamStart + 2U * kMs)};
        const std::vector<gr::property_map> atMax{eventNamed("arm", kStreamStart), eventNamed("edge", kStreamStart + 20U * kMs)};
        expect(eq(follow(atMin, settings).sequences, 1U));
        expect(eq(follow(atMax, settings).sequences, 0U));
    };

    "an A that waits past max_delay reports a timeout dated at its deadline, decided by a later event"_test = [] {
        const auto found = follow({eventNamed("arm", kStreamStart), eventNamed("heartbeat", kStreamStart + 30U * kMs), eventNamed("edge", kStreamStart + 31U * kMs)}, waitFor(2., 20.));
        expect(eq(found.timeouts, 1U));
        expect(eq(found.sequences, 0U)) << "the B after the deadline belongs to no A";
        expect(found.dated == std::vector<std::pair<std::uint64_t, std::string>>{{kStreamStart + 20U * kMs, "then_timeout"}});
        expect(eq(found.events.size(), 1UZ)) << fatal;
        const auto meta = gr::property_map_view{found.events[0]}.get_if<gr::property_map_view>(gr::tag::TRIGGER_META_INFO.key());
        expect(meta.has_value()) << fatal;
        const auto armedAt = meta->get_if<std::uint64_t>(gr::tag::TRIGGER_TIME.key());
        const auto armedBy = meta->get_if<std::string_view>(gr::tag::TRIGGER_NAME.key());
        expect(armedAt != nullptr and *armedAt == kStreamStart) << "the A that waited in vain, 20 ms before the deadline";
        expect(armedBy.has_value() and *armedBy == "arm");
    };

    "a timeout fires at the deadline itself when the upper bound is open, and past it when closed"_test = [] {
        const std::vector<gr::property_map> events{eventNamed("arm", kStreamStart), eventNamed("heartbeat", kStreamStart + 20U * kMs)};
        const auto                          open = follow(events, waitFor(0., 20., "maxOpen"));
        expect(eq(open.timeouts, 1U)) << "elapsed 20 ms >= max_delay";
        expect(open.dated == std::vector<std::pair<std::uint64_t, std::string>>{{kStreamStart + 20U * kMs, "then_timeout"}});
        expect(eq(open.pending, 0U));

        const auto closed = follow(events, waitFor(0., 20.));
        expect(eq(closed.timeouts, 0U)) << "a B at exactly 20 ms would still be in time";
        expect(eq(closed.pending, 1U)) << "and the A is still waiting when the run ends, which the counter says";
    };

    "simultaneous A and B advance the wait, whichever of them arrived first"_test = [] {
        const gr::property_map armAtT  = eventNamed("arm", kStreamStart);
        const gr::property_map edgeAtT = eventNamed("edge", kStreamStart);
        for (const auto& [order, events] : {std::pair{std::string("A first"), std::vector{armAtT, edgeAtT}}, std::pair{std::string("B first"), std::vector{edgeAtT, armAtT}}}) {
            const auto found = follow(events, waitFor(0., 20.));
            expect(eq(found.sequences, 1U)) << order << ": equal times are taken in the order of the tie-break, not of arrival";
        }
    };

    "an A held back for max_lateness at the end of the run is counted"_test = [] {
        auto settings                         = waitFor(0., 20.);
        settings[std::string("max_lateness")] = 1.;
        const auto found                      = follow({eventNamed("arm", kStreamStart)}, std::move(settings));
        expect(eq(found.pending, 1U));
        expect(eq(found.timeouts, 0U));
    };

    "a clock heartbeat past the deadline times an A out without any later event of the stream's own"_test = [] {
        const auto found = timedOutBy({10U, 40U}, waitFor(0., 20.));
        expect(eq(found.timeouts, 1U));
        expect(found.dated == std::vector<std::pair<std::uint64_t, std::string>>{{kStreamStart + 30U * kMs, "then_timeout"}}) << "A at 10 ms, deadline 20 ms later; the clock's own tick at 40 ms only decided it";
    };

    "a clock heartbeat exactly at A plus max_delay times out an open upper bound, and not a closed one"_test = [] {
        const auto open = timedOutBy({10U, 30U}, waitFor(0., 20., "maxOpen"));
        expect(eq(open.timeouts, 1U)) << ">= A + max_delay";
        expect(open.dated == std::vector<std::pair<std::uint64_t, std::string>>{{kStreamStart + 30U * kMs, "then_timeout"}});
        const auto closed = timedOutBy({10U, 30U}, waitFor(0., 20.));
        expect(eq(closed.timeouts, 0U)) << "a B at exactly 20 ms would still be in time";
        expect(eq(closed.pending, 1U));
    };

    "a clock heartbeat before the deadline leaves the wait open"_test = [] {
        const auto found = timedOutBy({10U, 25U}, waitFor(0., 20.));
        expect(eq(found.timeouts, 0U));
        expect(eq(found.pending, 1U));
    };

    "a repeated A is ignored by default, and restarts the wait when asked"_test = [] {
        const std::vector<gr::property_map> events{eventNamed("arm", kStreamStart), eventNamed("arm", kStreamStart + 15U * kMs), eventNamed("edge", kStreamStart + 25U * kMs)};
        expect(eq(follow(events, waitFor(0., 20.)).sequences, 0U)) << "25 ms after the first A, which the second did not move";
        expect(eq(follow(events, waitFor(0., 20., "closed", "restart")).sequences, 1U)) << "10 ms after the second A";
    };

    "a disarm event cancels the wait"_test = [] {
        const auto found = follow({eventNamed("arm", kStreamStart), eventNamed("disarm", kStreamStart + kMs), eventNamed("edge", kStreamStart + 3U * kMs)}, waitFor(0., 20.));
        expect(eq(found.disarmed, 1U));
        expect(eq(found.sequences, 0U));
    };

    "report selects what goes out: timeout drops the sequence report, sequence drops the timeout event, both is the default"_test = [] {
        const std::vector<gr::property_map> inTime{eventNamed("arm", kStreamStart), eventNamed("edge", kStreamStart + 5U * kMs)};
        const std::vector<gr::property_map> tooLate{eventNamed("arm", kStreamStart), eventNamed("heartbeat", kStreamStart + 30U * kMs)};

        auto timeoutOnly                   = waitFor(0., 20.);
        timeoutOnly[std::string("report")] = std::string("timeout");
        const auto dropped                 = follow(inTime, timeoutOnly);
        expect(dropped.dated.empty());
        expect(eq(dropped.sequences, 1U)) << "counted although not reported";

        auto sequenceOnly                   = waitFor(0., 20.);
        sequenceOnly[std::string("report")] = std::string("sequence");
        const auto silent                   = follow(tooLate, sequenceOnly);
        expect(silent.dated.empty());
        expect(eq(silent.timeouts, 1U)) << "counted although not reported";

        expect(eq(follow(inTime, waitFor(0., 20.)).dated.size(), 1UZ));
        expect(eq(follow(tooLate, waitFor(0., 20.)).dated.size(), 1UZ));
    };

    "the verdict waits for max_lateness: the watermark less the lateness has to reach the deadline"_test = [] {
        const std::vector<gr::property_map> early{eventNamed("arm", kStreamStart), eventNamed("heartbeat", kStreamStart + 30U * kMs)};
        expect(follow(early, watchFor(20., 20.)).dated.empty()) << "30 ms less 20 ms lateness is 10 ms, short of the 20 ms deadline";
        const std::vector<gr::property_map> late{eventNamed("arm", kStreamStart), eventNamed("heartbeat", kStreamStart + 50U * kMs)};
        const auto                          lateFound = follow(late, watchFor(20., 20.));
        expect(lateFound.dated == std::vector<std::pair<std::uint64_t, std::string>>{{kStreamStart + 20U * kMs, "then_timeout"}}) << "50 ms less 20 ms reaches it";
    };

    "an event that arrives late within max_lateness still ends the window"_test = [] {
        const auto found = follow(
            {
                eventNamed("arm", kStreamStart),
                eventNamed("heartbeat", kStreamStart + 30U * kMs),
                eventNamed("edge", kStreamStart + 10U * kMs),
                eventNamed("heartbeat", kStreamStart + 60U * kMs),
            },
            watchFor(20., 25.));
        expect(found.dated.empty()) << "the edge was in time by its own trigger_time";
        expect(eq(found.sequences, 1U));
    };

    "time passes only as events show it: with none after the window opened there is no verdict"_test = [] {
        const auto found = follow({eventNamed("arm", kStreamStart)}, watchFor(20.));
        expect(found.dated.empty());
        expect(eq(found.pending, 1U)) << "the open window is counted";
    };

    "without an arm filter the watchdog watches from the first event: one report per outage, ended by the next match"_test = [] {
        const std::vector<gr::property_map> events{eventNamed("edge", kStreamStart), eventNamed("edge", kStreamStart + 10U * kMs), eventNamed("heartbeat", kStreamStart + 40U * kMs), eventNamed("heartbeat", kStreamStart + 80U * kMs), //
            eventNamed("edge", kStreamStart + 90U * kMs), eventNamed("heartbeat", kStreamStart + 120U * kMs)};
        const auto                          found = follow(events, {{"trigger_filter", std::string("edge")}, {"max_delay", 20e-3}, {"report", std::string("timeout")}});
        expect(found.dated == std::vector<std::pair<std::uint64_t, std::string>>{{kStreamStart + 30U * kMs, "then_timeout"}, {kStreamStart + 110U * kMs, "then_timeout"}}) << "10 ms + 20 ms, quiet through 80 ms; then 90 ms + 20 ms";
        expect(eq(found.timeouts, 2U));
    };

    "without an arm filter a max_delay of no time is refused and the previous one stays"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     then = fixture.emplace<Then>({{"trigger_filter", std::string("edge")}, {"max_delay", 0.5}});
        then.settings().init();
        std::ignore = then.settings().applyStagedParameters();
        expect(then.settings().set({{"max_delay", 0.}}).empty());
        std::ignore = then.settings().activateContext();
        std::ignore = then.settings().applyStagedParameters();
        expect(eq(then.max_delay.value, 0.5)) << "a wait of no time would report every gap as a timeout";
    };
};
} // namespace qa_then

namespace qa_while {
using gr::trigger_test::kStreamStart;
using namespace gr::blocks::trigger;
using gr::trigger_test::eventNamed;
using gr::trigger_test::EventScript;
using gr::trigger_test::EventTap;

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
    source._tags.push_back({0UZ, gr::property_map{{std::string(gr::tag::TRIGGER_TIME.key()), kStreamStart}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}});
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

const boost::ut::suite<"While"> _while = [] {
    using namespace boost::ut;
    // high from the crossing at 2.5 samples (0 → 5) to the one at 5.5 (5 → 0)
    const std::vector<float> line{0.f, 0.f, 0.f, 5.f, 5.f, 5.f, 0.f, 0.f};

    "an event passes when the line was high at its trigger_time"_test = [&] {
        const auto found = qualify(line, {eventNamed("e", kStreamStart + 1U * kMs), eventNamed("e", kStreamStart + 3U * kMs), eventNamed("e", kStreamStart + 5U * kMs), eventNamed("e", kStreamStart + 7U * kMs)}, {{"threshold", 2.5f}});
        expect(found.passed == std::vector<std::uint64_t>{kStreamStart + 3U * kMs, kStreamStart + 5U * kMs});
        expect(eq(found.rejected, 2U)) << "1 ms and 7 ms saw the line low";
    };

    "the level is read at the crossing, not at the sample after it"_test = [&] {
        const auto found = qualify(line, {eventNamed("e", kStreamStart + 2'400'000U), eventNamed("e", kStreamStart + 2'600'000U)}, {{"threshold", 2.5f}});
        expect(found.passed == std::vector<std::uint64_t>{kStreamStart + 2'600'000U}) << "2.4 ms is before the 2.5 ms crossing, 2.6 ms after it";
    };

    "level low passes the complement"_test = [&] {
        const auto found = qualify(line, {eventNamed("e", kStreamStart + 1U * kMs), eventNamed("e", kStreamStart + 3U * kMs)}, {{"threshold", 2.5f}, {"level", std::string("low")}});
        expect(found.passed == std::vector<std::uint64_t>{kStreamStart + 1U * kMs});
    };

    "reference start reads the line at the leading edge of a pulse decided after the line fell"_test = [&] {
        // the line is high 2.5..5.5 ms; the pulse began at 3 ms and was decided at 7 ms, when the line was low again
        gr::property_map decided                               = eventNamed("e", kStreamStart + 7U * kMs);
        decided[std::string(gr::tag::TRIGGER_META_INFO.key())] = gr::property_map{
            {std::string(gr::tag::TRIGGER_NAME.key()), std::string("rising")},
            {std::string(gr::tag::TRIGGER_TIME.key()), kStreamStart + 3U * kMs},
            {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f},
        };
        const auto byDecision = qualify(line, {decided}, {{"threshold", 2.5f}});
        expect(byDecision.passed.empty() && eq(byDecision.rejected, 1U)) << "at 7 ms the line was low";
        const auto byStart = qualify(line, {decided}, {{"threshold", 2.5f}, {"reference", std::string("start")}});
        expect(byStart.passed == std::vector<std::uint64_t>{kStreamStart + 7U * kMs}) << "at 3 ms it was high; the event keeps its decision time";
        const auto fallback = qualify(line, {eventNamed("e", kStreamStart + 3U * kMs)}, {{"threshold", 2.5f}, {"reference", std::string("start")}});
        expect(fallback.passed == std::vector<std::uint64_t>{kStreamStart + 3U * kMs}) << "no start tuple: the decision time";
    };

    "an undated event, or one before the line's history, is not known and does not pass"_test = [&] {
        const auto found = qualify(line, {gr::trigger_test::undatedEvent("e"), eventNamed("e", kStreamStart - kMs)}, {{"threshold", 2.5f}});
        expect(found.passed.empty());
        expect(eq(found.unknown, 2U));
    };

    "inside the hysteresis band at the start the line is unknown, not low"_test = [] {
        // 2.5 lies inside 2..3, so until the line reaches 3 at sample 2 nothing about it is known
        const auto found = qualify({2.5f, 2.5f, 5.f, 5.f}, {eventNamed("e", kStreamStart), eventNamed("e", kStreamStart + 3U * kMs)}, {{"threshold", 2.5f}, {"hysteresis", 1.f}, {"level", std::string("low")}});
        expect(found.passed.empty()) << "neither event saw a low line";
        expect(eq(found.unknown, 1U)) << "the first saw no known level";
        expect(eq(found.rejected, 1U)) << "the second saw it high";
    };

    "a gap leaves the dropped samples unknown, and the line is read again from the first sample after it"_test = [&] {
        // samples 0..3 at 0..3 ms, then 6 ms were dropped: samples 4.. are dated 10 ms on, all of them high
        const gr::property_map redated{
            {std::string(gr::tag::TRIGGER_TIME.key()), kStreamStart + 10U * kMs},
            {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f},
            {std::string(gr::tag::N_DROPPED_SAMPLES.key()), gr::Size_t{6U}},
        };
        const std::vector<float>            high(8UZ, 5.f);
        const std::vector<gr::property_map> events{eventNamed("e", kStreamStart + 2U * kMs), eventNamed("e", kStreamStart + 6U * kMs), eventNamed("e", kStreamStart + 11U * kMs)};
        const auto                          found = qualify(high, events, {{"threshold", 2.5f}}, {{4UZ, redated}});
        expect(found.passed == std::vector<std::uint64_t>{kStreamStart + 2U * kMs, kStreamStart + 11U * kMs}) << "before the gap and after it the line is high";
        expect(eq(found.unknown, 1U)) << "6 ms lies in the dropped stretch: neither high nor low";
        expect(eq(found.gaps, 1U));
        const auto control = qualify(high, events, {{"threshold", 2.5f}}, {{4UZ, gr::property_map{{std::string(gr::tag::TRIGGER_TIME.key()), kStreamStart + 10U * kMs}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}}});
        expect(eq(control.passed.size(), 3UZ)) << "the same re-dating without dropped samples keeps the line high throughout";
        expect(eq(control.gaps, 0U));
    };
};
} // namespace qa_while

namespace qa_state_logic {
using gr::trigger_test::kStreamStart;
using namespace gr::blocks::trigger;
using gr::trigger_test::eventNamed;
using gr::trigger_test::EventScript;
using gr::trigger_test::EventTap;

[[nodiscard]] std::vector<std::pair<std::uint64_t, std::string>> logicOf(std::string operation, std::vector<std::string> filters, std::vector<gr::property_map> events) {
    gr::testing::GraphFixture fixture;
    auto&                     script = fixture.emplace<EventScript>();
    script._events                   = std::move(events);
    auto& logic                      = fixture.emplace<StateLogic>(gr::property_map{{"operation", std::move(operation)}, {"filters", std::move(filters)}});
    auto& tap                        = fixture.emplace<EventTap>();
    boost::ut::expect(fixture.connect<"evtOut", "evtIn">(script, logic).has_value());
    boost::ut::expect(fixture.connect<"evtOut", "evtIn">(logic, tap).has_value());
    boost::ut::expect(fixture.run().has_value());
    return tap.dated();
}

using Dated = std::vector<std::pair<std::uint64_t, std::string>>;

const boost::ut::suite<"StateLogic"> _stateLogic = [] {
    using namespace boost::ut;
    using enum Truth;
    const std::vector<std::string> ab{"[a_on, a_off]", "[b_on, b_off]"};

    "the three-valued truth tables of allOf, anyOf and notOf"_test = [] {
        const auto all = [](Truth lhs, Truth rhs) {
            const std::array<Truth, 2> states{lhs, rhs};
            return combine(StateOperation::allOf, states);
        };
        const auto any = [](Truth lhs, Truth rhs) {
            const std::array<Truth, 2> states{lhs, rhs};
            return combine(StateOperation::anyOf, states);
        };
        const auto neg = [](Truth state) {
            const std::array<Truth, 1> states{state};
            return combine(StateOperation::notOf, states);
        };
        expect(all(no, unknown) == no) << "false AND unknown is false";
        expect(all(unknown, no) == no);
        expect(all(yes, unknown) == unknown);
        expect(all(yes, yes) == yes);
        expect(all(yes, no) == no);
        expect(all(unknown, unknown) == unknown);
        expect(any(yes, unknown) == yes) << "true OR unknown is true";
        expect(any(unknown, yes) == yes);
        expect(any(no, unknown) == unknown);
        expect(any(no, no) == no);
        expect(any(unknown, unknown) == unknown);
        expect(neg(yes) == no);
        expect(neg(no) == yes);
        expect(neg(unknown) == unknown);
        expect(combine(StateOperation::allOf, {}) == unknown) << "no state, no answer";
    };

    "allOf enters when the last state opens and exits when any closes; an unknown state is not open"_test = [ab] {
        const Dated found = logicOf("allOf", ab,
            {
                eventNamed("a_on", kStreamStart),
                eventNamed("b_on", kStreamStart + 1U * kMs),
                eventNamed("a_off", kStreamStart + 2U * kMs),
                eventNamed("a_on", kStreamStart + 3U * kMs),
            });
        expect(found ==
               Dated{
                   {kStreamStart + 1U * kMs, "state_entry"},
                   {kStreamStart + 2U * kMs, "state_exit"},
                   {kStreamStart + 3U * kMs, "state_entry"},
               })
            << "a_on with b still unknown reports nothing";
    };

    "anyOf enters on the first open state while another is unknown, and exits when it closes into unknown"_test = [ab] {
        const Dated found = logicOf("anyOf", ab,
            {
                eventNamed("a_on", kStreamStart),
                eventNamed("a_off", kStreamStart + 1U * kMs),
                eventNamed("b_on", kStreamStart + 2U * kMs),
                eventNamed("b_off", kStreamStart + 3U * kMs),
            });
        expect(found == Dated{{kStreamStart, "state_entry"}, {kStreamStart + 1U * kMs, "state_exit"}, {kStreamStart + 2U * kMs, "state_entry"}, {kStreamStart + 3U * kMs, "state_exit"}});
    };

    "notOf enters when its state closes and exits when it opens"_test = [] {
        const Dated found = logicOf("notOf", {"[a_on, a_off]"},
            {
                eventNamed("a_on", kStreamStart),
                eventNamed("a_off", kStreamStart + 1U * kMs),
                eventNamed("a_on", kStreamStart + 2U * kMs),
            });
        expect(found == Dated{{kStreamStart + 1U * kMs, "state_entry"}, {kStreamStart + 2U * kMs, "state_exit"}});
    };
};
} // namespace qa_state_logic

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
