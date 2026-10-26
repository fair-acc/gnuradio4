#include "Marble.hpp"
#include "TriggerTest.hpp"
#include <boost/ut.hpp>
#include <cstdint>
#include <gnuradio-4.0/basic/ClockSource.hpp>
#include <gnuradio-4.0/test/EventMarbles.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/trigger/EventReduce.hpp>
#include <gnuradio-4.0/trigger/TagToEvent.hpp>
#include <string>
#include <vector>

namespace qa_accumulate {
using namespace gr::blocks::trigger;
using gr::testing::ProcessFunction;
using gr::testing::TagSink;

namespace {
/** 1 2 3 | 4 5 | 6, where the bars are the segment-ending triggers at samples 3 and 5 */
[[nodiscard]] const gr::property_map& kValues() {
    static const gr::property_map map{{"1", 1.f}, {"2", 2.f}, {"3", 3.f}, {"4", 4.f}, {"5", 5.f}, {"6", 6.f}};
    return map;
}
[[nodiscard]] const gr::property_map& kTags() {
    static const gr::property_map map{{"T", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("cut")}}}};
    return map;
}
constexpr std::string_view kScript = "1 2 3 T:4 5 T:6 |";

template<Accumulation mode, typename TOut = float>
[[nodiscard]] std::vector<TOut> segmentsOf(gr::property_map settings) {
    settings[std::string("segment_filter")] = std::string("cut");
    gr::testing::GraphFixture fixture;
    auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string(kScript)}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
    auto&                     fold   = fixture.emplace<Accumulate<float, mode>>(std::move(settings));
    auto&                     sink   = fixture.emplace<TagSink<TOut, ProcessFunction::USE_PROCESS_ONE>>();

    boost::ut::expect(fixture.connect<"out", "in">(source, fold).has_value());
    boost::ut::expect(fixture.connect<"out", "in">(fold, sink).has_value());
    boost::ut::expect(fixture.run().has_value());

    return std::vector<TOut>(sink._samples.begin(), sink._samples.end());
}
} // namespace

const boost::ut::suite<"Accumulate"> _accumulate = [] {
    using namespace boost::ut;

    "the last sample of each segment"_test = [] {
        const auto found = segmentsOf<Accumulation::last>({});
        expect(found == std::vector<float>{3.f, 5.f}) << "the sample before each cut; the third segment never ended";
    };

    "an unknown predicate is refused with an error message"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string(kScript)}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
        auto&                     fold   = fixture.emplace<Accumulate<float, Accumulation::last>>({{"segment_filter", std::string("cut")}, {"predicate", std::string("sideways")}});
        auto&                     sink   = fixture.emplace<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>();
        expect(fixture.connect<"out", "in">(source, fold).has_value());
        expect(fixture.connect<"out", "in">(fold, sink).has_value());
        gr::trigger_test::ErrorMessages sent{fixture.scheduler()};
        expect(fixture.run().has_value());
        expect(eq(sent.countContaining("sideways"), 1UZ));
    };

    "the sum of each segment"_test = [] {
        const auto found = segmentsOf<Accumulation::automatic>({{"operation", std::string("sum")}});
        expect(found == std::vector<float>{6.f, 9.f}) << "1+2+3, then 4+5";
    };

    "the mean of each segment"_test = [] {
        const auto found = segmentsOf<Accumulation::automatic>({{"operation", std::string("mean")}});
        expect(found == std::vector<float>{2.f, 4.5f});
    };

    "the largest and the smallest of each segment"_test = [] {
        expect(segmentsOf<Accumulation::automatic>({{"operation", std::string("maximum")}}) == std::vector<float>{3.f, 5.f});
        expect(segmentsOf<Accumulation::automatic>({{"operation", std::string("minimum")}}) == std::vector<float>{1.f, 4.f});
    };

    "how many samples each segment held"_test = [] {
        const auto found = segmentsOf<Accumulation::automatic>({{"operation", std::string("count")}});
        expect(found == std::vector<float>{3.f, 2.f}) << "three samples before the first cut, two between the cuts";
    };

    "whether every sample of the segment compared true"_test = [] {
        const auto all = segmentsOf<Accumulation::all, std::uint8_t>({{"predicate", std::string("less")}, {"threshold", 4.f}});
        expect(all == std::vector<std::uint8_t>{1U, 0U}) << "1 2 3 are all below 4; the segment holding 4 and 5 is not";
    };

    "whether any sample of the segment compared true"_test = [] {
        const auto any = segmentsOf<Accumulation::any, std::uint8_t>({{"predicate", std::string("greater")}, {"threshold", 2.5f}});
        expect(any == std::vector<std::uint8_t>{1U, 1U}) << "3 in the first segment, 4 and 5 in the second";
    };

    "with no trigger named the cadence ends the segments"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string(kScript)}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
        auto&                     fold   = fixture.emplace<Accumulate<float, Accumulation::automatic>>({{"operation", std::string("sum")}, {"n_samples", 2U}});
        auto&                     sink   = fixture.emplace<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>();
        expect(fixture.connect<"out", "in">(source, fold).has_value());
        expect(fixture.connect<"out", "in">(fold, sink).has_value());
        expect(fixture.run().has_value());

        const std::vector<float> found(sink._samples.begin(), sink._samples.end());
        expect(found == std::vector<float>{3.f, 7.f, 11.f}) << "1+2, 3+4, 5+6: a block with no trigger source is not a silent one";
    };

    "a cadence in seconds is the same limit where the rate is known"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string(kScript)}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
        auto&                     fold   = fixture.emplace<Accumulate<float, Accumulation::automatic>>({{"operation", std::string("count")}, {"n_samples", 0U}, {"timeout", 0.003f}, {"sample_rate", 1000.f}});
        auto&                     sink   = fixture.emplace<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>();
        expect(fixture.connect<"out", "in">(source, fold).has_value());
        expect(fixture.connect<"out", "in">(fold, sink).has_value());
        expect(fixture.run().has_value());

        const std::vector<float> found(sink._samples.begin(), sink._samples.end());
        expect(found == std::vector<float>{3.f, 3.f}) << "3 ms at 1 kHz is three samples, twice over";
    };

    "a trigger ends a segment before the cadence does"_test = [] {
        const auto found = segmentsOf<Accumulation::automatic>({{"operation", std::string("count")}, {"n_samples", 100U}});
        expect(found == std::vector<float>{3.f, 2.f}) << "whichever comes first, and here it is the trigger";
    };

    "the segments and what each amounted to, drawn"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string(kScript)}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
        auto&                     fold   = fixture.emplace<Accumulate<float, Accumulation::automatic>>({{"segment_filter", std::string("cut")}, {"operation", std::string("sum")}});
        auto&                     sink   = fixture.emplace<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>();
        expect(fixture.connect<"out", "in">(source, fold).has_value());
        expect(fixture.connect<"out", "in">(fold, sink).has_value());
        expect(fixture.run().has_value());

        gr::testing::MarbleDiagram diagram{"Reduce(sum): a trigger ends a segment, and the sum is published where it sat"};
        diagram.unit  = "sample";
        auto& samples = diagram.row("in");
        for (std::size_t i = 0UZ; i < 6UZ; ++i) {
            samples.at(i, std::format("{}", i + 1UZ));
        }
        samples.completes();
        diagram.condition("Reduce(operation = sum, segment_filter = \"cut\")");
        auto&                            out = diagram.row("out");
        const std::vector<std::uint64_t> cuts{3U, 5U};
        for (std::size_t i = 0UZ; i < sink._samples.size() && i < cuts.size(); ++i) {
            out.at(cuts[i], std::format("{:.0f}", sink._samples[i]));
        }
        out.completes();
        diagram.print();

        expect(eq(sink._samples.size(), 2UZ));
    };
};
} // namespace qa_accumulate

namespace qa_count {
using namespace gr::blocks::trigger;
using gr::trigger_test::eventNamed;
using gr::trigger_test::EventScript;
using gr::trigger_test::EventTap;
using gr::trigger_test::undatedEvent;

namespace {
/** events one millisecond apart, so the rate is exactly 1 kHz and can be asserted rather than approximated loosely */
[[nodiscard]] std::vector<gr::property_map> ticks(std::size_t count, std::string name = "edge") {
    std::vector<gr::property_map> events;
    for (std::size_t i = 0UZ; i < count; ++i) {
        events.push_back(eventNamed(name, 1'000'000'000U + static_cast<std::uint64_t>(i) * 1'000'000U));
    }
    return events;
}
} // namespace

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
            const std::uint64_t* third = gr::property_map_view{tap.metaOf(0)}.get_if<std::uint64_t>(std::string_view{"count"});
            expect(third != nullptr && *third == 3U) << "the report says which occurrence it was";
            const std::uint64_t* ninth = gr::property_map_view{tap.metaOf(2)}.get_if<std::uint64_t>(std::string_view{"count"});
            expect(ninth != nullptr && *ninth == 9U);
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
using namespace gr::blocks::trigger;
using gr::trigger_test::eventNamed;
using gr::trigger_test::EventScript;
using gr::trigger_test::EventTap;

namespace {
constexpr std::uint64_t kStart = 1'000'000'000U;
constexpr std::uint64_t kMs    = 1'000'000U;

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
/** a clock dated at kStart tags `tickAtMs` (the first tag arms, `arm`), a bridge turns them into events, and Then waits for an "edge" that never comes */
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
        {"start_time", kStart},
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
} // namespace

const boost::ut::suite<"Then (spec §6)"> _then = [] {
    using namespace boost::ut;

    "reference start dates B by the leading edge of a pulse decided long after it"_test = [] {
        // A at 0; B a pulse from 3 ms, decided at 12 ms: 12 ms after A by decision, 3 ms by its start tuple
        const std::vector<gr::property_map> events{eventNamed("arm", kStart), decidedAfterStart("edge", kStart + 12U * kMs, kStart + 3U * kMs)};
        auto                                byStart = waitFor(2., 5.);
        byStart["reference"]                        = std::string("start");
        const auto late                             = follow(events, waitFor(2., 5.));
        expect(eq(late.sequences, 0U)) << "decided 12 ms after A, beyond max_delay 5 ms";
        const auto found = follow(events, byStart);
        expect(eq(found.sequences, 1U)) << "3 ms from A, inside [2, 5] ms";
        expect(found.dated == std::vector<std::pair<std::uint64_t, std::string>>{{kStart + 12U * kMs, found.blockName}}) << "still dated by its decision";
    };

    "reference start without a start tuple falls back to the decision time"_test = [] {
        auto settings         = waitFor(2., 20.);
        settings["reference"] = std::string("start");
        const auto found      = follow({eventNamed("arm", kStart), eventNamed("edge", kStart + 5U * kMs)}, settings);
        expect(eq(found.sequences, 1U));
    };

    "B within [min, max] after A is reported, dated as B"_test = [] {
        const auto found = follow({eventNamed("arm", kStart), eventNamed("edge", kStart + 5U * kMs)}, waitFor(2., 20.));
        expect(eq(found.sequences, 1U));
        expect(found.dated == std::vector<std::pair<std::uint64_t, std::string>>{{kStart + 5U * kMs, found.blockName}});
    };

    "the report is decided at B and holds A's own trigger in trigger_meta_info, from which the delay follows"_test = [] {
        const auto found = follow({eventNamed("arm", kStart), eventNamed("edge", kStart + 5U * kMs)}, waitFor(2., 20.));
        expect(eq(found.events.size(), 1UZ)) << fatal;
        const gr::property_map_view report{found.events[0]};
        const auto                  meta = report.get_if<gr::property_map_view>(gr::tag::TRIGGER_META_INFO.key());
        expect(meta.has_value()) << fatal;
        const auto decided = report.get_if<std::uint64_t>(gr::tag::TRIGGER_TIME.key());
        const auto armedAt = meta->get_if<std::uint64_t>(gr::tag::TRIGGER_TIME.key());
        const auto armedBy = meta->get_if<std::string_view>(gr::tag::TRIGGER_NAME.key());
        expect(decided != nullptr and *decided == kStart + 5U * kMs) << "B's time";
        expect(armedAt != nullptr and *armedAt == kStart) << "A's time";
        expect(armedBy.has_value() and *armedBy == "arm") << "A's own name";
        expect(decided != nullptr and armedAt != nullptr and *decided - *armedAt == 5U * kMs) << "the delay is root time minus meta time: 5 ms";
        expect(!meta->contains(std::string_view{"armed_at"}) and !meta->contains(std::string_view{"elapsed"})) << "no stored duration";
        expect(!gr::trigger_test::carriesSource(found.events[0]));
    };

    "B before A, or earlier than min_delay, is ignored"_test = [] {
        const auto found = follow({eventNamed("edge", kStart), eventNamed("arm", kStart + kMs), eventNamed("edge", kStart + 2U * kMs), eventNamed("edge", kStart + 4U * kMs)}, waitFor(2., 20.));
        expect(eq(found.ignored, 2U)) << "the B before A, and the B 1 ms after it";
        expect(eq(found.sequences, 1U)) << "the B 3 ms after A";
    };

    "one B per A: a second B needs a second A"_test = [] {
        const auto found = follow({eventNamed("arm", kStart), eventNamed("edge", kStart + 3U * kMs), eventNamed("edge", kStart + 4U * kMs)}, waitFor(0., 20.));
        expect(eq(found.sequences, 1U));
        expect(eq(found.ignored, 1U));
    };

    "closed bounds accept B exactly at min and max delay; open bounds do not"_test = [] {
        const std::vector<gr::property_map> atMin{eventNamed("arm", kStart), eventNamed("edge", kStart + 2U * kMs)};
        const std::vector<gr::property_map> atMax{eventNamed("arm", kStart), eventNamed("edge", kStart + 20U * kMs)};
        expect(eq(follow(atMin, waitFor(2., 20.)).sequences, 1U));
        expect(eq(follow(atMax, waitFor(2., 20.)).sequences, 1U));
        expect(eq(follow(atMin, waitFor(2., 20., "open")).sequences, 0U));
        expect(eq(follow(atMax, waitFor(2., 20., "open")).sequences, 0U));
        expect(eq(follow(atMin, waitFor(2., 20., "maxOpen")).sequences, 1U)) << "only the upper end open";
    };

    "by default the delay range is [start, deadline): B at the deadline is late, B at the minimum is on time"_test = [] {
        auto settings = waitFor(2., 20.);
        settings.erase("bounds");
        const std::vector<gr::property_map> atMin{eventNamed("arm", kStart), eventNamed("edge", kStart + 2U * kMs)};
        const std::vector<gr::property_map> atMax{eventNamed("arm", kStart), eventNamed("edge", kStart + 20U * kMs)};
        expect(eq(follow(atMin, settings).sequences, 1U));
        expect(eq(follow(atMax, settings).sequences, 0U));
    };

    "an A that waits past max_delay reports a timeout dated at its deadline, decided by a later event"_test = [] {
        const auto found = follow({eventNamed("arm", kStart), eventNamed("heartbeat", kStart + 30U * kMs), eventNamed("edge", kStart + 31U * kMs)}, waitFor(2., 20.));
        expect(eq(found.timeouts, 1U));
        expect(eq(found.sequences, 0U)) << "the B after the deadline belongs to no A";
        expect(found.dated == std::vector<std::pair<std::uint64_t, std::string>>{{kStart + 20U * kMs, "then_timeout"}});
        expect(eq(found.events.size(), 1UZ)) << fatal;
        const auto meta = gr::property_map_view{found.events[0]}.get_if<gr::property_map_view>(gr::tag::TRIGGER_META_INFO.key());
        expect(meta.has_value()) << fatal;
        const auto armedAt = meta->get_if<std::uint64_t>(gr::tag::TRIGGER_TIME.key());
        const auto armedBy = meta->get_if<std::string_view>(gr::tag::TRIGGER_NAME.key());
        expect(armedAt != nullptr and *armedAt == kStart) << "the A that waited in vain, 20 ms before the deadline";
        expect(armedBy.has_value() and *armedBy == "arm");
    };

    "a timeout fires at the deadline itself when the upper bound is open, and past it when closed"_test = [] {
        const std::vector<gr::property_map> events{eventNamed("arm", kStart), eventNamed("heartbeat", kStart + 20U * kMs)};
        const auto                          open = follow(events, waitFor(0., 20., "maxOpen"));
        expect(eq(open.timeouts, 1U)) << "elapsed 20 ms >= max_delay";
        expect(open.dated == std::vector<std::pair<std::uint64_t, std::string>>{{kStart + 20U * kMs, "then_timeout"}});
        expect(eq(open.pending, 0U));

        const auto closed = follow(events, waitFor(0., 20.));
        expect(eq(closed.timeouts, 0U)) << "a B at exactly 20 ms would still be in time";
        expect(eq(closed.pending, 1U)) << "and the A is still waiting when the run ends, which the counter says";
    };

    "simultaneous A and B advance the wait, whichever of them arrived first"_test = [] {
        const gr::property_map armAtT  = eventNamed("arm", kStart);
        const gr::property_map edgeAtT = eventNamed("edge", kStart);
        for (const auto& [order, events] : {std::pair{std::string("A first"), std::vector{armAtT, edgeAtT}}, std::pair{std::string("B first"), std::vector{edgeAtT, armAtT}}}) {
            const auto found = follow(events, waitFor(0., 20.));
            expect(eq(found.sequences, 1U)) << order << ": equal times are taken in the order of the tie-break, not of arrival";
        }
    };

    "an A held back for max_lateness at the end of the run is counted"_test = [] {
        auto settings                         = waitFor(0., 20.);
        settings[std::string("max_lateness")] = 1.;
        const auto found                      = follow({eventNamed("arm", kStart)}, std::move(settings));
        expect(eq(found.pending, 1U));
        expect(eq(found.timeouts, 0U));
    };

    "a clock heartbeat past the deadline times an A out without any later event of the stream's own"_test = [] {
        const auto found = timedOutBy({10U, 40U}, waitFor(0., 20.));
        expect(eq(found.timeouts, 1U));
        expect(found.dated == std::vector<std::pair<std::uint64_t, std::string>>{{kStart + 30U * kMs, "then_timeout"}}) << "A at 10 ms, deadline 20 ms later; the clock's own tick at 40 ms only decided it";
    };

    "a clock heartbeat exactly at A plus max_delay times out an open upper bound, and not a closed one"_test = [] {
        const auto open = timedOutBy({10U, 30U}, waitFor(0., 20., "maxOpen"));
        expect(eq(open.timeouts, 1U)) << ">= A + max_delay";
        expect(open.dated == std::vector<std::pair<std::uint64_t, std::string>>{{kStart + 30U * kMs, "then_timeout"}});
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
        const std::vector<gr::property_map> events{eventNamed("arm", kStart), eventNamed("arm", kStart + 15U * kMs), eventNamed("edge", kStart + 25U * kMs)};
        expect(eq(follow(events, waitFor(0., 20.)).sequences, 0U)) << "25 ms after the first A, which the second did not move";
        expect(eq(follow(events, waitFor(0., 20., "closed", "restart")).sequences, 1U)) << "10 ms after the second A";
    };

    "a disarm event cancels the wait"_test = [] {
        const auto found = follow({eventNamed("arm", kStart), eventNamed("disarm", kStart + kMs), eventNamed("edge", kStart + 3U * kMs)}, waitFor(0., 20.));
        expect(eq(found.disarmed, 1U));
        expect(eq(found.sequences, 0U));
    };
};
} // namespace qa_then

namespace qa_event_builder {
using namespace gr::blocks::trigger;
using gr::trigger_test::CollectingSink;
using gr::trigger_test::EventTap;

namespace {
/** publishes a written list of fragments, so what the builder sees does not depend on work-call granularity */
struct FragmentSource : gr::Block<FragmentSource> {
    gr::PortOut<gr::DataSet<float>, gr::Async> out;

    GR_MAKE_REFLECTABLE(FragmentSource, out);

    std::vector<gr::DataSet<float>> _fragments;
    std::size_t                     _published = 0UZ;

    /**
     * one fragment per work call, so two of these interleave the way two front-ends do: a source that emptied itself
     * first would make its own events look overdue before the other had answered
     */
    gr::work::Status processBulk(gr::OutputSpanLike auto& outSpan) {
        std::size_t emitted = 0UZ;
        if (_published < _fragments.size() && !outSpan.empty()) {
            outSpan[0] = _fragments[_published++];
            emitted    = 1UZ;
        }
        outSpan.publish(emitted);
        if (_published >= _fragments.size()) {
            this->requestStop();
            return gr::work::Status::DONE;
        }
        return gr::work::Status::OK;
    }
};

/** one fragment: a named signal of `length` samples, stamped with the event it belongs to */
[[nodiscard]] gr::DataSet<float> fragmentOf(std::int64_t at, std::string name, std::size_t length, std::uint64_t id = 0U) {
    gr::DataSet<float> fragment;
    fragment.timestamp = at;
    fragment.axis_names.emplace_back("time");
    fragment.axis_units.emplace_back("s");
    fragment.axis_values.resize(1UZ);
    for (std::size_t i = 0UZ; i < length; ++i) {
        fragment.axis_values[0].push_back(static_cast<float>(i));
    }
    fragment.extents.push_back(static_cast<std::int32_t>(length));
    fragment.signal_names.emplace_back(std::move(name));
    fragment.signal_quantities.emplace_back("voltage");
    fragment.signal_units.emplace_back("V");
    for (std::size_t i = 0UZ; i < length; ++i) {
        fragment.signal_values.push_back(static_cast<float>(i));
    }
    fragment.signal_ranges.resize(1UZ);
    fragment.meta_information.resize(1UZ);
    if (id != 0U) {
        fragment.meta_information[0].insert_or_assign(std::string_view{"event_id"}, id);
    }
    fragment.timing_events.resize(1UZ);
    return fragment;
}

struct Outcome {
    std::size_t              events     = 0UZ;
    std::size_t              signals    = 0UZ;
    gr::Size_t               incomplete = 0U;
    gr::Size_t               dropped    = 0U;
    gr::Size_t               mismatched = 0U;
    std::vector<std::string> errors;
};

[[nodiscard]] Outcome build(std::vector<gr::DataSet<float>> left, std::vector<gr::DataSet<float>> right, gr::property_map settings) {
    settings[std::string("n_inputs")] = gr::Size_t(2);
    gr::testing::GraphFixture fixture;
    auto&                     a = fixture.emplace<FragmentSource>();
    a._fragments                = std::move(left);
    auto& b                     = fixture.emplace<FragmentSource>();
    b._fragments                = std::move(right);
    auto& builder               = fixture.emplace<EventBuilder<float>>(std::move(settings));
    auto& sink                  = fixture.emplace<CollectingSink<gr::DataSet<float>>>();
    auto& tap                   = fixture.emplace<EventTap>();

    boost::ut::expect(fixture.graph.connect(a, gr::PortDefinition{std::string("out")}, builder, gr::PortDefinition{std::string("in#0")}).has_value());
    boost::ut::expect(fixture.graph.connect(b, gr::PortDefinition{std::string("out")}, builder, gr::PortDefinition{std::string("in#1")}).has_value());
    boost::ut::expect(fixture.connect<"out", "in">(builder, sink).has_value());
    boost::ut::expect(fixture.connect<"evtOut", "evtIn">(builder, tap).has_value());
    gr::trigger_test::ErrorMessages sent{fixture.scheduler()};
    boost::ut::expect(fixture.run().has_value());

    return Outcome{
        .events     = sink._collected.size(),
        .signals    = sink._collected.empty() ? 0UZ : sink._collected.front().signal_names.size(),
        .incomplete = builder.n_incomplete.value,
        .dropped    = builder.n_dropped.value,
        .mismatched = builder.n_mismatched.value,
        .errors     = sent.texts(),
    };
}

constexpr std::int64_t kAt = 1'000'000'000;
} // namespace

const boost::ut::suite<"EventBuilder"> _eventBuilder = [] {
    using namespace boost::ut;

    "fragments sharing a timestamp become one event"_test = [] {
        const auto found = build({fragmentOf(kAt, "ch0", 4UZ)}, {fragmentOf(kAt, "ch1", 4UZ)}, {});

        expect(eq(found.events, 1UZ));
        expect(eq(found.signals, 2UZ)) << "one signal per fragment, in input order";
        expect(eq(found.incomplete, 0U));
    };

    "timestamps within the tolerance are the same event"_test = [] {
        const auto found = build({fragmentOf(kAt, "ch0", 4UZ)}, {fragmentOf(kAt + 30, "ch1", 4UZ)}, {{"tolerance", 50e-9}});

        expect(eq(found.events, 1UZ)) << "30 ns apart, inside a 50 ns tolerance";
    };

    "timestamps beyond the tolerance are different events"_test = [] {
        const auto found = build({fragmentOf(kAt, "ch0", 4UZ)}, {fragmentOf(kAt + 500'000, "ch1", 4UZ)}, {{"tolerance", 50e-9}, {"timeout", 1e-6}});

        expect(eq(found.events, 0UZ)) << "neither event was ever complete";
        expect(ge(found.incomplete, 1U)) << "and the older one is judged when the newer arrives from beyond the timeout";
        expect(ge(found.dropped, 1U));
    };

    "an incomplete event can be published as it stands"_test = [] {
        const auto found = build({fragmentOf(kAt, "ch0", 4UZ)}, {fragmentOf(kAt + 500'000, "ch1", 4UZ)}, //
            {{"tolerance", 50e-9}, {"timeout", 1e-6}, {"on_incomplete", std::string("emit_partial")}});

        expect(ge(found.events, 1UZ)) << "what was there is published rather than thrown away";
        expect(eq(found.dropped, 0U));
        expect(ge(found.incomplete, 1U)) << "and it is still counted as incomplete";
    };

    "an identifier can key the event instead of a time"_test = [] {
        const auto found = build({fragmentOf(0, "ch0", 4UZ, 42U)}, {fragmentOf(0, "ch1", 4UZ, 42U)}, {{"key", std::string("event_id")}});

        expect(eq(found.events, 1UZ)) << "neither fragment carries a usable timestamp, and neither needs to";
        expect(eq(found.signals, 2UZ));
    };

    "a fragment of another length is refused, and says why"_test = [] {
        const auto found = build({fragmentOf(kAt, "ch0", 4UZ)}, {fragmentOf(kAt, "ch1", 7UZ)}, {});

        expect(eq(found.mismatched, 1U)) << "a DataSet carries one extent per signal, so the seven cannot join the four";
        expect(eq(found.events, 0UZ));
        expect(found.errors.size() == 1UZ && found.errors[0].contains("extents disagreed")) << "and the refusal reaches the message output";
    };

    "a fragment carrying nothing to match on is counted"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     a = fixture.emplace<FragmentSource>();
        a._fragments                = {fragmentOf(0, "ch0", 4UZ)};
        auto& b                     = fixture.emplace<FragmentSource>();
        b._fragments                = {fragmentOf(kAt, "ch1", 4UZ)};
        auto& builder               = fixture.emplace<EventBuilder<float>>({{"n_inputs", 2U}});
        auto& sink                  = fixture.emplace<CollectingSink<gr::DataSet<float>>>();
        expect(fixture.graph.connect(a, gr::PortDefinition{std::string("out")}, builder, gr::PortDefinition{std::string("in#0")}).has_value());
        expect(fixture.graph.connect(b, gr::PortDefinition{std::string("out")}, builder, gr::PortDefinition{std::string("in#1")}).has_value());
        expect(fixture.connect<"out", "in">(builder, sink).has_value());
        expect(fixture.run().has_value());

        expect(eq(builder.n_unkeyed.value, 1U)) << "matching by arrival order would invent an event";
        expect(eq(sink._collected.size(), 0UZ));
    };

    "what was assembled, drawn"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     a = fixture.emplace<FragmentSource>();
        a._fragments                = {fragmentOf(kAt, "ch0", 4UZ), fragmentOf(kAt + 1'000'000, "ch0", 4UZ)};
        auto& b                     = fixture.emplace<FragmentSource>();
        b._fragments                = {fragmentOf(kAt + 20, "ch1", 4UZ), fragmentOf(kAt + 1'000'020, "ch1", 4UZ)};
        auto& builder               = fixture.emplace<EventBuilder<float>>({{"n_inputs", 2U}, {"tolerance", 50e-9}, {"timeout", 1e-4}});
        auto& sink                  = fixture.emplace<CollectingSink<gr::DataSet<float>>>();
        expect(fixture.graph.connect(a, gr::PortDefinition{std::string("out")}, builder, gr::PortDefinition{std::string("in#0")}).has_value());
        expect(fixture.graph.connect(b, gr::PortDefinition{std::string("out")}, builder, gr::PortDefinition{std::string("in#1")}).has_value());
        expect(fixture.connect<"out", "in">(builder, sink).has_value());
        expect(fixture.run().has_value());

        gr::testing::MarbleDiagram diagram{"EventBuilder: two front-ends, two events, matched within 50 ns"};
        diagram.row("in#0").at(static_cast<std::uint64_t>(kAt), "ch0").at(static_cast<std::uint64_t>(kAt) + 1'000'000U, "ch0").completes();
        diagram.row("in#1").at(static_cast<std::uint64_t>(kAt) + 20U, "ch1").at(static_cast<std::uint64_t>(kAt) + 1'000'020U, "ch1").completes();
        diagram.condition("EventBuilder(n_inputs = 2, tolerance = 50 ns)");
        auto& out = diagram.row("out");
        for (const gr::DataSet<float>& event : sink._collected) {
            out.at(static_cast<std::uint64_t>(event.timestamp), std::format("{} signals", event.signal_names.size()));
        }
        out.completes();
        diagram.print();

        expect(eq(sink._collected.size(), 2UZ));
        expect(eq(builder.n_events.value, 2U));
    };
};
} // namespace qa_event_builder

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
