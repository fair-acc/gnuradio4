#include <boost/ut.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/math/Histogram.hpp>
#include <gnuradio-4.0/test/EventMarbles.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/trigger/Coincidence.hpp>
#include <gnuradio-4.0/trigger/EventReduce.hpp>
#include <gnuradio-4.0/trigger/TimeInterval.hpp>
#include <gnuradio-4.0/trigger/ValueTrigger.hpp>

#include "Marble.hpp"
#include "TriggerTest.hpp"

namespace qa_coincidence {
using namespace gr::blocks::trigger;
using gr::trigger_test::eventNamed;
using gr::trigger_test::EventScript;
using gr::trigger_test::EventTap;

struct Composite {
    std::size_t                   seen = 0UZ;
    std::vector<std::uint64_t>    times;
    std::vector<gr::property_map> events;
};

struct Outcome {
    gr::Size_t coincidences;
    gr::Size_t vetoed;
    gr::Size_t undated;
    gr::Size_t incomplete;
    gr::Size_t pending;
    gr::Size_t dropped;
    Composite  composite;
};

/** `eventsPerWorkCall = 1` keeps the arrival order: the block orders by trigger_time what one work call drains */
[[nodiscard]] Outcome combine(std::vector<gr::property_map> events, gr::property_map settings, std::size_t eventsPerWorkCall = std::numeric_limits<std::size_t>::max()) {
    gr::testing::GraphFixture fixture;
    auto&                     script = fixture.emplace<EventScript>();
    script._events                   = std::move(events);
    script._maxPerWorkCall           = eventsPerWorkCall;
    auto& unit                       = fixture.emplace<Coincidence>(std::move(settings));
    auto& collector                  = fixture.emplace<EventTap>();
    boost::ut::expect(fixture.connect<"evtOut", "evtIn">(script, unit).has_value());
    boost::ut::expect(fixture.connect<"evtOut", "evtIn">(unit, collector).has_value());
    boost::ut::expect(fixture.run().has_value());
    return Outcome{unit.n_coincidences.value, unit.n_vetoed.value, unit.n_undated.value, unit.n_incomplete.value, unit.n_pending.value, unit.n_events_dropped.value, Composite{collector.size(), collector.times(), collector._events}};
}

[[nodiscard]] std::vector<std::string> conditionsAB() { return {std::string("a"), std::string("b")}; }

const boost::ut::suite<"Coincidence"> _coincidence = [] {
    using namespace boost::ut;

    "two conditions inside the window make one coincidence"_test = [] {
        const auto found = combine({eventNamed("a", 1'000'000'000U), eventNamed("b", 1'000'000'030U)}, {{"filters", conditionsAB()}, {"logic", std::string("all")}, {"window", 50e-9}});
        expect(eq(found.coincidences, 1U)) << "30 ns apart, inside a 50 ns window";
        expect(eq(found.composite.seen, 1UZ));
    };

    "reference start relates a pulse decided late by its leading edge"_test = [] {
        // a: an edge at 0; b: a pulse from 30 ns, decided at 80 ns -- apart by 80 ns by decision, by 30 ns by start
        gr::property_map decided                               = eventNamed("b", 1'000'000'080U);
        decided[std::string(gr::tag::TRIGGER_META_INFO.key())] = gr::property_map{
            {std::string(gr::tag::TRIGGER_NAME.key()), std::string("rising")},
            {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1'000'000'030U}},
            {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f},
        };
        const std::vector<gr::property_map> events{eventNamed("a", 1'000'000'000U), decided};
        const auto                          byDecision = combine(events, {{"filters", conditionsAB()}, {"logic", std::string("all")}, {"window", 50e-9}});
        expect(eq(byDecision.coincidences, 0U)) << "80 ns apart is beyond the 50 ns window";
        const auto byStart = combine(events, {
                                                 {"filters", conditionsAB()},
                                                 {"logic", std::string("all")},
                                                 {"window", 50e-9},
                                                 {"reference", std::string("start")},
                                             });
        expect(eq(byStart.coincidences, 1U)) << "30 ns from a's time by the start tuple";
        expect(byStart.composite.times == std::vector<std::uint64_t>{1'000'000'000U}) << "the composite is dated by the earliest reference";
    };

    "the same two outside the window make none"_test = [] {
        const auto found = combine({eventNamed("a", 1'000'000'000U), eventNamed("b", 1'000'000'080U)}, {{"filters", conditionsAB()}, {"logic", std::string("all")}, {"window", 50e-9}});
        expect(eq(found.coincidences, 0U)) << "80 ns apart is beyond the window, so neither still holds";
    };

    "the composite is dated by its first contributor"_test = [] {
        const auto found = combine({eventNamed("a", 1'000'000'000U), eventNamed("b", 1'000'000'030U)}, {{"filters", conditionsAB()}, {"window", 50e-9}, {"resolve", std::string("first")}});
        expect(eq(found.composite.times.size(), 1UZ));
        if (!found.composite.times.empty()) {
            expect(eq(found.composite.times[0], std::uint64_t{1'000'000'000U})) << "the causal instant, not the moment it became knowable";
        }
    };

    "resolve = last dates it when the condition became knowable"_test = [] {
        const auto found = combine({eventNamed("a", 1'000'000'000U), eventNamed("b", 1'000'000'030U)}, {{"filters", conditionsAB()}, {"window", 50e-9}, {"resolve", std::string("last")}});
        expect(eq(found.composite.times.size(), 1UZ));
        if (!found.composite.times.empty()) {
            expect(eq(found.composite.times[0], std::uint64_t{1'000'000'030U}));
        }
    };

    "any reports on the first condition to hold"_test = [] {
        const auto found = combine({eventNamed("a", 1'000'000'000U), eventNamed("b", 2'000'000'000U)}, {{"filters", conditionsAB()}, {"logic", std::string("any")}, {"window", 50e-9}});
        expect(eq(found.coincidences, 2U)) << "each on its own, the window never needing to hold them together";
    };

    "atLeast counts how many held, not which"_test = [] {
        const std::vector<std::string> three{std::string("a"), std::string("b"), std::string("c")};
        const auto                     found = combine({eventNamed("a", 1'000'000'000U), eventNamed("c", 1'000'000'020U)}, {{"filters", three}, {"logic", std::string("atLeast")}, {"k", 2U}, {"window", 50e-9}});
        expect(eq(found.coincidences, 1U)) << "two of the three, inside the window";
    };

    "exclusive waits for the window to close, then reports only a lone condition"_test = [] {
        const auto alone = combine({eventNamed("a", 1'000'000'000U), eventNamed("a", 1'000'500'000U)}, //
            {{"filters", conditionsAB()}, {"logic", std::string("exclusive")}, {"window", 50e-9}});
        expect(eq(alone.coincidences, 1U)) << "one condition held and no other, judged once its window had closed";

        const auto both = combine({eventNamed("a", 1'000'000'000U), eventNamed("b", 1'000'000'010U), eventNamed("a", 1'000'500'000U)}, //
            {{"filters", conditionsAB()}, {"logic", std::string("exclusive")}, {"window", 50e-9}});
        expect(eq(both.coincidences, 0U)) << "two held, so it was never exclusive -- and nothing was reported and taken back";
    };

    "emit_partial reports a group that closed short of every condition"_test = [] {
        std::vector<gr::property_map> events{eventNamed("a", 1'000'000'000U), eventNamed("a", 1'000'500'000U)};

        const auto dropped = combine(events, {{"filters", conditionsAB()}, {"logic", std::string("all")}, {"window", 50e-9}});
        expect(eq(dropped.coincidences, 0U)) << "by default an incomplete group is simply not reported";

        const auto partial = combine(std::move(events), {{"filters", conditionsAB()}, {"logic", std::string("all")}, {"window", 50e-9}, {"on_incomplete", std::string("emit_partial")}});
        expect(eq(partial.coincidences, 1U)) << "asked for it, the group is reported although b never came";
        expect(eq(partial.incomplete, 1U)) << "and counted as incomplete";
    };

    "a composite says in trigger_meta_info how many conditions held, and no source"_test = [] {
        const auto whole = combine({eventNamed("a", 1'000'000'000U), eventNamed("b", 1'000'000'030U)}, {{"filters", conditionsAB()}, {"logic", std::string("all")}, {"window", 50e-9}});
        expect(eq(whole.composite.events.size(), 1UZ)) << fatal;
        const gr::property_map_view complete{whole.composite.events[0]};
        const auto                  completeMeta = complete.get_if<gr::property_map_view>(gr::tag::TRIGGER_META_INFO.key());
        expect(completeMeta.has_value()) << fatal;
        expect(eq(completeMeta->get_if<gr::Size_t>(std::string_view{"n_present"}) != nullptr ? *completeMeta->get_if<gr::Size_t>(std::string_view{"n_present"}) : 0U, 2U)) << "a and b both held";
        expect(eq(completeMeta->get_if<gr::Size_t>(std::string_view{"n_expected"}) != nullptr ? *completeMeta->get_if<gr::Size_t>(std::string_view{"n_expected"}) : 0U, 2U));
        expect(!gr::trigger_test::carriesSource(whole.composite.events[0])) << "no source, at the root or in trigger_meta_info";
        expect(!complete.contains(std::string_view{"n_present"}) and !complete.contains(std::string_view{"n_expected"})) << "none of them at the root";

        const auto partial = combine({eventNamed("a", 1'000'000'000U), eventNamed("a", 1'000'500'000U)}, {{"filters", conditionsAB()}, {"logic", std::string("all")}, {"window", 50e-9}, {"on_incomplete", std::string("emit_partial")}});
        expect(eq(partial.composite.events.size(), 1UZ)) << fatal;
        const auto partialMeta = gr::property_map_view{partial.composite.events[0]}.get_if<gr::property_map_view>(gr::tag::TRIGGER_META_INFO.key());
        expect(partialMeta.has_value()) << fatal;
        expect(eq(partialMeta->get_if<gr::Size_t>(std::string_view{"n_present"}) != nullptr ? *partialMeta->get_if<gr::Size_t>(std::string_view{"n_present"}) : 99U, 1U)) << "only a held, of the two expected";
        expect(eq(partialMeta->get_if<gr::Size_t>(std::string_view{"n_expected"}) != nullptr ? *partialMeta->get_if<gr::Size_t>(std::string_view{"n_expected"}) : 0U, 2U));
    };

    "what the window does, drawn"_test = [] {
        std::vector<gr::property_map> events{eventNamed("a", 1'000'000'000U), eventNamed("b", 1'000'000'030U), //
            eventNamed("a", 1'000'000'500U), eventNamed("b", 1'000'000'520U),                                  //
            eventNamed("a", 1'000'001'000U), eventNamed("b", 1'000'001'200U)};
        const auto                    found = combine(events, {{"filters", conditionsAB()}, {"logic", std::string("all")}, {"window", 50e-9}});

        expect(eq(found.coincidences, 2U)) << "the third pair is 200 ns apart, beyond the window";

        gr::testing::MarbleDiagram diagram{"Coincidence: all conditions inside a 50 ns window"};
        auto&                      rowA = diagram.row("a");
        auto&                      rowB = diagram.row("b");
        for (const gr::property_map& event : events) {
            const auto name  = gr::property_map_view{event}.get_if<std::string_view>(gr::tag::TRIGGER_NAME.key());
            const auto stamp = gr::property_map_view{event}.get_if<std::uint64_t>(gr::tag::TRIGGER_TIME.key());
            if (name && stamp) {
                (*name == std::string_view{"a"} ? rowA : rowB).at(*stamp, std::string(*name));
            }
        }
        diagram.condition("Coincidence(all, within 50 ns)");
        auto& out = diagram.row("out");
        for (const std::uint64_t at : found.composite.times) {
            out.at(at, "coincidence");
        }
        diagram.print();
    };

    "a veto suppresses the coincidence it arrives with"_test = [] {
        const auto found = combine({eventNamed("veto", 1'000'000'000U), eventNamed("a", 1'000'000'010U), eventNamed("b", 1'000'000'020U)}, //
            {{"filters", conditionsAB()}, {"veto_filter", std::string("veto")}, {"logic", std::string("all")}, {"window", 50e-9}});
        expect(eq(found.coincidences, 0U));
        expect(eq(found.vetoed, 1U)) << "the conditions held, and the veto took the report away";
    };

    "a veto outside the window does not reach the coincidence"_test = [] {
        const auto found = combine({eventNamed("veto", 1'000'000'000U), eventNamed("a", 1'000'000'900U), eventNamed("b", 1'000'000'910U)}, //
            {{"filters", conditionsAB()}, {"veto_filter", std::string("veto")}, {"logic", std::string("all")}, {"window", 50e-9}});
        expect(eq(found.coincidences, 1U)) << "the veto had already fallen out of the window";
        expect(eq(found.vetoed, 0U));
    };

    "holdoff keeps a second report from following straight on"_test = [] {
        std::vector<gr::property_map> events{eventNamed("a", 1'000'000'000U), eventNamed("b", 1'000'000'010U), eventNamed("a", 1'000'000'100U), eventNamed("b", 1'000'000'110U)};
        const auto                    without = combine(events, {{"filters", conditionsAB()}, {"window", 50e-9}});
        expect(eq(without.coincidences, 2U)) << "two pairs, two reports";

        const auto with = combine(std::move(events), {{"filters", conditionsAB()}, {"window", 50e-9}, {"holdoff", 1e-6}});
        expect(eq(with.coincidences, 1U)) << "the second pair falls inside the dead time";
    };

    "an undated event cannot join a coincidence"_test = [] {
        std::vector<gr::property_map> events{gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("a")}}, eventNamed("b", 1'000'000'000U)};
        const auto                    found = combine(std::move(events), {{"filters", conditionsAB()}, {"logic", std::string("all")}, {"window", 50e-9}});
        expect(eq(found.coincidences, 0U));
        expect(eq(found.undated, 1U)) << "a coincidence is a statement about time";
    };

    "a group still open when the run ends is counted, not silently dropped"_test = [] {
        /**
         * one condition arrives and nothing closes its window: a block whose only input is asynchronous gets no
         * further work call once a stop has been requested, so this group can never be reported -- but it is visible
         */
        const auto found = combine({eventNamed("a", 1'000'000'000U)}, //
            {{"filters", conditionsAB()}, {"logic", std::string("all")}, {"window", 50e-9}, {"on_incomplete", std::string("emit_partial")}});

        expect(eq(found.coincidences, 0U)) << "nothing closed the window, so nothing was judged";
        expect(eq(found.pending, 1U)) << "and what was left open says so, rather than disappearing";
    };

    "a condition beyond a bare trigger name keeps the block on the host"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     byName    = fixture.emplace<Coincidence>({{"filters", conditionsAB()}});
        auto&                     byContext = fixture.emplace<Coincidence>({{"filters", std::vector<std::string>{"a/ring", "b"}}});
        expect(byName.hostOnlyReason().empty()) << "names alone are matched by hash on a device";
        expect(!byContext.hostOnlyReason().empty()) << "a context is matched by substring, which a hash cannot do";
    };

    "a wildcard and a negation match by name on the host, and keep the block off a device"_test = [] {
        const std::vector<std::string> beamAndTick{std::string("beam_*"), std::string("tick")};
        const auto                     together = combine({eventNamed("beam_in", 1'000'000'000U), eventNamed("tick", 1'000'000'030U)}, {
                                                                                                                       {"filters", beamAndTick},
                                                                                                                       {"logic", std::string("all")},
                                                                                                                       {"window", 50e-9},
                                                                                                                   });
        expect(eq(together.coincidences, 1U)) << "beam_in satisfies beam_*";

        const auto quiet = combine({eventNamed("noise", 1'000'000'000U), eventNamed("beam", 2'000'000'000U)}, {
                                                                                                                  {"filters", std::vector<std::string>{"!noise"}},
                                                                                                                  {"logic", std::string("any")},
                                                                                                                  {"window", 50e-9},
                                                                                                              });
        expect(eq(quiet.coincidences, 1U)) << "everything but noise, so only the beam";

        gr::testing::GraphFixture fixture;
        auto&                     wildcard = fixture.emplace<Coincidence>({{"filters", beamAndTick}});
        auto&                     negation = fixture.emplace<Coincidence>({{"filters", std::vector<std::string>{"!noise", "tick"}}});
        auto&                     vetoed   = fixture.emplace<Coincidence>({{"filters", conditionsAB()}, {"veto_filter", std::string("veto")}});
        expect(!wildcard.hostOnlyReason().empty()) << "a kernel compares trigger names by hash, which '*' is not";
        expect(!negation.hostOnlyReason().empty());
        expect(!vetoed.hostOnlyReason().empty()) << "the veto matcher runs on the host";
    };

    "an event with a partner inside its window is never reported as exclusive"_test = [] {
        const std::vector<gr::property_map> events{
            eventNamed("a", 1'000'000'000U),
            eventNamed("b", 1'000'000'040U),
            eventNamed("heartbeat", 1'000'000'060U),
            eventNamed("heartbeat", 1'000'000'200U),
        };
        const auto exclusive = combine(events, {{"filters", conditionsAB()}, {"logic", std::string("exclusive")}, {"window", 50e-9}});
        expect(eq(exclusive.coincidences, 0U)) << "a and b are 40 ns apart; b must not be reported alone once a has left the window";

        const auto exactlyOne = combine(events, {{"filters", conditionsAB()}, {"logic", std::string("exactly")}, {"k", 1U}, {"window", 50e-9}});
        expect(eq(exactlyOne.coincidences, 0U));

        const std::vector<std::string> abc{std::string("a"), std::string("b"), std::string("c")};
        const auto                     two = combine(
            {
                eventNamed("a", 1'000'000'000U),
                eventNamed("b", 1'000'000'020U),
                eventNamed("heartbeat", 1'000'000'200U),
            },
            {{"filters", abc}, {"logic", std::string("exactly")}, {"k", 2U}, {"window", 50e-9}});
        expect(eq(two.coincidences, 1U)) << "exactly two of three held";
        const auto three = combine(
            {
                eventNamed("a", 1'000'000'000U),
                eventNamed("b", 1'000'000'020U),
                eventNamed("c", 1'000'000'030U),
                eventNamed("heartbeat", 1'000'000'200U),
            },
            {{"filters", abc}, {"logic", std::string("exactly")}, {"k", 2U}, {"window", 50e-9}});
        expect(eq(three.coincidences, 0U)) << "three held is not exactly two";
    };

    "a coincidence consumes one event per condition and leaves the others for the next one"_test = [] {
        const auto found = combine({eventNamed("a", 1'000'000'000U), eventNamed("a", 1'000'000'020U), eventNamed("b", 1'000'000'030U), eventNamed("b", 1'000'000'060U)}, //
            {{"filters", conditionsAB()}, {"logic", std::string("all")}, {"window", 50e-9}});
        expect(eq(found.coincidences, 2U)) << "a0 with b30, then the a at 20 with the b at 60 (40 ns apart)";
        expect(found.composite.times == std::vector<std::uint64_t>{1'000'000'000U, 1'000'000'020U}) << "each dated by its own earliest member";
    };

    "emit_partial reports the closed group only and keeps the later events for theirs"_test = [] {
        const auto found = combine({eventNamed("a", 1'000'000'000U), eventNamed("a", 1'000'000'020U), eventNamed("heartbeat", 1'000'000'060U), eventNamed("b", 1'000'000'070U)}, //
            {{"filters", conditionsAB()}, {"logic", std::string("all")}, {"window", 50e-9}, {"on_incomplete", std::string("emit_partial")}});
        expect(eq(found.incomplete, 1U)) << "the a at 0 closed alone at 60 ns";
        expect(eq(found.coincidences, 2U)) << "and the a at 20 was not cleared with it: it meets the b at 70, 50 ns later";
    };

    "an event arriving late by less than max_lateness is not lost"_test = [] {
        const std::vector<gr::property_map> events{
            eventNamed("a", 1'000'000'000U),
            eventNamed("heartbeat", 1'000'000'090U),
            eventNamed("b", 1'000'000'030U),
            eventNamed("heartbeat", 1'000'000'300U),
        };
        const gr::property_map strict{{"filters", conditionsAB()}, {"logic", std::string("all")}, {"window", 50e-9}};
        const auto             without = combine(events, strict, 1UZ);
        expect(eq(without.coincidences, 0U)) << "b arrived after event time passed its window";
        expect(eq(without.dropped, 1U));

        gr::property_map tolerant             = strict;
        tolerant[std::string("max_lateness")] = 100e-9;
        const auto with                       = combine(events, tolerant, 1UZ);
        expect(eq(with.coincidences, 1U)) << "the store waits 100 ns for stragglers and hands the block the events in time order";
        expect(eq(with.dropped, 0U));
    };

    "events held back for max_lateness when the run ends are counted as pending"_test = [] {
        const auto found = combine({eventNamed("a", 1'000'000'000U), eventNamed("b", 1'000'000'010U)}, {
                                                                                                           {"filters", conditionsAB()},
                                                                                                           {"logic", std::string("all")},
                                                                                                           {"window", 50e-9},
                                                                                                           {"max_lateness", 1.},
                                                                                                       });
        expect(eq(found.coincidences, 0U));
        expect(eq(found.pending, 2U)) << "neither passed the watermark by a second";
    };

    "conditions past the eighth are not held"_test = [] {
        const std::vector<std::string> nine{"a", "b", "c", "d", "e", "f", "g", "h", "i"};
        const auto                     ninth = combine({eventNamed("i", 1'000'000'000U)}, {{"filters", nine}, {"logic", std::string("any")}, {"window", 50e-9}});
        const auto                     first = combine({eventNamed("a", 1'000'000'000U)}, {{"filters", nine}, {"logic", std::string("any")}, {"window", 50e-9}});
        expect(eq(ninth.coincidences, 0U)) << "the ninth filter was dropped when the settings were applied";
        expect(eq(first.coincidences, 1U));
    };

    "two events farther apart than the window never coincide, whichever arrives first"_test = [] {
        const gr::property_map         d0  = eventNamed("d0", 1'000'060'000U);
        const gr::property_map         ch0 = eventNamed("ch0", 1'000'080'000U);
        const std::vector<std::string> conditions{std::string("d0"), std::string("ch0")};
        const gr::property_map         settings{{"filters", conditions}, {"logic", std::string("all")}, {"window", 1e-6}};

        const auto inOrder = combine({d0, ch0}, settings, 1UZ);
        expect(eq(inOrder.coincidences, 0U)) << "d0 at 60 us and ch0 at 80 us are 20 us apart, beyond a 1 us window";

        const auto late = combine({ch0, d0}, settings, 1UZ);
        expect(eq(late.coincidences, 0U)) << "d0 arriving after ch0 is still 20 us before it, not a 20 us wide coincidence";
        expect(eq(late.dropped, 1U)) << "d0 came after event time had passed its window, so it is counted, not held";
    };

    "a coincidence inside the window is the same report whichever contributor arrives first"_test = [] {
        const gr::property_map a = eventNamed("a", 1'000'000'000U);
        const gr::property_map b = eventNamed("b", 1'000'000'030U);
        for (const std::string& resolve : {std::string("first"), std::string("last")}) {
            const std::uint64_t    expected = resolve == "first" ? 1'000'000'000U : 1'000'000'030U;
            const gr::property_map settings{{"filters", conditionsAB()}, {"logic", std::string("all")}, {"window", 50e-9}, {"resolve", resolve}};
            for (const auto& [order, events] : {std::pair{std::string("a then b"), std::vector{a, b}}, std::pair{std::string("b then a"), std::vector{b, a}}}) {
                const auto found = combine(events, settings, 1UZ);
                expect(eq(found.coincidences, 1U)) << resolve << ", " << order << ": 30 ns apart, inside a 50 ns window";
                expect(found.composite.times == std::vector<std::uint64_t>{expected}) << resolve << ", " << order << ": dated by trigger_time, not by arrival";
            }
        }
    };

    "holdoff is dead time in event time, whichever member of the next pair arrives first"_test = [] {
        const gr::property_map settings{{"filters", conditionsAB()}, {"logic", std::string("all")}, {"window", 50e-9}, {"holdoff", 1e-6}};
        const auto             afterFirstPair = [](std::uint64_t aAt, std::uint64_t bAt, bool aArrivesFirst) {
            std::vector<gr::property_map> events{eventNamed("a", 1'000'000'000U), eventNamed("b", 1'000'000'010U)};
            events.push_back(eventNamed(aArrivesFirst ? "a" : "b", aArrivesFirst ? aAt : bAt));
            events.push_back(eventNamed(aArrivesFirst ? "b" : "a", aArrivesFirst ? bAt : aAt));
            return events;
        };
        for (const bool aArrivesFirst : {true, false}) {
            const std::string order = aArrivesFirst ? "a arrives first" : "b arrives first";

            const auto inside = combine(afterFirstPair(1'000'000'990U, 1'000'001'020U, aArrivesFirst), settings, 1UZ);
            expect(eq(inside.coincidences, 1U)) << order << ": the first report is dated 1'000'000'000 ns, its dead time ends at 1'000'001'000, and the next pair is dated 1'000'000'990";
            expect(inside.composite.times == std::vector<std::uint64_t>{1'000'000'000U}) << order;

            const auto after = combine(afterFirstPair(1'000'001'000U, 1'000'001'030U, aArrivesFirst), settings, 1UZ);
            expect(eq(after.coincidences, 2U)) << order << ": a pair dated 1'000'001'000 begins as the dead time ends";
            expect(after.composite.times == std::vector<std::uint64_t>{1'000'000'000U, 1'000'001'000U}) << order;
        }
    };

    "a dated event matching no condition is a heartbeat that closes the window"_test = [] {
        const auto found = combine({eventNamed("a", 1'000'000'000U), eventNamed("heartbeat", 2'000'000'000U)}, //
            {{"filters", conditionsAB()}, {"logic", std::string("exclusive")}, {"window", 50e-9}});

        expect(eq(found.coincidences, 1U)) << "the heartbeat moved event time past the window, so the lone condition was judged";
        expect(eq(found.pending, 0U));
    };
};
} // namespace qa_coincidence

namespace qa_time_interval {
using namespace gr::blocks::trigger;
using gr::trigger_test::eventNamed;
using gr::trigger_test::EventScript;

template<typename T = double>
struct Measured {
    std::vector<T> values;
    gr::Size_t     unmatched = 0U;
    gr::Size_t     undated   = 0U;
};

template<typename T = double>
[[nodiscard]] Measured<T> measure(std::vector<gr::property_map> events, gr::property_map settings) {
    gr::testing::GraphFixture fixture;
    auto&                     script = fixture.emplace<EventScript>();
    script._events                   = std::move(events);
    auto& interval                   = fixture.emplace<TimeInterval<T>>(std::move(settings));
    auto& sink                       = fixture.emplace<gr::testing::TagSink<T, gr::testing::ProcessFunction::USE_PROCESS_ONE>>();
    boost::ut::expect(fixture.connect<"evtOut", "evtIn">(script, interval).has_value());
    boost::ut::expect(fixture.connect<"out", "in">(interval, sink).has_value());
    boost::ut::expect(fixture.run().has_value());
    return Measured<T>{std::vector<T>(sink._samples.begin(), sink._samples.end()), interval.n_unmatched.value, interval.n_undated.value};
}

const boost::ut::suite<"TimeInterval"> _timeInterval = [] {
    using namespace boost::ut;

    "each event is measured against the most recent reference"_test = [] {
        const auto found = measure({eventNamed("clk", 1'000'000'000U), eventNamed("pulse", 1'000'000'500U), eventNamed("clk", 1'001'000'000U), eventNamed("pulse", 1'001'000'250U)}, //
            {{"mode", std::string("toReference")}, {"reference_filter", std::string("clk")}, {"measure_filter", std::string("pulse")}});

        expect(eq(found.values.size(), 2UZ)) << "one measurement per pulse";
        if (found.values.size() == 2UZ) {
            expect(approx(found.values[0], 500e-9, 1e-15)) << "500 ns after the first clock";
            expect(approx(found.values[1], 250e-9, 1e-15)) << "250 ns after the second";
        }
    };

    "a measurement with no reference yet is counted, not invented"_test = [] {
        const auto found = measure({eventNamed("pulse", 1'000'000'000U), eventNamed("clk", 1'000'001'000U), eventNamed("pulse", 1'000'002'000U)}, //
            {{"mode", std::string("toReference")}, {"reference_filter", std::string("clk")}, {"measure_filter", std::string("pulse")}});

        expect(eq(found.values.size(), 1UZ)) << "only the pulse that followed a clock is measurable";
        expect(eq(found.unmatched, 1U)) << "and the one before it is counted as unmatched";
    };

    "a paired measurement consumes its reference, as a time of flight does"_test = [] {
        const auto found = measure({eventNamed("emit", 0U), eventNamed("echo", 1'000U), eventNamed("echo", 2'000U), eventNamed("emit", 10'000U), eventNamed("echo", 11'500U)}, //
            {{"mode", std::string("paired")}, {"reference_filter", std::string("emit")}, {"measure_filter", std::string("echo")}});

        expect(eq(found.values.size(), 2UZ)) << "one echo per emission, the second echo having no emission of its own";
        expect(eq(found.unmatched, 1U));
        if (found.values.size() == 2UZ) {
            expect(approx(found.values[0], 1e-6, 1e-15));
            expect(approx(found.values[1], 1.5e-6, 1e-15));
        }
    };

    "toPrevious measures the period between successive events"_test = [] {
        const auto found = measure({eventNamed("tick", 0U), eventNamed("tick", 1'000'000U), eventNamed("tick", 2'000'000U), eventNamed("tick", 3'000'500U)}, //
            {{"mode", std::string("toPrevious")}});

        expect(eq(found.values.size(), 3UZ)) << "three gaps between four events";
        if (found.values.size() == 3UZ) {
            expect(approx(found.values[0], 1e-3, 1e-12));
            expect(approx(found.values[2], 1.0005e-3, 1e-12)) << "the period that ran 500 ns long";
        }
    };

    "tie measures the departure from an ideal grid"_test = [] {
        const auto found = measure({eventNamed("tick", 0U), eventNamed("tick", 1'000'000U), eventNamed("tick", 2'000'500U)}, //
            {{"mode", std::string("tie")}, {"nominal_period", 1e-3}});

        expect(eq(found.values.size(), 2UZ));
        if (found.values.size() == 2UZ) {
            expect(approx(found.values[0], 0., 1e-12)) << "the first tick is on the grid";
            expect(approx(found.values[1], 500e-9, 1e-12)) << "the second is 500 ns off it";
        }
    };

    "tie counts grid positions, so a missing edge leaves the later ones on the grid"_test = [] {
        const auto found = measure({eventNamed("tick", 1'000'000'000U), eventNamed("tick", 1'000'001'010U), eventNamed("tick", 1'000'001'980U), eventNamed("tick", 1'000'004'030U)}, //
            {{"mode", std::string("tie")}, {"nominal_period", 1e-6}});

        expect(eq(found.values.size(), 3UZ)) << fatal;
        expect(approx(found.values[0], 10e-9, 1e-12)) << "1010 ns after the first, 10 ns past the first grid position";
        expect(approx(found.values[1], -20e-9, 1e-12)) << "1980 ns, 20 ns before the second";
        expect(approx(found.values[2], 30e-9, 1e-12)) << "4030 ns: the fourth position, the third edge being missing";
    };

    "an interval beyond the limit counts as unmatched rather than reported"_test = [] {
        const auto found = measure({eventNamed("clk", 0U), eventNamed("pulse", 1'000'000'000U)}, //
            {{"mode", std::string("toReference")}, {"reference_filter", std::string("clk")}, {"measure_filter", std::string("pulse")}, {"max_interval", 1e-3}});

        expect(found.values.empty()) << "a second-long gap against a millisecond limit means the pair was wrong";
        expect(eq(found.unmatched, 1U));
    };

    "an undated event is counted, never measured"_test = [] {
        std::vector<gr::property_map> events{eventNamed("clk", 1'000'000'000U), gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("pulse")}}};
        const auto                    found = measure(std::move(events), {{"reference_filter", std::string("clk")}, {"measure_filter", std::string("pulse")}});

        expect(found.values.empty());
        expect(eq(found.undated, 1U)) << "a difference against an event with no time would be invented";
    };

    "uncertainties add in quadrature when the output carries them"_test = [] {
        const auto found = measure<gr::UncertainValue<double>>({eventNamed("clk", 0U, 30U), eventNamed("pulse", 1'000U, 40U)}, //
            {{"reference_filter", std::string("clk")}, {"measure_filter", std::string("pulse")}});

        expect(eq(found.values.size(), 1UZ));
        if (!found.values.empty()) {
            expect(approx(gr::value(found.values[0]), 1e-6, 1e-15));
            expect(approx(gr::uncertainty(found.values[0]), 50e-9, 1e-15)) << "30 ns and 40 ns add to 50 ns";
        }
    };

    "a period between successive events carries both their uncertainties"_test = [] {
        const auto found = measure<gr::UncertainValue<double>>({eventNamed("pulse", 0U, 30U), eventNamed("pulse", 1'000U, 40U)}, {{"mode", std::string("toPrevious")}});

        expect(eq(found.values.size(), 1UZ)) << fatal;
        expect(approx(gr::uncertainty(found.values[0]), 50e-9, 1e-15)) << "the earlier event's 30 ns and the later one's 40 ns, not the later one's twice";
    };

    "an edge detector's own events measure the period between its edges"_test = [] {
        const gr::property_map values{{"lo", 1.0f}, {"hi", 5.0f}};
        const gr::property_map tags{{"T", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("sync")}, //
                                              {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1U}},                //
                                              {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}}};

        gr::testing::GraphFixture fixture;
        auto&                     source  = fixture.emplace<MarbleSource<float>>({{"script", std::string("T:lo hi lo hi lo |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&                     trigger = fixture.emplace<ValueTrigger<float, ValueCondition::level>>({{"threshold", 3.f}, {"hysteresis", 0.5f}, {"sample_rate", 1000.f}});
        trigger.in.max_samples            = 2UZ;
        auto& spent                       = fixture.emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        auto& interval                    = fixture.emplace<TimeInterval<double>>({{"mode", std::string("toPrevious")}});
        auto& measured                    = fixture.emplace<gr::testing::TagSink<double, gr::testing::ProcessFunction::USE_PROCESS_ONE>>();
        expect(fixture.connect<"out", "in">(source, trigger).has_value());
        expect(fixture.connect<"out", "in">(trigger, spent).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(trigger, interval).has_value());
        expect(fixture.connect<"out", "in">(interval, measured).has_value());
        expect(fixture.run().has_value());

        expect(eq(trigger.n_triggers.value, 2U)) << "two rising crossings";
        expect(eq(interval.n_undated.value, 0U)) << "both arrived dated";
        expect(eq(interval.n_intervals.value, 1U)) << "two edges give one period between them";
    };

    "what an interval measurement sees, drawn"_test = [] {
        std::vector<gr::property_map> script;
        std::vector<std::uint64_t>    clocks;
        std::vector<std::uint64_t>    pulses;
        for (std::size_t tick = 0UZ; tick < 6UZ; ++tick) {
            const std::uint64_t clock = 1'000'000'000U + static_cast<std::uint64_t>(tick) * 1'000'000U;
            const std::uint64_t pulse = clock + 100'000U + static_cast<std::uint64_t>(tick) * 40'000U;
            script.push_back(eventNamed("clk", clock));
            script.push_back(eventNamed("pulse", pulse));
            clocks.push_back(clock);
            pulses.push_back(pulse);
        }
        const auto found = measure(script, {{"mode", std::string("toReference")}, {"reference_filter", std::string("clk")}, {"measure_filter", std::string("pulse")}});

        expect(eq(found.values.size(), 6UZ)) << "one measurement per pulse";

        gr::testing::MarbleDiagram diagram{"TimeInterval: a pulse drifting against its reference"};
        auto&                      clockRow = diagram.row("clk");
        for (const std::uint64_t at : clocks) {
            clockRow.at(at, "clk");
        }
        auto& pulseRow = diagram.row("pulse");
        for (const std::uint64_t at : pulses) {
            pulseRow.at(at, "pulse");
        }
        diagram.condition("TimeInterval(toReference)");
        diagram.print();

        std::vector<double> x(found.values.size());
        std::vector<double> y(found.values.size());
        for (std::size_t i = 0UZ; i < found.values.size(); ++i) {
            x[i] = static_cast<double>(i);
            y[i] = found.values[i] * 1e6;
        }
        auto chart = gr::graphs::ImChart<90, 14>({{x.front(), x.back()}, {0., 400.}});
        chart.draw(x, y, "delay [us]");
        std::println("\nthe measured delay, one point per pulse -- timing turned into a signal:");
        chart.draw();
    };

    "nearest measures against whichever reference is closer, before or after"_test = [] {
        const auto found = measure({eventNamed("clk", 1'000'000'000U), eventNamed("pulse", 1'000'200'000U), eventNamed("pulse", 1'000'900'000U), eventNamed("clk", 1'001'000'000U)}, //
            {{"mode", std::string("nearest")}, {"reference_filter", std::string("clk")}, {"measure_filter", std::string("pulse")}});

        expect(eq(found.values.size(), 2UZ));
        if (found.values.size() == 2UZ) {
            expect(approx(found.values[0], 200e-6, 1e-12)) << "after its reference, so positive";
            expect(approx(found.values[1], -100e-6, 1e-12)) << "before the next one, so negative: the sign says which side";
        }
    };

    "nearest waits for the later reference rather than answering early"_test = [] {
        const auto found = measure({eventNamed("clk", 1'000'000'000U), eventNamed("pulse", 1'000'900'000U)}, //
            {{"mode", std::string("nearest")}, {"reference_filter", std::string("clk")}, {"measure_filter", std::string("pulse")}});

        expect(eq(found.values.size(), 0UZ)) << "no reference followed it, so the nearest one is not yet known";
    };

    "nearest measures a pulse that precedes every reference against the one after it"_test = [] {
        const auto found = measure({eventNamed("pulse", 1'000'700'000U), eventNamed("clk", 1'001'000'000U)}, //
            {{"mode", std::string("nearest")}, {"reference_filter", std::string("clk")}, {"measure_filter", std::string("pulse")}});

        expect(eq(found.values.size(), 1UZ));
        if (found.values.size() == 1UZ) {
            expect(approx(found.values[0], -300e-6, 1e-12)) << "no reference before it, so the later one is the nearest: 0.3 ms early";
        }
    };

    "nearest counts a pulse whose only reference lies beyond the limit"_test = [] {
        const auto found = measure({eventNamed("pulse", 1'000'000'000U), eventNamed("clk", 1'002'000'000U)}, //
            {{"mode", std::string("nearest")}, {"reference_filter", std::string("clk")}, {"measure_filter", std::string("pulse")}, {"max_interval", 1e-3}});

        expect(eq(found.values.size(), 0UZ)) << "2 ms exceeds the 1 ms limit";
        expect(eq(found.unmatched, 1U));
    };

    "the distribution of a period, which is what a jitter figure is read from"_test = [] {
        constexpr std::uint64_t               kPeriod = 1'000'000U;
        constexpr std::array<std::int64_t, 5> kJitter{0, 12'000, -8'000, 4'000, -16'000};
        std::vector<gr::property_map>         ticks;
        std::vector<std::uint64_t>            when;
        for (std::size_t k = 0UZ; k < 40UZ; ++k) {
            when.push_back(1'000'000'000U + static_cast<std::uint64_t>(k) * kPeriod + static_cast<std::uint64_t>(kJitter[k % kJitter.size()] + 20'000));
            ticks.push_back(eventNamed("tick", when.back()));
        }

        gr::testing::GraphFixture fixture;
        auto&                     script = fixture.emplace<EventScript>();
        script._events                   = ticks;
        auto& interval                   = fixture.emplace<TimeInterval<double>>({{"mode", std::string("toPrevious")}});
        auto& histogram                  = fixture.emplace<gr::blocks::math::Histogram<double>>({{"bin_min", 0.00095}, {"bin_max", 0.00105}, {"n_bins", 20U}, //
                             {"reset_on_publish", false}, {"disconnect_on_done", false}, {"axis_name", gr::Tensor<gr::pmt::Value>(gr::data_from, {gr::pmt::Value("period")})}, {"axis_unit", gr::Tensor<gr::pmt::Value>(gr::data_from, {gr::pmt::Value("s")})}});

        expect(fixture.connect<"evtOut", "evtIn">(script, interval).has_value());
        expect(fixture.connect<"out", "in">(interval, histogram).has_value());
        expect(fixture.run().has_value());

        expect(eq(histogram._entriesInRange, 39U)) << "39 gaps between 40 ticks, every one inside the range";
        expect(eq(histogram._belowRange, 0U));
        expect(eq(histogram._aboveRange, 0U));
        expect(approx(histogram._mean, 1e-3, 2e-6)) << "the jitter averages out, leaving the nominal period";
        expect(histogram._stdDev > 1e-6) << "the spread is the jitter figure itself";

        {
            gr::testing::MarbleDiagram diagram{"TimeInterval into Histogram: a period measured, then distributed"};
            auto&                      tickRow = diagram.row("tick");
            for (const std::uint64_t at : when) {
                tickRow.at(at, "tick");
            }
            diagram.condition(std::format("TimeInterval(toPrevious) -> Histogram(20 bins over [0.95, 1.05] ms), mean {:.4f} ms, stddev {:.4f} ms", //
                histogram._mean * 1e3, histogram._stdDev * 1e3));
            diagram.print();
        }

        std::vector<double> centres(histogram._counts.size());
        std::vector<double> counts(histogram._counts.size());
        const double        width = (0.00105 - 0.00095) / static_cast<double>(histogram._counts.size());
        for (std::size_t bin = 0UZ; bin < histogram._counts.size(); ++bin) {
            centres[bin] = (0.00095 + (static_cast<double>(bin) + 0.5) * width) * 1e3;
            counts[bin]  = static_cast<double>(histogram._counts[bin]);
        }
        auto chart = gr::graphs::ImChart<90, 14>({{centres.front(), centres.back()}, {0., *std::ranges::max_element(counts)}});
        chart.draw<gr::graphs::Style::Bars>(centres, counts, "counts");
        std::println("\nthe distribution of the measured period -- the jitter histogram:");
        chart.draw();
    };
};
} // namespace qa_time_interval

namespace qa_event_builder {
using gr::trigger_test::kStreamStart;
using namespace gr::blocks::trigger;
using gr::trigger_test::CollectingSink;
using gr::trigger_test::EventTap;

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

constexpr std::int64_t kAt = static_cast<std::int64_t>(kStreamStart);

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
