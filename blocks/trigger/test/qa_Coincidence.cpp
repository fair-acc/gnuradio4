#include <boost/ut.hpp>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gnuradio-4.0/test/EventMarbles.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/trigger/Coincidence.hpp>

#include "TriggerTest.hpp"

using namespace gr::blocks::trigger;
using gr::trigger_test::eventNamed;
using gr::trigger_test::EventScript;
using gr::trigger_test::EventTap;

namespace {
struct Composite {
    std::size_t                seen = 0UZ;
    std::vector<std::uint64_t> times;
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
    return Outcome{unit.n_coincidences.value, unit.n_vetoed.value, unit.n_undated.value, unit.n_incomplete.value, unit.n_pending.value, unit.n_events_dropped.value, Composite{collector.size(), collector.times()}};
}

/**
 * returned by value: a namespace-scope container read from a statically executed test body depends on
 * initialisation order, and reads garbage when it loses
 */
[[nodiscard]] std::vector<std::string> conditionsAB() { return {std::string("a"), std::string("b")}; }
} // namespace

const boost::ut::suite<"Coincidence"> _coincidence = [] {
    using namespace boost::ut;

    "two conditions inside the window make one coincidence"_test = [] {
        const auto found = combine({eventNamed("a", 1'000'000'000U), eventNamed("b", 1'000'000'030U)}, {{"filters", conditionsAB()}, {"logic", std::string("all")}, {"window", 50e-9}});
        expect(eq(found.coincidences, 1U)) << "30 ns apart, inside a 50 ns window";
        expect(eq(found.composite.seen, 1UZ));
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

    "at_least counts how many held, not which"_test = [] {
        const std::vector<std::string> three{std::string("a"), std::string("b"), std::string("c")};
        const auto                     found = combine({eventNamed("a", 1'000'000'000U), eventNamed("c", 1'000'000'020U)}, {{"filters", three}, {"logic", std::string("at_least")}, {"k", 2U}, {"window", 50e-9}});
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

int main() { /* tests are statically executed */ }
