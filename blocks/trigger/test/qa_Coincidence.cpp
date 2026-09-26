#include <boost/ut.hpp>

#include <cstdint>
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
    Composite  composite;
};

[[nodiscard]] Outcome combine(std::vector<gr::property_map> events, gr::property_map settings) {
    gr::testing::GraphFixture fixture;
    auto&                     script = fixture.emplace<EventScript>();
    script._events                   = std::move(events);
    auto& unit                       = fixture.emplace<Coincidence>(std::move(settings));
    auto& collector                  = fixture.emplace<EventTap>();
    boost::ut::expect(fixture.connect<"evtOut", "evtIn">(script, unit).has_value());
    boost::ut::expect(fixture.connect<"evtOut", "evtIn">(unit, collector).has_value());
    boost::ut::expect(fixture.run().has_value());
    return Outcome{unit.n_coincidences.value, unit.n_vetoed.value, unit.n_undated.value, unit.n_incomplete.value, unit.n_pending.value, Composite{collector.size(), collector.times()}};
}

/// returned by value: a namespace-scope container read from a statically executed test body depends on
/// initialisation order, and reads garbage when it loses
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
        // the trailing event is what closes the earlier group: nothing else says its window has passed
        const auto alone = combine({eventNamed("a", 1'000'000'000U), eventNamed("a", 1'000'500'000U)}, //
            {{"filters", conditionsAB()}, {"logic", std::string("exclusive")}, {"window", 50e-9}});
        expect(eq(alone.coincidences, 1U)) << "one condition held and no other, judged once its window had closed";

        const auto both = combine({eventNamed("a", 1'000'000'000U), eventNamed("b", 1'000'000'010U), eventNamed("a", 1'000'500'000U)}, //
            {{"filters", conditionsAB()}, {"logic", std::string("exclusive")}, {"window", 50e-9}});
        expect(eq(both.coincidences, 0U)) << "two held, so it was never exclusive -- and nothing was reported and taken back";
    };

    "emit_partial reports a group that closed short of every condition"_test = [] {
        std::vector<gr::property_map> events{eventNamed("a", 1'000'000'000U), eventNamed("a", 1'000'500'000U)}; // b never arrives

        const auto dropped = combine(events, {{"filters", conditionsAB()}, {"logic", std::string("all")}, {"window", 50e-9}});
        expect(eq(dropped.coincidences, 0U)) << "by default an incomplete group is simply not reported";

        const auto partial = combine(std::move(events), {{"filters", conditionsAB()}, {"logic", std::string("all")}, {"window", 50e-9}, {"on_incomplete", std::string("emit_partial")}});
        expect(eq(partial.coincidences, 1U)) << "asked for it, the group is reported although b never came";
        expect(eq(partial.incomplete, 1U)) << "and counted as incomplete";
    };

    "what the window does, drawn"_test = [] {
        // three groups: the first two inside a 50 ns window, the third too far apart to count
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
        // one condition arrives and nothing closes its window: a block whose only input is asynchronous gets no
        // further work call once a stop has been requested, so this group can never be reported -- but it is visible
        const auto found = combine({eventNamed("a", 1'000'000'000U)}, //
            {{"filters", conditionsAB()}, {"logic", std::string("all")}, {"window", 50e-9}, {"on_incomplete", std::string("emit_partial")}, {"flush_after", 0.}});

        expect(eq(found.coincidences, 0U)) << "nothing closed the window, so nothing was judged";
        expect(eq(found.pending, 1U)) << "and what was left open says so, rather than disappearing";
    };

    "an idle flush judges a group that no further event will close"_test = [] {
        // `flush_after` is measured on the block's own clock; a run this short never reaches it, which is what the
        // assertion pins -- the flush must not fire early and report a group that was still growing
        const auto found = combine({eventNamed("a", 1'000'000'000U), eventNamed("b", 1'000'000'010U)}, //
            {{"filters", conditionsAB()}, {"logic", std::string("exclusive")}, {"window", 50e-9}, {"flush_after", 3600.}});

        expect(eq(found.coincidences, 0U)) << "an hour's idle time is not reached in a test, so nothing is flushed";
        expect(eq(found.pending, 2U)) << "both contributions are still held";
    };
};

int main() { /* tests are statically executed */ }
