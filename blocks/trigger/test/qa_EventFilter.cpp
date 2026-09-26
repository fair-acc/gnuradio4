#include <boost/ut.hpp>

#include <string>
#include <vector>

#include <gnuradio-4.0/test/EventMarbles.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/trigger/EventFilter.hpp>

#include "TriggerTest.hpp"

using namespace gr::blocks::trigger;
using gr::trigger_test::eventNamed;
using gr::trigger_test::EventScript;
using gr::trigger_test::EventTap;

namespace {
struct Outcome {
    std::vector<std::string> names;
    gr::Size_t               passed  = 0U;
    gr::Size_t               blocked = 0U;
    gr::Size_t               vetoed  = 0U;
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

    return Outcome{.names = tap.names(), .passed = sieve.n_passed.value, .blocked = sieve.n_blocked.value, .vetoed = sieve.n_vetoed.value};
}
} // namespace

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

    "a veto wins over the filter that accepted the event"_test = [] {
        const auto found = sift({eventNamed("edge", 100U), eventNamed("edge", 200U)}, //
            {{"filter", std::string("edge")}, {"veto_filter", std::string("edge")}});

        expect(eq(found.names.size(), 0UZ)) << "an event both accept is dropped";
        expect(eq(found.vetoed, 2U));
        expect(eq(found.passed, 0U));
    };

    "what passes can be renamed, so a downstream filter is written once"_test = [] {
        const auto found = sift({eventNamed("edge", 100U)}, {{"filter", std::string("edge")}, {"trigger_name", std::string("clean_edge")}});

        expect(eq(found.names.size(), 1UZ));
        if (!found.names.empty()) {
            expect(eq(found.names[0], std::string("clean_edge")));
        }
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
        auto& sieve                      = fixture.emplace<EventFilter>({{"filter", std::string("edge")}, {"trigger_name", std::string("clean_edge")}});
        auto& tap                        = fixture.emplace<EventTap>();
        expect(fixture.connect<"evtOut", "evtIn">(script, sieve).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(sieve, tap).has_value());
        expect(fixture.run().has_value());

        gr::testing::MarbleDiagram diagram{"EventFilter: edge accepted, noise blocked, what passes renamed"};
        auto&                      in = diagram.row("evtIn");
        for (const auto& [at, name] : std::vector<std::pair<std::uint64_t, std::string>>{{100U, "edge"}, {200U, "noise"}, {300U, "edge"}, {400U, "edge"}}) {
            in.at(at, name);
        }
        in.completes();
        diagram.condition("EventFilter(filter = \"edge\", trigger_name = \"clean_edge\")");
        auto& out = diagram.row("evtOut");
        for (const auto& [at, name] : tap.dated()) {
            out.at(at, name);
        }
        out.completes();
        diagram.print();

        expect(eq(tap.countOf("clean_edge"), 3UZ));
    };
};

int main() { /* tests are statically executed */ }
