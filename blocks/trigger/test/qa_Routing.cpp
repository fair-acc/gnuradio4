#include <boost/ut.hpp>

#include <algorithm>
#include <string>
#include <vector>

#include <gnuradio-4.0/basic/Mux.hpp>
#include <gnuradio-4.0/test/EventMarbles.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/trigger/PredicateSwitch.hpp>

#include "Marble.hpp"
#include "TriggerTest.hpp"

namespace qa_demux {
using namespace gr::blocks::trigger;
using gr::basic::Demux;
using gr::basic::Routing;
using gr::trigger_test::EventScript;

[[nodiscard]] const gr::property_map& values() {
    static const gr::property_map map{{"a", 1.f}, {"b", 2.f}};
    return map;
}
/** R names the ramp, F the flat top, Z a state no output claims */
[[nodiscard]] const gr::property_map& tags() {
    static const gr::property_map map{{"R", gr::property_map{{std::string(gr::tag::CONTEXT.key()), std::string("RAMP")}}}, //
        {"F", gr::property_map{{std::string(gr::tag::CONTEXT.key()), std::string("FLATTOP")}}},                            //
        {"Z", gr::property_map{{std::string(gr::tag::CONTEXT.key()), std::string("UNKNOWN")}}}};
    return map;
}

struct Outcome {
    std::size_t ramp      = 0UZ;
    std::size_t flatTop   = 0UZ;
    gr::Size_t  unmatched = 0U;
    gr::Size_t  switches  = 0U;
    std::size_t rampTags  = 0UZ;
};

[[nodiscard]] Outcome route(std::string script, const gr::property_map& sampleTags = tags(), std::string selectKey = std::string(gr::tag::CONTEXT.key())) {
    gr::testing::GraphFixture fixture;
    auto&                     source  = fixture.emplace<MarbleSource<float>>({{"script", std::move(script)}, {"sample_values", values()}, {"sample_tags", sampleTags}});
    auto&                     split   = fixture.emplace<Demux<float, 0UZ, Routing::context>>({{"contexts", std::vector<std::string>{"RAMP", "FLATTOP"}}, {"select_key", std::move(selectKey)}});
    auto&                     ramp    = fixture.emplace<MarbleSink<float>>({{"sample_values", values()}, {"sample_tags", sampleTags}});
    auto&                     flatTop = fixture.emplace<MarbleSink<float>>({{"sample_values", values()}, {"sample_tags", sampleTags}});

    boost::ut::expect(fixture.connect<"out", "in">(source, split).has_value());
    boost::ut::expect(fixture.graph.connect(split, gr::PortDefinition{std::string("out#0")}, ramp, gr::PortDefinition{std::string("in")}).has_value());
    boost::ut::expect(fixture.graph.connect(split, gr::PortDefinition{std::string("out#1")}, flatTop, gr::PortDefinition{std::string("in")}).has_value());
    boost::ut::expect(fixture.run().has_value());

    return Outcome{.ramp = ramp._samples.size(), .flatTop = flatTop._samples.size(), .unmatched = split.n_unmatched.value, .switches = split.n_switches.value, .rampTags = ramp._tags.size()};
}

const boost::ut::suite<"Demux"> _demux = [] {
    using namespace boost::ut;

    "any tag key selects the output when select_key names it"_test = [] {
        const gr::property_map modeTags{{"R", gr::property_map{{std::string("mode"), std::string("RAMP")}}}, //
            {"F", gr::property_map{{std::string("mode"), std::string("FLATTOP")}}}};
        const auto             found = route("R:a a a F:b b |", modeTags, "mode");

        expect(eq(found.ramp, 3UZ));
        expect(eq(found.flatTop, 2UZ));
        expect(eq(found.switches, 2U));

        const auto ignored = route("R:a a a F:b b |", modeTags);
        expect(eq(ignored.unmatched, 5U)) << "with the default key, a 'mode' tag selects nothing";
    };

    "each context's samples go to its own output"_test = [] {
        const auto found = route("R:a a a F:b b |");

        expect(eq(found.ramp, 3UZ)) << "the ramp's tag and the two samples after it";
        expect(eq(found.flatTop, 2UZ));
        expect(eq(found.unmatched, 0U));
        expect(eq(found.switches, 2U));
    };

    "the samples before the first context are counted, not guessed at"_test = [] {
        const auto found = route("a a R:a a |");

        expect(eq(found.unmatched, 2U)) << "a stream whose state is unknown is not a stream in the first state";
        expect(eq(found.ramp, 2UZ));
        expect(eq(found.flatTop, 0UZ));
    };

    "a context no output claims drops its samples and says how many"_test = [] {
        const auto found = route("R:a a Z:a a a F:b |");

        expect(eq(found.ramp, 2UZ));
        expect(eq(found.unmatched, 3U)) << "the unknown state's samples";
        expect(eq(found.flatTop, 1UZ));
    };

    "the switching tag travels with the samples it switched"_test = [] {
        const auto found = route("R:a a F:b b |");

        expect(eq(found.rampTags, 1UZ)) << "so a branch still knows which state it is in";
    };

    "switching back and forth keeps every sample in order"_test = [] {
        const auto found = route("R:a F:b R:a F:b R:a |");

        expect(eq(found.ramp, 3UZ));
        expect(eq(found.flatTop, 2UZ));
        expect(eq(found.switches, 5U));
        expect(eq(found.unmatched, 0U));
    };

    "an injected event switches the context of a stream that carries no tags"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string("a a a a a a a a |")}, {"sample_values", values()}, {"sample_tags", tags()}});
        auto&                     script = fixture.emplace<EventScript>();
        script._events                   = {gr::property_map{{std::string(gr::tag::CONTEXT.key()), std::string("FLATTOP")}}};
        script._stopWhenDone             = false;
        auto& split                      = fixture.emplace<Demux<float, 0UZ, Routing::context>>({{"contexts", std::vector<std::string>{"RAMP", "FLATTOP"}}});
        auto& ramp                       = fixture.emplace<MarbleSink<float>>({{"sample_values", values()}, {"sample_tags", tags()}});
        auto& flatTop                    = fixture.emplace<MarbleSink<float>>({{"sample_values", values()}, {"sample_tags", tags()}});

        expect(fixture.connect<"out", "in">(source, split).has_value());
        expect(fixture.graph.connect(script, gr::PortDefinition{std::string("evtOut")}, split, gr::PortDefinition{std::string("evtIn")}).has_value());
        expect(fixture.graph.connect(split, gr::PortDefinition{std::string("out#0")}, ramp, gr::PortDefinition{std::string("in")}).has_value());
        expect(fixture.graph.connect(split, gr::PortDefinition{std::string("out#1")}, flatTop, gr::PortDefinition{std::string("in")}).has_value());
        expect(fixture.run().has_value());

        expect(ge(flatTop._samples.size(), 1UZ)) << "the injected context routed the samples that followed it";
        expect(eq(ramp._samples.size(), 0UZ)) << "and nothing reached the output the stream never named";
    };

    "where each sample went, drawn"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source  = fixture.emplace<MarbleSource<float>>({{"script", std::string("a a R:a a a F:b b Z:b b F:b |")}, {"sample_values", values()}, {"sample_tags", tags()}});
        auto&                     split   = fixture.emplace<Demux<float, 0UZ, Routing::context>>({{"contexts", std::vector<std::string>{"RAMP", "FLATTOP"}}});
        auto&                     ramp    = fixture.emplace<MarbleSink<float>>({{"sample_values", values()}, {"sample_tags", tags()}});
        auto&                     flatTop = fixture.emplace<MarbleSink<float>>({{"sample_values", values()}, {"sample_tags", tags()}});
        expect(fixture.connect<"out", "in">(source, split).has_value());
        expect(fixture.graph.connect(split, gr::PortDefinition{std::string("out#0")}, ramp, gr::PortDefinition{std::string("in")}).has_value());
        expect(fixture.graph.connect(split, gr::PortDefinition{std::string("out#1")}, flatTop, gr::PortDefinition{std::string("in")}).has_value());
        expect(fixture.run().has_value());

        gr::testing::MarbleDiagram diagram{"Demux: one stream, three machine states, two branches"};
        diagram.unit = "sample";
        diagram.row("in").at(2U, "RAMP").at(5U, "FLATTOP").at(7U, "UNKNOWN").at(9U, "FLATTOP").completes();
        diagram.condition(std::format("Demux(contexts = RAMP, FLATTOP), {} unmatched", split.n_unmatched.value));
        diagram.row("out#0").at(2U, "RAMP").completes();
        diagram.row("out#1").at(5U, "FLATTOP").at(9U, "FLATTOP").completes();
        diagram.print();

        expect(eq(ramp._samples.size(), 3UZ));
        expect(eq(flatTop._samples.size(), 3UZ));
        expect(eq(split.n_unmatched.value, 4U)) << "two before the first context, two in the unknown one";
    };
};
} // namespace qa_demux

namespace qa_mux {
using namespace gr::blocks::trigger;
using gr::basic::Mux;
using gr::basic::Routing;
using gr::trigger_test::EventScript;

[[nodiscard]] const gr::property_map& low() {
    static const gr::property_map map{{"a", 1.f}, {"b", 2.f}, {"c", 3.f}, {"d", 4.f}};
    return map;
}

[[nodiscard]] const gr::property_map& high() {
    static const gr::property_map map{{"a", 11.f}, {"b", 12.f}, {"c", 13.f}, {"d", 14.f}};
    return map;
}

[[nodiscard]] gr::property_map contextNamed(std::string context) { //
    return gr::property_map{{std::string(gr::tag::CONTEXT.key()), std::move(context)}, {std::string(gr::tag::TRIGGER_NAME.key()), std::string("switch")}};
}

const boost::ut::suite<"Mux"> _mux = [] {
    using namespace boost::ut;

    "the selected input reaches the output and the other does not"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     ramp    = fixture.emplace<MarbleSource<float>>({{"script", std::string("a b c d |")}, {"sample_values", low()}});
        auto&                     flatTop = fixture.emplace<MarbleSource<float>>({{"script", std::string("a b c d |")}, {"sample_values", high()}});
        auto&                     control = fixture.emplace<EventScript>();
        auto&                     mux     = fixture.emplace<Mux<float, 0UZ, Routing::context>>({{"contexts", std::vector<std::string>{"RAMP", "FLATTOP"}}});
        auto&                     sink    = fixture.emplace<gr::trigger_test::CollectingSink<float>>();
        control._events.push_back(contextNamed("RAMP"));
        control._stopWhenDone = false;

        expect(fixture.graph.connect(control, gr::PortDefinition{std::string("evtOut")}, mux, gr::PortDefinition{std::string("evtIn")}).has_value());
        expect(fixture.graph.connect(ramp, gr::PortDefinition{std::string("out")}, mux, gr::PortDefinition{std::string("in#0")}).has_value());
        expect(fixture.graph.connect(flatTop, gr::PortDefinition{std::string("out")}, mux, gr::PortDefinition{std::string("in#1")}).has_value());
        expect(fixture.graph.connect(mux, gr::PortDefinition{std::string("out")}, sink, gr::PortDefinition{std::string("in")}).has_value());
        expect(fixture.run().has_value());

        expect(!sink._collected.empty()) << "the selected input was forwarded";
        expect(std::ranges::all_of(sink._collected, [](float sample) { return sample < 10.f; })) << "every sample came from the low source, which is the one RAMP names";
        expect(eq(mux.n_switches.value, gr::Size_t{1}));
        expect(gt(mux.n_dropped.value, gr::Size_t{0})) << "the un-selected input was read and dropped, as Demux does with an unclaimed context";
    };

    "a context naming no input is counted and changes nothing"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     ramp    = fixture.emplace<MarbleSource<float>>({{"script", std::string("a b |")}, {"sample_values", low()}});
        auto&                     control = fixture.emplace<EventScript>();
        auto&                     mux     = fixture.emplace<Mux<float, 0UZ, Routing::context>>({{"contexts", std::vector<std::string>{"RAMP"}}});
        auto&                     sink    = fixture.emplace<gr::trigger_test::CollectingSink<float>>();
        control._events.push_back(contextNamed("CALIBRATION"));
        control._stopWhenDone = false;

        expect(fixture.graph.connect(control, gr::PortDefinition{std::string("evtOut")}, mux, gr::PortDefinition{std::string("evtIn")}).has_value());
        expect(fixture.graph.connect(ramp, gr::PortDefinition{std::string("out")}, mux, gr::PortDefinition{std::string("in#0")}).has_value());
        expect(fixture.graph.connect(mux, gr::PortDefinition{std::string("out")}, sink, gr::PortDefinition{std::string("in")}).has_value());
        expect(fixture.run().has_value());

        expect(eq(mux.n_unmatched.value, gr::Size_t{1}));
        expect(eq(mux.n_switches.value, gr::Size_t{0}));
        expect(sink._collected.empty()) << "nothing is selected, so nothing is forwarded";
    };

    "with no context yet nothing is forwarded"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     ramp = fixture.emplace<MarbleSource<float>>({{"script", std::string("a b |")}, {"sample_values", low()}});
        auto&                     mux  = fixture.emplace<Mux<float, 0UZ, Routing::context>>({{"contexts", std::vector<std::string>{"RAMP"}}});
        auto&                     sink = fixture.emplace<gr::trigger_test::CollectingSink<float>>();

        expect(fixture.graph.connect(ramp, gr::PortDefinition{std::string("out")}, mux, gr::PortDefinition{std::string("in#0")}).has_value());
        expect(fixture.graph.connect(mux, gr::PortDefinition{std::string("out")}, sink, gr::PortDefinition{std::string("in")}).has_value());
        expect(fixture.run().has_value());

        expect(sink._collected.empty()) << "a stream whose state is unknown is not a stream in the first state";
    };

    "the input list follows the context list"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     mux = fixture.emplace<Mux<float, 0UZ, Routing::context>>({{"contexts", std::vector<std::string>{"RAMP", "FLATTOP", "CALIBRATION"}}});
        expect(eq(mux.in.size(), 3UZ));
    };
};
} // namespace qa_mux

namespace qa_predicate_switch {
using namespace gr::blocks::trigger;

[[nodiscard]] const gr::property_map& values() {
    static const gr::property_map map{{"a", 1.f}, {"b", 5.f}, {"c", -1.f}};
    return map;
}
/** L carries a level the predicates may read, H a higher one */
[[nodiscard]] const gr::property_map& tags() {
    static const gr::property_map map{{"L", gr::property_map{{"level", 0.5f}}}, {"H", gr::property_map{{"level", 2.f}}}};
    return map;
}

struct Routed {
    std::string first;
    std::string second;
    gr::Size_t  unmatched;
};

[[nodiscard]] Routed route(std::string script, std::vector<std::string> predicates, std::size_t chunk = 0UZ) {
    gr::testing::GraphFixture fixture;
    auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::move(script)}, {"sample_values", values()}, {"sample_tags", tags()}});
    auto&                     split  = fixture.emplace<PredicateSwitch<float>>({{"predicates", std::move(predicates)}});
    auto&                     first  = fixture.emplace<MarbleSink<float>>({{"sample_values", values()}, {"sample_tags", tags()}});
    auto&                     second = fixture.emplace<MarbleSink<float>>({{"sample_values", values()}, {"sample_tags", tags()}});
    if (chunk > 0UZ) {
        split.in.max_samples = chunk;
    }
    boost::ut::expect(fixture.connect<"out", "in">(source, split).has_value());
    boost::ut::expect(fixture.graph.connect(split, gr::PortDefinition{std::string("out#0")}, first, gr::PortDefinition{std::string("in")}).has_value());
    boost::ut::expect(fixture.graph.connect(split, gr::PortDefinition{std::string("out#1")}, second, gr::PortDefinition{std::string("in")}).has_value());
    boost::ut::expect(fixture.run().has_value());
    return {first.script(), second.script(), split.n_unmatched.value};
}

const boost::ut::suite<"PredicateSwitch"> _predicateSwitch = [] {
    using namespace boost::ut;

    "each sample goes to the first predicate that holds for it"_test = [] {
        const auto found = route("a b c b a |", {"x > 4", "x > 0"});
        expect(eq(found.first, std::string("b b |")));
        expect(eq(found.second, std::string("a a |")));
        expect(eq(found.unmatched, 1U)) << "the negative sample matches neither";
    };

    "a predicate reads a field of the tags seen so far"_test = [] {
        const auto found = route("L:a a H:a a |", {"x > level", "x <= level"});
        expect(eq(found.first, std::string("L:a a |"))) << "the tag travels with the sample it marks";
        expect(eq(found.second, std::string("H:a a |")));
        expect(eq(found.unmatched, 0U));
    };

    "a field no tag has carried yet matches nothing"_test = [] {
        const auto found = route("a a L:a |", {"x > level", "x > 10"});
        expect(eq(found.first, std::string("L:a |")));
        expect(eq(found.second, std::string("|")));
        expect(eq(found.unmatched, 2U));
    };

    "a predicate that does not compile never matches, and the others still route"_test = [] {
        const auto found = route("a b |", {"x >", "x > 0"});
        expect(eq(found.first, std::string("|")));
        expect(eq(found.second, std::string("a b |")));
    };

    "the routing does not depend on where the stream is cut"_test = [] {
        const auto whole = route("L:a b c H:a b a |", {"x > level", "x > 0"});
        for (const std::size_t chunk : {1UZ, 2UZ, 3UZ}) {
            const auto cut = route("L:a b c H:a b a |", {"x > level", "x > 0"}, chunk);
            expect(eq(cut.first, whole.first)) << chunk;
            expect(eq(cut.second, whole.second)) << chunk;
            expect(eq(cut.unmatched, whole.unmatched)) << chunk;
        }
    };
};
} // namespace qa_predicate_switch

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
