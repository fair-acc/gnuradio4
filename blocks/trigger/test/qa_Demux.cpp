#include <boost/ut.hpp>

#include <string>
#include <vector>

#include <gnuradio-4.0/test/EventMarbles.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/trigger/Demux.hpp>
#include <gnuradio-4.0/trigger/Marble.hpp>

#include "TriggerTest.hpp"

using namespace gr::blocks::trigger;
using gr::trigger_test::EventScript;

namespace {
[[nodiscard]] const gr::property_map& kValues() {
    static const gr::property_map map{{"a", 1.f}, {"b", 2.f}};
    return map;
}
/// R names the ramp, F the flat top, Z a state no output claims
[[nodiscard]] const gr::property_map& kTags() {
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

[[nodiscard]] Outcome route(std::string script) {
    gr::testing::GraphFixture fixture;
    auto&                     source  = fixture.emplace<MarbleSource<float>>({{"script", std::move(script)}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
    auto&                     split   = fixture.emplace<Demux<float>>({{"contexts", std::vector<std::string>{"RAMP", "FLATTOP"}}});
    auto&                     ramp    = fixture.emplace<MarbleSink<float>>({{"sample_values", kValues()}, {"sample_tags", kTags()}});
    auto&                     flatTop = fixture.emplace<MarbleSink<float>>({{"sample_values", kValues()}, {"sample_tags", kTags()}});

    boost::ut::expect(fixture.connect<"out", "in">(source, split).has_value());
    boost::ut::expect(fixture.graph.connect(split, gr::PortDefinition{std::string("out#0")}, ramp, gr::PortDefinition{std::string("in")}).has_value());
    boost::ut::expect(fixture.graph.connect(split, gr::PortDefinition{std::string("out#1")}, flatTop, gr::PortDefinition{std::string("in")}).has_value());
    boost::ut::expect(fixture.run().has_value());

    return Outcome{.ramp = ramp._samples.size(), .flatTop = flatTop._samples.size(), .unmatched = split.n_unmatched.value, .switches = split.n_switches.value, .rampTags = ramp._tags.size()};
}
} // namespace

const boost::ut::suite<"Demux"> _demux = [] {
    using namespace boost::ut;

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
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string("a a a a a a a a |")}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
        auto&                     script = fixture.emplace<EventScript>();
        script._events                   = {gr::property_map{{std::string(gr::tag::CONTEXT.key()), std::string("FLATTOP")}}};
        script._stopWhenDone             = false;
        auto& split                      = fixture.emplace<Demux<float>>({{"contexts", std::vector<std::string>{"RAMP", "FLATTOP"}}});
        auto& ramp                       = fixture.emplace<MarbleSink<float>>({{"sample_values", kValues()}, {"sample_tags", kTags()}});
        auto& flatTop                    = fixture.emplace<MarbleSink<float>>({{"sample_values", kValues()}, {"sample_tags", kTags()}});

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
        auto&                     source  = fixture.emplace<MarbleSource<float>>({{"script", std::string("a a R:a a a F:b b Z:b b F:b |")}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
        auto&                     split   = fixture.emplace<Demux<float>>({{"contexts", std::vector<std::string>{"RAMP", "FLATTOP"}}});
        auto&                     ramp    = fixture.emplace<MarbleSink<float>>({{"sample_values", kValues()}, {"sample_tags", kTags()}});
        auto&                     flatTop = fixture.emplace<MarbleSink<float>>({{"sample_values", kValues()}, {"sample_tags", kTags()}});
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

int main() { /* tests are statically executed */ }
