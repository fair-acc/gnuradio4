#include <boost/ut.hpp>

#include <string>
#include <vector>

#include <gnuradio-4.0/test/GraphFixture.hpp>

#include "Marble.hpp"
#include <gnuradio-4.0/trigger/Gate.hpp>
#include <gnuradio-4.0/trigger/TagToEvent.hpp>

#include "TriggerTest.hpp"

using namespace gr::blocks::trigger;
using gr::trigger_test::completed;
using gr::trigger_test::EventTap;

namespace {
constexpr std::uint64_t kAnchorTime = 1'000'000'000U;
constexpr float         kRate       = 1000.f;

/** emits its values on the control port in the first work call, as an operator's open/close line would */
struct ControlScript : gr::Block<ControlScript> {
    gr::PortOut<std::uint8_t, gr::Async> out;

    GR_MAKE_REFLECTABLE(ControlScript, out);

    std::vector<std::uint8_t> _values;
    bool                      _sent = false;

    gr::work::Status processBulk(gr::OutputSpanLike auto& outSpan) {
        const std::size_t n = _sent ? 0UZ : std::min(_values.size(), outSpan.size());
        for (std::size_t i = 0UZ; i < n; ++i) {
            outSpan[i] = _values[i];
        }
        outSpan.publish(n);
        _sent = true;
        return gr::work::Status::OK;
    }
};

/** the normative baselines from the functional specification, as executable marbles */
struct Baseline {
    std::string_view name;
    std::string_view script;
    std::string_view emitted;
    gr::property_map settings;
};
} // namespace

const boost::ut::suite<"Gate"> _gate = [] {
    using namespace boost::ut;

    const gr::property_map values{{"a", 1.0f}, {"b", 2.0f}, {"c", 3.0f}, {"d", 4.0f}, {"e", 5.0f}, {"f", 6.0f}, {"g", 7.0f}, {"h", 8.0f}};
    const gr::property_map tags{{"T", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("start")}, {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1U}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}}};

    auto play = [&](const Baseline& baseline) {
        gr::testing::GraphFixture fixture;
        gr::property_map          settings   = baseline.settings;
        settings[std::string("open_filter")] = std::string("start");
        auto& source                         = fixture.emplace<MarbleSource<float>>({{"script", std::string(baseline.script)}, {"sample_values", values}, {"sample_tags", tags}});
        auto& gate                           = fixture.emplace<Gate<float>>(std::move(settings));
        auto& sink                           = fixture.emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        expect(fixture.connect<"out", "in">(source, gate).has_value());
        expect(fixture.connect<"out", "in">(gate, sink).has_value());
        expect(fixture.run().has_value()) << baseline.name;
        expect(eq(sink.script(), completed(baseline.emitted))) << baseline.name;
        gr::trigger_test::drawSubset(std::format("Gate: {}", baseline.name), std::format("Gate({})", baseline.name), baseline.script, sink.script());
    };

    "a gate reports each state change on evtOut"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string("a b T:c d T:e f |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&                     gate   = fixture.emplace<Gate<float>>({{"mode", std::string("toggle")}, {"open_filter", std::string("start")}});
        auto&                     sink   = fixture.emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        auto&                     states = fixture.emplace<EventTap>();
        expect(fixture.connect<"out", "in">(source, gate).has_value());
        expect(fixture.connect<"out", "in">(gate, sink).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(gate, states).has_value());
        expect(fixture.run().has_value());

        const std::vector<std::string> names = states.names();
        expect(ge(names.size(), 1UZ)) << "a gate that opens must say so";
        if (!names.empty()) {
            expect(eq(names.front(), std::string("opened")));
        }
    };

    "once forwards n_open samples from the trigger and ignores a retrigger"_test = [&] { //
        play({"once", "a b T:c d e f T:g h |", "T:c d e", {{"mode", std::string("once")}, {"n_open", 3U}}});
    };

    "toggle includes the opening sample and excludes the closing one"_test = [&] { //
        play({"toggle", "a b T:c d T:e f T:g |", "T:c d T:g", {{"mode", std::string("toggle")}}});
    };

    "cooldown emits the trigger sample then alternates suppressed and forwarded runs"_test = [&] { //
        play({"cooldown", "a T:b c T:d e f T:g |", "T:b f T:g", {{"mode", std::string("cooldown")}, {"n_open", 2U}, {"n_cooldown", 3U}}});
    };

    "wait resumes on the n_delay-th sample after the trigger"_test = [&] { //
        play({"wait", "a T:b c d e f |", "d e f", {{"mode", std::string("wait")}, {"n_delay", 2U}}});
    };

    "takeUntil forwards up to but not including the trigger"_test = [&] { //
        play({"takeUntil", "a b c T:d e f |", "a b c", {{"mode", std::string("takeUntil")}}});
    };

    "skipUntil forwards from the trigger sample onwards"_test = [&] { //
        play({"skipUntil", "a b c T:d e f |", "T:d e f", {{"mode", std::string("skipUntil")}}});
    };

    "a tag matching both filters closes the gate"_test = [&] { //
        play({"close wins", "a b T:c d e |", "a b", {{"mode", std::string("once")}, {"close_filter", std::string("start")}, {"initial_state", true}, {"n_open", 1U}}});
    };

    "wait ignores a trigger during the delay it is already serving"_test = [&] { //
        play({"wait/ignore", "a T:b T:c d e f |", "d e f", {{"mode", std::string("wait")}, {"n_delay", 2U}}});
    };

    "restart makes that trigger begin the delay again"_test = [&] { //
        play({"wait/restart", "a T:b T:c d e f |", "e f", {{"mode", std::string("wait")}, {"n_delay", 2U}, {"retrigger", std::string("restart")}}});
    };

    "hold is refused when nothing can reopen the gate"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string("a b T:c d e |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&                     gate   = fixture.emplace<Gate<float>>({{"mode", std::string("once")}, {"open_filter", std::string("start")}, {"n_open", 1U}, {"closed_policy", std::string("hold")}});
        auto&                     sink   = fixture.emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        expect(fixture.connect<"out", "in">(source, gate).has_value());
        expect(fixture.connect<"out", "in">(gate, sink).has_value());
        gr::trigger_test::ErrorMessages sent{fixture.scheduler()};
        expect(fixture.run().has_value());

        expect(eq(sent.countContaining("closed_policy"), 1UZ)) << "the refusal is reported as an error message";
        expect(eq(std::string_view{gate.closed_policy.value}, std::string_view("drop"))) << "neither evtIn nor control is connected, so holding could never end";
        expect(eq(sink.script(), completed("T:c"))) << "and the gate drops as it would have without the setting";
    };

    "a held gate forwards the samples a dropping gate discards"_test = [&] {
        auto run = [&](std::string_view policy) {
            gr::testing::GraphFixture fixture;
            auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string("a b T:c d e |")}, {"sample_values", values}, {"sample_tags", tags}});
            auto&                     bridge = fixture.emplace<TagToEvent<float>>({{"filter", std::string("start")}});
            auto&                     gate   = fixture.emplace<Gate<float>>({{"mode", std::string("once")}, {"open_filter", std::string("start")}, {"n_open", 2U}, {"closed_policy", std::string(policy)}});
            auto&                     sink   = fixture.emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
            expect(fixture.connect<"out", "in">(source, bridge).has_value());
            expect(fixture.connect<"out", "in">(source, gate).has_value());
            expect(fixture.connect<"evtOut", "evtIn">(bridge, gate).has_value());
            expect(fixture.connect<"out", "in">(gate, sink).has_value());
            expect(fixture.run().has_value()) << policy;
            return sink.script();
        };

        expect(eq(run("hold"), completed("a b"))) << "holding keeps the samples that arrived before the gate opened";
        expect(eq(run("drop"), completed("T:c d"))) << "dropping consumes them, so the run starts at the trigger instead";
    };

    "a mode with no open run refuses to restart one"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     gate = fixture.emplace<Gate<float>>({{"mode", std::string("toggle")}, {"open_filter", std::string("start")}, {"retrigger", std::string("restart")}});
        expect(eq(std::string_view{gate.retrigger.value}, std::string_view("ignore")));
    };

    "only toggle acts on both edges, so interval matching is refused elsewhere"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     gate = fixture.emplace<Gate<float>>({{"mode", std::string("once")}, {"open_filter", std::string("start")}, {"match_mode", std::string("interval")}});
        expect(eq(std::string_view{gate.match_mode.value}, std::string_view("")));
    };

    "a trigger that cannot be placed in time still acts, and is counted"_test = [&] {
        const gr::property_map untimedTags{{"T", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("start")}}}};

        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string("a b c T:d e f |")}, {"sample_values", values}, {"sample_tags", untimedTags}});
        auto&                     gate   = fixture.emplace<Gate<float>>({{"mode", std::string("skipUntil")}, {"open_filter", std::string("start")}});
        auto&                     sink   = fixture.emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", untimedTags}});
        expect(fixture.connect<"out", "in">(source, gate).has_value());
        expect(fixture.connect<"out", "in">(gate, sink).has_value());
        gr::trigger_test::ErrorMessages sent{fixture.scheduler()};
        expect(fixture.run().has_value());

        expect(eq(sink.script(), completed("T:d e f"))) << "the transition the trigger asked for still happens";
        expect(eq(gate.n_invalid_triggers.value, 1U)) << "and the trigger is counted as one that could not be placed in time";
        expect(eq(sent.countContaining("lacking"), 1UZ)) << "and reported as an error message";
    };

    "a discarded trigger and the tags lost with a suppressed sample are counted"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string("a b T:c d e f T:g h |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&                     gate   = fixture.emplace<Gate<float>>({{"mode", std::string("once")}, {"open_filter", std::string("start")}, {"n_open", 3U}});
        auto&                     sink   = fixture.emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        expect(fixture.connect<"out", "in">(source, gate).has_value());
        expect(fixture.connect<"out", "in">(gate, sink).has_value());
        expect(fixture.run().has_value());

        expect(eq(sink.script(), completed("T:c d e"))) << "the baseline is unchanged";
        expect(eq(gate.n_triggers_ignored.value, 1U)) << "the second trigger was discarded by retrigger = ignore";
        expect(eq(gate.n_tags_dropped.value, 1U)) << "and its tag went with the sample the gate suppressed";
    };

    "the same trigger opens the gate whether it arrives as a tag or on evtIn"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string("a b c d e f |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&                     ticks  = fixture.emplace<MarbleSource<float>>({{"script", std::string("T:a |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&                     bridge = fixture.emplace<TagToEvent<float>>({{"filter", std::string("start")}});
        auto&                     gate   = fixture.emplace<Gate<float>>({{"mode", std::string("skipUntil")}, {"open_filter", std::string("start")}});
        auto&                     sink   = fixture.emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        auto&                     spent  = fixture.emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        expect(fixture.connect<"out", "in">(ticks, bridge).has_value());
        expect(fixture.connect<"out", "in">(bridge, spent).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(bridge, gate).has_value());
        expect(fixture.connect<"out", "in">(source, gate).has_value());
        expect(fixture.connect<"out", "in">(gate, sink).has_value());
        expect(fixture.run().has_value());

        expect(eq(sink._samples.size(), 6UZ)) << "the injected event opened a gate no tag on this stream ever reaches";
        expect(eq(gate.n_suppressed.value, 0U)) << "and it did so on the first sample of the call it arrived in";
    };

    "a count of no samples is refused, since zero already means unbounded"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     gate = fixture.emplace<Gate<float>>({{"mode", std::string("once")}, {"open_filter", std::string("start")}, {"n_open", 0U}, {"n_delay", 0U}});
        expect(eq(gate.n_open.value, 1U));
        expect(eq(gate.n_delay.value, 1U));
    };

    /** sample i of a stream anchored by tag A on its first sample is dated kAnchorTime + i ms */
    const gr::property_map anchorTags{{"A", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("anchor")}, {std::string(gr::tag::TRIGGER_TIME.key()), kAnchorTime}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}, {std::string("sample_rate"), kRate}}}};

    auto playEvent = [&](gr::property_map event, std::string_view emitted, gr::Size_t* nInvalid = nullptr) {
        gr::testing::GraphFixture fixture;
        auto&                     script = fixture.emplace<gr::trigger_test::EventScript>();
        script._events                   = {std::move(event)};
        script._stopWhenDone             = false;
        auto& source                     = fixture.emplace<MarbleSource<float>>({
            {"script", std::string("A:a b c d e f g h |")},
            {"sample_values", values},
            {"sample_tags", anchorTags},
        });
        auto& gate                       = fixture.emplace<Gate<float>>({{"mode", std::string("skipUntil")}, {"open_filter", std::string("start")}});
        auto& sink                       = fixture.emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", anchorTags}});
        expect(fixture.graph.connect(script, gr::PortDefinition{std::string("evtOut")}, gate, gr::PortDefinition{std::string("evtIn")}).has_value());
        expect(fixture.connect<"out", "in">(source, gate).has_value());
        expect(fixture.connect<"out", "in">(gate, sink).has_value());
        gr::trigger_test::ErrorMessages sent{fixture.scheduler()};
        expect(fixture.run().has_value());
        expect(eq(sink.script(), completed(emitted)));
        if (nInvalid != nullptr) {
            *nInvalid = gate.n_invalid_triggers.value;
        }
    };

    "a dated event acts at the sample its trigger_time falls on"_test = [&] { //
        playEvent(gr::trigger_test::eventNamed("start", kAnchorTime + 3'000'000U), "d e f g h");
    };

    "an undated event acts at the start of the call and is counted"_test = [&] {
        gr::Size_t nInvalid = 0U;
        playEvent(gr::trigger_test::undatedEvent("start"), "A:a b c d e f g h", &nInvalid);
        expect(eq(nInvalid, 1U));
    };

    "the control port opens and closes the gate"_test = [&] {
        auto run = [&](std::vector<std::uint8_t> wanted, bool initiallyOpen) {
            gr::testing::GraphFixture fixture;
            auto&                     control = fixture.emplace<ControlScript>();
            control._values                   = std::move(wanted);
            auto& source                      = fixture.emplace<MarbleSource<float>>({{"script", std::string("a b c d e f |")}, {"sample_values", values}, {"sample_tags", tags}});
            auto& gate                        = fixture.emplace<Gate<float>>({
                {"mode", std::string("skipUntil")},
                {"open_filter", std::string("start")},
                {"initial_state", initiallyOpen},
            });
            auto& sink                        = fixture.emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
            expect(fixture.graph.connect(control, gr::PortDefinition{std::string("out")}, gate, gr::PortDefinition{std::string("control")}).has_value());
            expect(fixture.connect<"out", "in">(source, gate).has_value());
            expect(fixture.connect<"out", "in">(gate, sink).has_value());
            expect(fixture.run().has_value());
            return sink._samples.size();
        };
        expect(eq(run({1U}, false), 6UZ)) << "a nonzero control value opens a closed gate";
        expect(eq(run({0U}, true), 0UZ)) << "zero closes an open one";
    };

    "cooldown with nothing to suppress keeps forwarding"_test = [&] { //
        play({"cooldown/none", "a T:b c d e |", "T:b c d e", {{"mode", std::string("cooldown")}, {"n_open", 1U}, {"n_cooldown", 0U}}});
    };
};

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
