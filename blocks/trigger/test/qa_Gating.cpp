#include <boost/ut.hpp>

#include <complex>
#include <optional>
#include <string>
#include <vector>

#include <gnuradio-4.0/test/EventMarbles.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/trigger/Gate.hpp>
#include <gnuradio-4.0/trigger/Predicate.hpp>
#include <gnuradio-4.0/trigger/SampleAndHold.hpp>
#include <gnuradio-4.0/trigger/StreamOps.hpp>
#include <gnuradio-4.0/trigger/TagToEvent.hpp>
#include <gnuradio-4.0/trigger/TakeSkip.hpp>
#include <gnuradio-4.0/trigger/TriggerWatchdog.hpp>

#include "Marble.hpp"
#include "TriggerTest.hpp"

namespace qa_gate {
using gr::trigger_test::kSampleRate;
using gr::trigger_test::kStreamStart;
using namespace gr::blocks::trigger;
using gr::trigger_test::completed;
using gr::trigger_test::EventTap;

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

const boost::ut::suite<"Gate"> _gate = [] {
    using namespace boost::ut;

    const gr::property_map values{{"a", 1.0f}, {"b", 2.0f}, {"c", 3.0f}, {"d", 4.0f}, {"e", 5.0f}, {"f", 6.0f}, {"g", 7.0f}, {"h", 8.0f}};
    const gr::property_map tags{{"T", gr::property_map{{std::string(gr::tag::kTriggerName.key()), std::string("start")}, {std::string(gr::tag::kTriggerTime.key()), std::uint64_t{1U}}, {std::string(gr::tag::kTriggerOffset.key()), 0.f}}}};

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
        const gr::property_map untimedTags{{"T", gr::property_map{{std::string(gr::tag::kTriggerName.key()), std::string("start")}}}};

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

    /** sample i of a stream anchored by tag A on its first sample is dated kStreamStart + i ms */
    const gr::property_map anchorTags{{"A", gr::property_map{{std::string(gr::tag::kTriggerName.key()), std::string("anchor")}, {std::string(gr::tag::kTriggerTime.key()), kStreamStart}, {std::string(gr::tag::kTriggerOffset.key()), 0.f}, {std::string("sample_rate"), kSampleRate}}}};

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
        playEvent(gr::trigger_test::eventNamed("start", kStreamStart + 3'000'000U), "d e f g h");
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
} // namespace qa_gate

namespace qa_take_skip {
using namespace gr::blocks::trigger;
using gr::trigger_test::completed;

const boost::ut::suite<"TakeN and SkipN"> _takeSkip = [] {
    using namespace boost::ut;

    const gr::property_map values{{"a", 1.0f}, {"b", 2.0f}, {"c", 3.0f}, {"d", 4.0f}, {"e", 5.0f}, {"f", 6.0f}};
    const gr::property_map tags{{"T", gr::property_map{{std::string(gr::tag::kTriggerName.key()), std::string("start")}, {std::string(gr::tag::kTriggerTime.key()), std::uint64_t{1U}}, {std::string(gr::tag::kTriggerOffset.key()), 0.f}}}};

    auto play = [&]<typename TBlock>(std::string_view script, std::string_view emitted, gr::property_map settings) {
        gr::testing::GraphFixture fixture;
        settings[std::string("filter")] = std::string("start");
        auto& source                    = fixture.emplace<MarbleSource<float>>({{"script", std::string(script)}, {"sample_values", values}, {"sample_tags", tags}});
        auto& block                     = fixture.emplace<TBlock>(std::move(settings));
        auto& sink                      = fixture.emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        expect(fixture.connect<"out", "in">(source, block).has_value());
        expect(fixture.connect<"out", "in">(block, sink).has_value());
        expect(fixture.run().has_value());
        expect(eq(sink.script(), completed(emitted)));
        gr::trigger_test::drawSubset("TakeN/SkipN: what the settings let through", std::format("the run the trigger asked for"), script, sink.script());
    };

    "TakeN forwards n samples from the matching tag"_test = [&] { //
        play.template operator()<TakeN<float>>("a b T:c d e f |", "T:c d", {{"n", 2U}});
    };

    "SkipN suppresses n samples from the matching tag"_test = [&] { //
        play.template operator()<SkipN<float>>("a b T:c d e f |", "a b e f", {{"n", 2U}});
    };

    "a retrigger restarts the run by default"_test = [&] { //
        play.template operator()<TakeN<float>>("a T:b c T:d e f |", "T:b c T:d e", {{"n", 2U}});
    };

    "extend adds to the run instead of restarting it"_test = [&] { //
        play.template operator()<TakeN<float>>("a T:b T:c d e f |", "T:b T:c d e", {{"n", 2U}, {"retrigger", std::string("extend")}});
    };

    "the counters are lifetime statistics: only the transition to INITIALISED zeroes them"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string("a b T:c d e f |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&                     block  = fixture.emplace<TakeN<float>>({{"filter", std::string("start")}, {"n", 2U}});
        auto&                     sink   = fixture.emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        expect(fixture.connect<"out", "in">(source, block).has_value());
        expect(fixture.connect<"out", "in">(block, sink).has_value());
        expect(fixture.run().has_value());
        expect(gt(block.n_passed.value, 0U)) << "the run forwarded samples";
        expect(gt(block.n_suppressed.value, 0U)) << "and suppressed the rest";
        expect(block.changeStateTo(gr::lifecycle::State::INITIALISED).has_value());
        expect(eq(block.n_passed.value, 0U)) << "reset() zeroes the lifetime counters";
        expect(eq(block.n_suppressed.value, 0U));
    };

    "hold is refused where no asynchronous input could release a held sample"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     takeN = fixture.emplace<TakeN<float>>({{"filter", std::string("start")}, {"n", 2U}, {"policy", std::string("hold")}});
        /** not `skip`, which is a boost::ut name */
        auto& skipN = fixture.emplace<SkipN<float>>({{"filter", std::string("start")}, {"n", 2U}, {"policy", std::string("hold")}});
        expect(eq(std::string_view{takeN.policy.value}, std::string_view("drop")));
        expect(eq(std::string_view{skipN.policy.value}, std::string_view("drop")));
    };
};
} // namespace qa_take_skip

namespace qa_sample_and_hold {
using namespace gr::blocks::trigger;
using gr::trigger_test::completed;

const boost::ut::suite<"SampleAndHold"> _sampleAndHold = [] {
    using namespace boost::ut;

    const gr::property_map values{{"a", 1.0f}, {"b", 2.0f}, {"c", 3.0f}, {"d", 4.0f}, {"e", 5.0f}};
    const gr::property_map tags{{"T", gr::property_map{{std::string(gr::tag::kTriggerName.key()), std::string("start")}, {std::string(gr::tag::kTriggerTime.key()), std::uint64_t{1U}}, {std::string(gr::tag::kTriggerOffset.key()), 0.f}}}};

    auto play = [&](std::string_view script, std::string_view emitted, gr::property_map settings) {
        gr::testing::GraphFixture fixture;
        settings[std::string("filter")] = std::string("start");
        auto& source                    = fixture.emplace<MarbleSource<float>>({{"script", std::string(script)}, {"sample_values", values}, {"sample_tags", tags}});
        auto& block                     = fixture.emplace<SampleAndHold<float>>(std::move(settings));
        auto& sink                      = fixture.emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        expect(fixture.connect<"out", "in">(source, block).has_value());
        expect(fixture.connect<"out", "in">(block, sink).has_value());
        expect(fixture.run().has_value());
        expect(eq(sink.script(), completed(emitted)));
        gr::trigger_test::drawSubset("SampleAndHold: what the settings let through", std::format("the sample the trigger captured, held"), script, sink.script());
    };

    "the initial value holds until the first match, then the captured sample does"_test = [&] { //
        play("a b T:c d e |", "a a T:c c c", {{"initial_value", 1.0f}});
    };

    "a second match captures again"_test = [&] { //
        play("T:a b T:c d |", "T:a a T:c c", {{"initial_value", 1.0f}});
    };

    "tag_output=false withholds the capture tag"_test = [&] { //
        play("a T:c d |", "a c c", {{"initial_value", 1.0f}, {"tag_output", false}});
    };

    "a complex sample is held whole, both parts"_test = [] {
        using C = std::complex<float>;
        using namespace gr::testing;
        GraphFixture fixture;
        auto&        source = fixture.emplace<TagSource<C, ProcessFunction::USE_PROCESS_BULK>>({{"values", std::vector<C>{{1.f, -1.f}, {2.f, -2.f}, {3.f, -3.f}, {4.f, -4.f}}}, {"n_samples_max", gr::Size_t(4)}, {"mark_tag", false}});
        source._tags        = {OwningTag(2UZ, {{std::string(gr::tag::kTriggerName.key()), std::string("start")}, {std::string(gr::tag::kTriggerTime.key()), std::uint64_t{1U}}, {std::string(gr::tag::kTriggerOffset.key()), 0.f}})};
        auto& block         = fixture.emplace<SampleAndHold<C>>({{"filter", std::string("start")}, {"initial_value", C{0.f, 9.f}}});
        auto& sink          = fixture.emplace<TagSink<C, ProcessFunction::USE_PROCESS_BULK>>();
        expect(fixture.connect<"out", "in">(source, block).has_value());
        expect(fixture.connect<"out", "in">(block, sink).has_value());
        expect(fixture.run().has_value());
        expect(std::vector<C>(sink._samples.begin(), sink._samples.end()) == std::vector<C>{{0.f, 9.f}, {0.f, 9.f}, {3.f, -3.f}, {3.f, -3.f}}) << "initial until the trigger, then the triggered sample";
    };
};
} // namespace qa_sample_and_hold

namespace qa_predicate {
using namespace gr::blocks::trigger;
using gr::testing::ProcessFunction;
using gr::testing::TagSink;
using gr::testing::TagSource;

[[nodiscard]] const gr::property_map& values() {
    static const gr::property_map map{{"1", 1.f}, {"2", 2.f}, {"3", 3.f}, {"7", 7.f}, {"8", 8.f}};
    return map;
}

[[nodiscard]] const gr::property_map& tags() {
    static const gr::property_map map{{"S", gr::property_map{{std::string(gr::tag::kTriggerName.key()), std::string("start")}}}};
    return map;
}

template<typename TBlock>
[[nodiscard]] std::vector<float> through(std::vector<float> samples, gr::property_map settings) {
    gr::testing::GraphFixture fixture;
    const gr::Size_t          n      = static_cast<gr::Size_t>(samples.size());
    auto&                     source = fixture.template emplace<TagSource<float>>({{"n_samples_max", n}, {"values", std::move(samples)}, {"mark_tag", false}});
    auto&                     block  = fixture.template emplace<TBlock>(std::move(settings));
    auto&                     sink   = fixture.template emplace<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>();

    boost::ut::expect(fixture.template connect<"out", "in">(source, block).has_value());
    boost::ut::expect(fixture.template connect<"out", "in">(block, sink).has_value());
    boost::ut::expect(fixture.run().has_value());

    return std::vector<float>(sink._samples.begin(), sink._samples.end());
}

const boost::ut::suite<"TakeWhile"> _takeWhile = [] {
    using namespace boost::ut;

    "samples pass until the first that fails, and nothing after it"_test = [] {
        const auto found = through<TakeWhile<float>>({1.f, 2.f, 3.f, 7.f, 1.f, 2.f}, {{"predicate", std::string("less")}, {"threshold", 5.f}});

        expect(found == std::vector<float>{1.f, 2.f, 3.f}) << "the 1 and 2 after the 7 are gone too: that is the difference from a filter";
    };

    "a stream that never fails passes whole"_test = [] {
        const auto found = through<TakeWhile<float>>({1.f, 2.f, 3.f}, {{"predicate", std::string("less")}, {"threshold", 5.f}});
        expect(found == std::vector<float>{1.f, 2.f, 3.f});
    };

    "a failure on the first sample emits nothing"_test = [] {
        const auto found = through<TakeWhile<float>>({7.f, 1.f, 2.f}, {{"predicate", std::string("less")}, {"threshold", 5.f}});
        expect(found.empty());
    };

    "an unknown predicate is refused with an error message and the previous one kept"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<TagSource<float>>({{"n_samples_max", 3U}, {"values", std::vector<float>{1.f, 2.f, 3.f}}, {"mark_tag", false}});
        auto&                     block  = fixture.emplace<TakeWhile<float>>({{"predicate", std::string("sideways")}, {"threshold", 5.f}});
        auto&                     sink   = fixture.emplace<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>();
        expect(fixture.connect<"out", "in">(source, block).has_value());
        expect(fixture.connect<"out", "in">(block, sink).has_value());
        gr::trigger_test::ErrorMessages sent{fixture.scheduler()};
        expect(fixture.run().has_value());
        expect(eq(sent.countContaining("sideways"), 1UZ));
    };

    "an expression overrides the comparison"_test = [] {
        const auto found = through<TakeWhile<float>>({1.f, 2.f, 3.f, 8.f, 6.f, 2.f}, {{"expression", std::string("x < 5 or x > 7")}});
        expect(found == std::vector<float>{1.f, 2.f, 3.f, 8.f}) << "8 passes where 'less than 5' alone would not; 6 fails both halves and ends it";
    };

    "a segment trigger re-arms it, so it is useful on a stream that never ends"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string("1 2 7 1 S:1 2 7 1 |")}, {"sample_values", values()}, {"sample_tags", tags()}});
        auto&                     block  = fixture.emplace<TakeWhile<float>>({{"predicate", std::string("less")}, {"threshold", 5.f}, {"segment_filter", std::string("start")}});
        auto&                     sink   = fixture.emplace<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>();
        expect(fixture.connect<"out", "in">(source, block).has_value());
        expect(fixture.connect<"out", "in">(block, sink).has_value());
        expect(fixture.run().has_value());

        const std::vector<float> found(sink._samples.begin(), sink._samples.end());
        expect(found == std::vector<float>{1.f, 2.f, 1.f, 2.f}) << "each segment stops at its own 7";
        expect(eq(block.n_segments_ended.value, 2U));
    };

    "a segment trigger inside a work call re-arms at its own sample, not at the start of the call"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<TagSource<float>>({{"n_samples_max", 8U}, {"values", std::vector<float>{1.f, 2.f, 7.f, 1.f, 1.f, 2.f, 7.f, 1.f}}, {"mark_tag", false}});
        source._tags.push_back({4UZ, gr::property_map{{std::string(gr::tag::kTriggerName.key()), std::string("start")}}});
        auto& block = fixture.emplace<TakeWhile<float>>({{"predicate", std::string("less")}, {"threshold", 5.f}, {"segment_filter", std::string("start")}});
        auto& sink  = fixture.emplace<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>();
        expect(fixture.connect<"out", "in">(source, block).has_value());
        expect(fixture.connect<"out", "in">(block, sink).has_value());
        expect(fixture.run().has_value());

        const std::vector<float> found(sink._samples.begin(), sink._samples.end());
        expect(found == std::vector<float>{1.f, 2.f, 1.f, 2.f}) << "the 1 at sample 3 stays dropped, the segment opened at sample 4 passes 1, 2";
    };
};

const boost::ut::suite<"SkipWhile"> _skipWhile = [] {
    using namespace boost::ut;

    "samples are suppressed until the first that fails, then everything passes"_test = [] {
        const auto found = through<SkipWhile<float>>({7.f, 8.f, 1.f, 7.f, 2.f}, {{"predicate", std::string("greater")}, {"threshold", 5.f}});

        expect(found == std::vector<float>{1.f, 7.f, 2.f}) << "the 7 after the 1 passes: skipping does not resume, which is Rx's asymmetry";
    };

    "a stream that always holds is suppressed entirely"_test = [] {
        const auto found = through<SkipWhile<float>>({7.f, 8.f, 7.f}, {{"predicate", std::string("greater")}, {"threshold", 5.f}});
        expect(found.empty());
    };

    "a segment trigger starts the skipping again"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string("7 1 2 S:7 1 2 |")}, {"sample_values", values()}, {"sample_tags", tags()}});
        auto&                     block  = fixture.emplace<SkipWhile<float>>({{"predicate", std::string("greater")}, {"threshold", 5.f}, {"segment_filter", std::string("start")}});
        auto&                     sink   = fixture.emplace<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>();
        expect(fixture.connect<"out", "in">(source, block).has_value());
        expect(fixture.connect<"out", "in">(block, sink).has_value());
        expect(fixture.run().has_value());

        const std::vector<float> found(sink._samples.begin(), sink._samples.end());
        expect(found == std::vector<float>{1.f, 2.f, 1.f, 2.f}) << "the 7 opening each segment is skipped again";
    };
};

const boost::ut::suite<"ElementAt"> _elementAt = [] {
    using namespace boost::ut;

    "the nth sample of the stream, counting from zero"_test = [] {
        expect(through<ElementAt<float>>({1.f, 2.f, 3.f, 7.f}, {{"n", 2U}}) == std::vector<float>{3.f});
        expect(through<ElementAt<float>>({1.f, 2.f, 3.f}, {{"n", 0U}}) == std::vector<float>{1.f}) << "zero is the first sample, not a disabled setting";
    };

    "a stream shorter than n emits nothing"_test = [] { expect(through<ElementAt<float>>({1.f, 2.f}, {{"n", 9U}}).empty()); };

    "with a segment filter it emits the nth of every segment"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string("S:1 2 3 S:7 8 1 |")}, {"sample_values", values()}, {"sample_tags", tags()}});
        auto&                     block  = fixture.emplace<ElementAt<float>>({{"n", 1U}, {"segment_filter", std::string("start")}});
        auto&                     sink   = fixture.emplace<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>();
        expect(fixture.connect<"out", "in">(source, block).has_value());
        expect(fixture.connect<"out", "in">(block, sink).has_value());
        expect(fixture.run().has_value());

        const std::vector<float> found(sink._samples.begin(), sink._samples.end());
        expect(found == std::vector<float>{2.f, 8.f}) << "the second sample of each segment";
        expect(eq(block.n_emitted.value, 2U));
    };

    "what each of the three did, drawn"_test = [] {
        const std::string script = "1 2 3 7 8 1 2 |";
        for (const auto& [title, settings] : std::vector<std::pair<std::string, gr::property_map>>{
                 {"TakeWhile(less than 5)", {{"predicate", std::string("less")}, {"threshold", 5.f}}},
             }) {
            gr::testing::GraphFixture fixture;
            auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", script}, {"sample_values", values()}, {"sample_tags", tags()}});
            auto&                     block  = fixture.emplace<TakeWhile<float>>(settings);
            auto&                     sink   = fixture.emplace<MarbleSink<float>>({{"sample_values", values()}, {"sample_tags", tags()}});
            expect(fixture.connect<"out", "in">(source, block).has_value());
            expect(fixture.connect<"out", "in">(block, sink).has_value());
            expect(fixture.run().has_value());
            gr::trigger_test::drawSubset(std::format("{}: everything after the first failure is gone", title), title, script, sink.script());
        }

        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", script}, {"sample_values", values()}, {"sample_tags", tags()}});
        auto&                     block  = fixture.emplace<SkipWhile<float>>({{"predicate", std::string("less")}, {"threshold", 5.f}});
        auto&                     sink   = fixture.emplace<MarbleSink<float>>({{"sample_values", values()}, {"sample_tags", tags()}});
        expect(fixture.connect<"out", "in">(source, block).has_value());
        expect(fixture.connect<"out", "in">(block, sink).has_value());
        expect(fixture.run().has_value());
        gr::trigger_test::drawSubset("SkipWhile(less than 5): nothing resumes the skipping", "SkipWhile(less than 5)", script, sink.script());

        expect(ge(block.n_skipped.value, 3U));
    };
};
} // namespace qa_predicate

namespace qa_tail {
using namespace gr::blocks::trigger;

[[nodiscard]] const gr::property_map& values() {
    static const gr::property_map map{{"1", 1.f}, {"2", 2.f}, {"3", 3.f}, {"4", 4.f}, {"5", 5.f}, {"6", 6.f}};
    return map;
}

[[nodiscard]] const gr::property_map& tags() {
    static const gr::property_map map{{"E", gr::property_map{{std::string(gr::tag::kTriggerName.key()), std::string("end")}, {std::string(gr::tag::kTriggerTime.key()), std::uint64_t{1U}}, {std::string(gr::tag::kTriggerOffset.key()), 0.f}}}, //
        {"X", gr::property_map{{std::string(gr::tag::kTriggerName.key()), std::string("other")}, {std::string(gr::tag::kTriggerTime.key()), std::uint64_t{1U}}, {std::string(gr::tag::kTriggerOffset.key()), 0.f}}}};
    return map;
}

template<typename TBlock>
[[nodiscard]] std::vector<float> tailed(std::string script, gr::property_map settings, std::size_t chunk = 0UZ, std::size_t room = 0UZ) {
    gr::testing::GraphFixture fixture;
    auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::move(script)}, {"sample_values", values()}, {"sample_tags", tags()}});
    auto&                     block  = fixture.emplace<TBlock>(std::move(settings));
    auto&                     sink   = fixture.emplace<gr::trigger_test::CollectingSink<float>>();
    if (chunk > 0UZ) {
        block.in.max_samples = chunk;
    }
    if (room > 0UZ) {
        block.out.max_samples = room;
    }
    boost::ut::expect(fixture.connect<"out", "in">(source, block).has_value());
    boost::ut::expect(fixture.connect<"out", "in">(block, sink).has_value());
    boost::ut::expect(fixture.run().has_value());
    return sink._collected;
}

const boost::ut::suite<"Tail"> _tail = [] {
    using namespace boost::ut;

    "TakeLast keeps the tail of the stream, SkipLast keeps everything before it"_test = [] {
        expect(tailed<TakeLast<float>>("1 2 3 4 |", {{"n", 2U}}) == std::vector<float>{3.f, 4.f});
        expect(tailed<SkipLast<float>>("1 2 3 4 |", {{"n", 2U}}) == std::vector<float>{1.f, 2.f}) << "the two together are the whole stream, which is what makes them a pair";
    };

    "a matching trigger ends a segment, and each segment has its own tail"_test = [] {
        expect(tailed<TakeLast<float>>("1 2 3 E:4 5 6 |", {{"filter", std::string("end")}, {"n", 2U}}) == std::vector<float>{2.f, 3.f, 5.f, 6.f}) << "the tag ends the first segment before its own sample, and the stream ends the second";
        expect(tailed<SkipLast<float>>("1 2 3 E:4 5 6 |", {{"filter", std::string("end")}, {"n", 2U}}) == std::vector<float>{1.f, 4.f});
    };

    "a trigger the filter does not name does not end anything"_test = [] { //
        expect(tailed<TakeLast<float>>("1 2 X:3 4 |", {{"filter", std::string("end")}, {"n", 2U}}) == std::vector<float>{3.f, 4.f});
    };

    "a tail longer than the segment is the whole segment"_test = [] {
        expect(tailed<TakeLast<float>>("1 2 |", {{"n", 5U}}) == std::vector<float>{1.f, 2.f});
        expect(tailed<SkipLast<float>>("1 2 |", {{"n", 5U}}) == std::vector<float>{}) << "nothing is old enough to be released";
    };

    "a tail of one"_test = [] {
        expect(tailed<TakeLast<float>>("1 2 3 |", {{"n", 1U}}) == std::vector<float>{3.f});
        expect(tailed<SkipLast<float>>("1 2 3 |", {{"n", 1U}}) == std::vector<float>{1.f, 2.f});
    };

    "a single sample"_test = [] {
        expect(tailed<TakeLast<float>>("4 |", {{"n", 2U}}) == std::vector<float>{4.f});
        expect(tailed<SkipLast<float>>("4 |", {{"n", 2U}}) == std::vector<float>{});
    };

    "the counters account for every sample"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string("1 2 3 4 |")}, {"sample_values", values()}});
        auto&                     block  = fixture.emplace<TakeLast<float>>({{"n", 2U}});
        auto&                     sink   = fixture.emplace<gr::trigger_test::CollectingSink<float>>();
        expect(fixture.connect<"out", "in">(source, block).has_value());
        expect(fixture.connect<"out", "in">(block, sink).has_value());
        expect(fixture.run().has_value());

        expect(eq(block.n_passed.value, gr::Size_t{2}));
        expect(eq(block.n_dropped.value, gr::Size_t{2}));
        expect(eq(block.n_segments.value, gr::Size_t{1})) << "the end of the stream ends the last segment";
    };

    "the tail does not depend on where the stream was cut"_test = [] {
        for (const std::size_t chunk : {1UZ, 2UZ, 3UZ, 5UZ}) {
            expect(tailed<TakeLast<float>>("1 2 3 E:4 5 6 |", {{"filter", std::string("end")}, {"n", 2U}}, chunk) == std::vector<float>{2.f, 3.f, 5.f, 6.f}) << std::format("TakeLast at a work-call size of {}", chunk);
            expect(tailed<SkipLast<float>>("1 2 3 E:4 5 6 |", {{"filter", std::string("end")}, {"n", 2U}}, chunk) == std::vector<float>{1.f, 4.f}) << std::format("SkipLast at a work-call size of {}", chunk);
        }
    };
};
} // namespace qa_tail

namespace qa_trigger_watchdog {
using namespace gr::blocks::trigger;
using gr::trigger_test::EventTap;

const boost::ut::suite<"TriggerWatchdog"> _triggerWatchdog = [] {
    using namespace boost::ut;

    const gr::property_map values{{"a", 1.0f}, {"b", 2.0f}, {"c", 3.0f}, {"d", 4.0f}, {"e", 5.0f}, {"f", 6.0f}};
    const gr::property_map tags{{"T", gr::property_map{{std::string(gr::tag::kTriggerName.key()), std::string("start")}, {std::string(gr::tag::kTriggerTime.key()), std::uint64_t{1U}}, {std::string(gr::tag::kTriggerOffset.key()), 0.f}}}};

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

    "anchored, the watchdog reports an absence once, only after its anchor and only without a match "_test = [&] {
        const gr::property_map anchored{{"T", gr::property_map{{std::string(gr::tag::kTriggerName.key()), std::string("start")}, {std::string(gr::tag::kTriggerTime.key()), std::uint64_t{1U}}, {std::string(gr::tag::kTriggerOffset.key()), 0.f}}}, {"A", gr::property_map{{std::string(gr::tag::kTriggerName.key()), std::string("arm")}, {std::string(gr::tag::kTriggerTime.key()), std::uint64_t{1U}}, {std::string(gr::tag::kTriggerOffset.key()), 0.f}}}, {"G", gr::property_map{{std::string(gr::tag::kNDroppedSamples.key()), gr::Size_t{5U}}}}};
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
        const gr::property_map untimedTags{{"T", gr::property_map{{std::string(gr::tag::kTriggerName.key()), std::string("start")}}}};

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
            gr::property_map anchorTag{{std::string(gr::tag::kTriggerName.key()), std::string("start")}, {std::string(gr::tag::kTriggerTime.key()), at}, {std::string(gr::tag::kTriggerOffset.key()), 0.f}};
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
} // namespace qa_trigger_watchdog

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
