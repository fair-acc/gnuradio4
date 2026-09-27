#include <boost/ut.hpp>

#include <string>
#include <vector>

#include <gnuradio-4.0/test/DeviceTestHelper.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/trigger/Gate.hpp>
#include <gnuradio-4.0/trigger/Marble.hpp>
#include <gnuradio-4.0/trigger/SampleAndHold.hpp>
#include <gnuradio-4.0/trigger/SchmittTrigger.hpp>
#include <gnuradio-4.0/trigger/TagBridge.hpp>
#include <gnuradio-4.0/trigger/TakeSkip.hpp>
#include <gnuradio-4.0/trigger/TimeInterval.hpp>
#include <gnuradio-4.0/trigger/TriggerWatchdog.hpp>
#include <gnuradio-4.0/trigger/ValueTrigger.hpp>

#include "TriggerTest.hpp"

using namespace gr::blocks::trigger;
using gr::testing::operator""_domain_test;
using gr::trigger_test::completed;
using gr::trigger_test::eventNamed;
using gr::trigger_test::EventScript;

int main() {
    using namespace boost::ut;

    const gr::property_map values{{"a", 1.0f}, {"b", 2.0f}, {"c", 3.0f}, {"d", 4.0f}, {"e", 5.0f}, {"f", 6.0f}};
    const gr::property_map tags{{"T", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("start")}, {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1U}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}}};

    // the trigger blocks are host blocks: none submits a kernel. What must hold on every served domain is that
    // they build under that toolchain and answer identically, whichever domain the graph around them runs in.
    "a gate answers the same on every served domain"_domain_test = [&](auto& fixture, std::string_view domain) {
        auto& source = fixture.template emplace<MarbleSource<float>>({{"script", std::string("a b T:c d e f |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto& gate   = fixture.template emplace<Gate<float>>({{"mode", std::string("skip_until")}, {"open_filter", std::string("start")}});
        auto& sink   = fixture.template emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        expect(fixture.template connect<"out", "in">(source, gate).has_value());
        expect(fixture.template connect<"out", "in">(gate, sink).has_value());
        expect(fixture.run().has_value()) << domain;
        expect(eq(sink.script(), completed("T:c d e f"))) << domain;
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "a take/hold chain answers the same on every served domain"_domain_test = [&](auto& fixture, std::string_view domain) {
        auto& source = fixture.template emplace<MarbleSource<float>>({{"script", std::string("a T:b c d e |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto& take   = fixture.template emplace<TakeN<float>>({{"filter", std::string("start")}, {"n", 3U}});
        auto& hold   = fixture.template emplace<SampleAndHold<float>>({{"filter", std::string("start")}, {"initial_value", 1.0f}});
        auto& sink   = fixture.template emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        expect(fixture.template connect<"out", "in">(source, take).has_value());
        expect(fixture.template connect<"out", "in">(take, hold).has_value());
        expect(fixture.template connect<"out", "in">(hold, sink).has_value());
        expect(fixture.run().has_value()) << domain;
        expect(eq(sink.script(), completed("T:b b b"))) << domain << ": the captured sample is held for the whole run";
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "the tag/event bridge and the watchdog run on every served domain"_domain_test = [&](auto& fixture, std::string_view domain) {
        auto& source   = fixture.template emplace<MarbleSource<float>>({{"script", std::string("T:a b c d |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto& watchdog = fixture.template emplace<TriggerWatchdog<float>>({{"filter", std::string("start")}, {"timeout_samples", 2U}});
        auto& bridge   = fixture.template emplace<TagToMessage<float>>({{"filter", std::string("start")}});
        auto& sink     = fixture.template emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        expect(fixture.template connect<"out", "in">(source, watchdog).has_value());
        expect(fixture.template connect<"out", "in">(watchdog, bridge).has_value());
        expect(fixture.template connect<"out", "in">(bridge, sink).has_value());
        expect(fixture.run().has_value()) << domain;
        expect(eq(sink._samples.size(), 4UZ)) << domain << ": the stream passes through untouched";
        expect(eq(watchdog.n_timeouts.value, 1U)) << domain << ": one outage after the trigger stops";
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    // the blocks added since: a trigger formed from the signal, and a time measured from the triggers it forms
    "a value trigger forms the same triggers on every served domain"_domain_test = [&](auto& fixture, std::string_view domain) {
        auto& source  = fixture.template emplace<MarbleSource<float>>({{"script", std::string("a e a e a |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto& trigger = fixture.template emplace<ValueTrigger<float, ValueCondition::level>>({{"threshold", 3.f}, {"hysteresis", 0.5f}, {"sample_rate", 1000.f}});
        auto& sink    = fixture.template emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        expect(fixture.template connect<"out", "in">(source, trigger).has_value());
        expect(fixture.template connect<"out", "in">(trigger, sink).has_value());
        expect(fixture.run().has_value()) << domain;
        expect(eq(sink._samples.size(), 5UZ)) << domain << ": the stream passes through untouched";
        expect(eq(trigger.n_triggers.value, 2U)) << domain << ": two crossings of the threshold, whichever domain ran it";
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "an interval measurement answers the same on every served domain"_domain_test = [&](auto& fixture, std::string_view domain) {
        auto& script   = fixture.template emplace<EventScript>();
        script._events = {eventNamed("tick", 1'000'000'000U), eventNamed("tick", 1'002'000'000U), eventNamed("tick", 1'004'000'000U)};
        auto& interval = fixture.template emplace<TimeInterval<double>>({{"mode", std::string("to_previous")}});
        auto& measured = fixture.template emplace<gr::testing::TagSink<double, gr::testing::ProcessFunction::USE_PROCESS_ONE>>();
        expect(fixture.template connect<"evtOut", "evtIn">(script, interval).has_value());
        expect(fixture.template connect<"out", "in">(interval, measured).has_value());
        expect(fixture.run().has_value()) << domain;

        expect(eq(interval.n_undated.value, 0U)) << domain << ": every event arrived dated";
        expect(eq(interval.n_intervals.value, 2U)) << domain << ": three ticks give two periods";
        expect(eq(measured._samples.size(), 2UZ)) << domain;
        if (measured._samples.size() == 2UZ) {
            expect(approx(measured._samples[0], 2e-3, 1e-12)) << domain;
            expect(approx(measured._samples[1], 2e-3, 1e-12)) << domain;
        }
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "an edge detector feeding an interval measurement answers the same on every served domain"_domain_test = [&](auto& fixture, std::string_view domain) {
        // the leading tag dates the stream: without an anchor the trigger can form an edge but cannot say when it was
        auto& source           = fixture.template emplace<MarbleSource<float>>({{"script", std::string("T:a e a e a |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto& trigger          = fixture.template emplace<ValueTrigger<float, ValueCondition::level>>({{"threshold", 3.f}, {"hysteresis", 0.5f}, {"sample_rate", 1000.f}});
        auto& spent            = fixture.template emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        auto& interval         = fixture.template emplace<TimeInterval<double>>({{"mode", std::string("to_previous")}});
        auto& measured         = fixture.template emplace<gr::testing::TagSink<double, gr::testing::ProcessFunction::USE_PROCESS_ONE>>();
        trigger.in.max_samples = 2UZ; // an event output grants one slot per work call, so each edge needs a call of its own
        expect(fixture.template connect<"out", "in">(source, trigger).has_value());
        expect(fixture.template connect<"out", "in">(trigger, spent).has_value());
        expect(fixture.template connect<"evtOut", "evtIn">(trigger, interval).has_value());
        expect(fixture.template connect<"out", "in">(interval, measured).has_value());
        expect(fixture.run().has_value()) << domain;

        expect(eq(trigger.n_triggers.value, 2U)) << domain << ": the trigger found both edges";
        expect(eq(interval.n_undated.value, 0U)) << domain << ": every edge arrived dated";
        expect(eq(interval.n_intervals.value, 1U)) << domain << ": two edges give one period between them";
        expect(eq(measured._samples.size(), 1UZ)) << domain;
        if (measured._samples.size() == 1UZ) {
            expect(approx(measured._samples[0], 2e-3, 1e-9)) << domain << ": two samples apart at 1 kHz";
        }
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    return 0;
}
