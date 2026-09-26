#include <boost/ut.hpp>

#include <string>
#include <vector>

#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/test/EventMarbles.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/trigger/Marble.hpp>
#include <gnuradio-4.0/trigger/ValueTrigger.hpp>

#include "TriggerTest.hpp"

using namespace gr::blocks::trigger;
using gr::trigger_test::EventTap;

namespace {
struct Outcome {
    gr::Size_t  triggers;
    gr::Size_t  dropped;
    std::size_t events;
};

template<ValueCondition condition>
[[nodiscard]] Outcome run(std::vector<float> samples, gr::property_map settings) {
    gr::testing::GraphFixture fixture;
    settings[std::string("sample_rate")] = 1000.f;
    auto& source                         = fixture.emplace<gr::testing::TagSource<float>>({{"values", samples}, {"n_samples_max", static_cast<gr::Size_t>(samples.size())}});
    auto& trigger                        = fixture.emplace<ValueTrigger<float, condition>>(std::move(settings));
    auto& sink                           = fixture.emplace<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_ONE>>();
    auto& edges                          = fixture.emplace<EventTap>();
    boost::ut::expect(fixture.connect<"out", "in">(source, trigger).has_value());
    boost::ut::expect(fixture.connect<"out", "in">(trigger, sink).has_value());
    boost::ut::expect(fixture.connect<"evtOut", "evtIn">(trigger, edges).has_value());
    boost::ut::expect(fixture.run().has_value());
    return {trigger.n_triggers.value, trigger.n_events_dropped.value, edges.names().size()};
}
} // namespace

const boost::ut::suite<"ValueTrigger"> _valueTrigger = [] {
    using namespace boost::ut;

    "a level trigger fires once per rising crossing"_test = [] {
        const auto found = run<ValueCondition::level>({0.f, 0.f, 3.f, 3.f, 0.f, 0.f, 3.f, 0.f}, {{"threshold", 1.f}, {"hysteresis", 0.5f}});
        expect(eq(found.triggers, 2U)) << "two rising crossings, the falling ones ignored";
        expect(eq(found.events + static_cast<std::size_t>(found.dropped), 2UZ)) << "each reaches the event output, or is counted as lost";
    };

    "both edges doubles what a rising-only trigger sees"_test = [] {
        const auto found = run<ValueCondition::level>({0.f, 3.f, 0.f, 3.f, 0.f}, {{"threshold", 1.f}, {"hysteresis", 0.5f}, {"edge", std::string("both")}});
        expect(ge(found.triggers, 3U)) << "rising and falling crossings alike";
        expect(eq(found.events + static_cast<std::size_t>(found.dropped), static_cast<std::size_t>(found.triggers))) << "nothing is lost without being counted";
    };

    "a window trigger fires when the signal leaves the band"_test = [] {
        const auto found = run<ValueCondition::window>({1.5f, 1.5f, 5.f, 5.f, 1.5f}, {{"threshold", 1.f}, {"threshold_upper", 2.f}, {"hysteresis", 0.1f}, {"edge", std::string("both")}});
        expect(ge(found.triggers, 2U)) << "one leaving the band, one re-entering it";
    };

    "a pulse-width trigger accepts the width it was asked for and counts the rest"_test = [] {
        // one two-sample pulse and one four-sample pulse, accepting only widths of three or less
        const auto found = run<ValueCondition::pulse_width>({0.f, 3.f, 3.f, 0.f, 0.f, 3.f, 3.f, 3.f, 3.f, 0.f}, //
            {{"threshold", 1.f}, {"hysteresis", 0.5f}, {"width_max_samples", 3U}});
        expect(eq(found.triggers, 1U)) << "the short pulse qualifies, the long one does not";
    };

    "a runt trigger fires on a pulse that never reaches the upper threshold"_test = [] {
        // first pulse reaches 5 (not a runt), second only 2 (a runt) against an upper threshold of 4
        const auto found = run<ValueCondition::runt>({0.f, 5.f, 0.f, 2.f, 0.f}, //
            {{"threshold", 1.f}, {"threshold_upper", 4.f}, {"hysteresis", 0.5f}});
        expect(eq(found.triggers, 1U)) << "only the pulse that fell short is a runt";
    };

    "a dropout trigger fires once when crossings stop"_test = [] {
        const auto found = run<ValueCondition::dropout>({0.f, 3.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, //
            {{"threshold", 1.f}, {"hysteresis", 0.5f}, {"width_max_samples", 3U}});
        expect(eq(found.triggers, 1U)) << "one outage, reported once however long it lasts";
    };

    "the runtime mode reaches the same decision as the fixed one"_test = [] {
        const auto fixed   = run<ValueCondition::level>({0.f, 3.f, 0.f, 3.f, 0.f}, {{"threshold", 1.f}, {"hysteresis", 0.5f}});
        const auto runtime = run<ValueCondition::AUTO>({0.f, 3.f, 0.f, 3.f, 0.f}, {{"threshold", 1.f}, {"hysteresis", 0.5f}, {"condition_mode", std::string("level")}});
        expect(eq(fixed.triggers, runtime.triggers)) << "AUTO delegates to the setting and agrees with the template parameter";
    };

    "a slew-rate trigger fires where the signal changes faster than it may"_test = [] {
        // a gentle ramp, then a step of 5 in one sample: at 1 kHz that is 5000 units per second
        const auto found = run<ValueCondition::slew_rate>({0.f, 0.1f, 0.2f, 0.3f, 5.3f, 5.4f, 5.5f}, //
            {{"slew_samples", 1U}, {"slew_max_rate", 1000.f}, {"threshold", 100.f}, {"hysteresis", 1.f}});

        expect(eq(found.triggers, 1U)) << "one trigger for the excursion, not one per sample of it";
        expect(eq(found.events, 1UZ));
    };

    "a slew-rate trigger stays quiet while the change is within its band"_test = [] {
        const auto found = run<ValueCondition::slew_rate>({0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f}, //
            {{"slew_samples", 1U}, {"slew_max_rate", 1000.f}, {"threshold", 100.f}, {"hysteresis", 1.f}});

        expect(eq(found.triggers, 0U)) << "0.1 per sample at 1 kHz is 100 per second, well inside the limit";
    };

    "what the conditions look like, drawn"_test = [] {
        // one waveform, judged by three conditions: a level crossing, a short pulse, and a runt that never gets high
        const gr::property_map values{{"z", 0.f}, {"m", 2.f}, {"h", 5.f}};
        const gr::property_map tags{{"T", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("acq")}, //
                                              {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1'000'000'000U}},   //
                                              {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f},                           //
                                              {std::string("sample_rate"), 1000.f}}}};
        const std::string      script = "T:z h h z z h h h h z z m m z z h z |";

        auto judge = [&](gr::property_map settings) {
            gr::testing::GraphFixture fixture;
            auto&                     source  = fixture.emplace<MarbleSource<float>>({{"script", script}, {"sample_values", values}, {"sample_tags", tags}});
            auto&                     trigger = fixture.emplace<ValueTrigger<float, ValueCondition::AUTO>>(std::move(settings));
            auto&                     sink    = fixture.emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
            auto&                     edges   = fixture.emplace<EventTap>();
            expect(fixture.connect<"out", "in">(source, trigger).has_value());
            expect(fixture.connect<"out", "in">(trigger, sink).has_value());
            expect(fixture.connect<"evtOut", "evtIn">(trigger, edges).has_value());
            expect(fixture.run().has_value());
            return edges.dated();
        };

        const auto level = judge({{"condition_mode", std::string("level")}, {"threshold", 1.f}, {"hysteresis", 0.5f}, {"sample_rate", 1000.f}, {"trigger_name", std::string("level")}});
        const auto width = judge({{"condition_mode", std::string("pulse_width")}, {"threshold", 1.f}, {"hysteresis", 0.5f}, {"width_max_samples", 3U}, {"sample_rate", 1000.f}, {"trigger_name", std::string("short")}});
        const auto runt  = judge({{"condition_mode", std::string("runt")}, {"threshold", 1.f}, {"threshold_upper", 4.f}, {"hysteresis", 0.5f}, {"sample_rate", 1000.f}, {"trigger_name", std::string("runt")}});

        expect(ge(level.size(), 3UZ)) << "three rising crossings in the waveform";
        expect(ge(runt.size(), 1UZ)) << "and one pulse that never reached the upper threshold";

        gr::testing::MarbleDiagram diagram{"ValueTrigger: one waveform, three conditions (threshold 1, upper 4)"};
        diagram.row("level").all(level);
        diagram.row("width<=3").all(width);
        diagram.row("runt").all(runt);
        diagram.print();

        const std::vector<float> wave{0.f, 5.f, 5.f, 0.f, 0.f, 5.f, 5.f, 5.f, 5.f, 0.f, 0.f, 2.f, 2.f, 0.f, 0.f, 5.f, 0.f};
        std::vector<double>      x(wave.size());
        std::vector<double>      y(wave.size());
        for (std::size_t i = 0UZ; i < wave.size(); ++i) {
            x[i] = static_cast<double>(i);
            y[i] = static_cast<double>(wave[i]);
        }
        auto chart = gr::graphs::ImChart<90, 14>({{x.front(), x.back()}, {-0.5, 6.}});
        chart.draw(x, y, "signal");
        std::println("\nthe waveform those conditions judge, by sample index:");
        chart.draw();
    };
};

int main() { /* tests are statically executed */ }
