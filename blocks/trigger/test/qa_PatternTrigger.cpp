#include <boost/ut.hpp>

#include <string>
#include <vector>

#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/test/EventMarbles.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/trigger/Marble.hpp>
#include <gnuradio-4.0/trigger/PatternTrigger.hpp>

#include "TriggerTest.hpp"

using namespace gr::blocks::trigger;
using gr::trigger_test::EventTap;

namespace {
struct Outcome {
    gr::Size_t                 triggers;
    gr::Size_t                 rejected;
    std::size_t                seen;
    std::vector<std::uint64_t> times;
};

/// runs one pattern over two channels, a sample at a time so each report has a work call of its own
[[nodiscard]] Outcome judge(std::vector<float> channelA, std::vector<float> channelB, gr::property_map settings) {
    gr::testing::GraphFixture fixture;
    settings[std::string("n_inputs")]    = 2U;
    settings[std::string("sample_rate")] = 1000.f;
    auto& sourceA                        = fixture.emplace<gr::testing::TagSource<float>>({{"values", channelA}, {"n_samples_max", static_cast<gr::Size_t>(channelA.size())}});
    auto& sourceB                        = fixture.emplace<gr::testing::TagSource<float>>({{"values", channelB}, {"n_samples_max", static_cast<gr::Size_t>(channelB.size())}});
    auto& trigger                        = fixture.emplace<PatternTrigger<float>>(std::move(settings));
    auto& collector                      = fixture.emplace<EventTap>();
    // the pattern's inputs are a vector, so the destination port is named by index rather than by name
    boost::ut::expect(fixture.graph.connect(sourceA, gr::PortDefinition{0UZ}, trigger, gr::PortDefinition{0UZ, 0UZ}).has_value());
    boost::ut::expect(fixture.graph.connect(sourceB, gr::PortDefinition{0UZ}, trigger, gr::PortDefinition{0UZ, 1UZ}).has_value());
    boost::ut::expect(fixture.connect<"evtOut", "evtIn">(trigger, collector).has_value());
    boost::ut::expect(fixture.run().has_value());
    return Outcome{trigger.n_triggers.value, trigger.n_rejected.value, collector.size(), collector.times()};
}
} // namespace

const boost::ut::suite<"PatternTrigger"> _patternTrigger = [] {
    using namespace boost::ut;

    "a pattern fires where both channels are in the named state"_test = [] {
        //            A: lo hi hi lo      B: hi lo lo hi      "10" holds at samples 1 and 2
        const auto found = judge({0.f, 5.f, 5.f, 0.f}, {5.f, 0.f, 0.f, 5.f}, {{"pattern", std::string("10")}, {"thresholds", std::vector<float>{2.5f}}, {"hysteresis", 0.5f}});
        expect(eq(found.triggers, 1U)) << "the pattern becomes true once, however long it then stands";
    };

    "a don't-care channel is not consulted"_test = [] {
        const auto both = judge({0.f, 5.f, 0.f}, {0.f, 0.f, 0.f}, {{"pattern", std::string("10")}, {"thresholds", std::vector<float>{2.5f}}, {"hysteresis", 0.5f}});
        const auto only = judge({0.f, 5.f, 0.f}, {0.f, 5.f, 0.f}, {{"pattern", std::string("1X")}, {"thresholds", std::vector<float>{2.5f}}, {"hysteresis", 0.5f}});
        expect(eq(both.triggers, 1U));
        expect(eq(only.triggers, 1U)) << "the second channel is high where '10' would have refused it";
    };

    "a pattern that never stands reports nothing"_test = [] {
        const auto found = judge({0.f, 5.f, 0.f}, {0.f, 5.f, 0.f}, {{"pattern", std::string("10")}, {"thresholds", std::vector<float>{2.5f}}, {"hysteresis", 0.5f}});
        expect(eq(found.triggers, 0U)) << "both channels go high together, so '10' is never true";
    };

    "leaves fires where the pattern stops standing"_test = [] {
        const auto found = judge({0.f, 5.f, 5.f, 0.f}, {5.f, 0.f, 0.f, 5.f}, //
            {{"pattern", std::string("10")}, {"thresholds", std::vector<float>{2.5f}}, {"hysteresis", 0.5f}, {"when", std::string("leaves")}});
        expect(eq(found.triggers, 1U)) << "once, where it became false again";
    };

    "holds refuses a pattern that does not stand long enough"_test = [] {
        // the pattern stands for two samples; a setup time of four samples must reject it
        const auto refused = judge({0.f, 5.f, 5.f, 0.f}, {5.f, 0.f, 0.f, 5.f}, //
            {{"pattern", std::string("10")}, {"thresholds", std::vector<float>{2.5f}}, {"hysteresis", 0.5f}, {"when", std::string("holds")}, {"width_min_samples", 4U}});
        expect(eq(refused.triggers, 0U));
        expect(eq(refused.rejected, 1U)) << "it stood, but not for the setup time";

        const auto accepted = judge({0.f, 5.f, 5.f, 0.f}, {5.f, 0.f, 0.f, 5.f}, //
            {{"pattern", std::string("10")}, {"thresholds", std::vector<float>{2.5f}}, {"hysteresis", 0.5f}, {"when", std::string("holds")}, {"width_min_samples", 2U}});
        expect(eq(accepted.triggers, 1U)) << "two samples is the setup time it was given";
    };

    "each channel may carry its own threshold"_test = [] {
        // A crosses 2.5, B must stay below 8 -- a single threshold would judge B high
        const auto found = judge({0.f, 5.f, 0.f}, {6.f, 6.f, 6.f}, //
            {{"pattern", std::string("10")}, {"thresholds", std::vector<float>{2.5f, 8.f}}, {"hysteresis", 0.5f}});
        expect(eq(found.triggers, 1U)) << "B sits below its own threshold, so it counts as low";
    };

    "a pattern whose length does not match the channels consults none of them"_test = [] {
        const auto found = judge({0.f, 5.f, 0.f}, {0.f, 0.f, 0.f}, {{"pattern", std::string("101")}, {"thresholds", std::vector<float>{2.5f}}, {"hysteresis", 0.5f}});
        expect(eq(found.triggers, 1U)) << "every channel falls back to 'X', so the pattern is true from the first sample";
    };

    "a reported pattern is dated and named"_test = [] {
        const auto found = judge({0.f, 5.f, 5.f, 0.f}, {5.f, 0.f, 0.f, 5.f}, {{"pattern", std::string("10")}, {"thresholds", std::vector<float>{2.5f}}, {"hysteresis", 0.5f}});
        expect(eq(found.seen, 1UZ)) << "the report reaches the event output";
        expect(found.times.empty()) << "and carries no time, this stream having no anchor to date it from";
    };

    "what the pattern sees, drawn"_test = [] {
        // A is high over samples 1..4 and 9..11; B is low over 1..2 and 8..11, so "10" stands twice, briefly and then longer
        const gr::property_map values{{"z", 0.f}, {"h", 5.f}};
        const gr::property_map tags{{"T", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("acq")}, //
                                              {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1'000'000'000U}},   //
                                              {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f},                           //
                                              {std::string("sample_rate"), 1000.f}}}};
        const std::string      scriptA = "T:z h h h h z z z z h h h z |";
        const std::string      scriptB = "T:h z z h h h h h z z z z h |";

        auto judge = [&](std::string when, gr::Size_t setup) {
            gr::testing::GraphFixture fixture;
            auto&                     sourceA   = fixture.emplace<MarbleSource<float>>({{"script", scriptA}, {"sample_values", values}, {"sample_tags", tags}});
            auto&                     sourceB   = fixture.emplace<MarbleSource<float>>({{"script", scriptB}, {"sample_values", values}, {"sample_tags", tags}});
            auto&                     trigger   = fixture.emplace<PatternTrigger<float>>({{"n_inputs", 2U}, //
                                      {"pattern", std::string("10")},                                       //
                                      {"thresholds", std::vector<float>{2.5f}},                             //
                                      {"hysteresis", 0.5f},                                                 //
                                      {"sample_rate", 1000.f},                                              //
                                      {"when", std::move(when)},                                            //
                                      {"width_min_samples", setup},                                         //
                                      {"trigger_name", std::string(setup == 0U ? "enters" : "holds 3")}});
            auto&                     collector = fixture.emplace<EventTap>();
            expect(fixture.graph.connect(sourceA, gr::PortDefinition{std::string("out")}, trigger, gr::PortDefinition{std::string("in#0")}).has_value());
            expect(fixture.graph.connect(sourceB, gr::PortDefinition{std::string("out")}, trigger, gr::PortDefinition{std::string("in#1")}).has_value());
            expect(fixture.graph.connect(trigger, gr::PortDefinition{std::string("evtOut")}, collector, gr::PortDefinition{std::string("evtIn")}).has_value());
            expect(fixture.run().has_value());
            return collector.dated("pattern");
        };

        const auto enters = judge("enters", 0U);
        const auto holds  = judge("holds", 3U);
        expect(ge(enters.size(), 1UZ)) << "the pattern stands at least once";

        gr::testing::MarbleDiagram diagram{"PatternTrigger: pattern \"10\" over two channels"};
        diagram.row("enters").all(enters);
        diagram.row("holds 3").all(holds);
        diagram.condition("A above 2.5 and B below it, at the same sample");
        diagram.print();

        const std::vector<float> a{0.f, 5.f, 5.f, 5.f, 5.f, 0.f, 0.f, 0.f, 0.f, 5.f, 5.f, 5.f, 0.f};
        const std::vector<float> b{5.f, 0.f, 0.f, 5.f, 5.f, 5.f, 5.f, 5.f, 0.f, 0.f, 0.f, 0.f, 5.f};
        std::vector<double>      x(a.size());
        std::vector<double>      ya(a.size());
        std::vector<double>      yb(a.size());
        for (std::size_t i = 0UZ; i < a.size(); ++i) {
            x[i]  = static_cast<double>(i);
            ya[i] = static_cast<double>(a[i]);
            yb[i] = static_cast<double>(b[i]) + 7.;
        }
        auto chart = gr::graphs::ImChart<90, 16>({{x.front(), x.back()}, {-0.5, 13.}});
        chart.draw(x, ya, "A");
        chart.draw(x, yb, "B (offset)");
        std::println("\nthe two channels, by sample index (B drawn above A):");
        chart.draw();
    };
};

int main() { /* tests are statically executed */ }
