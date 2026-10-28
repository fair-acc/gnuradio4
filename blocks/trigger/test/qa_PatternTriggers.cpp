#include <boost/ut.hpp>

#include <array>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/test/EventMarbles.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/trigger/ClockedTriggers.hpp>
#include <gnuradio-4.0/trigger/PatternTrigger.hpp>

#include "Marble.hpp"
#include "TriggerTest.hpp"

namespace qa_pattern_trigger {
using gr::trigger_test::kStreamStart;
using namespace gr::blocks::trigger;
using gr::trigger_test::EventTap;

struct Outcome {
    gr::Size_t                 triggers;
    gr::Size_t                 rejected;
    std::size_t                seen;
    std::vector<std::uint64_t> times;
};

/** runs one pattern over two channels, a sample at a time so each report has a work call of its own */
[[nodiscard]] Outcome judge(std::vector<float> channelA, std::vector<float> channelB, gr::property_map settings) {
    gr::testing::GraphFixture fixture;
    settings[std::string("n_inputs")]    = 2U;
    settings[std::string("sample_rate")] = 1000.f;
    auto& sourceA                        = fixture.emplace<gr::testing::TagSource<float>>({{"values", channelA}, {"n_samples_max", static_cast<gr::Size_t>(channelA.size())}});
    auto& sourceB                        = fixture.emplace<gr::testing::TagSource<float>>({{"values", channelB}, {"n_samples_max", static_cast<gr::Size_t>(channelB.size())}});
    auto& trigger                        = fixture.emplace<PatternTrigger<float>>(std::move(settings));
    auto& collector                      = fixture.emplace<EventTap>();
    boost::ut::expect(fixture.graph.connect(sourceA, gr::PortDefinition{0UZ}, trigger, gr::PortDefinition{0UZ, 0UZ}).has_value());
    boost::ut::expect(fixture.graph.connect(sourceB, gr::PortDefinition{0UZ}, trigger, gr::PortDefinition{0UZ, 1UZ}).has_value());
    boost::ut::expect(fixture.connect<"evtOut", "evtIn">(trigger, collector).has_value());
    boost::ut::expect(fixture.run().has_value());
    return Outcome{trigger.n_triggers.value, trigger.n_rejected.value, collector.size(), collector.times()};
}

const boost::ut::suite<"PatternTrigger"> _patternTrigger = [] {
    using namespace boost::ut;

    "a pattern fires where both channels are in the named state"_test = [] {
        const auto found = judge({0.f, 5.f, 5.f, 0.f}, {5.f, 0.f, 0.f, 5.f}, {{"pattern", std::string("10")}, {"thresholds", std::vector<float>{2.5f}}, {"hysteresis", 0.5f}});
        expect(eq(found.triggers, 1U)) << "the pattern becomes true once, however long it then stands";
    };

    "a don't-care channel is not consulted"_test = [] {
        const auto both = judge({0.f, 5.f, 0.f}, {0.f, 0.f, 0.f}, {{"pattern", std::string("10")}, {"thresholds", std::vector<float>{2.5f}}, {"hysteresis", 0.5f}});
        const auto only = judge({0.f, 5.f, 0.f}, {0.f, 5.f, 0.f}, {{"pattern", std::string("1X")}, {"thresholds", std::vector<float>{2.5f}}, {"hysteresis", 0.5f}});
        expect(eq(both.triggers, 1U));
        expect(eq(only.triggers, 1U)) << "the second channel is high where '10' would have refused it";
    };

    "logic any fires when one channel's term holds, where all would not"_test = [] {
        const gr::property_map settings{{"pattern", std::string("11")}, {"thresholds", std::vector<float>{2.5f}}, {"hysteresis", 0.5f}};
        gr::property_map       any = settings;
        any["logic"]               = std::string("any");
        expect(eq(judge({0.f, 5.f, 0.f}, {0.f, 0.f, 0.f}, settings).triggers, 0U));
        expect(eq(judge({0.f, 5.f, 0.f}, {0.f, 0.f, 0.f}, any).triggers, 1U));
    };

    "an unknown line matches neither 1 nor 0"_test = [] {
        // channel A starts inside the band 2..3, so it is unknown, then becomes high: '01' never stands
        const auto found = judge({2.5f, 5.f}, {5.f, 5.f}, {{"pattern", std::string("01")}, {"thresholds", std::vector<float>{2.5f}}, {"hysteresis", 1.f}});
        expect(eq(found.triggers, 0U)) << "an unknown A is not a low A";
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
        const auto refused = judge({0.f, 5.f, 5.f, 0.f}, {5.f, 0.f, 0.f, 5.f}, //
            {
                {"pattern", std::string("10")},
                {"thresholds", std::vector<float>{2.5f}},
                {"hysteresis", 0.5f},
                {"when", std::string("holds")},
                {"duration_min_samples", 4U},
            });
        expect(eq(refused.triggers, 0U));
        expect(eq(refused.rejected, 1U)) << "it stood, but not for the setup time";

        const auto accepted = judge({0.f, 5.f, 5.f, 0.f}, {5.f, 0.f, 0.f, 5.f}, //
            {
                {"pattern", std::string("10")},
                {"thresholds", std::vector<float>{2.5f}},
                {"hysteresis", 0.5f},
                {"when", std::string("holds")},
                {"duration_min_samples", 2U},
            });
        expect(eq(accepted.triggers, 1U)) << "two samples is the setup time it was given";
    };

    "skew_samples lets a term that has just left its state still count, so channels that differ in delay coincide"_test = [] {
        const std::vector<float> a{0.f, 5.f, 5.f, 0.f, 0.f, 0.f};
        const std::vector<float> b{0.f, 0.f, 0.f, 0.f, 5.f, 5.f};
        const gr::property_map   both{{"pattern", std::string("11")}, {"thresholds", std::vector<float>{2.5f}}, {"hysteresis", 0.5f}};
        const auto               withSkew = [&](gr::Size_t skew) {
            gr::property_map settings             = both;
            settings[std::string("skew_samples")] = skew;
            return judge(a, b, settings).triggers;
        };
        expect(eq(withSkew(0U), 0U)) << "A was last high at sample 2, B first high at 4: never the same sample";
        expect(eq(withSkew(1U), 0U)) << "at sample 4 A left the state two samples ago, one is not enough";
        expect(eq(withSkew(2U), 1U)) << "two samples is exactly the distance from A's last high sample to B's first";
    };

    "a gap abandons a pattern in progress and the part after it does not inherit the part before"_test = [] {
        const gr::property_map values{{"z", 0.f}, {"h", 5.f}};
        const gr::property_map gap{{"G", gr::property_map{{std::string(gr::tag::kNDroppedSamples.key()), gr::Size_t{7U}}}}};
        const auto             run = [&](std::string scriptA) {
            gr::testing::GraphFixture fixture;
            auto&                     sourceA = fixture.emplace<MarbleSource<float>>({{"script", std::move(scriptA)}, {"sample_values", values}, {"sample_tags", gap}});
            auto&                     sourceB = fixture.emplace<MarbleSource<float>>({{"script", std::string("z z z z z z z |")}, {"sample_values", values}});
            auto&                     trigger = fixture.emplace<PatternTrigger<float>>({
                {"n_inputs", 2U},
                {"pattern", std::string("10")},
                {"thresholds", std::vector<float>{2.5f}},
                {"hysteresis", 0.5f},
                {"sample_rate", 1000.f},
                {"when", std::string("holds")},
                {"duration_min_samples", 4U},
            });
            auto&                     tap     = fixture.emplace<EventTap>();
            expect(fixture.graph.connect(sourceA, gr::PortDefinition{std::string("out")}, trigger, gr::PortDefinition{std::string("in#0")}).has_value());
            expect(fixture.graph.connect(sourceB, gr::PortDefinition{std::string("out")}, trigger, gr::PortDefinition{std::string("in#1")}).has_value());
            expect(fixture.graph.connect(trigger, gr::PortDefinition{std::string("evtOut")}, tap, gr::PortDefinition{std::string("evtIn")}).has_value());
            expect(fixture.run().has_value());
            return std::array{trigger.n_triggers.value, trigger.n_abandoned.value, trigger.n_rejected.value};
        };
        expect(run("z h h h h h z |") == std::array<gr::Size_t, 3UZ>{1U, 0U, 0U}) << "A stands for five samples, four are needed";
        expect(run("z h h G:h h h z |") ==
               std::array<gr::Size_t, 3UZ>{
                   0U,
                   1U,
                   1U,
               })
            << "two samples stood before the gap and were abandoned; the three after it fall short of four";
    };

    "each channel may carry its own threshold"_test = [] {
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

    "a pattern that held is decided at the sample that completed the setup time, about its first matching sample"_test = [] {
        const gr::property_map values{{"z", 0.f}, {"h", 5.f}};
        const gr::property_map tags{{"T", gr::property_map{{std::string(gr::tag::kTriggerName.key()), std::string("acq")}, {std::string(gr::tag::kTriggerTime.key()), kStreamStart}, {std::string(gr::tag::kTriggerOffset.key()), 0.f}, {std::string("sample_rate"), 1000.f}}}};

        auto report = [&](std::string when, gr::Size_t setup) {
            gr::testing::GraphFixture fixture;
            // pattern "10": A high and B low, true at samples 1 and 2
            auto& sourceA   = fixture.emplace<MarbleSource<float>>({{"script", std::string("T:z h h z z |")}, {"sample_values", values}, {"sample_tags", tags}});
            auto& sourceB   = fixture.emplace<MarbleSource<float>>({{"script", std::string("T:z z z z z |")}, {"sample_values", values}, {"sample_tags", tags}});
            auto& trigger   = fixture.emplace<PatternTrigger<float>>({{"n_inputs", 2U}, {"pattern", std::string("10")}, {"thresholds", std::vector<float>{2.5f}}, {"hysteresis", 0.5f}, {"sample_rate", 1000.f}, //
                  {"when", std::move(when)}, {"duration_min_samples", setup}, {"trigger_name", std::string("held")}});
            auto& collector = fixture.emplace<EventTap>();
            expect(fixture.graph.connect(sourceA, gr::PortDefinition{std::string("out")}, trigger, gr::PortDefinition{std::string("in#0")}).has_value());
            expect(fixture.graph.connect(sourceB, gr::PortDefinition{std::string("out")}, trigger, gr::PortDefinition{std::string("in#1")}).has_value());
            expect(fixture.graph.connect(trigger, gr::PortDefinition{std::string("evtOut")}, collector, gr::PortDefinition{std::string("evtIn")}).has_value());
            expect(fixture.run().has_value());
            return collector._events;
        };
        const auto timeOf = [](const gr::property_map_view& map) {
            const auto* const time = map.get_if<std::uint64_t>(gr::tag::kTriggerTime.key());
            return time != nullptr ? std::optional{*time} : std::nullopt;
        };

        const auto held = report("holds", 2U);
        expect(eq(held.size(), 1UZ)) << fatal;
        const gr::property_map_view event{held[0]};
        const auto                  meta = event.get_if<gr::property_map_view>(gr::tag::kTriggerMetaInfo.key());
        expect(meta.has_value()) << fatal;
        expect(timeOf(event) == std::optional<std::uint64_t>{kStreamStart + 2'000'000U}) << "the second matching sample, sample 2 at 1 kHz from 1 s";
        expect(timeOf(*meta) == std::optional<std::uint64_t>{kStreamStart + 1'000'000U}) << "the first matching sample, one sample earlier";
        const auto referenceName = meta->get_if<std::string_view>(gr::tag::kTriggerName.key());
        expect(referenceName.has_value() and *referenceName == "held");
        expect(!gr::trigger_test::carriesSource(held[0]));

        const auto entered = report("enters", 0U);
        expect(eq(entered.size(), 1UZ)) << fatal;
        expect(timeOf(gr::property_map_view{entered[0]}) == std::optional<std::uint64_t>{kStreamStart + 1'000'000U}) << "entering is decided at the first matching sample";
        const auto enteredMeta = gr::property_map_view{entered[0]}.get_if<gr::property_map_view>(gr::tag::kTriggerMetaInfo.key());
        expect(!enteredMeta.has_value()) << "and about no earlier moment: there is nothing to put in trigger_meta_info";
        expect(!gr::trigger_test::carriesSource(entered[0]));
    };

    "what the pattern sees, drawn"_test = [] {
        const gr::property_map values{{"z", 0.f}, {"h", 5.f}};
        const gr::property_map tags{{"T", gr::property_map{{std::string(gr::tag::kTriggerName.key()), std::string("acq")}, //
                                              {std::string(gr::tag::kTriggerTime.key()), std::uint64_t{1'000'000'000U}},   //
                                              {std::string(gr::tag::kTriggerOffset.key()), 0.f},                           //
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
                                      {"duration_min_samples", setup},                                      //
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
} // namespace qa_pattern_trigger

namespace qa_clocked_triggers {
using namespace gr::blocks::trigger;
using gr::testing::TagSource;
using gr::trigger_test::EventTap;

/** a square clock, one sample low and one high, so an edge sits on every second sample */
[[nodiscard]] std::vector<float> clockOf(std::size_t nEdges) {
    std::vector<float> wave;
    for (std::size_t i = 0UZ; i < nEdges; ++i) {
        wave.push_back(0.f);
        wave.push_back(5.f);
    }
    return wave;
}

constexpr std::size_t kNoGap = std::numeric_limits<std::size_t>::max();

/** `gapAt`: the data sample after which samples were dropped */
template<typename TBlock>
[[nodiscard]] std::size_t runWith(std::vector<float> data, std::vector<float> clock, gr::property_map settings, gr::trigger_test::Kept<TBlock>* kept = nullptr, std::size_t gapAt = kNoGap) {
    auto             owner    = std::make_unique<gr::testing::GraphFixture<>>();
    auto&            fixture  = *owner;
    const gr::Size_t nSamples = static_cast<gr::Size_t>(std::min(data.size(), clock.size()));
    auto&            dataSrc  = fixture.template emplace<TagSource<float>>({{"n_samples_max", nSamples}, {"values", std::move(data)}, {"mark_tag", false}});
    auto&            clkSrc   = fixture.template emplace<TagSource<float>>({{"n_samples_max", nSamples}, {"values", std::move(clock)}, {"mark_tag", false}});
    if (gapAt != kNoGap) {
        dataSrc._tags.push_back({gapAt, gr::property_map{{std::string(gr::tag::kNDroppedSamples.key()), gr::Size_t{7U}}}});
    }
    auto& block = fixture.template emplace<TBlock>(std::move(settings));
    auto& sink  = fixture.template emplace<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_ONE>>();
    auto& tap   = fixture.template emplace<EventTap>();

    boost::ut::expect(fixture.graph.connect(dataSrc, gr::PortDefinition{std::string("out")}, block, gr::PortDefinition{std::string("in")}).has_value());
    boost::ut::expect(fixture.graph.connect(clkSrc, gr::PortDefinition{std::string("out")}, block, gr::PortDefinition{std::string("clk_in")}).has_value());
    boost::ut::expect(fixture.template connect<"out", "in">(block, sink).has_value());
    boost::ut::expect(fixture.template connect<"evtOut", "evtIn">(block, tap).has_value());
    boost::ut::expect(fixture.run().has_value());

    if (kept != nullptr) {
        kept->block   = std::addressof(block);
        kept->fixture = std::move(owner);
    }
    return tap.size();
}

const boost::ut::suite<"SerialPatternTrigger"> _serialPatternTrigger = [] {
    using namespace boost::ut;

    const gr::property_map settings{{"threshold", 2.5f}, {"hysteresis", 1.f}, {"clock_threshold", 2.5f}, {"clock_edge", std::string("rising")}, {"sample_rate", 1000.f}};

    "the word the clock spelled out is found"_test = [&] {
        const std::vector<float> data{5.f, 5.f, 0.f, 0.f, 5.f, 5.f, 5.f, 5.f};
        gr::property_map         withPattern = settings;
        withPattern[std::string("pattern")]  = std::string("1011");

        gr::trigger_test::Kept<SerialPatternTrigger<float>> block;
        const std::size_t                                   found = runWith<SerialPatternTrigger<float>>(data, clockOf(4UZ), withPattern, &block);

        expect(eq(found, 1UZ)) << "the four bits spell the pattern exactly once";
        expect(block != nullptr && block->n_bits.value == 4U) << "one bit per clock edge";
    };

    "a word that does not match is not reported"_test = [&] {
        const std::vector<float> data{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        gr::property_map         withPattern = settings;
        withPattern[std::string("pattern")]  = std::string("1011");

        expect(eq(runWith<SerialPatternTrigger<float>>(data, clockOf(4UZ), withPattern), 0UZ));
    };

    "an X accepts whichever bit arrives"_test = [&] {
        const std::vector<float> data{5.f, 5.f, 0.f, 0.f, 5.f, 5.f, 0.f, 0.f};
        gr::property_map         withPattern = settings;
        withPattern[std::string("pattern")]  = std::string("1X1X");

        expect(eq(runWith<SerialPatternTrigger<float>>(data, clockOf(4UZ), withPattern), 1UZ)) << "1?1? matches 1 0 1 0";
    };

    "a gap restarts the register: the bits before it do not join the bits after it"_test = [&] {
        const std::vector<float> data{5.f, 5.f, 0.f, 0.f, 5.f, 5.f, 5.f, 5.f};
        gr::property_map         withPattern = settings;
        withPattern[std::string("pattern")]  = std::string("1011");

        gr::trigger_test::Kept<SerialPatternTrigger<float>> control;
        expect(eq(runWith<SerialPatternTrigger<float>>(data, clockOf(4UZ), withPattern, &control), 1UZ)) << "without a gap the four bits spell the word";
        expect(control != nullptr && control->n_gaps.value == 0U);

        gr::trigger_test::Kept<SerialPatternTrigger<float>> gapped;
        expect(eq(runWith<SerialPatternTrigger<float>>(data, clockOf(4UZ), withPattern, &gapped, 4UZ), 0UZ)) << "the bits 1,0 stood before the gap, 1,1 follow it: no word";
        expect(gapped != nullptr && gapped->n_gaps.value == 1U) << "counted";
    };

    "the data line passes through untouched"_test = [&] {
        const std::vector<float> data{5.f, 5.f, 0.f, 0.f};
        gr::property_map         withPattern = settings;
        withPattern[std::string("pattern")]  = std::string("10");

        gr::testing::GraphFixture fixture;
        auto&                     dataSrc = fixture.emplace<TagSource<float>>({{"n_samples_max", 4U}, {"values", data}, {"mark_tag", false}});
        auto&                     clkSrc  = fixture.emplace<TagSource<float>>({{"n_samples_max", 4U}, {"values", clockOf(2UZ)}, {"mark_tag", false}});
        auto&                     block   = fixture.emplace<SerialPatternTrigger<float>>(withPattern);
        auto&                     sink    = fixture.emplace<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_ONE>>();
        expect(fixture.graph.connect(dataSrc, gr::PortDefinition{std::string("out")}, block, gr::PortDefinition{std::string("in")}).has_value());
        expect(fixture.graph.connect(clkSrc, gr::PortDefinition{std::string("out")}, block, gr::PortDefinition{std::string("clk_in")}).has_value());
        expect(fixture.connect<"out", "in">(block, sink).has_value());
        expect(fixture.run().has_value());

        expect(eq(sink._samples.size(), 4UZ)) << "what is triggered on is the word, not the signal";
    };
};

const boost::ut::suite<"SetupHoldTrigger"> _setupHoldTrigger = [] {
    using namespace boost::ut;

    const gr::property_map settings{{"threshold", 2.5f}, {"hysteresis", 1.f}, {"clock_threshold", 2.5f}, {"clock_edge", std::string("rising")}, //
        {"setup_samples", 2U}, {"hold_samples", 2U}, {"sample_rate", 1000.f}};

    "data settled well before the edge is no violation"_test = [&] {
        const std::vector<float>                        data{5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f};
        gr::trigger_test::Kept<SetupHoldTrigger<float>> block;

        expect(eq(runWith<SetupHoldTrigger<float>>(data, clockOf(5UZ), settings, &block), 0UZ));
        expect(block != nullptr && block->n_checks.value == 5U) << "every edge was judged";
    };

    "data that moves inside the window is a violation"_test = [&] {
        const std::vector<float>                        data{0.f, 0.f, 0.f, 0.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f};
        gr::trigger_test::Kept<SetupHoldTrigger<float>> block;

        const std::size_t found = runWith<SetupHoldTrigger<float>>(data, clockOf(5UZ), settings, &block);
        expect(ge(found, 1UZ)) << "the change sits in the setup half of one edge and the hold half of another";
        expect(block != nullptr && block->n_violations.value == found);
    };

    "a violation is reported once its hold half has arrived, not before"_test = [&] {
        const std::vector<float>                        data{0.f, 0.f, 0.f, 0.f, 5.f, 5.f};
        gr::trigger_test::Kept<SetupHoldTrigger<float>> block;

        const std::size_t found = runWith<SetupHoldTrigger<float>>(data, clockOf(3UZ), settings, &block);
        expect(block != nullptr) << "the block outlives the run";
        expect(le(found, block->n_checks.value)) << "an edge whose hold samples never arrived is not judged at all";
    };

    "a gap drops the windows in progress: the change before it proves nothing about the edges after it"_test = [&] {
        const std::vector<float> data{0.f, 0.f, 0.f, 0.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f};

        gr::trigger_test::Kept<SetupHoldTrigger<float>> control;
        expect(ge(runWith<SetupHoldTrigger<float>>(data, clockOf(5UZ), settings, &control), 1UZ)) << "the change at 4 sits inside the windows of the edges at 3 and 5";

        gr::trigger_test::Kept<SetupHoldTrigger<float>> gapped;
        expect(eq(runWith<SetupHoldTrigger<float>>(data, clockOf(5UZ), settings, &gapped, 5UZ), 0UZ)) << "a gap at 5 forgets the change at 4 and the open hold window of the edge at 3";
        expect(gapped != nullptr && gapped->n_gaps.value == 1U);
    };

    "the window and the violation in it, drawn"_test = [&] {
        const std::vector<float>  data{0.f, 0.f, 0.f, 0.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f};
        gr::testing::GraphFixture fixture;
        auto&                     dataSrc = fixture.emplace<TagSource<float>>({{"n_samples_max", 10U}, {"values", data}, {"mark_tag", false}});
        auto&                     clkSrc  = fixture.emplace<TagSource<float>>({{"n_samples_max", 10U}, {"values", clockOf(5UZ)}, {"mark_tag", false}});
        auto&                     block   = fixture.emplace<SetupHoldTrigger<float>>(settings);
        auto&                     tap     = fixture.emplace<EventTap>();
        expect(fixture.graph.connect(dataSrc, gr::PortDefinition{std::string("out")}, block, gr::PortDefinition{std::string("in")}).has_value());
        expect(fixture.graph.connect(clkSrc, gr::PortDefinition{std::string("out")}, block, gr::PortDefinition{std::string("clk_in")}).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(block, tap).has_value());
        expect(fixture.run().has_value());

        gr::testing::MarbleDiagram diagram{"SetupHoldTrigger: the data moves at sample 4, inside two clock windows"};
        diagram.unit   = "sample";
        auto& clockRow = diagram.row("clk");
        for (std::uint64_t at = 1U; at < 10U; at += 2U) {
            clockRow.at(at, "edge");
        }
        clockRow.completes();
        diagram.row("in").at(4U, "data changes").completes();
        diagram.condition(std::format("SetupHoldTrigger(setup = 2, hold = 2), {} of {} edges violated", block.n_violations.value, block.n_checks.value));
        auto& out = diagram.row("evtOut");
        for (std::size_t i = 0UZ; i < tap.size(); ++i) {
            out.at(3U + 2U * i, "violation");
        }
        out.completes();
        diagram.print();

        expect(ge(block.n_violations.value, 1U));
    };
};
} // namespace qa_clocked_triggers

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
