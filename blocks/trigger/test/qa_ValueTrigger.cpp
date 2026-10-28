#include <boost/ut.hpp>

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/test/EventMarbles.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/trigger/ValueTrigger.hpp>

#include "Marble.hpp"
#include "TriggerTest.hpp"

namespace qa_value_trigger {
using namespace gr::blocks::trigger;
using gr::trigger_test::EventTap;
using gr::trigger_test::kNsPerSample;
using gr::trigger_test::kStreamStart;

/** a time in samples is (ns - kStreamStart) / kNsPerSample, so every expectation is a hand-computed crossing */
constexpr double kTolerance = 1e-4; /// samples: float interpolation, rounded to whole ns

struct Run {
    std::vector<gr::property_map> events;
    std::vector<gr::property_map> tags; /// the trigger tags the stream carried
    gr::Size_t                    triggers  = 0U;
    gr::Size_t                    rejected  = 0U;
    gr::Size_t                    abandoned = 0U;
    std::string                   blockName; /// the trigger block's `name`, the default trigger_name
    std::vector<std::string>      errors;    /// the error messages the block sent
};

template<ValueCondition condition, typename T = float>
[[nodiscard]] Run measured(std::vector<T> samples, gr::property_map settings, std::vector<gr::testing::OwningTag> tags = {}) {
    gr::testing::GraphFixture fixture;
    settings[std::string("sample_rate")] = 1000.f;
    auto& source                         = fixture.emplace<gr::testing::TagSource<T>>({
        {"values", samples},
        {"n_samples_max", static_cast<gr::Size_t>(samples.size())},
        {"mark_tag", false},
    });
    source._tags.push_back({0UZ, gr::property_map{{std::string(gr::tag::kTriggerTime.key()), kStreamStart}, {std::string(gr::tag::kTriggerOffset.key()), 0.f}}});
    for (gr::testing::OwningTag& tag : tags) {
        source._tags.push_back(std::move(tag));
    }
    auto& trigger = fixture.emplace<ValueTrigger<T, condition>>(std::move(settings));
    auto& sink    = fixture.emplace<gr::testing::TagSink<T, gr::testing::ProcessFunction::USE_PROCESS_ONE>>();
    auto& tap     = fixture.emplace<EventTap>();
    boost::ut::expect(fixture.connect<"out", "in">(source, trigger).has_value());
    boost::ut::expect(fixture.connect<"out", "in">(trigger, sink).has_value());
    boost::ut::expect(fixture.connect<"evtOut", "evtIn">(trigger, tap).has_value());
    gr::trigger_test::ErrorMessages sent{fixture.scheduler()};
    boost::ut::expect(fixture.run().has_value());
    std::vector<gr::property_map> triggerTags;
    for (const gr::testing::OwningTag& tag : sink._tags) {
        if (tag.map.contains(std::string(gr::tag::kTriggerName.key()))) {
            triggerTags.push_back(tag.map);
        }
    }
    return {
        tap._events,
        std::move(triggerTags),
        trigger.n_triggers.value,
        trigger.n_rejected.value,
        trigger.n_abandoned.value,
        std::string(trigger.name.value),
        sent.texts(),
    };
}

/** a trigger tuple's time -- trigger_time + trigger_offset -- in samples since sample 0, or NaN when it does not carry one */
[[nodiscard]] double samplesOf(const gr::property_map_view& tuple) {
    const auto ns     = tuple.get_if<std::uint64_t>(gr::tag::kTriggerTime.key());
    const auto offset = tuple.get_if<float>(gr::tag::kTriggerOffset.key());
    return ns ? (static_cast<double>(*ns) - static_cast<double>(kStreamStart) + (offset ? static_cast<double>(*offset) * 1e9 : 0.)) / kNsPerSample : std::numeric_limits<double>::quiet_NaN();
}

[[nodiscard]] std::optional<gr::property_map_view> metaOf(const gr::property_map& event) { return gr::property_map_view{event}.get_if<gr::property_map_view>(gr::tag::kTriggerMetaInfo.key()); }

/** the decision (root trigger_time) and, for a delayed decision, the moment it is about (the tuple in trigger_meta_info) with the polarity name of its edge */
struct Expected {
    double                moment;
    std::optional<double> reference     = std::nullopt;
    std::string_view      referenceName = {};
    std::string_view      label         = {}; /// `class` in trigger_meta_info; empty: none
    bool                  clipped       = false;
};

[[nodiscard]] bool flagged(const std::optional<gr::property_map_view>& meta, std::string_view flag) {
    const auto flags = meta ? meta->get_if<gr::property_map_view>(gr::tag::kAcquisitionFlags.key()) : std::nullopt;
    const auto value = flags ? flags->get_if<bool>(flag) : nullptr;
    return value != nullptr && *value;
}

[[nodiscard]] std::string_view textOf(const std::optional<gr::property_map_view>& meta, std::string_view key) {
    const auto text = meta ? meta->get_if<std::string_view>(key) : std::nullopt;
    return text ? *text : std::string_view{};
}

void expectEvents(const Run& found, const std::vector<Expected>& expected) {
    using namespace boost::ut;
    expect(eq(found.events.size(), expected.size())) << "number of events";
    for (std::size_t i = 0UZ; i < std::min(found.events.size(), expected.size()); ++i) {
        const gr::property_map_view event{found.events[i]};
        expect(approx(samplesOf(event), expected[i].moment, kTolerance)) << "event " << i << " decision (trigger_time + trigger_offset)";
        const auto offset = event.get_if<float>(gr::tag::kTriggerOffset.key());
        expect(offset != nullptr and *offset >= 0.f and *offset < 1e-9f) << "event " << i << " trigger_offset is the sub-ns residual";
        expect(!event.contains(gr::tag::kTriggerStopTime.key())) << "event " << i << " carries no trigger_stop_time";
        const auto meta = metaOf(found.events[i]);
        expect(!gr::trigger_test::carriesSource(found.events[i])) << "event " << i << " carries no source";
        expect(meta.has_value()) << "event " << i << " carries the comparators' boundaries and the estimator in trigger_meta_info" << fatal;
        expect(eq(meta->contains(gr::tag::kTriggerTime.key()), expected[i].reference.has_value())) << "event " << i << " holds a reference tuple only for a delayed decision";
        expect(eq(textOf(meta, "class"), expected[i].label)) << "event " << i << " class label";
        expect(eq(flagged(meta, "clipped"), expected[i].clipped)) << "event " << i << " clipped flag";
        if (expected[i].reference) {
            expect(approx(samplesOf(*meta), *expected[i].reference, kTolerance)) << "event " << i << " reference (trigger_meta_info.trigger_time + trigger_offset)";
            const auto name = meta->get_if<std::string_view>(gr::tag::kTriggerName.key());
            expect(name.has_value() and *name == expected[i].referenceName) << "event " << i << " reference edge name, expected " << expected[i].referenceName;
        }
    }
}

[[nodiscard]] gr::property_map with(gr::property_map settings, std::string_view comparison, double minimum, double maximum) {
    settings[std::string("duration_condition")] = std::string(comparison);
    settings[std::string("duration_min")]       = minimum;
    settings[std::string("duration_max")]       = maximum;
    return settings;
}

const boost::ut::suite<"ValueTrigger"> _valueTrigger = [] {
    using namespace boost::ut;

    "the runtime mode reaches the same decision as the fixed one"_test = [] {
        const auto fixed   = measured<ValueCondition::level>({0.f, 3.f, 0.f, 3.f, 0.f}, {{"threshold", 1.f}, {"hysteresis", 0.5f}});
        const auto runtime = measured<ValueCondition::automatic>({0.f, 3.f, 0.f, 3.f, 0.f}, {{"threshold", 1.f}, {"hysteresis", 0.5f}, {"condition_mode", std::string("level")}});
        expect(eq(fixed.triggers, runtime.triggers)) << "automatic delegates to the setting and agrees with the template parameter";
    };

    "a slew-rate trigger fires where the signal changes faster than it may"_test = [] {
        const auto found = measured<ValueCondition::slewRate>({0.f, 0.1f, 0.2f, 0.3f, 5.3f, 5.4f, 5.5f}, //
            {{"slew_samples", 1U}, {"slew_max_rate", 1000.f}, {"threshold", 100.f}, {"hysteresis", 1.f}});

        expect(eq(found.triggers, 1U)) << "one trigger for the excursion, not one per sample of it";
        expect(eq(found.events.size(), 1UZ)) << fatal;
        expect(approx(samplesOf(gr::property_map_view{found.events[0]}), 4., kTolerance)) << "decided at sample 4, where 5 per sample at 1 kHz first exceeds 1000 per second";
    };

    "a slew-rate trigger stays quiet while the change is within its band"_test = [] {
        const auto found = measured<ValueCondition::slewRate>({0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f}, //
            {{"slew_samples", 1U}, {"slew_max_rate", 1000.f}, {"threshold", 100.f}, {"hysteresis", 1.f}});

        expect(eq(found.triggers, 0U)) << "0.1 per sample at 1 kHz is 100 per second, well inside the limit";
    };

    "what the conditions look like, drawn"_test = [] {
        const gr::property_map values{{"z", 0.f}, {"m", 2.f}, {"h", 5.f}};
        const gr::property_map tags{{"T", gr::property_map{{std::string(gr::tag::kTriggerName.key()), std::string("acq")}, //
                                              {std::string(gr::tag::kTriggerTime.key()), std::uint64_t{1'000'000'000U}},   //
                                              {std::string(gr::tag::kTriggerOffset.key()), 0.f},                           //
                                              {std::string("sample_rate"), 1000.f}}}};
        const std::string      script = "T:z h h z z h h h h z z m m z z h z |";

        auto judge = [&](gr::property_map settings) {
            gr::testing::GraphFixture fixture;
            auto&                     source  = fixture.emplace<MarbleSource<float>>({{"script", script}, {"sample_values", values}, {"sample_tags", tags}});
            auto&                     trigger = fixture.emplace<ValueTrigger<float, ValueCondition::automatic>>(std::move(settings));
            auto&                     sink    = fixture.emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
            auto&                     edges   = fixture.emplace<EventTap>();
            expect(fixture.connect<"out", "in">(source, trigger).has_value());
            expect(fixture.connect<"out", "in">(trigger, sink).has_value());
            expect(fixture.connect<"evtOut", "evtIn">(trigger, edges).has_value());
            expect(fixture.run().has_value());
            return edges.dated();
        };

        const auto level = judge({{"condition_mode", std::string("level")}, {"threshold", 1.f}, {"hysteresis", 0.5f}, {"sample_rate", 1000.f}, {"trigger_name", std::string("level")}});
        const auto width = judge({{"condition_mode", std::string("pulseWidth")}, {"threshold", 1.f}, {"hysteresis", 0.5f}, {"duration_condition", std::string("lessEqual")}, {"duration_max", 3.}, {"sample_rate", 1000.f}, {"trigger_name", std::string("short")}});
        const auto runt  = judge({{"condition_mode", std::string("runt")}, {"threshold", 1.f}, {"threshold_upper", 4.f}, {"hysteresis", 0.5f}, {"sample_rate", 1000.f}, {"trigger_name", std::string("runt")}});

        expect(eq(level.size(), 4UZ)) << "three full pulses and the 2.0 plateau rise through the level";
        expect(eq(runt.size(), 1UZ)) << "and one pulse that never reached the upper threshold";

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

const boost::ut::suite<"ValueTrigger conditions, dated"> _specified = [] {
    using namespace boost::ut;
    const gr::property_map edgeAt1{{"threshold", 1.f}, {"hysteresis", 1.f}};  // rising at 1.5, falling at 0.5
    const gr::property_map levelAt1{{"threshold", 1.f}, {"hysteresis", 0.f}}; // high is x >= 1
    const gr::property_map band1to3{{"threshold", 1.f}, {"threshold_upper", 3.f}, {"hysteresis", 0.f}};

    "an edge is dated where the boundary is crossed"_test = [&] {
        // rising boundary 1.5: crossed three quarters of the way from sample 1 (0) to sample 2 (2), and from 5 to 6
        expectEvents(measured<ValueCondition::level>({0.f, 0.f, 2.f, 2.f, 0.f, 0.f, 2.f}, edgeAt1), {{1.75}, {5.75}});
    };

    "a falling edge is dated at the lower boundary"_test = [&] {
        gr::property_map settings = edgeAt1;
        settings["edge"]          = std::string("falling");
        // falling boundary 0.5: from 2 (sample 3) to 0 (sample 4) it is crossed three quarters of the way
        expectEvents(measured<ValueCondition::level>({0.f, 0.f, 2.f, 2.f, 0.f}, settings), {{3.75}});
    };

    "the stream tag carries the decision and no source; a plain crossing's trigger_meta_info holds no reference"_test = [&] {
        const auto found = measured<ValueCondition::level>({0.f, 0.f, 2.f, 2.f, 0.f, 0.f, 2.f}, edgeAt1);
        expect(eq(found.tags.size(), 2UZ));
        for (std::size_t i = 0UZ; i < std::min(found.tags.size(), 2UZ); ++i) {
            const gr::property_map_view stamp{found.tags[i]};
            expect(approx(samplesOf(stamp), i == 0UZ ? 1.75 : 5.75, kTolerance)) << "tag " << i << " is dated by the crossing";
            expect(!gr::trigger_test::carriesSource(found.tags[i]));
            const auto meta = stamp.get_if<gr::property_map_view>(gr::tag::kTriggerMetaInfo.key());
            expect(meta.has_value() and !meta->contains(gr::tag::kTriggerTime.key()));
        }
    };

    "a delayed decision's stream tag holds its reference in trigger_meta_info, without a source"_test = [&] {
        const auto found = measured<ValueCondition::pulseWidth>({0.f, 2.f, 2.f, 2.f, 2.f, 0.f}, with(levelAt1, "greaterEqual", 3., 0.));
        expect(eq(found.tags.size(), 1UZ));
        if (found.tags.empty()) {
            return;
        }
        const gr::property_map_view stamp{found.tags[0]};
        const auto                  meta = stamp.get_if<gr::property_map_view>(gr::tag::kTriggerMetaInfo.key());
        expect(meta.has_value()) << fatal;
        expect(approx(samplesOf(stamp), 4.5, kTolerance)) << "the trailing edge, 4.5 samples in";
        expect(approx(samplesOf(*meta), 0.5, kTolerance)) << "the leading edge";
        expect(approx(samplesOf(stamp) - samplesOf(*meta), 4., kTolerance)) << "the width is the decision minus the reference: 4.5 - 0.5";
        expect(!gr::trigger_test::carriesSource(found.tags[0]));
    };

    "with zero hysteresis a plateau at the level is one rising and one falling edge"_test = [&] {
        gr::property_map settings = levelAt1;
        settings["edge"]          = std::string("both");
        expectEvents(measured<ValueCondition::level>({0.f, 1.f, 1.f, 1.f, 0.f}, settings), {{1.}, {3.}});
    };

    "a stream starting above the level reports no rising edge"_test = [&] { expectEvents(measured<ValueCondition::level>({2.f, 2.f, 2.f}, edgeAt1), {}); };

    "a window reports entering and leaving its closed band"_test = [&] {
        gr::property_map settings = band1to3;
        settings["edge"]          = std::string("both");
        // entry at the level 1 (0 → 2), exit above 3 (2 → 4), re-entry (4 → 2), exit below (2 → 0): each crossed half way
        expectEvents(measured<ValueCondition::window>({0.f, 2.f, 2.f, 4.f, 2.f, 0.f}, settings), {{0.5}, {2.5}, {3.5}, {4.5}});
    };

    "a window's band is closed: a sample at its upper level is inside"_test = [&] {
        gr::property_map settings = band1to3;
        settings["edge"]          = std::string("falling");
        expectEvents(measured<ValueCondition::window>({2.f, 3.f, 3.f, 2.f}, settings), {});
    };

    "a window's dwell test qualifies an exit by the time spent inside"_test = [&] {
        gr::property_map settings = with(band1to3, "greaterEqual", 2., 0.);
        settings["edge"]          = std::string("falling");
        // inside from 0.5 to 1.5 (1 sample, rejected) and from 2.5 to 5.5 (3 samples, reported): decided at the exit, about the entry
        const auto found = measured<ValueCondition::window>({0.f, 2.f, 0.f, 2.f, 2.f, 2.f, 0.f}, settings);
        expectEvents(found, {{5.5, 2.5, "RISING"}});
        expect(eq(found.rejected, 1U));
    };

    "a positive pulse is reported at its trailing edge, its leading edge in the meta-info, and its width compared"_test = [&] {
        const auto found = measured<ValueCondition::pulseWidth>({0.f, 2.f, 2.f, 0.f, 0.f, 2.f, 2.f, 2.f, 2.f, 0.f}, with(levelAt1, "lessEqual", 0., 2.));
        // pulse 1: 0.5 to 2.5, width 2 <= 2; pulse 2: 4.5 to 8.5, width 4, rejected
        expectEvents(found, {{2.5, 0.5, "RISING"}});
        expect(eq(found.rejected, 1U));
    };

    "edge both reports a positive and a negative pulse, the trailing edge of one leading the other"_test = [&] {
        gr::property_map settings = levelAt1;
        settings["edge"]          = std::string("both");
        // high from 0.5 to 2.5, low from 2.5 to 4.5, high from 4.5 to 6.5
        expectEvents(measured<ValueCondition::pulseWidth>({0.f, 2.f, 2.f, 0.f, 0.f, 2.f, 2.f, 0.f}, settings), {{2.5, 0.5, "RISING"}, {4.5, 2.5, "FALLING"}, {6.5, 4.5, "RISING"}});
    };

    "an inverted range is refused: a window whose upper threshold lies below its lower reports nothing"_test = [&] {
        gr::property_map settings{{"threshold", 3.f}, {"threshold_upper", 1.f}, {"hysteresis", 0.f}, {"edge", std::string("both")}};
        expectEvents(measured<ValueCondition::window>({0.f, 2.f, 2.f, 4.f, 2.f, 0.f}, settings), {});
    };

    "a negative pulse mirrors a positive one"_test = [&] {
        gr::property_map settings = with(levelAt1, "inside", 1., 3.);
        settings["edge"]          = std::string("falling");
        expectEvents(measured<ValueCondition::pulseWidth>({2.f, 0.f, 0.f, 2.f, 2.f}, settings), {{2.5, 0.5, "FALLING"}});
    };

    "a pulse cut by a non-finite sample is abandoned, not measured"_test = [&] {
        const auto found = measured<ValueCondition::pulseWidth>({0.f, 2.f, std::numeric_limits<float>::quiet_NaN(), 2.f, 0.f}, levelAt1);
        expectEvents(found, {});
        expect(eq(found.abandoned, 1U));
    };

    "a pulse cut by dropped samples is abandoned, not measured"_test = [&] {
        const gr::testing::OwningTag gap{2UZ, gr::property_map{{std::string(gr::tag::kNDroppedSamples.key()), gr::Size_t{100U}}}};
        const auto                   found = measured<ValueCondition::pulseWidth>({0.f, 2.f, 2.f, 2.f, 0.f}, levelAt1, {gap});
        expectEvents(found, {});
        expect(eq(found.abandoned, 1U));
    };

    "a runt returns across its first threshold without touching the second"_test = [&] {
        gr::property_map settings{{"threshold", 1.f}, {"threshold_upper", 4.f}, {"hysteresis", 0.f}};
        // the first pulse reaches 5 and the third touches 4 exactly, both disqualified; the second rises to 2 only
        const auto found = measured<ValueCondition::runt>({0.f, 5.f, 0.f, 2.f, 0.f, 4.f, 0.f}, settings);
        expectEvents(found, {{3.5, 2.5, "RISING"}});
        expect(eq(found.rejected, 0U)) << "n_rejected counts duration rejects only: a pulse that touched the second threshold is no runt, not a rejection";
    };

    "a negative runt falls across the upper threshold and returns without touching the lower"_test = [&] {
        gr::property_map settings{{"threshold", 1.f}, {"threshold_upper", 4.f}, {"hysteresis", 0.f}, {"edge", std::string("falling")}};
        expectEvents(measured<ValueCondition::runt>({5.f, 3.f, 5.f}, settings), {{1.5, 0.5, "FALLING"}});
    };

    "a rising transition is timed from one threshold to the other"_test = [&] {
        // 1 is crossed two thirds of the way from 0 to 1.5, 3 half way from 2.5 to 3.5: 2.5 - 2/3 > 1.5
        const auto found = measured<ValueCondition::transition>({0.f, 1.5f, 2.5f, 3.5f, 0.f, 4.f}, with(band1to3, "greater", 1.5, 0.));
        expectEvents(found, {{2.5, 2. / 3., "RISING"}});
        expect(eq(found.rejected, 1U)) << "the jump from 0 to 4 crosses both in 0.5 samples, too fast";
    };

    "a falling transition mirrors a rising one"_test = [&] {
        gr::property_map settings = with(band1to3, "greater", 1.5, 0.);
        settings["edge"]          = std::string("falling");
        expectEvents(measured<ValueCondition::transition>({4.f, 2.5f, 1.5f, 0.5f}, settings), {{2.5, 2. / 3., "FALLING"}});
    };

    "a transition that turns back before the second threshold is abandoned"_test = [&] {
        const auto found = measured<ValueCondition::transition>({0.f, 2.f, 0.f}, band1to3);
        expectEvents(found, {{1.5, 0.5, "RISING", "reversed"}});
        expect(eq(found.abandoned, 1U)) << "a reversal is classified and still counted";
    };

    "a level timeout is decided at its deadline, once per episode, about the crossing that began it"_test = [&] {
        // high from 0.5; >= 3.5 samples first holds at sample 4
        expectEvents(measured<ValueCondition::levelTimeout>(
                         {
                             0.f,
                             2.f,
                             2.f,
                             2.f,
                             2.f,
                             2.f,
                             0.f,
                         },
                         with(levelAt1, "greaterEqual", 3.5, 0.)),
            {{4., 0.5, "RISING", "ongoing"}});
    };

    "a strict level timeout waits until the duration is exceeded"_test = [&] { expectEvents(measured<ValueCondition::levelTimeout>({0.f, 2.f, 2.f, 2.f, 2.f, 2.f, 0.f}, with(levelAt1, "greater", 3.5, 0.)), {{4., 0.5, "RISING", "ongoing"}}); };

    "an edge timeout is armed by an edge and decided when no next one comes"_test = [&] { expectEvents(measured<ValueCondition::edgeTimeout>({0.f, 2.f, 0.f, 0.f, 0.f, 0.f}, with(levelAt1, "greaterEqual", 3., 0.)), {{3.5, 0.5, "RISING"}}); };

    "an interval compares consecutive rising edges, decided at the second, about the first"_test = [&] {
        // rising edges at 0.5, 3.5 and 5.5: 3 lies in [3, 5], 2 does not
        const auto found = measured<ValueCondition::interval>({0.f, 2.f, 0.f, 0.f, 2.f, 0.f, 2.f}, with(levelAt1, "inside", 3., 5.));
        expectEvents(found, {{3.5, 0.5, "RISING"}});
        expect(eq(found.rejected, 1U));
    };

    "an upstream tag's trigger_name does not rename the trigger: the configured name, else the block's name, stays"_test = [&] {
        const auto check = [&](const gr::property_map& settings, std::string_view configuredName) {
            const gr::testing::OwningTag upstream{0UZ, gr::property_map{{std::string(gr::tag::kTriggerName.key()), std::string("upstream")}, {std::string(gr::tag::kTriggerTime.key()), kStreamStart}, {std::string(gr::tag::kTriggerOffset.key()), 0.f}}};
            const auto                   found = measured<ValueCondition::level>({0.f, 0.f, 2.f, 2.f, 0.f}, settings, {upstream});
            expect(eq(found.events.size(), 1UZ)) << fatal;
            const std::string_view expectedName = configuredName.empty() ? std::string_view{found.blockName} : configuredName;
            const auto             eventName    = gr::property_map_view{found.events[0]}.get_if<std::string_view>(gr::tag::kTriggerName.key());
            expect(eventName.has_value() and *eventName == expectedName) << "the event is named " << expectedName << ", got " << (eventName ? *eventName : std::string_view{"(none)"});
            const auto namedLike = [&](std::string_view name) {
                return std::ranges::count_if(found.tags, [&](const gr::property_map& stamp) {
                    const auto tagName = gr::property_map_view{stamp}.get_if<std::string_view>(gr::tag::kTriggerName.key());
                    return tagName && *tagName == name;
                });
            };
            expect(eq(namedLike(expectedName), 1)) << "the stream tag the trigger produced is named " << expectedName;
        };
        gr::property_map named = edgeAt1;
        named["trigger_name"]  = std::string("up");
        check(named, "up");
        check(edgeAt1, {});
    };

    "an inverted duration range is refused, and the trigger keeps no test"_test = [&] {
        const auto found = measured<ValueCondition::pulseWidth>({0.f, 2.f, 0.f}, with(levelAt1, "inside", 5., 3.));
        expect(eq(found.events.size(), 1UZ)) << "the pulse (0.5 to 1.5) judged by no test" << fatal;
        expect(eq(found.errors.size(), 1UZ)) << fatal;
        expect(found.errors[0].contains("inverted"));
        expect(approx(samplesOf(gr::property_map_view{found.events[0]}), 1.5, kTolerance));
    };

    "a clipped plateau gives no confirmed width: it is reported at once, flagged clipped, whatever the duration test says"_test = [&] {
        gr::property_map settings = with(levelAt1, "lessEqual", 0., 2.);
        settings["trigger_name"]  = std::string("t");
        const auto control        = measured<ValueCondition::pulseWidth>({0.f, 2.f, 5.f, 5.f, 2.f, 0.f}, settings);
        expectEvents(control, {});
        expect(eq(control.rejected, 1U)) << "unclipped, the 4-sample pulse fails <= 2";
        settings["clip_level"] = 5.f;
        const auto found       = measured<ValueCondition::pulseWidth>({0.f, 2.f, 5.f, 5.f, 2.f, 0.f}, settings);
        expectEvents(found, {{4.5, 0.5, "RISING", "", true}});
        expect(eq(found.rejected, 0U));
    };

    "a pulse that stays below the clip level is judged as before"_test = [&] {
        gr::property_map settings = with(levelAt1, "lessEqual", 0., 2.);
        settings["clip_level"]    = 5.f;
        expectEvents(measured<ValueCondition::pulseWidth>({0.f, 2.f, 2.f, 0.f}, settings), {{2.5, 0.5, "RISING"}});
    };

    "a pulse cut by the end of the stream is never reported complete"_test = [&] { expectEvents(measured<ValueCondition::pulseWidth>({0.f, 2.f, 2.f, 2.f}, levelAt1), {}); };

    "a pulse, a runt and a transition already under way at the start of the stream are never reported complete"_test = [&] {
        expectEvents(measured<ValueCondition::pulseWidth>({2.f, 2.f, 0.f, 0.f}, levelAt1), {});
        expectEvents(measured<ValueCondition::runt>({2.f, 2.f, 0.f, 0.f}, band1to3), {});
        expectEvents(measured<ValueCondition::transition>({4.f, 4.f, 0.f, 0.f}, band1to3), {});
        gr::property_map both = levelAt1;
        both["edge"]          = std::string("both");
        expectEvents(measured<ValueCondition::pulseWidth>({2.f, 2.f, 0.f, 0.f}, both), {});
    };

    "a runt and a transition with hysteresis report the crossings of both comparator boundaries"_test = [&] {
        // lower 1 ± 0.5: rising at 1.5, falling at 0.5; upper 3 ± 0.5: rising at 3.5
        const gr::property_map settings{{"threshold", 1.f}, {"threshold_upper", 3.f}, {"hysteresis", 1.f}, {"trigger_name", std::string("t")}};
        // 1.5 lies three quarters of the way from sample 1 (0) to sample 2 (2); 3.5 a quarter of the way from sample 2 (2) to sample 3 (8)
        const auto transition = measured<ValueCondition::transition>({0.f, 0.f, 2.f, 8.f}, settings);
        expectEvents(transition, {{2.25, 1.75, "RISING"}});
        const auto meta = metaOf(transition.events.front());
        expect(meta.has_value()) << fatal;
        expect(approx(*meta->get_if<double>("boundary_rising"), 1.5, 1e-9));
        expect(approx(*meta->get_if<double>("boundary_falling"), 0.5, 1e-9));
        expect(eq(textOf(meta, "estimator"), std::string_view{"linear"}));
        expect(approx(*meta->get_if<double>("slope"), 2. / 0.5e-3, 1e-6)) << "(3 - 1) units over 0.5 samples at 1 kHz";

        const gr::property_map runtSettings{{"threshold", 1.f}, {"threshold_upper", 4.f}, {"hysteresis", 1.f}};
        // up through 1.5 three quarters of the way from sample 1 to 2, back down through 0.5 three quarters of the way from sample 2 to 3
        expectEvents(measured<ValueCondition::runt>({0.f, 0.f, 2.f, 0.f, 0.f}, runtSettings), {{2.75, 1.75, "RISING"}});
    };

    "an event is never dropped for a long name: the meta-info degrades, the root tuple stays"_test = [&] {
        gr::property_map settings = with(levelAt1, "lessEqual", 0., 2.);
        settings["trigger_name"]  = std::string(300, 'n');
        settings["clip_level"]    = 5.f;
        const auto found          = measured<ValueCondition::pulseWidth>({0.f, 2.f, 5.f, 5.f, 2.f, 0.f}, settings);
        expect(eq(found.events.size(), 1UZ)) << fatal;
        expect(approx(samplesOf(gr::property_map_view{found.events[0]}), 4.5, kTolerance));
    };

    "a reversed transition falling back is classified like a rising one"_test = [&] {
        gr::property_map settings = band1to3;
        settings["edge"]          = std::string("falling");
        expectEvents(measured<ValueCondition::transition>({4.f, 2.f, 4.f}, settings), {{1.5, 0.5, "FALLING", "reversed"}});
    };

    "a runt's duration is compared strictly or not as asked"_test = [&] {
        const gr::property_map settings{{"threshold", 1.f}, {"threshold_upper", 4.f}, {"hysteresis", 0.f}};
        // widths 1 (0.5 to 1.5) and 3 (3.5 to 6.5)
        const std::vector<float> wave{0.f, 2.f, 0.f, 0.f, 2.f, 2.f, 2.f, 0.f};
        const auto               atLeast = measured<ValueCondition::runt>(wave, with(settings, "greaterEqual", 3., 0.));
        expectEvents(atLeast, {{6.5, 3.5, "RISING"}});
        expect(eq(atLeast.rejected, 1U));
        const auto above = measured<ValueCondition::runt>(wave, with(settings, "greater", 3., 0.));
        expectEvents(above, {});
        expect(eq(above.rejected, 2U)) << "a width of exactly 3 is not greater than 3";
    };

    "an interval of exactly the limit passes or fails by the kind of bound"_test = [&] {
        const std::vector<float> wave{0.f, 1.f, 0.f, 0.f, 1.f, 0.f}; // rising crossings at 1 and 4: an interval of exactly 3
        expectEvents(measured<ValueCondition::interval>(wave, with(levelAt1, "greaterEqual", 3., 0.)), {{4., 1., "RISING"}});
        expectEvents(measured<ValueCondition::interval>(wave, with(levelAt1, "lessEqual", 0., 3.)), {{4., 1., "RISING"}});
        expectEvents(measured<ValueCondition::interval>(wave, with(levelAt1, "inside", 3., 3.)), {{4., 1., "RISING"}});
        const auto strictlyMore = measured<ValueCondition::interval>(wave, with(levelAt1, "greater", 3., 0.));
        expectEvents(strictlyMore, {});
        expect(eq(strictlyMore.rejected, 1U));
        expectEvents(measured<ValueCondition::interval>(wave, with(levelAt1, "less", 0., 3.)), {});
        expectEvents(measured<ValueCondition::interval>(wave, with(levelAt1, "outside", 3., 3.)), {});
    };

    "a timeout of 5 samples is dated at start + 5 and decided at the sample that reaches it, or exceeds it when strict"_test = [&] {
        // the level is reached at sample 1; start + 5 = 6
        const std::vector<float> reachedAt6{0.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
        expectEvents(measured<ValueCondition::levelTimeout>(reachedAt6, with(levelAt1, "greaterEqual", 5., 0.)), {{6., 1., "RISING", "ongoing"}});
        expectEvents(measured<ValueCondition::levelTimeout>(reachedAt6, with(levelAt1, "greater", 5., 0.)), {});
        const std::vector<float> exceededAt7{0.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
        expectEvents(measured<ValueCondition::levelTimeout>(exceededAt7, with(levelAt1, "greater", 5., 0.)), {{6., 1., "RISING", "ongoing"}});

        const std::vector<float> edgeThenSilence{0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        expectEvents(measured<ValueCondition::edgeTimeout>(edgeThenSilence, with(levelAt1, "greaterEqual", 5., 0.)), {{6., 1., "RISING"}});
        expectEvents(measured<ValueCondition::edgeTimeout>(edgeThenSilence, with(levelAt1, "greater", 5., 0.)), {});
        const std::vector<float> longerSilence{0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        expectEvents(measured<ValueCondition::edgeTimeout>(longerSilence, with(levelAt1, "greater", 5., 0.)), {{6., 1., "RISING"}});
    };

    "a ringing edge is one event with hysteresis and several without"_test = [&] {
        const std::vector<float> ringing{0.f, 2.f, 0.9f, 2.f, 0.9f, 2.f};
        expectEvents(measured<ValueCondition::level>(ringing, edgeAt1), {{0.75}}); // 1.5 lies three quarters of the way from 0 to 2
        expect(eq(measured<ValueCondition::level>(ringing, levelAt1).events.size(), 3UZ)) << "high at x >= 1: three rising crossings";
    };

    "debounce by level timeout keeps the original candidate time through ringing and restarts after a real release"_test = [&] {
        // within the hysteresis band the candidate at 0.75 survives: decided at its deadline 0.75 + 3
        expectEvents(measured<ValueCondition::levelTimeout>(
                         {
                             0.f,
                             2.f,
                             1.f,
                             2.f,
                             2.f,
                             2.f,
                             2.f,
                             2.f,
                         },
                         with(edgeAt1, "greaterEqual", 3., 0.)),
            {{3.75, 0.75, "RISING", "ongoing"}});
        // a release below the band starts a new episode: 0.5 and 1.5 (released), then 2.5 → deadline 5.5
        expectEvents(measured<ValueCondition::levelTimeout>(
                         {
                             0.f,
                             2.f,
                             0.f,
                             2.f,
                             2.f,
                             2.f,
                             2.f,
                             2.f,
                             2.f,
                         },
                         with(levelAt1, "greaterEqual", 3., 0.)),
            {{5.5, 2.5, "RISING", "ongoing"}});
    };

    "a duration in seconds without a sample rate decides nothing and is counted"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     source  = fixture.emplace<gr::testing::TagSource<float>>({
            {"values", std::vector<float>{0.f, 2.f, 2.f, 0.f}},
            {"n_samples_max", gr::Size_t{4U}},
            {"mark_tag", false},
        });
        auto&                     trigger = fixture.emplace<ValueTrigger<float, ValueCondition::pulseWidth>>({
            {"threshold", 1.f},
            {"hysteresis", 0.f},
            {"duration_condition", std::string("lessEqual")},
            {"duration_max", 1.},
            {"duration_unit", std::string("seconds")},
        });
        auto&                     sink    = fixture.emplace<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_ONE>>();
        expect(fixture.connect<"out", "in">(source, trigger).has_value());
        expect(fixture.connect<"out", "in">(trigger, sink).has_value());
        expect(fixture.run().has_value());
        expect(eq(trigger.n_triggers.value, 0U));
        expect(eq(trigger.n_rejected.value, 0U)) << "it is no duration reject";
        expect(eq(trigger.n_unmeasurable.value, 1U));
    };

    "an unknown edge or duration comparison is refused, the previous value kept, and an error message sent"_test = [&] {
        gr::property_map badEdge = edgeAt1;
        badEdge["edge"]          = std::string("sideways");
        const auto found         = measured<ValueCondition::level>({0.f, 0.f, 2.f, 2.f, 0.f, 0.f, 2.f}, badEdge);
        expect(eq(found.events.size(), 2UZ)) << "the two rising edges of the kept 'rising'" << fatal;
        expect(eq(found.errors.size(), 1UZ)) << fatal;
        expect(found.errors[0].contains("sideways"));

        gr::property_map badComparison      = levelAt1;
        badComparison["duration_condition"] = std::string("sideways");
        const auto second                   = measured<ValueCondition::pulseWidth>({0.f, 2.f, 0.f}, badComparison);
        expect(eq(second.events.size(), 1UZ)) << "the untested pulse" << fatal;
        expect(eq(second.errors.size(), 1UZ)) << fatal;
        expect(second.errors[0].contains("sideways"));
    };
};

const boost::ut::suite<"ValueTrigger, integer limits"> _integers = [] {
    using namespace boost::ut;
    "a window over int16_t reaches the type's lowest value without overflow"_test = [] {
        gr::testing::GraphFixture       fixture;
        const std::vector<std::int16_t> samples{0, 10, std::numeric_limits<std::int16_t>::lowest(), 10};
        auto&                           source  = fixture.emplace<gr::testing::TagSource<std::int16_t>>({{"values", samples}, {"n_samples_max", gr::Size_t{4U}}, {"mark_tag", false}});
        auto&                           trigger = fixture.emplace<ValueTrigger<std::int16_t, ValueCondition::window>>({{"threshold", std::int16_t{5}}, {"threshold_upper", std::numeric_limits<std::int16_t>::max()}, {"edge", std::string("both")}, {"sample_rate", 1000.f}});
        auto&                           sink    = fixture.emplace<gr::testing::TagSink<std::int16_t, gr::testing::ProcessFunction::USE_PROCESS_ONE>>();
        expect(fixture.connect<"out", "in">(source, trigger).has_value());
        expect(fixture.connect<"out", "in">(trigger, sink).has_value());
        expect(fixture.run().has_value());
        expect(eq(trigger.n_triggers.value, 3U)) << "into the band at 10, out at the lowest value, back in at 10";
    };
};

const boost::ut::suite<"ValueTrigger, odd integer hysteresis"> _oddHysteresis = [] {
    using namespace boost::ut;
    "int16_t with an odd hysteresis keeps the band edges at level ± h/2, not at ± floor(h/2)"_test = [] {
        // level 0, h = 3: rising at x >= 1.5, falling at x <= -1.5; the edges lie half way between 1 and 2 and between -1 and -2
        const gr::property_map settings{{"threshold", std::int16_t{0}}, {"hysteresis", std::int16_t{3}}, {"edge", std::string("both")}};
        expectEvents(measured<ValueCondition::level, std::int16_t>({-3, 1, 2, 0, -1, -2, 0}, settings), {{1.5}, {4.5}});
    };
};
} // namespace qa_value_trigger

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
