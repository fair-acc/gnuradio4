#include <boost/ut.hpp>

#include <algorithm>
#include <cstdint>
#include <format>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gnuradio-4.0/trigger/ClockedTriggers.hpp>
#include <gnuradio-4.0/trigger/EventReduce.hpp>
#include <gnuradio-4.0/trigger/EventShaping.hpp>
#include <gnuradio-4.0/trigger/Gate.hpp>
#include <gnuradio-4.0/trigger/PatternTrigger.hpp>
#include <gnuradio-4.0/trigger/Predicate.hpp>
#include <gnuradio-4.0/trigger/SampleAndHold.hpp>
#include <gnuradio-4.0/trigger/SchmittTrigger.hpp>
#include <gnuradio-4.0/trigger/StreamOps.hpp>
#include <gnuradio-4.0/trigger/TagToEvent.hpp>
#include <gnuradio-4.0/trigger/TakeSkip.hpp>
#include <gnuradio-4.0/trigger/TriggerWatchdog.hpp>
#include <gnuradio-4.0/trigger/ValueTrigger.hpp>
#include <gnuradio-4.0/trigger/While.hpp>

#include "TriggerTest.hpp"

namespace qa_acceptance {
using namespace gr::blocks::trigger;
using gr::trigger_test::ErrorMessages;
using gr::trigger_test::EventScript;
using gr::trigger_test::EventTap;
using gr::trigger_test::kNsPerSample;
using gr::trigger_test::kSampleRate;
using gr::trigger_test::kStreamStart;
using gr::trigger_test::PartitionSource;

/**
 * one way to cut a stream: `maxSamples` caps what the block under test reads per work call (0 = uncapped),
 * `seed` makes the source publish in pseudo-random runs of 1 to 9 samples (0 = whatever the source has at once)
 */
struct Cut {
    std::size_t   maxSamples = 0UZ;
    std::uint32_t seed       = 0U;
};

/** the cuts every chunk-independence scenario runs: the whole stream, single samples, small and prime caps, seeded random runs, and both at once */
[[nodiscard]] const std::vector<Cut>& chunkCuts() {
    static const std::vector<Cut> cuts{
        {0UZ, 0U},
        {1UZ, 0U},
        {2UZ, 0U},
        {3UZ, 0U},
        {5UZ, 0U},
        {7UZ, 0U},
        {0UZ, 1U},
        {0UZ, 2U},
        {0UZ, 3U},
        {0UZ, 4U},
        {4UZ, 5U},
        {2UZ, 6U},
    };
    return cuts;
}

/** the anchor every such stream starts with */
[[nodiscard]] std::pair<std::size_t, gr::property_map> streamAnchor() { return {0UZ, gr::property_map{{std::string(gr::tag::TRIGGER_TIME.key()), kStreamStart}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}, {std::string("sample_rate"), kSampleRate}}}; }

[[nodiscard]] std::pair<std::size_t, gr::property_map> streamGap(std::size_t at) { return {at, gr::property_map{{std::string(gr::tag::N_DROPPED_SAMPLES.key()), gr::Size_t{7U}}}}; }

/** what an event stream and a block's counters came to, with times in samples since the stream start; `reference` is -1 when the event has none */
struct Detection {
    double      at        = 0.;
    double      reference = -1.;
    std::string name;
};

struct DetectorRun {
    std::vector<Detection>  events{};
    std::vector<gr::Size_t> counters{};
    std::vector<float>      passed{}; /// the samples a gating block let through; empty where that is not asked
};

[[nodiscard]] double samplesSinceStart(const gr::property_map_view& tuple) {
    const auto ns     = tuple.get_if<std::uint64_t>(gr::tag::TRIGGER_TIME.key());
    const auto offset = tuple.get_if<float>(gr::tag::TRIGGER_OFFSET.key());
    if (!ns) {
        return -1.;
    }
    const double offsetNs = offset ? static_cast<double>(*offset) * 1e9 : 0.;
    return (static_cast<double>(*ns) - static_cast<double>(kStreamStart) + offsetNs) / kNsPerSample;
}

[[nodiscard]] std::vector<Detection> detectionsOf(const EventTap& tap) {
    std::vector<Detection> found;
    for (const gr::property_map& event : tap._events) {
        const gr::property_map_view view{event};
        const auto                  meta = view.get_if<gr::property_map_view>(gr::tag::TRIGGER_META_INFO.key());
        const auto                  name = view.get_if<std::string_view>(gr::tag::TRIGGER_NAME.key());
        found.push_back({samplesSinceStart(view), meta ? samplesSinceStart(*meta) : -1., name ? std::string(*name) : std::string{}});
    }
    return found;
}

[[nodiscard]] std::string describe(const DetectorRun& run) {
    std::string text = "events";
    for (const Detection& event : run.events) {
        text += std::format(" ({}, ref {})", event.at, event.reference);
    }
    text += " counters";
    for (const gr::Size_t counter : run.counters) {
        text += std::format(" {}", counter);
    }
    return text + std::format(" passed {} samples", run.passed.size());
}

/**
 * Runs `run` under every cut and expects the answer of the specification each time, and the same answer bit for bit
 * across cuts; `truth` is the analytic event list, in samples since the stream start, and the counters it implies.
 */
template<typename TRun>
void expectChunkIndependent(std::string_view what, TRun&& run, const DetectorRun& truth, double toleranceSamples = 1e-4) {
    using namespace boost::ut;
    std::optional<DetectorRun> first;
    for (const Cut& cut : chunkCuts()) {
        const DetectorRun found = run(cut);
        const std::string where = std::format("{} (max_samples {}, seed {}): {}", what, cut.maxSamples, cut.seed, describe(found));
        expect(eq(found.events.size(), truth.events.size())) << where << ": number of events";
        for (std::size_t i = 0UZ; i < std::min(found.events.size(), truth.events.size()); ++i) {
            expect(approx(found.events[i].at, truth.events[i].at, toleranceSamples)) << where << ": event " << i << " decision";
            expect(approx(found.events[i].reference, truth.events[i].reference, toleranceSamples)) << where << ": event " << i << " reference";
        }
        expect(found.counters == truth.counters) << where << ": counters";
        expect(found.passed == truth.passed) << where << ": passed samples";
        if (!first) {
            first = found;
            continue;
        }
        const bool sameEvents = std::ranges::equal(found.events, first->events, [](const Detection& a, const Detection& b) { return a.at == b.at && a.reference == b.reference && a.name == b.name; });
        expect(sameEvents) << where << ": events differ bit for bit from the first cut's";
    }
}

/** the sample and tag symbols every acceptance script is written in */
[[nodiscard]] const gr::property_map& acceptanceValues() {
    static const gr::property_map map{{"a", 0.f}, {"b", 1.f}, {"c", 5.f}, {"d", 0.f}, {"e", 5.f}, {"f", 0.f}};
    return map;
}

[[nodiscard]] const gr::property_map& acceptanceTags() {
    static const gr::property_map map{
        {"T", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("start")}, {std::string(gr::tag::TRIGGER_TIME.key()), kStreamStart}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}},
        {"U", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("other")}, {std::string(gr::tag::TRIGGER_TIME.key()), kStreamStart}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}},
        {"V", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("start")}}},
    };
    return map;
}

/** the tag placements the specification asks every block to survive */
[[nodiscard]] const std::vector<std::pair<std::string, std::string>>& tagEdgeCases() {
    static const std::vector<std::pair<std::string, std::string>> cases{
        {"a tag on the first sample", "T:a b c d |"},
        {"a tag on the last sample", "a b c T:d |"},
        {"tags on consecutive samples", "T:a T:b c d |"},
        {"two tags on one sample", "T+U:a b c d |"},
        {"a tag the filter ignores", "U:a b c d |"},
        {"no tags at all", "a b c d |"},
        {"a trigger carrying no time", "a V:b c d |"},
        {"a single sample", "T:a |"},
    };
    return cases;
}

/**
 * Runs one script through a block at several work-call sizes and hands back what came out each time.
 *
 * A block decides on spans, and the scheduler is free to cut the stream anywhere: what a block does must not depend on
 * where. Every entry of the result is the sink's script for one chunk size, so a test asserts they are all the same
 * rather than asserting any particular one -- which is the property, and it is the one no amount of reading the code
 * establishes.
 */
template<typename TBlock>
[[nodiscard]] std::vector<std::string> scriptsAtChunkSizes(const std::string& script, const gr::property_map& settings, const std::vector<std::size_t>& chunks = {0UZ, 1UZ, 2UZ, 3UZ}) {
    std::vector<std::string> found;
    for (const std::size_t chunk : chunks) {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.template emplace<gr::blocks::trigger::MarbleSource<float>>({{"script", script}, {"sample_values", acceptanceValues()}, {"sample_tags", acceptanceTags()}});
        auto&                     block  = fixture.template emplace<TBlock>(settings);
        auto&                     sink   = fixture.template emplace<gr::blocks::trigger::MarbleSink<float>>({{"sample_values", acceptanceValues()}, {"sample_tags", acceptanceTags()}});
        if (chunk > 0UZ) {
            if (chunk < block.in.min_samples) {
                continue;
            }
            block.in.max_samples = chunk;
        }
        boost::ut::expect(fixture.template connect<"out", "in">(source, block).has_value());
        boost::ut::expect(fixture.template connect<"out", "in">(block, sink).has_value());
        ErrorMessages reported{fixture.scheduler()}; // a block's error message would otherwise end the run
        boost::ut::expect(fixture.run().has_value()) << script;
        found.push_back(sink.script());
    }
    return found;
}

/** asserts a block answers the same however the stream was cut, over every tag placement the specification lists */
template<typename TBlock>
void acceptsEveryTagPlacement(std::string_view blockName, const gr::property_map& settings) {
    using namespace boost::ut;
    for (const auto& [what, script] : tagEdgeCases()) {
        const std::vector<std::string> answers = scriptsAtChunkSizes<TBlock>(script, settings);
        expect(!answers.empty());
        for (std::size_t i = 1UZ; i < answers.size(); ++i) {
            expect(eq(answers[i], answers[0])) << std::format("{}: {} ('{}') answered differently at another work-call size", blockName, what, script);
        }
        expect(!answers[0].empty()) << std::format("{}: {} produced no script at all, terminator included", blockName, what);
    }
}
using Tags = std::vector<std::pair<std::size_t, gr::property_map>>;

void capReads(auto& port, const Cut& cut) {
    if (cut.maxSamples > 0UZ) {
        port.max_samples = cut.maxSamples;
    }
}

template<typename T>
PartitionSource<T>& playStream(auto& fixture, std::vector<T> samples, Tags tags, std::uint32_t seed) {
    auto& source    = fixture.template emplace<PartitionSource<T>>();
    source._samples = std::move(samples);
    source._tags    = std::move(tags);
    source._seed    = seed;
    return source;
}

[[nodiscard]] Tags anchoredAnd(Tags more = {}) {
    more.insert(more.begin(), streamAnchor());
    return more;
}

/** a value trigger over a stream cut by `cut`; the counters are triggers, rejected, gaps, abandoned */
template<ValueCondition condition>
[[nodiscard]] DetectorRun valueRun(const std::vector<float>& samples, gr::property_map settings, Tags tags, const Cut& cut) {
    gr::testing::GraphFixture fixture;
    settings[std::string("sample_rate")] = 1000.f;
    auto& source                         = playStream<float>(fixture, samples, anchoredAnd(std::move(tags)), cut.seed);
    auto& trigger                        = fixture.emplace<ValueTrigger<float, condition>>(std::move(settings));
    auto& sink                           = fixture.emplace<gr::trigger_test::CollectingSink<float>>();
    auto& tap                            = fixture.emplace<EventTap>();
    capReads(trigger.in, cut);
    boost::ut::expect(fixture.connect<"out", "in">(source, trigger).has_value());
    boost::ut::expect(fixture.connect<"out", "in">(trigger, sink).has_value());
    boost::ut::expect(fixture.connect<"evtOut", "evtIn">(trigger, tap).has_value());
    ErrorMessages sent{fixture.scheduler()};
    boost::ut::expect(fixture.run().has_value());
    return {.events = detectionsOf(tap), .counters = {trigger.n_triggers.value, trigger.n_rejected.value, trigger.n_gaps.value, trigger.n_abandoned.value}};
}

[[nodiscard]] gr::property_map durationTest(gr::property_map settings, std::string_view comparison, double minimum, double maximum) {
    settings[std::string("duration_condition")] = std::string(comparison);
    settings[std::string("duration_min")]       = minimum;
    settings[std::string("duration_max")]       = maximum;
    return settings;
}

template<ValueCondition condition>
void valueChunkIndependent(std::string_view what, const std::vector<float>& samples, const gr::property_map& settings, const DetectorRun& truth, const Tags& tags = {}) {
    expectChunkIndependent(what, [&](const Cut& cut) { return valueRun<condition>(samples, settings, tags, cut); }, truth);
}
/** events handed to a block a few per work call, as the cut dictates */
[[nodiscard]] std::size_t eventsPerCall(const Cut& cut) {
    if (cut.maxSamples > 0UZ) {
        return cut.maxSamples;
    }
    if (cut.seed > 0U) {
        return 1UZ + cut.seed % 3UZ;
    }
    return std::numeric_limits<std::size_t>::max();
}

[[nodiscard]] gr::property_map eventAtMs(std::string name, std::uint64_t ms) { return gr::trigger_test::eventNamed(std::move(name), kStreamStart + ms * 1'000'000U); }

template<typename TBlock>
[[nodiscard]] DetectorRun eventRun(std::vector<gr::property_map> events, gr::property_map settings, const Cut& cut, auto&& countersOf) {
    gr::testing::GraphFixture fixture;
    auto&                     script = fixture.emplace<EventScript>();
    script._events                   = std::move(events);
    script._maxPerWorkCall           = eventsPerCall(cut);
    auto& block                      = fixture.emplace<TBlock>(std::move(settings));
    auto& tap                        = fixture.emplace<EventTap>();
    boost::ut::expect(fixture.connect<"evtOut", "evtIn">(script, block).has_value());
    boost::ut::expect(fixture.connect<"evtOut", "evtIn">(block, tap).has_value());
    boost::ut::expect(fixture.run().has_value());
    return {.events = detectionsOf(tap), .counters = countersOf(block)};
}

/** a stream cut by `cut`, gated by events handed over a few per work call; `TBlock` reads `in`, `evtIn` and `evtOut` */
template<typename TBlock>
[[nodiscard]] DetectorRun gatedRun(std::vector<float> samples, std::vector<gr::property_map> events, gr::property_map settings, const Cut& cut, auto&& countersOf) {
    gr::testing::GraphFixture fixture;
    auto&                     source = playStream<float>(fixture, std::move(samples), anchoredAnd(), cut.seed);
    auto&                     script = fixture.emplace<EventScript>();
    script._events                   = std::move(events);
    script._stopWhenDone             = false;
    auto& block                      = fixture.emplace<TBlock>(std::move(settings));
    auto& sink                       = fixture.emplace<gr::trigger_test::CollectingSink<float>>();
    auto& tap                        = fixture.emplace<EventTap>();
    capReads(block.in, cut);
    boost::ut::expect(fixture.connect<"out", "in">(source, block).has_value());
    boost::ut::expect(fixture.connect<"out", "in">(block, sink).has_value());
    boost::ut::expect(fixture.connect<"evtOut", "evtIn">(script, block).has_value());
    boost::ut::expect(fixture.connect<"evtOut", "evtIn">(block, tap).has_value());
    boost::ut::expect(fixture.run().has_value());
    return {.events = detectionsOf(tap), .counters = countersOf(block), .passed = sink._collected};
}

/** two channels cut independently, into `PatternTrigger` */
[[nodiscard]] DetectorRun patternRun(const std::vector<float>& channelA, const std::vector<float>& channelB, gr::property_map settings, Tags tagsA, const Cut& cut) {
    gr::testing::GraphFixture fixture;
    settings[std::string("n_inputs")]    = 2U;
    settings[std::string("sample_rate")] = 1000.f;
    auto& sourceA                        = playStream<float>(fixture, channelA, anchoredAnd(std::move(tagsA)), cut.seed);
    auto& sourceB                        = playStream<float>(fixture, channelB, anchoredAnd(), cut.seed == 0U ? 0U : cut.seed * 7U + 3U);
    auto& trigger                        = fixture.emplace<PatternTrigger<float>>(std::move(settings));
    auto& tap                            = fixture.emplace<EventTap>();
    for (auto& port : trigger.in) {
        capReads(port, cut);
    }
    boost::ut::expect(fixture.graph.connect(sourceA, gr::PortDefinition{std::string("out")}, trigger, gr::PortDefinition{std::string("in#0")}).has_value());
    boost::ut::expect(fixture.graph.connect(sourceB, gr::PortDefinition{std::string("out")}, trigger, gr::PortDefinition{std::string("in#1")}).has_value());
    boost::ut::expect(fixture.connect<"evtOut", "evtIn">(trigger, tap).has_value());
    boost::ut::expect(fixture.run().has_value());
    return {.events = detectionsOf(tap), .counters = {trigger.n_triggers.value, trigger.n_rejected.value, trigger.n_gaps.value, trigger.n_abandoned.value}};
}

/** a data line and a clock into a clocked trigger; `countersOf` reads the block */
template<typename TBlock>
[[nodiscard]] DetectorRun clockedRun(const std::vector<float>& data, const std::vector<float>& clock, gr::property_map settings, Tags dataTags, const Cut& cut, auto&& countersOf) {
    gr::testing::GraphFixture fixture;
    settings[std::string("sample_rate")] = 1000.f;
    auto& dataSource                     = playStream<float>(fixture, data, anchoredAnd(std::move(dataTags)), cut.seed);
    auto& clockSource                    = playStream<float>(fixture, clock, anchoredAnd(), cut.seed == 0U ? 0U : cut.seed * 7U + 3U);
    auto& block                          = fixture.emplace<TBlock>(std::move(settings));
    auto& sink                           = fixture.emplace<gr::trigger_test::CollectingSink<float>>();
    auto& tap                            = fixture.emplace<EventTap>();
    capReads(block.in, cut);
    capReads(block.clk_in, cut);
    boost::ut::expect(fixture.graph.connect(dataSource, gr::PortDefinition{std::string("out")}, block, gr::PortDefinition{std::string("in")}).has_value());
    boost::ut::expect(fixture.graph.connect(clockSource, gr::PortDefinition{std::string("out")}, block, gr::PortDefinition{std::string("clk_in")}).has_value());
    boost::ut::expect(fixture.connect<"out", "in">(block, sink).has_value());
    boost::ut::expect(fixture.connect<"evtOut", "evtIn">(block, tap).has_value());
    boost::ut::expect(fixture.run().has_value());
    return {.events = detectionsOf(tap), .counters = countersOf(block)};
}

const boost::ut::suite<"acceptance: the same answer however the stream is cut"> _chunkInvariance = [] {
    using namespace boost::ut;

    const gr::property_map onStart{{"filter", std::string("start")}};

    "Gate over each of its six modes"_test = [&] {
        for (const std::string mode : {"once", "toggle", "cooldown", "wait", "takeUntil", "skipUntil"}) {
            acceptsEveryTagPlacement<Gate<float>>(std::format("Gate({})", mode), //
                {{"mode", mode}, {"open_filter", std::string("start")}, {"n_open", 2U}, {"n_delay", 2U}, {"n_cooldown", 2U}});
        }
    };

    "TakeN and SkipN"_test = [&] {
        acceptsEveryTagPlacement<TakeN<float>>("TakeN", {{"filter", std::string("start")}, {"n", 2U}});
        acceptsEveryTagPlacement<SkipN<float>>("SkipN", {{"filter", std::string("start")}, {"n", 2U}});
    };

    "SampleAndHold"_test = [&] { acceptsEveryTagPlacement<SampleAndHold<float>>("SampleAndHold", {{"filter", std::string("start")}, {"initial_value", 0.f}}); };

    "Debounce and Throttle"_test = [&] {
        acceptsEveryTagPlacement<Debounce<float>>("Debounce", {{"filter", std::string("start")}, {"n_samples", 2U}});
        acceptsEveryTagPlacement<Throttle<float>>("Throttle", {{"filter", std::string("start")}, {"n_samples", 2U}});
    };

    "the predicate three"_test = [&] {
        acceptsEveryTagPlacement<TakeWhile<float>>("TakeWhile", {{"predicate", std::string("less")}, {"threshold", 3.f}});
        acceptsEveryTagPlacement<SkipWhile<float>>("SkipWhile", {{"predicate", std::string("less")}, {"threshold", 3.f}});
        acceptsEveryTagPlacement<ElementAt<float>>("ElementAt", {{"n", 1U}});
        acceptsEveryTagPlacement<ElementAt<float>>("ElementAt(per segment)", {{"n", 1U}, {"segment_filter", std::string("start")}});
    };

    "Distinct and DistinctUntilChanged"_test = [&] {
        acceptsEveryTagPlacement<Distinct<float>>("Distinct", {{"max_values", 8U}});
        acceptsEveryTagPlacement<DistinctUntilChanged<float>>("DistinctUntilChanged", {});
    };

    "Accumulate over its cadence"_test = [&] {
        acceptsEveryTagPlacement<Accumulate<float, Accumulation::automatic>>("Reduce(sum)", {{"operation", std::string("sum")}, {"n_samples", 2U}});
        acceptsEveryTagPlacement<Accumulate<float, Accumulation::last>>("Last(per segment)", {{"segment_filter", std::string("start")}, {"n_samples", 0U}});
    };

    "SampleFilter over each comparison"_test = [&] {
        for (const std::string predicate : {"greater", "greaterEqual", "less", "lessEqual", "equal", "notEqual"}) {
            acceptsEveryTagPlacement<SampleFilter<float>>(std::format("SampleFilter({})", predicate), {{"predicate", predicate}, {"threshold", 1.}});
        }
    };

    "Scan, with and without a reset"_test = [&] {
        acceptsEveryTagPlacement<Scan<float>>("Scan(sum)", {{"operation", std::string("sum")}});
        acceptsEveryTagPlacement<Scan<float>>("Scan(mean, reset)", {{"operation", std::string("mean")}, {"reset_filter", std::string("start")}});
    };

    "TakeLast and SkipLast"_test = [&] {
        acceptsEveryTagPlacement<TakeLast<float>>("TakeLast", {{"filter", std::string("start")}, {"n", 2U}});
        acceptsEveryTagPlacement<SkipLast<float>>("SkipLast", {{"filter", std::string("start")}, {"n", 2U}});
        acceptsEveryTagPlacement<TakeLast<float>>("TakeLast(whole stream)", {{"n", 2U}});
    };

    "Repeat"_test = [&] { acceptsEveryTagPlacement<Repeat<float>>("Repeat", {{"n_repeats", 2U}, {"capacity", 8U}, {"segment_filter", std::string("start")}}); };

    "the blocks that pass the stream through untouched"_test = [&] {
        acceptsEveryTagPlacement<TagToEvent<float>>("TagToEvent", onStart);
        acceptsEveryTagPlacement<TriggerWatchdog<float>>("TriggerWatchdog", {{"filter", std::string("start")}, {"timeout_samples", 2U}});
        /**
         * an interpolating trigger declares its window through in.min_samples, so the harness only offers it spans it
         * admits; qa_SchmittTriggerBlock holds it to the same edges over a script long enough for several of them
         */
        acceptsEveryTagPlacement<SchmittTrigger<float, gr::trigger::InterpolationMethod::BASIC_LINEAR_INTERPOLATION>>("SchmittTrigger", {{"threshold", 2.5f}, {"hysteresis", 4.f}});
        acceptsEveryTagPlacement<SchmittTrigger<float, gr::trigger::InterpolationMethod::NO_INTERPOLATION>>("SchmittTrigger(no interpolation)", {{"threshold", 2.5f}, {"hysteresis", 4.f}});
        acceptsEveryTagPlacement<ValueTrigger<float, ValueCondition::level>>("LevelTrigger", {{"threshold", 2.5f}, {"hysteresis", 1.f}});
    };
};

const boost::ut::suite<"acceptance: a detector answers the same however the stream is cut"> _detectors = [] {
    using namespace boost::ut;
    const gr::property_map levelAt1{{"threshold", 1.f}, {"hysteresis", 0.f}};
    const gr::property_map band1to3{{"threshold", 1.f}, {"threshold_upper", 3.f}, {"hysteresis", 0.f}};
    // every expectation is a hand-computed crossing, in samples since the stream start; counters: triggers, rejected, gaps, abandoned

    "pulseWidth: a pulse accepted by its width, one rejected, boundaries inside both"_test = [&] {
        valueChunkIndependent<ValueCondition::pulseWidth>("pulseWidth", {0.f, 2.f, 2.f, 0.f, 0.f, 2.f, 2.f, 2.f, 2.f, 0.f}, durationTest(levelAt1, "lessEqual", 0., 2.), //
            {.events = {{2.5, 0.5, {}}}, .counters = {1U, 1U, 0U, 0U}});
    };

    "pulseWidth: a gap inside a pulse leaves no event and one abandoned pulse"_test = [&] {
        valueChunkIndependent<ValueCondition::pulseWidth>("pulseWidth(gap)", {0.f, 2.f, 2.f, 2.f, 0.f, 0.f, 2.f, 2.f, 0.f}, levelAt1, //
            {.events = {{7.5, 5.5, {}}}, .counters = {1U, 0U, 1U, 1U}}, {streamGap(2UZ)});
    };

    "runt: a pulse that reaches the second threshold is no runt, one that does not is"_test = [&] {
        valueChunkIndependent<ValueCondition::runt>("runt", {0.f, 5.f, 0.f, 2.f, 0.f, 4.f, 0.f}, {{"threshold", 1.f}, {"threshold_upper", 4.f}, {"hysteresis", 0.f}}, //
            {.events = {{3.5, 2.5, {}}}, .counters = {1U, 0U, 0U, 0U}});
    };

    "transition: timed between the thresholds, a too fast one rejected"_test = [&] {
        valueChunkIndependent<ValueCondition::transition>("transition", {0.f, 1.5f, 2.5f, 3.5f, 0.f, 4.f}, durationTest(band1to3, "greater", 1.5, 0.), //
            {.events = {{2.5, 2. / 3., {}}}, .counters = {1U, 1U, 0U, 0U}});
    };

    "transition: a reversal is classified and counted"_test = [&] {
        valueChunkIndependent<ValueCondition::transition>("transition(reversal)", {0.f, 2.f, 0.f, 0.f, 2.f, 0.f}, band1to3, //
            {.events = {{1.5, 0.5, {}}, {4.5, 3.5, {}}}, .counters = {2U, 0U, 0U, 2U}});
    };

    "levelTimeout: decided at its deadline, once per episode"_test = [&] {
        valueChunkIndependent<ValueCondition::levelTimeout>("levelTimeout", {0.f, 2.f, 2.f, 2.f, 2.f, 2.f, 0.f, 0.f, 2.f, 2.f, 0.f}, durationTest(levelAt1, "greaterEqual", 3.5, 0.), //
            {.events = {{4., 0.5, {}}}, .counters = {1U, 0U, 0U, 0U}});
    };

    "edgeTimeout: armed by an edge, decided when no next one comes"_test = [&] {
        valueChunkIndependent<ValueCondition::edgeTimeout>("edgeTimeout", {0.f, 2.f, 0.f, 0.f, 0.f, 0.f}, durationTest(levelAt1, "greaterEqual", 3., 0.), //
            {.events = {{3.5, 0.5, {}}}, .counters = {1U, 0U, 0U, 0U}});
    };

    "window: entering and leaving a band, two crossings inside one chunk and across chunks"_test = [&] {
        gr::property_map settings = band1to3;
        settings["edge"]          = std::string("both");
        valueChunkIndependent<ValueCondition::window>("window", {0.f, 2.f, 2.f, 4.f, 2.f, 0.f}, settings, //
            {.events = {{0.5, -1., {}}, {2.5, -1., {}}, {3.5, -1., {}}, {4.5, -1., {}}}, .counters = {4U, 0U, 0U, 0U}});
    };

    "interval: consecutive rising edges compared, decided at the second"_test = [&] {
        valueChunkIndependent<ValueCondition::interval>("interval", {0.f, 2.f, 0.f, 0.f, 2.f, 0.f, 2.f}, durationTest(levelAt1, "inside", 3., 5.), //
            {.events = {{3.5, 0.5, {}}}, .counters = {1U, 1U, 0U, 0U}});
    };
};

const boost::ut::suite<"acceptance: multi-channel, clocked and event-driven consumers answer the same however their inputs are cut"> _consumers = [] {
    using namespace boost::ut;

    "PatternTrigger: a held pattern reported once, a short one rejected, a gap abandoning one in progress"_test = [] {
        // A is high over 1..5 and 8..12; B stays low. The gap on sample 10 cuts the second run: 8 and 9 are abandoned, 10..12 fall short of 4
        const std::vector<float> channelA{0.f, 5.f, 5.f, 5.f, 5.f, 5.f, 0.f, 0.f, 5.f, 5.f, 5.f, 5.f, 5.f, 0.f, 0.f, 0.f};
        const std::vector<float> channelB(16UZ, 0.f);
        const gr::property_map   settings{
              {"pattern", std::string("10")},
              {"thresholds", std::vector<float>{2.5f}},
              {"hysteresis", 0.5f},
              {"when", std::string("holds")},
              {"duration_min_samples", 4U},
        };
        expectChunkIndependent("PatternTrigger", [&](const Cut& cut) { return patternRun(channelA, channelB, settings, {streamGap(10UZ)}, cut); }, //
            {.events = {{4., 1., {}}}, .counters = {1U, 1U, 1U, 1U}});
        expectChunkIndependent("PatternTrigger(no gap)", [&](const Cut& cut) { return patternRun(channelA, channelB, settings, {}, cut); }, //
            {.events = {{4., 1., {}}, {11., 8., {}}}, .counters = {2U, 0U, 0U, 0U}});
    };

    "SerialPatternTrigger: the word spelled by the clock edges, with and without a gap before the second"_test = [] {
        // rising edges on odd samples; the bits at edges 1, 3, 5, 7, 9, 11, 13, 15 are 1 0 1 1 1 0 1 1: the word 1011 ends at edges 7 and 15
        const std::vector<float> data{5.f, 5.f, 0.f, 0.f, 5.f, 5.f, 5.f, 5.f, 0.f, 5.f, 0.f, 0.f, 0.f, 5.f, 0.f, 5.f};
        std::vector<float>       clock;
        for (std::size_t i = 0UZ; i < data.size() / 2UZ; ++i) {
            clock.push_back(0.f);
            clock.push_back(5.f);
        }
        const gr::property_map settings{
            {"pattern", std::string("1011")},
            {"threshold", 2.5f},
            {"hysteresis", 1.f},
            {"clock_threshold", 2.5f},
            {"clock_edge", std::string("rising")},
        };
        const auto counters = [](const auto& block) { return std::vector<gr::Size_t>{block.n_triggers.value, block.n_gaps.value}; };
        expectChunkIndependent("SerialPatternTrigger", [&](const Cut& cut) { return clockedRun<SerialPatternTrigger<float>>(data, clock, settings, {}, cut, counters); }, //
            {.events = {{7., -1., {}}, {15., -1., {}}}, .counters = {2U, 0U}});
        expectChunkIndependent("SerialPatternTrigger(gap)", [&](const Cut& cut) { return clockedRun<SerialPatternTrigger<float>>(data, clock, settings, {streamGap(4UZ)}, cut, counters); }, //
            {.events = {{15., -1., {}}}, .counters = {1U, 1U}});
    };

    "SetupHoldTrigger: a change inside the windows of two edges is reported at both, a gap forgets it"_test = [] {
        // the data moves at sample 4: inside the hold of the edge at 3 and the setup of the edge at 5, with setup = hold = 2
        const std::vector<float> data{0.f, 0.f, 0.f, 0.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f};
        std::vector<float>       clock;
        for (std::size_t i = 0UZ; i < data.size() / 2UZ; ++i) {
            clock.push_back(0.f);
            clock.push_back(5.f);
        }
        const gr::property_map settings{
            {"threshold", 2.5f},
            {"hysteresis", 1.f},
            {"clock_threshold", 2.5f},
            {"clock_edge", std::string("rising")},
            {"setup_samples", 2U},
            {"hold_samples", 2U},
        };
        const auto counters = [](const auto& block) { return std::vector<gr::Size_t>{block.n_violations.value, block.n_gaps.value}; };
        expectChunkIndependent("SetupHoldTrigger", [&](const Cut& cut) { return clockedRun<SetupHoldTrigger<float>>(data, clock, settings, {}, cut, counters); }, //
            {.events = {{3., -1., {}}, {5., -1., {}}}, .counters = {2U, 0U}});
        expectChunkIndependent("SetupHoldTrigger(gap)", [&](const Cut& cut) { return clockedRun<SetupHoldTrigger<float>>(data, clock, settings, {streamGap(5UZ)}, cut, counters); }, //
            {.events = {}, .counters = {0U, 1U}});
    };

    "Count: every third match is reported however the events arrive"_test = [] {
        std::vector<gr::property_map> events;
        for (std::uint64_t i = 0U; i < 10U; ++i) {
            events.push_back(eventAtMs("edge", 2U * i));
            events.push_back(eventAtMs("other", 2U * i + 1U));
        }
        expectChunkIndependent("Count", [&](const Cut& cut) { return eventRun<Count>(events, {{"filter", std::string("edge")}, {"mode", std::string("every")}, {"n", 3U}}, cut, [](const Count& block) { return std::vector<gr::Size_t>{block.n_matches.value}; }); }, //
            {.events = {{4., -1., {}}, {10., -1., {}}, {16., -1., {}}}, .counters = {10U}});
    };

    "Then: B in time, an A that timed out, an A disarmed, and an A answered"_test = [] {
        const std::vector<gr::property_map> events{
            eventAtMs("arm", 0U),
            eventAtMs("edge", 5U),
            eventAtMs("arm", 10U),
            eventAtMs("heartbeat", 40U),
            eventAtMs("arm", 50U),
            eventAtMs("disarm", 52U),
            eventAtMs("edge", 54U),
            eventAtMs("arm", 60U),
            eventAtMs("edge", 62U),
        };
        const gr::property_map settings{
            {"arm_filter", std::string("arm")},
            {"trigger_filter", std::string("edge")},
            {"disarm_filter", std::string("disarm")},
            {"min_delay", 2e-3},
            {"max_delay", 20e-3},
            {"bounds", std::string("closed")},
        };
        const auto counters = [](const Then& block) { return std::vector<gr::Size_t>{block.n_sequences.value, block.n_timeouts.value, block.n_disarmed.value, block.n_ignored.value, block.n_pending.value}; };
        expectChunkIndependent("Then", [&](const Cut& cut) { return eventRun<Then>(events, settings, cut, counters); }, //
            {.events = {{5., 0., {}}, {30., 10., {}}, {62., 60., {}}}, .counters = {2U, 1U, 1U, 1U, 0U}});
    };

    "While: events pass where the line was high at their trigger_time, whichever way stream and events are cut"_test = [] {
        // high from the crossing at 2.5 to 5.5, at 8.5 to 10.5, and from 13.5 on
        const std::vector<float>            line{0.f, 0.f, 0.f, 5.f, 5.f, 5.f, 0.f, 0.f, 0.f, 5.f, 5.f, 0.f, 0.f, 0.f, 5.f, 5.f};
        const std::vector<gr::property_map> events{
            eventAtMs("e", 1U),
            eventAtMs("e", 3U),
            eventAtMs("e", 5U),
            eventAtMs("e", 7U),
            eventAtMs("e", 9U),
            eventAtMs("e", 11U),
            eventAtMs("e", 14U),
        };
        const auto counters = [](const While<float>& block) { return std::vector<gr::Size_t>{block.n_passed.value, block.n_rejected.value, block.n_unknown.value, block.n_gaps.value}; };
        expectChunkIndependent("While", [&](const Cut& cut) { return gatedRun<While<float>>(line, events, {{"threshold", 2.5f}, {"sample_rate", 1000.f}}, cut, counters); }, //
            {.events = {{3., -1., {}}, {5., -1., {}}, {9., -1., {}}, {14., -1., {}}}, .counters = {4U, 3U, 0U, 0U}, .passed = line});
    };

    "Gate: an event on evtIn acts at the sample its trigger_time falls on, however the stream is cut"_test = [] {
        std::vector<float> samples;
        for (float v = 1.f; v <= 12.f; v += 1.f) {
            samples.push_back(v);
        }
        const std::vector<gr::property_map> events{eventAtMs("start", 3U)};
        const auto                          counters = [](const Gate<float>& block) { return std::vector<gr::Size_t>{block.n_passed.value, block.n_suppressed.value}; };
        expectChunkIndependent("Gate(skipUntil)", [&](const Cut& cut) { return gatedRun<Gate<float>>(samples, events, {{"mode", std::string("skipUntil")}, {"open_filter", std::string("start")}}, cut, counters); }, //
            {.events = {{3., -1., {}}}, .counters = {9U, 3U}, .passed = {4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f}});
        expectChunkIndependent("Gate(once)", [&](const Cut& cut) { return gatedRun<Gate<float>>(samples, events, {{"mode", std::string("once")}, {"open_filter", std::string("start")}, {"n_open", 4U}}, cut, counters); }, //
            {.events = {{3., -1., {}}, {7., -1., {}}}, .counters = {4U, 8U}, .passed = {4.f, 5.f, 6.f, 7.f}});
    };
};

const boost::ut::suite<"acceptance: what a block does when a port is not there"> _disconnected = [] {
    using namespace boost::ut;

    "a block whose event input is unconnected still runs"_test = [] {
        const auto gated = scriptsAtChunkSizes<Gate<float>>("T:a b c d |", {{"mode", std::string("once")}, {"open_filter", std::string("start")}, {"n_open", 2U}});
        expect(!gated.empty() && !gated[0].empty()) << "an unconnected event bus is not an empty one that never ends";

        const auto debounced = scriptsAtChunkSizes<Debounce<float>>("T:a b c d |", {{"filter", std::string("start")}, {"n_samples", 2U}});
        expect(!debounced.empty() && !debounced[0].empty());
    };

    "a block whose event output is unconnected still runs"_test = [] {
        const auto reported = scriptsAtChunkSizes<TagToEvent<float>>("T:a b c |", {{"filter", std::string("start")}});
        expect(!reported.empty() && !reported[0].empty()) << "the events go nowhere, and that is not an error";
    };

    "the stream terminator reaches the sink in every case"_test = [] {
        const auto answers = scriptsAtChunkSizes<SampleAndHold<float>>("T:a b c |", {{"filter", std::string("start")}});
        for (const std::string& answer : answers) {
            expect(answer.ends_with("|")) << std::format("'{}' does not say how the stream ended", answer);
        }
    };
};

const boost::ut::suite<"acceptance: a settings change between runs"> _settings = [] {
    using namespace boost::ut;

    "a changed setting takes effect and leaves no state behind"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     block = fixture.emplace<TakeN<float>>({{"filter", std::string("start")}, {"n", 2U}});
        block.settings().init();
        std::ignore = block.settings().applyStagedParameters();
        expect(eq(block.n.value, 2U));

        expect(block.settings().set({{"n", gr::Size_t(3)}}).empty());
        std::ignore = block.settings().activateContext();
        std::ignore = block.settings().applyStagedParameters();
        expect(eq(block.n.value, 3U)) << "the new setting is the one that applies";

        const auto two   = scriptsAtChunkSizes<TakeN<float>>("a T:b c d e |", {{"filter", std::string("start")}, {"n", 2U}});
        const auto three = scriptsAtChunkSizes<TakeN<float>>("a T:b c d e |", {{"filter", std::string("start")}, {"n", 3U}});
        expect(neq(two[0], three[0])) << "and the difference is visible in what comes out";
    };
};
} // namespace qa_acceptance

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
