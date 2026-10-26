#ifndef GNURADIO_TRIGGER_TEST_TRIGGERTEST_HPP
#define GNURADIO_TRIGGER_TEST_TRIGGERTEST_HPP

#include <algorithm>
#include <cstdint>
#include <format>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <boost/ut.hpp>

#include "Marble.hpp"
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Message.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/test/EventMarbles.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>

/**
 * the fixtures every qa_ file in this directory shares: one scripted event source, one recording event sink, one
 * collecting stream sink, and the acceptance property that a block answers the same however the stream was cut
 */
namespace gr::trigger_test {

/** a block of a finished graph, readable while this owns the graph: a block dies with its `GraphFixture` */
template<typename TBlock>
struct Kept {
    std::unique_ptr<::gr::testing::GraphFixture<>> fixture;
    TBlock*                                        block = nullptr;

    TBlock*     operator->() const { return block; }
    friend bool operator==(const Kept& kept, std::nullptr_t) { return kept.block == nullptr; }
};

/** a marble script's whitespace-separated tokens, each still carrying its `TAG:` prefix if it had one */
[[nodiscard]] inline std::vector<std::string_view> tokensOf(std::string_view script) {
    std::vector<std::string_view> tokens;
    for (const auto token : std::views::split(script, ' ')) {
        const std::string_view text(std::to_address(token.begin()), token.size());
        if (!text.empty()) {
            tokens.push_back(text);
        }
    }
    return tokens;
}

/** a token without its tag prefix: `T+U:a` is the sample `a` */
[[nodiscard]] inline std::string_view valueOf(std::string_view token) {
    const std::size_t marker = token.rfind(':');
    return marker == std::string_view::npos ? token : token.substr(marker + 1UZ);
}

[[nodiscard]] inline bool isTerminator(std::string_view token) { return token == "|" || token.front() == '#'; }

/** the samples of a script, tags and terminator stripped: what a block emitted, ignoring where it marked it */
[[nodiscard]] inline std::vector<std::string> symbolsOf(std::string_view script) {
    auto samples = tokensOf(script) | std::views::filter(std::not_fn(isTerminator)) | std::views::transform([](std::string_view token) { return std::string(valueOf(token)); });
    return std::vector<std::string>(samples.begin(), samples.end());
}

/** a baseline closed by the terminator the sink writes, so a case about samples still compares whole scripts */
[[nodiscard]] inline std::string completed(std::string_view samples) { return samples.empty() ? std::string("|") : std::string(samples) + " |"; }

/** plays a written list of events, which is what a timing receiver or an edge detector looks like from downstream */
struct EventScript : gr::Block<EventScript> {
    /**
     * an event output claims one slot per work call unless told otherwise, and a script must place all of its events
     * before whatever drives the graph ends
     */
    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 16UZ}};

    GR_MAKE_REFLECTABLE(EventScript, evtOut);

    std::vector<gr::property_map> _events;
    std::size_t                   _published    = 0UZ;
    bool                          _stopWhenDone = true;
    /**
     * lower this to 1 to space the events out over work calls, which is what separates two requests that would
     * otherwise land on the same sample and merge
     */
    std::size_t _maxPerWorkCall = std::numeric_limits<std::size_t>::max();

    gr::work::Status processBulk(gr::OutputSpanLike auto& evtSpan) {
        std::size_t emitted = 0UZ;
        while (_published < _events.size() && emitted < std::min(evtSpan.size(), _maxPerWorkCall)) {
            if (!gr::emit(evtSpan, emitted, gr::property_map_view{_events[_published]})) {
                break;
            }
            ++_published;
            ++emitted;
        }
        evtSpan.publish(emitted);
        if (_stopWhenDone && _published >= _events.size()) {
            this->requestStop();
            return gr::work::Status::DONE;
        }
        return gr::work::Status::OK;
    }
};

/** records whole events as they arrive on a bus, so a test can assert the order, the dating and the payload */
struct EventTap : gr::Block<EventTap> {
    gr::EventPortIn evtIn;

    GR_MAKE_REFLECTABLE(EventTap, evtIn);

    std::vector<gr::property_map> _events;

    gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan) {
        for (const gr::property_map_view& event : evtSpan) {
            if (!event.empty()) {
                _events.emplace_back(event);
            }
        }
        if (!evtSpan.consume(evtSpan.size())) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }

    [[nodiscard]] std::size_t size() const { return _events.size(); }

    /** the trigger name of every event, in arrival order; an unnamed event contributes an empty string */
    [[nodiscard]] std::vector<std::string> names() const {
        auto named = _events | std::views::transform([](const gr::property_map& event) {
            const auto triggerName = gr::property_map_view{event}.get_if<std::string_view>(std::string_view{gr::tag::TRIGGER_NAME.key()});
            return triggerName ? std::string(*triggerName) : std::string{};
        });
        return std::vector<std::string>(named.begin(), named.end());
    }

    /** the trigger times of the events that carry one, in arrival order */
    [[nodiscard]] std::vector<std::uint64_t> times() const {
        std::vector<std::uint64_t> found;
        for (const gr::property_map& event : _events) {
            if (const auto at = gr::property_map_view{event}.get_if<std::uint64_t>(std::string_view{gr::tag::TRIGGER_TIME.key()})) {
                found.push_back(*at);
            }
        }
        return found;
    }

    /** (time, name) for every dated event, which is what a test of ordering asserts on */
    [[nodiscard]] std::vector<std::pair<std::uint64_t, std::string>> dated(std::string_view whenUnnamed = "") const {
        std::vector<std::pair<std::uint64_t, std::string>> found;
        for (const gr::property_map& event : _events) {
            const gr::property_map_view view{event};
            if (const auto at = view.get_if<std::uint64_t>(std::string_view{gr::tag::TRIGGER_TIME.key()})) {
                const auto triggerName = view.get_if<std::string_view>(std::string_view{gr::tag::TRIGGER_NAME.key()});
                found.emplace_back(*at, triggerName ? std::string(*triggerName) : std::string(whenUnnamed));
            }
        }
        return found;
    }

    [[nodiscard]] std::size_t countOf(std::string_view wanted) const {
        return static_cast<std::size_t>(std::ranges::count_if(_events, [wanted](const gr::property_map& event) { //
            const auto triggerName = gr::property_map_view{event}.get_if<std::string_view>(std::string_view{gr::tag::TRIGGER_NAME.key()});
            return triggerName && *triggerName == wanted;
        }));
    }

    /** how many events carry `key` at all: the schema question, asked of one key */
    [[nodiscard]] std::size_t countCarrying(std::string_view key) const {
        return static_cast<std::size_t>(std::ranges::count_if(_events, [key](const gr::property_map& event) { return gr::property_map_view{event}.contains(key); }));
    }

    [[nodiscard]] gr::property_map metaOf(std::size_t index) const {
        if (index >= _events.size()) {
            return {};
        }
        const auto meta = gr::property_map_view{_events[index]}.get_if<gr::property_map>(std::string_view{gr::tag::TRIGGER_META_INFO.key()});
        return meta ? *meta : gr::property_map{};
    }
};

/**
 * one way to cut a stream: `maxSamples` caps what the block under test reads per work call (0 = uncapped),
 * `seed` makes the source publish in pseudo-random runs of 1 to 9 samples (0 = whatever the source has at once)
 */
struct Cut {
    std::size_t   maxSamples = 0UZ;
    std::uint32_t seed       = 0U;
};

/** the cuts every chunk-independence scenario runs: the whole stream, single samples, small and prime caps, seeded random runs, and both at once */
[[nodiscard]] inline const std::vector<Cut>& chunkCuts() {
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

/** plays a vector of samples with the given tags, publishing in the runs a `seed` dictates: where a stream is cut is the test's choice, not the scheduler's */
template<typename T>
struct PartitionSource : gr::Block<PartitionSource<T>> {
    gr::PortOut<T> out;

    GR_MAKE_REFLECTABLE(PartitionSource, out);

    std::vector<T>                                        _samples;
    std::vector<std::pair<std::size_t, gr::property_map>> _tags; /// (sample index, tag)
    std::uint32_t                                         _seed     = 0U;
    std::size_t                                           _position = 0UZ;

    [[nodiscard]] std::size_t runAt(std::size_t position) const noexcept {
        if (_seed == 0U) {
            return std::numeric_limits<std::size_t>::max();
        }
        std::uint64_t mixed = (static_cast<std::uint64_t>(_seed) << 32U) ^ position;
        mixed               = (mixed ^ (mixed >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        mixed               = (mixed ^ (mixed >> 27U)) * 0x94d049bb133111ebULL;
        return 1UZ + static_cast<std::size_t>((mixed ^ (mixed >> 31U)) % 9ULL);
    }

    gr::work::Status processBulk(gr::OutputSpanLike auto& outSpan) {
        const std::size_t n = std::min({outSpan.size(), _samples.size() - _position, runAt(_position)});
        for (std::size_t i = 0UZ; i < n; ++i) {
            outSpan[i] = _samples[_position + i];
        }
        for (const auto& [index, tag] : _tags) {
            if (index >= _position && index < _position + n) {
                outSpan.publishTag(tag, index - _position);
            }
        }
        outSpan.publish(n);
        _position += n;
        if (_position >= _samples.size()) {
            this->requestStop();
            return gr::work::Status::DONE;
        }
        return gr::work::Status::OK;
    }
};

/** the sample 0 time of every chunk-independence stream, at 1 kHz: sample i is at kStreamStart + i ms */
inline constexpr std::uint64_t kStreamStart = 1'000'000'000U;
inline constexpr double        kNsPerSample = 1e6;

/** the anchor every such stream starts with */
[[nodiscard]] inline std::pair<std::size_t, gr::property_map> streamAnchor() { return {0UZ, gr::property_map{{std::string(gr::tag::TRIGGER_TIME.key()), kStreamStart}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}, {std::string("sample_rate"), 1000.f}}}; }

[[nodiscard]] inline std::pair<std::size_t, gr::property_map> streamGap(std::size_t at) { return {at, gr::property_map{{std::string(gr::tag::N_DROPPED_SAMPLES.key()), gr::Size_t{7U}}}}; }

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

[[nodiscard]] inline double samplesSinceStart(const gr::property_map_view& tuple) {
    const auto ns     = tuple.get_if<std::uint64_t>(gr::tag::TRIGGER_TIME.key());
    const auto offset = tuple.get_if<float>(gr::tag::TRIGGER_OFFSET.key());
    if (!ns) {
        return -1.;
    }
    const double offsetNs = offset ? static_cast<double>(*offset) * 1e9 : 0.;
    return (static_cast<double>(*ns) - static_cast<double>(kStreamStart) + offsetNs) / kNsPerSample;
}

[[nodiscard]] inline std::vector<Detection> detectionsOf(const EventTap& tap) {
    std::vector<Detection> found;
    for (const gr::property_map& event : tap._events) {
        const gr::property_map_view view{event};
        const auto                  meta = view.get_if<gr::property_map_view>(gr::tag::TRIGGER_META_INFO.key());
        const auto                  name = view.get_if<std::string_view>(gr::tag::TRIGGER_NAME.key());
        found.push_back({samplesSinceStart(view), meta ? samplesSinceStart(*meta) : -1., name ? std::string(*name) : std::string{}});
    }
    return found;
}

[[nodiscard]] inline std::string describe(const DetectorRun& run) {
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

/** whether an event names a `source`, at its root or inside `trigger_meta_info`: none does */
[[nodiscard]] inline bool carriesSource(const gr::property_map& event) {
    const gr::property_map_view view{event};
    const auto                  meta = view.get_if<gr::property_map_view>(std::string_view{gr::tag::TRIGGER_META_INFO.key()});
    return view.contains(std::string_view{"source"}) || (meta && meta->contains(std::string_view{"source"}));
}

/** whether an event carries a `trigger_meta_info` with nothing in it: it is written only where there is content */
[[nodiscard]] inline bool carriesEmptyMeta(const gr::property_map& event) {
    const auto meta = gr::property_map_view{event}.get_if<gr::property_map_view>(std::string_view{gr::tag::TRIGGER_META_INFO.key()});
    return meta && meta->empty();
}

/** keeps whatever arrived on a stream port, so a test can measure the windows, sets or samples rather than only count them */
template<typename T>
struct CollectingSink : gr::Block<CollectingSink<T>> {
    gr::PortIn<T> in;

    GR_MAKE_REFLECTABLE(CollectingSink, in);

    std::vector<T> _collected;

    void processOne(T item) { _collected.push_back(std::move(item)); }
};

/** the error messages the blocks of a fixture's graph sent, read from the scheduler's message output; construct from the scheduler before it runs */
struct ErrorMessages {
    gr::MsgPortIn port;

    template<typename TScheduler>
    explicit ErrorMessages(TScheduler& scheduler) {
        boost::ut::expect(boost::ut::fatal(scheduler.msgOut.connect(port).has_value()));
    }

    [[nodiscard]] std::vector<std::string> texts() {
        std::vector<std::string> found;
        const std::size_t        available = port.streamReader().available();
        if (available == 0UZ) {
            return found;
        }
        auto span = port.streamReader().get(available);
        for (const gr::Message& message : span) {
            if (!message.data.has_value()) {
                found.push_back(message.data.error().message);
            } else if (const auto reason = message.data->template get_if<std::string_view>(gr::blocks::trigger::detail::kErrorKey)) {
                found.emplace_back(*reason);
            }
        }
        std::ignore = span.consume(available);
        return found;
    }

    [[nodiscard]] std::size_t countContaining(std::string_view fragment) {
        const auto found = texts();
        return static_cast<std::size_t>(std::ranges::count_if(found, [fragment](const std::string& text) { return text.contains(fragment); }));
    }
};

[[nodiscard]] inline gr::property_map eventNamed(std::string name, std::uint64_t atNs) { return gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::move(name)}, {std::string(gr::tag::TRIGGER_TIME.key()), atNs}}; }

/** as above, and stamped with the uncertainty on that time */
[[nodiscard]] inline gr::property_map eventNamed(std::string name, std::uint64_t atNs, std::uint64_t errorNs) { return gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::move(name)}, {std::string(gr::tag::TRIGGER_TIME.key()), atNs}, {std::string(gr::tag::TRIGGER_TIME_ERROR.key()), errorNs}}; }

[[nodiscard]] inline gr::property_map undatedEvent(std::string named) { return gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::move(named)}}; }

/**
 * the sample and tag symbols every acceptance script is written in; a function-local static, because a namespace-scope
 * map is not yet built when a statically executed suite body reads it -- which is why every such map in this directory
 * is a function
 */
[[nodiscard]] inline const gr::property_map& acceptanceValues() {
    static const gr::property_map map{{"a", 0.f}, {"b", 1.f}, {"c", 5.f}, {"d", 0.f}, {"e", 5.f}, {"f", 0.f}};
    return map;
}

[[nodiscard]] inline const gr::property_map& acceptanceTags() {
    static const gr::property_map map{
        {"T", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("start")}, {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1'000'000'000U}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}},
        {"U", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("other")}, {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1'000'000'000U}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}},
        {"V", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("start")}}},
    };
    return map;
}

/** the tag placements the specification asks every block to survive */
[[nodiscard]] inline const std::vector<std::pair<std::string, std::string>>& tagEdgeCases() {
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

/**
 * Draws what a block let through, against where it came from.
 *
 * Both scripts name their samples with distinct symbols, so a sample on the output can be drawn at the position it
 * held on the input: which is the whole question for a block that forwards, suppresses or holds samples rather than
 * changing them. The output row is where a marble diagram earns its place -- an assertion on the script says *what*
 * survived, the drawing says *when*.
 */
inline void drawSubset(std::string_view title, std::string_view condition, std::string_view inScript, std::string_view outScript) {
    const std::vector<std::string> before = symbolsOf(inScript);
    const std::vector<std::string> after  = symbolsOf(outScript);

    gr::testing::MarbleDiagram diagram{std::string(title)};
    diagram.unit = "sample";
    auto& inRow  = diagram.row("in");
    for (std::size_t i = 0UZ; i < before.size(); ++i) {
        inRow.at(i, before[i]);
    }
    inRow.completes();
    diagram.condition(std::string(condition));

    auto&       outRow = diagram.row("out");
    std::size_t from   = 0UZ;
    for (const std::string& symbol : after) {
        const auto found = std::ranges::find(before.begin() + static_cast<std::ptrdiff_t>(from), before.end(), symbol);
        if (found == before.end()) {
            continue;
        }
        const std::size_t at = static_cast<std::size_t>(std::ranges::distance(before.begin(), found));
        outRow.at(at, symbol);
        from = at;
    }
    outRow.completes();
    diagram.print();
}

} // namespace gr::trigger_test

#endif // GNURADIO_TRIGGER_TEST_TRIGGERTEST_HPP
