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

/** the fixtures every qa_ file in this directory shares: one scripted event source, one recording event sink, one collecting stream sink */
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
            const auto triggerName = gr::property_map_view{event}.get_if<std::string_view>(std::string_view{gr::tag::kTriggerName.key()});
            return triggerName ? std::string(*triggerName) : std::string{};
        });
        return std::vector<std::string>(named.begin(), named.end());
    }

    /** the trigger times of the events that carry one, in arrival order */
    [[nodiscard]] std::vector<std::uint64_t> times() const {
        std::vector<std::uint64_t> found;
        for (const gr::property_map& event : _events) {
            if (const auto at = gr::property_map_view{event}.get_if<std::uint64_t>(std::string_view{gr::tag::kTriggerTime.key()})) {
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
            if (const auto at = view.get_if<std::uint64_t>(std::string_view{gr::tag::kTriggerTime.key()})) {
                const auto triggerName = view.get_if<std::string_view>(std::string_view{gr::tag::kTriggerName.key()});
                found.emplace_back(*at, triggerName ? std::string(*triggerName) : std::string(whenUnnamed));
            }
        }
        return found;
    }

    [[nodiscard]] std::size_t countOf(std::string_view wanted) const {
        return static_cast<std::size_t>(std::ranges::count_if(_events, [wanted](const gr::property_map& event) { //
            const auto triggerName = gr::property_map_view{event}.get_if<std::string_view>(std::string_view{gr::tag::kTriggerName.key()});
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
        const auto meta = gr::property_map_view{_events[index]}.get_if<gr::property_map>(std::string_view{gr::tag::kTriggerMetaInfo.key()});
        return meta ? *meta : gr::property_map{};
    }
};

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

/** the clock of the 1 kHz streams the tests anchor at 1 s: sample i is at kStreamStart + i * kNsPerSample */
inline constexpr std::uint64_t kStreamStart = 1'000'000'000U;
inline constexpr double        kNsPerSample = 1e6;
inline constexpr float         kSampleRate  = 1000.f;

/** whether an event names a `source`, at its root or inside `trigger_meta_info`: none does */
[[nodiscard]] inline bool carriesSource(const gr::property_map& event) {
    const gr::property_map_view view{event};
    const auto                  meta = view.get_if<gr::property_map_view>(std::string_view{gr::tag::kTriggerMetaInfo.key()});
    return view.contains(std::string_view{"source"}) || (meta && meta->contains(std::string_view{"source"}));
}

/** whether an event carries a `trigger_meta_info` with nothing in it: it is written only where there is content */
[[nodiscard]] inline bool carriesEmptyMeta(const gr::property_map& event) {
    const auto meta = gr::property_map_view{event}.get_if<gr::property_map_view>(std::string_view{gr::tag::kTriggerMetaInfo.key()});
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

[[nodiscard]] inline gr::property_map eventNamed(std::string name, std::uint64_t atNs) { return gr::property_map{{std::string(gr::tag::kTriggerName.key()), std::move(name)}, {std::string(gr::tag::kTriggerTime.key()), atNs}}; }

/** as above, and stamped with the uncertainty on that time */
[[nodiscard]] inline gr::property_map eventNamed(std::string name, std::uint64_t atNs, std::uint64_t errorNs) { return gr::property_map{{std::string(gr::tag::kTriggerName.key()), std::move(name)}, {std::string(gr::tag::kTriggerTime.key()), atNs}, {std::string(gr::tag::kTriggerTimeError.key()), errorNs}}; }

[[nodiscard]] inline gr::property_map undatedEvent(std::string named) { return gr::property_map{{std::string(gr::tag::kTriggerName.key()), std::move(named)}}; }

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
