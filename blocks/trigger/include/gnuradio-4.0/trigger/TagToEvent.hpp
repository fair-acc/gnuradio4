#ifndef GNURADIO_TRIGGER_TAGTOEVENT_HPP
#define GNURADIO_TRIGGER_TAGTOEVENT_HPP

#include <algorithm>
#include <complex>
#include <cstdint>
#include <format>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/TriggerMatcher.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>

namespace gr::blocks::trigger {

/** glob match of the whole `text`: `*` any run of characters (also none), `?` exactly one */
[[nodiscard]] constexpr bool matchesGlob(std::string_view text, std::string_view pattern) noexcept {
    std::size_t t        = 0UZ;
    std::size_t p        = 0UZ;
    std::size_t starAt   = std::string_view::npos;
    std::size_t starText = 0UZ;
    while (t < text.size()) {
        if (p < pattern.size() && (pattern[p] == '?' || pattern[p] == text[t])) {
            ++t;
            ++p;
        } else if (p < pattern.size() && pattern[p] == '*') {
            starAt   = p++;
            starText = t;
        } else if (starAt != std::string_view::npos) {
            p = starAt + 1UZ;
            t = ++starText;
        } else {
            return false;
        }
    }
    while (p < pattern.size() && pattern[p] == '*') {
        ++p;
    }
    return p == pattern.size();
}

GR_REGISTER_BLOCK(gr::blocks::trigger::TagToEvent, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double> ])

template<typename T>
struct TagToEvent : gr::Block<TagToEvent<T>> {
    using Description = Doc<R"(@brief turn the tags a trigger filter or a key pattern selects into events, leaving the stream untouched

    in      ─a──T:b──c──U:d─▶        (no RxMarbles equivalent: a change of carrier, not of content)
    out     ─a──T:b──c──U:d─▶
    evtOut  ─────E──────────▶        filter = the tag's name

The bridge out of the stream: a condition found in the samples becomes an event several blocks can read, without the stream
itself changing. A tag the trigger `filter` accepts becomes a dated trigger event (its name, or `trigger_name` when set; with
`include_tag` the tag's fields too); a tag with a key matching one of `key_patterns` goes out whole, a diagnostic tap of what
travels past a point (e.g. `!gr:*`, the keys outside the canonical ones). Events wait in a bounded queue while `evtOut` is full,
the oldest dropped and counted.
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}};
    gr::PortIn<T>    in;
    gr::PortOut<T>   out;

    A<std::pmr::string, "filter", Doc<"trigger filter, e.g. 'start/ctx' or 'start^/stop'; empty: none">>                                                  filter;
    A<std::pmr::string, "match mode", Doc<"'pulse' or 'interval'">>                                                                                       match_mode;
    A<std::vector<std::string>, "key_patterns", Doc<"globs, any one matching: `*` any run, `?` one character, a leading `!` the keys the rest does not">> key_patterns;
    gr::TriggerName                                                                                                                                       trigger_name;
    A<bool, "include tag", Doc<"shallow-merge the original tag fields into a filter's event">>                                                            include_tag        = false;
    A<gr::Size_t, "max pending events", Doc<"events held while evtOut is full">>                                                                          max_pending_events = 64U;
    A<gr::Size_t, "n events dropped", Doc<"events the queue lost, oldest first">>                                                                         n_events_dropped   = 0U;
    A<gr::Size_t, "n invalid triggers", Doc<"undated triggers">>                                                                                          n_invalid_triggers = 0U;

    GR_MAKE_REFLECTABLE(TagToEvent, evtOut, in, out, filter, match_mode, key_patterns, trigger_name, include_tag, max_pending_events, n_events_dropped, n_invalid_triggers);

    gr::trigger::BasicTriggerNameCtxMatcher::MatchState _matcher{};
    detail::PendingEvents                               _pending;

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        std::ignore       = detail::compileOptionalFilter(filter.value, match_mode.value, _matcher, "TagToEvent");
        _pending.capacity = static_cast<std::size_t>(max_pending_events);
        _pending.clear();
        n_events_dropped   = 0U;
        n_invalid_triggers = 0U;
    }

    gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& evtSpan, gr::OutputSpanLike auto& outSpan) {
        const std::size_t n = std::min(inSpan.size(), outSpan.size());
        for (const auto& tag : inSpan.rawTags()) {
            if (tag.index - inSpan.streamIndex >= n) {
                continue;
            }
            if (const auto result = gr::trigger::BasicTriggerNameCtxMatcher::match(_matcher, tag.map); result != gr::trigger::MatchResult::Ignore) {
                if (detail::isIncompleteTrigger(tag.map)) {
                    const std::string reason = std::format("a trigger lacking '{}' or '{}' was reported", gr::tag::TRIGGER_TIME.key(), gr::tag::TRIGGER_OFFSET.key());
                    n_invalid_triggers       = n_invalid_triggers + 1U;
                    gr::log::warning("TagToEvent: {}", reason);
                    _pending.push(detail::makeErrorEvent(reason, this->unique_name.value()));
                }
                _pending.push(triggerEvent(tag.map, result));
            } else if (keysMatch(tag.map)) {
                _pending.push(gr::property_map{tag.map});
            }
        }
        const std::size_t published = _pending.drainInto(evtSpan, 0UZ, this->name, this->unique_name);
        n_events_dropped            = static_cast<gr::Size_t>(_pending.dropped);

        std::ranges::copy(inSpan | std::views::take(n), outSpan.begin());
        outSpan.publish(n);
        evtSpan.publish(published);
        if (!inSpan.consume(n)) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }

private:
    [[nodiscard]] bool keysMatch(const gr::property_map_view& tagMap) const {
        const auto keyMatches = [this](std::string_view key) { return std::ranges::any_of(key_patterns.value, [key](std::string_view pattern) { return pattern.starts_with('!') ? !matchesGlob(key, pattern.substr(1UZ)) : matchesGlob(key, pattern); }); };
        return std::ranges::any_of(tagMap, [&](const auto& entry) { return keyMatches(std::string_view{entry.first}); });
    }

    [[nodiscard]] gr::property_map triggerEvent(const gr::property_map_view& tagMap, gr::trigger::MatchResult result) const {
        const auto       fromTag = tagMap.template get_if<std::string_view>(gr::tag::TRIGGER_NAME.key());
        gr::property_map event   = detail::makeEvent(trigger_name.value.empty() && fromTag ? *fromTag : std::string_view{trigger_name.value}, this->unique_name.value(), tagMap);
        if (match_mode.value == "interval") {
            event[std::string("state")] = std::pmr::string(result == gr::trigger::MatchResult::Matching ? "active" : "inactive");
        }
        if (include_tag) {
            for (const auto& key : tagMap.keys()) {
                if (auto value = tagMap.find_value(key); value && !event.contains(key)) {
                    event[std::string(key)] = gr::pmt::Value(value.value());
                }
            }
        }
        return event;
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_TAGTOEVENT_HPP
