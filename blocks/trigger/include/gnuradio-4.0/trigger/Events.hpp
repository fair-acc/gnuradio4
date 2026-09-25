#ifndef GNURADIO_TRIGGER_EVENTS_HPP
#define GNURADIO_TRIGGER_EVENTS_HPP

#include <cstddef>
#include <deque>
#include <expected>
#include <format>
#include <memory_resource>
#include <string>
#include <string_view>

#include <gnuradio-4.0/Logger.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/TriggerMatcher.hpp>

namespace gr::blocks::trigger::detail {

using MatchState = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;

[[nodiscard]] inline std::expected<MatchState, gr::Error> compileFilter(std::string_view definition, std::string_view matchMode) {
    std::optional<gr::trigger::BasicTriggerNameCtxMatcher::MatchMode> mode;
    if (matchMode == "pulse") {
        mode = gr::trigger::BasicTriggerNameCtxMatcher::MatchMode::pulse;
    } else if (matchMode == "interval") {
        mode = gr::trigger::BasicTriggerNameCtxMatcher::MatchMode::interval;
    }
    return gr::trigger::BasicTriggerNameCtxMatcher::compile(definition, mode);
}

inline void compileFilterInto(std::string_view definition, std::string_view matchMode, MatchState& into, std::string_view blockName) {
    if (const auto compiled = compileFilter(definition, matchMode)) {
        into = *compiled;
    } else {
        gr::log::warning("{}: {}", blockName, compiled.error().message);
    }
}

[[nodiscard]] inline bool compileOptionalFilter(std::string_view definition, std::string_view matchMode, MatchState& into, std::string_view blockName) {
    if (!definition.empty()) {
        if (const auto compiled = compileFilter(definition, matchMode)) {
            into = *compiled;
            return true;
        } else {
            gr::log::warning("{}: {}", blockName, compiled.error().message);
        }
    }
    into = MatchState{};
    return false;
}

[[nodiscard]] inline bool matches(MatchState& state, const gr::property_map_view& candidate) { //
    return gr::trigger::BasicTriggerNameCtxMatcher::match(state, candidate) == gr::trigger::MatchResult::Matching;
}

[[nodiscard]] inline bool acceptsTrigger(MatchState& state, bool filtered, const gr::property_map_view& candidate) {
    if (!candidate.contains(std::string_view{gr::tag::TRIGGER_NAME.key()})) {
        return false;
    }
    return !filtered || matches(state, candidate);
}

[[nodiscard]] constexpr bool isIncompleteTrigger(const gr::property_map_view& tagMap) noexcept { //
    return !tagMap.contains(gr::tag::TRIGGER_TIME.key()) || !tagMap.contains(gr::tag::TRIGGER_OFFSET.key());
}

[[nodiscard]] inline gr::property_map makeEvent(std::string_view name, std::string_view source, const gr::property_map_view& from = gr::property_map_view{}) {
    gr::property_map event;
    event[std::string(gr::tag::TRIGGER_NAME.key())] = std::pmr::string(name);
    event[std::string("source")]                    = std::pmr::string(source);
    for (const auto& key : {gr::tag::TRIGGER_TIME.key(), gr::tag::TRIGGER_OFFSET.key(), gr::tag::TRIGGER_TIME_ERROR.key(), gr::tag::CONTEXT.key()}) {
        if (const auto value = from.find_value(key)) {
            event[std::string(key)] = gr::pmt::Value(value.value());
        }
    }
    return event;
}

[[nodiscard]] inline gr::property_map makeErrorEvent(std::string_view reason, std::string_view source) {
    gr::property_map details;
    details[std::string("reason")] = std::string(reason);
    details[std::string("source")] = std::pmr::string(source);

    gr::property_map event;
    event[std::string(gr::tag::TRIGGER_NAME.key())]      = std::pmr::string("error");
    event[std::string(gr::tag::TRIGGER_META_INFO.key())] = std::move(details);
    return event;
}

[[nodiscard]] inline bool reportError(auto& evtOutSpan, std::size_t index, std::string_view blockName, std::string_view source, std::string_view reason) {
    gr::log::warning("{}: {}", blockName, reason);
    if (index >= evtOutSpan.size()) {
        return false;
    }
    return gr::emitEvent(evtOutSpan, index, makeErrorEvent(reason, source)).has_value();
}

[[nodiscard]] inline std::size_t auditTrigger(const gr::property_map_view& condition, auto& invalidCount, auto& evtOutSpan, std::size_t index, std::string_view blockName, std::string_view source) {
    if (!isIncompleteTrigger(condition)) {
        return 0UZ;
    }
    invalidCount = invalidCount + 1U;
    return reportError(evtOutSpan, index, blockName, source, std::format("a trigger lacking '{}' or '{}' was acted on", gr::tag::TRIGGER_TIME.key(), gr::tag::TRIGGER_OFFSET.key())) ? 1UZ : 0UZ;
}

inline void auditTrigger(const gr::property_map_view& condition, auto& invalidCount, std::string_view blockName) {
    if (isIncompleteTrigger(condition)) {
        invalidCount = invalidCount + 1U;
        gr::log::warning("{}: a trigger lacking '{}' or '{}' was acted on", blockName, gr::tag::TRIGGER_TIME.key(), gr::tag::TRIGGER_OFFSET.key());
    }
}

/**
 * The events a block has decided on but not yet handed over.
 *
 * An output span holds what it holds: a block that finds several triggers in one work call cannot publish them all,
 * and dropping the surplus loses exactly the events a downstream measurement needed. They wait here instead, bounded,
 * losing the oldest first when the consumer cannot keep up, and one slot is always kept back so the loss is reported
 * however full the output is.
 */
struct PendingEvents {
    std::size_t                  capacity = 64UZ;
    std::deque<gr::property_map> queue;
    std::uint32_t                dropped        = 0U;
    bool                         lossUnreported = false;

    void clear() noexcept {
        queue.clear();
        dropped        = 0U;
        lossUnreported = false;
    }

    void push(gr::property_map&& event) {
        while (!queue.empty() && queue.size() >= capacity) {
            queue.pop_front();
            ++dropped;
            lossUnreported = true;
        }
        queue.push_back(std::move(event));
    }

    [[nodiscard]] std::size_t size() const noexcept { return queue.size(); }

    [[nodiscard]] std::size_t drainInto(auto& evtSpan, std::size_t index, std::string_view blockName, std::string_view source) {
        const std::size_t capacityLeft = evtSpan.size();
        const std::size_t reserved     = lossUnreported && capacityLeft > index ? 1UZ : 0UZ;
        std::size_t       published    = index;
        while (!queue.empty() && published + reserved < capacityLeft) {
            if (!gr::emitEvent(evtSpan, published, gr::property_map_view{queue.front()})) {
                break;
            }
            queue.pop_front();
            ++published;
        }
        if (lossUnreported && published < capacityLeft) {
            const std::string reason = std::format("{} event(s) dropped: the queue of {} was full", dropped, capacity);
            if (reportError(evtSpan, published, blockName, source, reason)) {
                ++published;
                lossUnreported = false;
            }
        }
        return published - index;
    }
};

} // namespace gr::blocks::trigger::detail

#endif // GNURADIO_TRIGGER_EVENTS_HPP
