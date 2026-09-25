#ifndef GNURADIO_TRIGGER_EVENTS_HPP
#define GNURADIO_TRIGGER_EVENTS_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <expected>
#include <format>
#include <iterator>
#include <memory_resource>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include <gnuradio-4.0/Logger.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/TriggerMatcher.hpp>
#include <gnuradio-4.0/device/ExecutionStrategy.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>

namespace gr::blocks::trigger {

enum class EdgeSelection : std::uint8_t { rising, falling, both };

} // namespace gr::blocks::trigger

namespace gr::blocks::trigger::detail {

/** events a block queues between work calls: fixed, as a kernel's slots are */
inline constexpr std::size_t kMaxQueuedEvents = 64UZ;

using MatchState = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;

[[nodiscard]] inline std::expected<MatchState, gr::Error> compileFilter(std::string_view definition, std::string_view matchMode) { return gr::trigger::BasicTriggerNameCtxMatcher::compile(definition, gr::meta::parseEnum<gr::trigger::BasicTriggerNameCtxMatcher::MatchMode>(matchMode)); }

inline bool compileFilterInto(std::string_view definition, std::string_view matchMode, MatchState& into, std::string_view blockName) {
    const auto compiled = compileFilter(definition, matchMode);
    if (!compiled) {
        gr::log::warning("{}: {}", blockName, compiled.error().message);
        return false;
    }
    into = *compiled;
    return true;
}

[[nodiscard]] inline bool compileOptionalFilter(std::string_view definition, std::string_view matchMode, MatchState& into, std::string_view blockName) {
    if (!definition.empty() && compileFilterInto(definition, matchMode, into, blockName)) {
        return true;
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

/**
 * a bounded string kept in the block itself, so a device mirror carries it -- a block's `unique_name`, a message the
 * host composed; truncated to the array
 */
struct InlineText {
    std::array<char, 256UZ> chars{};

    void assign(std::string_view name) noexcept {
        chars.fill('\0');
        std::ranges::copy(name.substr(0UZ, chars.size() - 1UZ), chars.begin());
    }

    [[nodiscard]] constexpr std::string_view view() const noexcept {
        /** `find` on chars becomes a memchr a GPU cannot link */
        const auto end = std::ranges::find_if(chars, [](char c) { return c == '\0'; });
        return {chars.data(), static_cast<std::size_t>(end - chars.begin())};
    }
};

inline constexpr std::uint32_t    kEventKeys         = 6U;
inline constexpr std::string_view kIncompleteTrigger = "a trigger lacking 'gr:trigger_time' or 'gr:trigger_offset' was acted on";

/** the event a block reports for a trigger it acted on, dated as the trigger was; false when a field did not fit */
[[nodiscard]] inline bool writeEvent(gr::property_map_view& event, std::string_view name, std::string_view source, const gr::property_map_view& from) noexcept {
    bool       complete = event.try_emplace(gr::tag::TRIGGER_NAME.key(), name) && event.try_emplace(std::string_view{"source"}, source);
    const auto copy     = [&]<typename TValue>(std::string_view key) {
        if (const auto value = from.template get_if<TValue>(key)) {
            complete = complete && event.try_emplace(key, *value);
        }
    };
    copy.template operator()<std::uint64_t>(gr::tag::TRIGGER_TIME.key());
    copy.template operator()<float>(gr::tag::TRIGGER_OFFSET.key());
    copy.template operator()<std::uint64_t>(gr::tag::TRIGGER_TIME_ERROR.key());
    copy.template operator()<std::string_view>(gr::tag::CONTEXT.key());
    return complete;
}

/** advances `next` past the tags before `at` and those on it, and returns where the tags on `at` begin */
template<typename TIterator, typename TSentinel>
[[nodiscard]] constexpr TIterator tagsOn(TIterator& next, TSentinel end, std::size_t at) noexcept {
    while (next != end && (*next).index < at) {
        ++next;
    }
    const TIterator first = next;
    while (next != end && (*next).index == at) {
        ++next;
    }
    return first;
}

/**
 * the tags on one sample as one condition, the way the host's condition map merges them: a single tag is viewed in
 * place; several are merged into `scratch`, the latest tag winning, over the fields a trigger decision reads
 */
template<typename TIterator>
[[nodiscard]] gr::property_map_view mergedAt(TIterator first, TIterator last, std::span<std::byte> scratch) noexcept {
    if (first == last || std::next(first) == last) {
        return first == last ? gr::property_map_view{} : gr::property_map_view{(*first).map};
    }
    gr::property_map_view merged     = gr::formatEvent(scratch, kEventKeys);
    const auto            copyLatest = [&]<typename TValue>(std::string_view key, std::string_view shortKey) {
        std::optional<TValue> latest;
        for (auto tag = first; tag != last; ++tag) {
            const gr::property_map_view tagMap{(*tag).map};
            if (auto value = tagMap.template get_if<TValue>(key); value) {
                latest = *value;
            } else if (auto bare = tagMap.template get_if<TValue>(shortKey); bare) {
                latest = *bare;
            }
        }
        if (latest) {
            std::ignore = merged.try_emplace(key, *latest);
        }
    };
    copyLatest.template operator()<std::string_view>(gr::tag::TRIGGER_NAME.key(), gr::tag::TRIGGER_NAME.shortKey());
    copyLatest.template operator()<std::string_view>(gr::tag::CONTEXT.key(), gr::tag::CONTEXT.shortKey());
    copyLatest.template operator()<std::uint64_t>(gr::tag::TRIGGER_TIME.key(), gr::tag::TRIGGER_TIME.shortKey());
    copyLatest.template operator()<float>(gr::tag::TRIGGER_OFFSET.key(), gr::tag::TRIGGER_OFFSET.shortKey());
    copyLatest.template operator()<std::uint64_t>(gr::tag::TRIGGER_TIME_ERROR.key(), gr::tag::TRIGGER_TIME_ERROR.shortKey());
    copyLatest.template operator()<float>(gr::tag::SAMPLE_RATE.key(), gr::tag::SAMPLE_RATE.shortKey());
    return merged;
}

/** one event a block keeps between work calls, copied by value so a device mirror holds it too */
struct HeldEvent {
    alignas(gr::pmt::kBlobAlignment) std::array<std::byte, gr::kDeviceEventSlotBytes> blob{};
    std::size_t size = 0UZ;

    [[nodiscard]] constexpr bool empty() const noexcept { return size == 0UZ; }
    constexpr void               clear() noexcept { size = 0UZ; }

    /** false when the event is larger than a device event slot */
    template<bool kOnHost>
    [[nodiscard]] bool assign(const gr::property_map_view& event) {
        if (copy(event.blob())) {
            return true;
        }
        if constexpr (kOnHost) {
            gr::property_map compact{event};
            compact.shrink_to_fit();
            return copy(compact.blob());
        }
        return false;
    }

    [[nodiscard]] bool copy(std::span<const std::byte> image) noexcept {
        if (image.size() > blob.size()) {
            return false;
        }
        std::ranges::copy(image, blob.begin());
        size = image.size();
        return true;
    }

    [[nodiscard]] gr::property_map_view view() const noexcept { return gr::pmt::ValueMap::makeView(std::span<const std::byte>(blob.data(), size)); }
};

inline constexpr std::string_view kEventTooLarge = "a trigger larger than a device event slot cannot be held and was not";

[[nodiscard]] inline bool writeErrorEvent(gr::property_map_view& event, std::string_view reason, std::string_view source) noexcept {
    if (!event.try_emplace(gr::tag::TRIGGER_NAME.key(), std::string_view{"error"})) {
        return false;
    }
    gr::property_map_view details = event.try_emplace_map(gr::tag::TRIGGER_META_INFO.key(), 2U, static_cast<std::uint32_t>(reason.size() + source.size() + 64UZ));
    return details.try_emplace(std::string_view{"reason"}, reason) && details.try_emplace(std::string_view{"source"}, source);
}

[[nodiscard]] inline gr::property_map makeEvent(std::string_view name, std::string_view source, const gr::property_map_view& from = gr::property_map_view{}) {
    const auto       context = from.template get_if<std::string_view>(gr::tag::CONTEXT.key());
    gr::property_map event;
    event.reserve(kEventKeys, static_cast<std::uint32_t>(256UZ + name.size() + source.size() + (context ? context->size() : 0UZ)));
    [[maybe_unused]] const bool complete = writeEvent(event, name, source, from);
    assert(complete && "the reservation holds every field");
    return event;
}

[[nodiscard]] inline gr::property_map makeErrorEvent(std::string_view reason, std::string_view source) {
    gr::property_map event;
    event.reserve(2U, static_cast<std::uint32_t>(512UZ + 2UZ * (reason.size() + source.size())));
    [[maybe_unused]] const bool complete = writeErrorEvent(event, reason, source);
    assert(complete && "the reservation holds every field");
    return event;
}

/**
 * The events a block has decided on but its output span cannot yet hold.
 * Bounded, losing the oldest first when the consumer cannot keep up;
 * one slot is always kept back so the loss is reported however full the output is.
 */
struct PendingEvents {
    std::size_t                  capacity = kMaxQueuedEvents;
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
            if (!gr::emit(evtSpan, published, gr::property_map_view{queue.front()})) {
                break;
            }
            queue.pop_front();
            ++published;
        }
        if (lossUnreported && published < capacityLeft) {
            const std::string reason = std::format("{} event(s) dropped: the queue of {} was full", dropped, capacity);
            gr::log::warning("{}: {}", blockName, reason);
            if (gr::emit(evtSpan, published, makeErrorEvent(reason, source))) {
                ++published;
                lossUnreported = false;
            }
        }
        return published - index;
    }
};

} // namespace gr::blocks::trigger::detail

#endif // GNURADIO_TRIGGER_EVENTS_HPP
