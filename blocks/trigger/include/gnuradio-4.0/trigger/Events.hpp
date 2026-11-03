#ifndef GNURADIO_TRIGGER_EVENTS_HPP
#define GNURADIO_TRIGGER_EVENTS_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <expected>
#include <format>
#include <iterator>
#include <memory_resource>
#include <optional>
#include <source_location>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>

#include <gnuradio-4.0/Logger.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/TriggerMatcher.hpp>
#include <gnuradio-4.0/device/ExecutionStrategy.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>

namespace gr::blocks::trigger {

enum class EdgeSelection : std::uint8_t { rising, falling, both };
/** a trigger while a run or window is open: ignore it, start the run again, or lengthen it */
enum class Retrigger : std::uint8_t { ignore, restart, extend };

} // namespace gr::blocks::trigger

namespace gr::blocks::trigger::detail {

/** events a block queues between work calls: fixed, as a kernel's slots are */
inline constexpr std::size_t kMaxQueuedEvents = 64UZ;

/** a duration in seconds as whole ns, rounded; a negative one is 0 (kernel-safe: std::round, not llround) */
[[nodiscard]] constexpr std::uint64_t nanoseconds(double seconds) noexcept { return seconds > 0. ? static_cast<std::uint64_t>(std::round(seconds * 1e9)) : 0U; }

using MatchState = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;

[[nodiscard]] inline std::expected<MatchState, gr::Error> compileFilter(std::string_view definition, std::string_view matchMode) { return gr::trigger::BasicTriggerNameCtxMatcher::compile(definition, gr::meta::parseEnum<gr::trigger::BasicTriggerNameCtxMatcher::MatchMode>(matchMode)); }

[[nodiscard]] inline bool matches(MatchState& state, const gr::property_map_view& candidate) { //
    return gr::trigger::BasicTriggerNameCtxMatcher::match(state, candidate) == gr::trigger::MatchResult::Matching;
}

[[nodiscard]] inline bool acceptsTrigger(MatchState& state, bool filtered, const gr::property_map_view& candidate) {
    if (!candidate.contains(std::string_view{gr::tag::kTriggerName.key()})) {
        return false;
    }
    return !filtered || matches(state, candidate);
}

[[nodiscard]] constexpr bool isIncompleteTrigger(const gr::property_map_view& tagMap) noexcept { //
    return !tagMap.contains(gr::tag::kTriggerTime.key()) || !tagMap.contains(gr::tag::kTriggerOffset.key());
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

inline constexpr std::string_view kErrorKey = "error";

/** the last refusal a block reported from `settingsChanged`, so that re-applying the same settings does not report it again, and whether it still waits for the message output */
struct ErrorMemory {
    InlineText last;
    bool       waiting = false;
};

/**
 * a recoverable failure the flow graph should see: a warning in the log and a non-fatal notify message
 * {"error": reason} on the block's message output; host only, a device kernel counts instead.
 * Initial settings are applied before the scheduler connects that output, so a block with an `_deferredError`
 * keeps a refusal made then and sends it from `sendDeferredError` when it starts
 */
template<typename TBlock>
void reportError(const TBlock& block, std::string_view endpoint, std::string_view reason) noexcept {
    TBlock& sender = const_cast<TBlock&>(block); // a const processBulk runs on a non-const block
    if constexpr (requires { sender._deferredError; }) {
        if (endpoint == "settingsChanged()") {
            if (sender._deferredError.last.view() == reason) {
                return;
            }
            sender._deferredError.last.assign(reason);
        }
        gr::log::warning("{}: {}", block.name.value, reason);
        if (!sender.msgOut.isConnected()) {
            sender._deferredError.waiting = true;
            return;
        }
    } else {
        gr::log::warning("{}: {}", block.name.value, reason);
    }
    sender.emitMessage(endpoint, gr::property_map{{std::string(kErrorKey), std::string(reason)}});
}

/** a filter that does not compile is refused through `reportError` and leaves `into` as it was */
template<typename TBlock>
bool compileFilterInto(const TBlock& block, std::string_view definition, std::string_view matchMode, MatchState& into) {
    const auto compiled = compileFilter(definition, matchMode);
    if (!compiled) {
        reportError(block, "settingsChanged()", compiled.error().message);
        return false;
    }
    into = *compiled;
    return true;
}

/** an empty or refused definition leaves no filter */
template<typename TBlock>
[[nodiscard]] bool compileOptionalFilter(const TBlock& block, std::string_view definition, std::string_view matchMode, MatchState& into) {
    if (!definition.empty() && compileFilterInto(block, definition, matchMode, into)) {
        return true;
    }
    into = MatchState{};
    return false;
}

template<typename TBlock>
void sendDeferredError(TBlock& block) noexcept {
    if (block._deferredError.waiting) {
        block._deferredError.waiting = false;
        block.emitMessage("start()", gr::property_map{{std::string(kErrorKey), std::string(block._deferredError.last.view())}});
    }
}

inline constexpr std::uint32_t    kEventKeys         = 6U;
inline constexpr std::string_view kIncompleteTrigger = "a trigger lacking 'gr:trigger_time' or 'gr:trigger_offset' was acted on";

/// an empty `trigger_name` takes the block's `name` when the block starts, set as a setting so that no upstream tag renames it
template<typename TBlock>
void nameTriggerAfterBlock(TBlock& block) {
    if (block.trigger_name.value.empty()) {
        const gr::property_map named{{std::string(gr::tag::kTriggerName.shortKey()), std::string(block.name.value)}};
        std::ignore = block.settings().set(named);        // stored as set: it stops following the tags
        std::ignore = block.settings().activateContext(); // staged, and applied by the first work() call before any tag
    }
}

inline constexpr std::uint32_t kReferenceKeys = 3U;

/** a delayed decision's reference -- the trigger it is about -- as that trigger's own name, time and offset in the decision's `trigger_meta_info`; false when a field did not fit */
[[nodiscard]] inline bool writeReference(gr::property_map_view& details, const gr::property_map_view& reference) noexcept {
    const auto name   = reference.template get_if<std::string_view>(gr::tag::kTriggerName.key());
    const auto time   = reference.template get_if<std::uint64_t>(gr::tag::kTriggerTime.key());
    const auto offset = reference.template get_if<float>(gr::tag::kTriggerOffset.key());
    return (!name || details.try_emplace(gr::tag::kTriggerName.key(), *name)) && (!time || details.try_emplace(gr::tag::kTriggerTime.key(), *time)) && (!offset || details.try_emplace(gr::tag::kTriggerOffset.key(), *offset));
}

inline constexpr auto kNoDetails = [](gr::property_map_view& /*details*/) noexcept { return true; };

/**
 * the event a block reports for a trigger it acted on, dated as the trigger was; `trigger_meta_info` exists only when
 * `nDetails` > 0 and holds the keys `writeDetails` adds, as many as fit; false only when the root tuple did not fit
 */
template<typename TDetails = std::remove_cvref_t<decltype(kNoDetails)>>
[[nodiscard]] inline bool writeEvent(gr::property_map_view& event, std::string_view name, const gr::property_map_view& from, std::uint32_t nDetails = 0U, TDetails&& writeDetails = TDetails{}) noexcept {
    bool       complete = event.try_emplace(gr::tag::kTriggerName.key(), name);
    const auto copy     = [&]<typename TValue>(std::string_view key) {
        if (const auto value = from.template get_if<TValue>(key)) {
            complete = complete && event.try_emplace(key, *value);
        }
    };
    copy.template operator()<std::uint64_t>(gr::tag::kTriggerTime.key());
    copy.template operator()<float>(gr::tag::kTriggerOffset.key());
    copy.template operator()<std::uint64_t>(gr::tag::kTriggerTimeError.key());
    copy.template operator()<std::string_view>(gr::tag::kContext.key());
    if (nDetails == 0U || !complete) {
        return complete;
    }
    /** degrade, never drop: the root tuple stands; `writeDetails` adds its keys in priority order and the ones that no longer fit are left out from the end */
    for (std::uint32_t nKept = nDetails; nKept > 0U; --nKept) {
        gr::property_map_view details = event.try_emplace_map(gr::tag::kTriggerMetaInfo.key(), nKept, static_cast<std::uint32_t>(64UZ * nKept));
        if (!details.blob().empty()) {
            std::ignore = writeDetails(details);
            break;
        }
    }
    return true;
}

/** a tag map reports dropped samples: the stream is not contiguous at this sample */
[[nodiscard]] inline bool isGap(const gr::property_map_view& map) noexcept {
    const auto dropped = map.get_if<gr::Size_t>(gr::tag::kNDroppedSamples.key());
    return dropped && *dropped > 0U;
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
    copyLatest.template operator()<std::string_view>(gr::tag::kTriggerName.key(), gr::tag::kTriggerName.shortKey());
    copyLatest.template operator()<std::string_view>(gr::tag::kContext.key(), gr::tag::kContext.shortKey());
    copyLatest.template operator()<std::uint64_t>(gr::tag::kTriggerTime.key(), gr::tag::kTriggerTime.shortKey());
    copyLatest.template operator()<float>(gr::tag::kTriggerOffset.key(), gr::tag::kTriggerOffset.shortKey());
    copyLatest.template operator()<std::uint64_t>(gr::tag::kTriggerTimeError.key(), gr::tag::kTriggerTimeError.shortKey());
    copyLatest.template operator()<float>(gr::tag::kSampleRate.key(), gr::tag::kSampleRate.shortKey());
    return merged;
}

/** one event a block keeps between work calls, copied by value so a device mirror holds it too */
struct HeldEvent {
    alignas(gr::pmt::kBlobAlignment) std::array<std::byte, gr::kDeviceEventSlotBytes> blob{};
    std::size_t size = 0UZ;

    [[nodiscard]] constexpr bool empty() const noexcept { return size == 0UZ; }
    constexpr void               clear() noexcept { size = 0UZ; }

    /** false when the event is larger than a device event slot */
    template<bool onHost>
    [[nodiscard]] bool assign(const gr::property_map_view& event) {
        if (copy(event.blob())) {
            return true;
        }
        if constexpr (onHost) { // shrink_to_fit keeps a nested map's slack: `trigger_meta_info` is rebuilt to its size
            const auto packed = [](const auto& from) {
                gr::property_map to;
                for (const auto& [key, value] : from) {
                    std::ignore = to.insert_or_assign(key, value);
                }
                to.shrink_to_fit();
                return to;
            };
            gr::property_map compact = packed(event);
            if (const auto meta = compact.template get_if<gr::property_map>(gr::tag::kTriggerMetaInfo.key())) {
                std::ignore = compact.insert_or_assign(gr::tag::kTriggerMetaInfo.key(), packed(*meta));
                compact     = packed(compact); // the replaced map's bytes stay behind until copied out
            }
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

    [[nodiscard]] gr::property_map_view view() const noexcept { return gr::pmt::ValueMapView::viewOf(std::span<const std::byte>(blob.data(), size)); }
};

inline constexpr std::string_view kEventTooLarge = "a trigger larger than a device event slot cannot be held and was not";

template<typename TDetails = std::remove_cvref_t<decltype(kNoDetails)>>
[[nodiscard]] inline gr::property_map makeEvent(std::string_view name, const gr::property_map_view& from = gr::property_map_view{}, std::uint32_t nDetails = 0U, TDetails&& writeDetails = TDetails{}) {
    const auto        context = from.template get_if<std::string_view>(gr::tag::kContext.key());
    gr::property_map  event;
    const std::size_t metaBytes = nDetails > 0U ? 64UZ + 128UZ * nDetails : 0UZ; // the child map's header, entries and values
    event.reserve(kEventKeys, static_cast<std::uint32_t>(256UZ + name.size() + (context ? context->size() : 0UZ) + metaBytes));
    [[maybe_unused]] const bool complete = writeEvent(event, name, from, nDetails, std::forward<TDetails>(writeDetails));
    assert(complete && "the reservation holds every field");
    return event;
}

/** removes the last entry of an event's `trigger_meta_info`, and the map itself with its only entry; false when there is none */
[[nodiscard]] inline bool dropLastMetaKey(gr::property_map& event) {
    const auto meta = event.template get_if<gr::property_map>(gr::tag::kTriggerMetaInfo.key());
    if (!meta || meta->empty()) {
        return false;
    }
    gr::property_map kept;
    std::size_t      nKept = 0UZ;
    for (const auto& [key, value] : *meta) {
        if (++nKept < meta->size()) {
            std::ignore = kept.insert_or_assign(key, value);
        }
    }
    if (kept.empty()) {
        return event.erase(gr::tag::kTriggerMetaInfo.key()) > 0UZ;
    }
    return event.insert_or_assign(gr::tag::kTriggerMetaInfo.key(), std::move(kept)).second;
}

/**
 * The events a block has decided on but its output span cannot yet hold.
 * Bounded, losing the oldest first when the consumer cannot keep up;
 * the loss is reported once as a message.
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
        if (!queue.empty() && queue.size() >= capacity) {
            queue.pop_front();
            ++dropped;
            lossUnreported = true;
        }
        queue.push_back(std::move(event));
    }

    [[nodiscard]] std::size_t size() const noexcept { return queue.size(); }

    /** a loss is reported as a message of `source`, the block */
    [[nodiscard]] std::size_t drainInto(auto& evtSpan, std::size_t index, const auto& source) {
        const std::size_t capacityLeft = evtSpan.size();
        std::size_t       published    = index;
        while (!queue.empty() && published < capacityLeft) {
            auto emitted = gr::emit(evtSpan, published, gr::property_map_view{queue.front()});
            while (!emitted && emitted.error() == gr::EmitError::tooLarge && dropLastMetaKey(queue.front())) {
                emitted = gr::emit(evtSpan, published, gr::property_map_view{queue.front()});
            }
            if (!emitted && emitted.error() == gr::EmitError::tooLarge) { // not even the root tuple fits
                queue.pop_front();
                ++dropped;
                lossUnreported = true;
                continue;
            }
            if (!emitted) {
                break;
            }
            queue.pop_front();
            ++published;
        }
        reportLoss(source);
        return published - index;
    }

    /** a loss since the last report becomes one message of `source`, the block */
    void reportLoss(const auto& source) {
        if (lossUnreported) {
            reportError(source, "processBulk()", std::format("{} event(s) dropped: the queue of {} was full", dropped, capacity));
            lossUnreported = false;
        }
    }
};

} // namespace gr::blocks::trigger::detail

#endif // GNURADIO_TRIGGER_EVENTS_HPP
