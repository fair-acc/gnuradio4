#ifndef GNURADIO_TRIGGER_EVENTSTORE_HPP
#define GNURADIO_TRIGGER_EVENTSTORE_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>

namespace gr::blocks::trigger {

struct StoredEvent {
    gr::property_map             event;
    std::optional<std::uint64_t> at;

    [[nodiscard]] bool dated() const noexcept { return at.has_value(); }
};

/**
 * simultaneous events in one fixed order whichever producer's arrived first: `trigger_offset`, then `trigger_name`, then `context`;
 * an event lacking a field ranks as if it held 0 or the empty text
 */
[[nodiscard]] inline bool precedesSimultaneous(const gr::property_map_view& lhs, const gr::property_map_view& rhs) {
    const auto keyOf = [](const gr::property_map_view& event) {
        const auto offset  = event.template get_if<float>(gr::tag::kTriggerOffset.key());
        const auto name    = event.template get_if<std::string_view>(gr::tag::kTriggerName.key());
        const auto context = event.template get_if<std::string_view>(gr::tag::kContext.key());
        return std::tuple{offset ? *offset : 0.f, name ? *name : std::string_view{}, context ? *context : std::string_view{}};
    };
    return keyOf(lhs) < keyOf(rhs);
}

/**
 * A block's bounded, time-ordered copy of the events it drained from its bus: dated events before undated ones, then by `trigger_time`
 * and, at equal times, by `precedesSimultaneous`; undated ones in arrival order.
 * The bus is drained on sight so back-pressure never stalls the producers sharing the ring; once full the store drops and counts its oldest entry.
 * Ordering across work calls: the watermark is the latest `trigger_time` seen, and an event is `ready()`
 * once the watermark has passed its `trigger_time` by `lateness`, the longest one producer's events reach this store after
 * another's of the same time; a dated event that matches nothing is a heartbeat that moves the watermark on. An event later
 * than that still arrives out of order, which a consumer that subtracts times must allow for.
 *
 *   store.drain(evtSpan); for (const StoredEvent& event : store.ordered() | std::views::take(store.ready())) { ... } store.retire(handled);
 */
struct EventStore {
    std::size_t                  capacity = detail::kMaxQueuedEvents;
    std::uint32_t                dropped  = 0U;
    std::uint64_t                lateness = 0U; /// ns
    std::optional<std::uint64_t> watermark;     /// ns

    std::deque<StoredEvent> entries;

    void clear() noexcept {
        entries.clear();
        dropped   = 0U;
        watermark = std::nullopt;
    }

    void drain(gr::InputSpanLike auto& evtSpan) {
        for (const gr::property_map_view& incoming : evtSpan) {
            if (incoming.empty()) {
                continue;
            }
            StoredEvent stored{.event = gr::property_map{incoming}, .at = std::nullopt};
            if (const auto stamp = incoming.template get_if<std::uint64_t>(gr::tag::kTriggerTime.key())) {
                stored.at = *stamp;
            }
            push(std::move(stored));
        }
        std::ignore = evtSpan.consume(evtSpan.size());
        order();
    }

    void push(StoredEvent&& stored) {
        if (stored.at && (!watermark || *stored.at > *watermark)) {
            watermark = *stored.at;
        }
        if (!entries.empty() && entries.size() >= capacity) {
            entries.pop_front();
            ++dropped;
        }
        entries.push_back(std::move(stored));
    }

    void order() {
        std::ranges::stable_sort(entries, [](const StoredEvent& lhs, const StoredEvent& rhs) {
            if (lhs.dated() != rhs.dated()) {
                return lhs.dated();
            }
            if (!lhs.dated() || *lhs.at != *rhs.at) {
                return lhs.dated() && *lhs.at < *rhs.at;
            }
            return precedesSimultaneous(gr::property_map_view{lhs.event}, gr::property_map_view{rhs.event});
        });
    }

    [[nodiscard]] const std::deque<StoredEvent>& ordered() const noexcept { return entries; }

    /** the leading ordered entries no later event can still precede; undated ones follow the dated, in arrival order */
    [[nodiscard]] std::size_t ready() const noexcept {
        std::size_t n = 0UZ;
        for (const StoredEvent& stored : entries) {
            if (stored.dated() && (!watermark || *stored.at + lateness > *watermark)) {
                break;
            }
            ++n;
        }
        return n;
    }
    [[nodiscard]] bool        empty() const noexcept { return entries.empty(); }
    [[nodiscard]] std::size_t size() const noexcept { return entries.size(); }

    void retire(std::size_t count) { entries.erase(entries.begin(), entries.begin() + static_cast<std::ptrdiff_t>(std::min(count, entries.size()))); }
};

/**
 * Hands a body the events of a work call in the store's order -- dated before undated, then by `trigger_time`,
 * stable. On the host the bus is drained into `store` and `onEvent` may stop early by returning false; in a kernel the
 * first `limit` events of the span are ordered in place, all of them handled and consumed, so a kernel's `limit` must be
 * what the call can finish (one output slot per event, say). Returns how many were handled.
 */
template<bool onHost>
std::size_t forEachEvent(EventStore& store, gr::InputSpanLike auto& evtSpan, std::size_t limit, auto&& onEvent) {
    if constexpr (onHost) {
        store.drain(evtSpan);
        const std::size_t ready   = store.ready();
        std::size_t       handled = 0UZ;
        for (const StoredEvent& stored : store.ordered()) {
            if (handled >= std::min(limit, ready) || !onEvent(gr::property_map_view{stored.event})) {
                break;
            }
            ++handled;
        }
        store.retire(handled);
        return handled;
    } else {
        constexpr std::size_t kMaxOrdered = detail::kMaxQueuedEvents;
        const std::size_t     count       = std::min({limit, evtSpan.size(), kMaxOrdered});
        const auto            timeOf      = [&](std::size_t i) { return evtSpan[i].template get_if<std::uint64_t>(gr::tag::kTriggerTime.key()); };
        const auto            precedes    = [&](std::size_t mine, std::size_t other) {
            const auto mineAt  = timeOf(mine);
            const auto otherAt = timeOf(other);
            if (!mineAt || !otherAt) {
                return mineAt && !otherAt;
            }
            return *mineAt != *otherAt ? *mineAt < *otherAt : precedesSimultaneous(gr::property_map_view{evtSpan[mine]}, gr::property_map_view{evtSpan[other]});
        };
        std::array<std::size_t, kMaxOrdered> order{};
        for (std::size_t i = 0UZ; i < count; ++i) {
            std::size_t at = i;
            for (; at > 0UZ && precedes(i, order[at - 1UZ]); --at) {
                order[at] = order[at - 1UZ];
            }
            order[at] = i;
        }
        for (std::size_t k = 0UZ; k < count; ++k) {
            if (!evtSpan[order[k]].empty()) {
                std::ignore = onEvent(gr::property_map_view{evtSpan[order[k]]});
            }
        }
        std::ignore = evtSpan.consume(count);
        return count;
    }
}

/**
 * Hands a body the triggers of a work call in stream order: the injected events first, at the call's first sample, then
 * the tags on each sample merged into one condition. `onTags` sees every tagged sample, `onTrigger(trigger, at)` returns
 * false when it cannot act (no output slot left) and the walk stops there. Returns the samples the call may consume:
 * those before the trigger it stopped at.
 */
template<bool onHost>
std::size_t forEachTrigger(EventStore& store, gr::InputSpanLike auto& evtSpan, const gr::InputSpanLike auto& inSpan, std::size_t nSamples, std::size_t eventLimit, auto&& onTrigger, auto&& onTags) {
    bool stopped = false;
    std::ignore  = forEachEvent<onHost>(store, evtSpan, eventLimit, [&](const gr::property_map_view& event) {
        stopped = !onTrigger(event, inSpan.streamIndex);
        return !stopped;
    });
    if (stopped) {
        return 0UZ;
    }
    auto                                                                              tags = inSpan.rawTags();
    auto                                                                              next = std::ranges::begin(tags);
    alignas(gr::pmt::kBlobAlignment) std::array<std::byte, gr::kDeviceEventSlotBytes> scratch{};
    while (next != std::ranges::end(tags)) {
        const std::size_t at = (*next).index;
        if (at >= inSpan.streamIndex + nSamples) {
            break;
        }
        const auto                  first     = detail::tagsOn(next, std::ranges::end(tags), at);
        const gr::property_map_view condition = detail::mergedAt(first, next, scratch);
        if (at < inSpan.streamIndex) {
            continue;
        }
        onTags(condition, at);
        if (!onTrigger(condition, at)) {
            return at - inSpan.streamIndex;
        }
    }
    return nSamples;
}

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_EVENTSTORE_HPP
