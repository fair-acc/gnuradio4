#ifndef GNURADIO_TRIGGER_EVENTSTORE_HPP
#define GNURADIO_TRIGGER_EVENTSTORE_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
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
 * A block's bounded, time-ordered copy of the events it drained from its bus: dated events before undated ones, then by `trigger_time`, undated ones in arrival order.
 * The bus is drained on sight so back-pressure never stalls the producers sharing the ring; once full the store drops and counts its oldest entry.
 *
 *   store.drain(evtSpan); for (const StoredEvent& event : store.ordered()) { ... } store.retire(handled);
 */
struct EventStore {
    std::size_t   capacity = detail::kMaxQueuedEvents;
    std::uint32_t dropped  = 0U;

    std::deque<StoredEvent> entries;

    void clear() noexcept {
        entries.clear();
        dropped = 0U;
    }

    void drain(gr::InputSpanLike auto& evtSpan) {
        for (const gr::property_map_view& incoming : evtSpan) {
            if (incoming.empty()) {
                continue;
            }
            StoredEvent stored{.event = gr::property_map{incoming}, .at = std::nullopt};
            if (const auto stamp = incoming.template get_if<std::uint64_t>(gr::tag::TRIGGER_TIME.key())) {
                stored.at = *stamp;
            }
            push(std::move(stored));
        }
        std::ignore = evtSpan.consume(evtSpan.size());
        order();
    }

    void push(StoredEvent&& stored) {
        if (!entries.empty() && entries.size() >= capacity) {
            entries.pop_front();
            ++dropped;
        }
        entries.push_back(std::move(stored));
    }

    void order() {
        std::ranges::stable_sort(entries, {}, [](const StoredEvent& stored) { return std::pair{!stored.dated(), stored.at.value_or(0U)}; });
    }

    [[nodiscard]] const std::deque<StoredEvent>& ordered() const noexcept { return entries; }
    [[nodiscard]] bool                           empty() const noexcept { return entries.empty(); }
    [[nodiscard]] std::size_t                    size() const noexcept { return entries.size(); }

    void retire(std::size_t count) { entries.erase(entries.begin(), entries.begin() + static_cast<std::ptrdiff_t>(std::min(count, entries.size()))); }
};

/**
 * Hands a body the events of a work call in the store's order -- dated before undated, then by `trigger_time`,
 * stable. On the host the bus is drained into `store` and `handle` may stop early by returning false; in a kernel the
 * first `limit` events of the span are ordered in place, all of them handled and consumed, so a kernel's `limit` must be
 * what the call can finish (one output slot per event, say). Returns how many were handled.
 */
template<bool kOnHost>
std::size_t forEachEvent(EventStore& store, gr::InputSpanLike auto& evtSpan, std::size_t limit, auto&& handle) {
    if constexpr (kOnHost) {
        store.drain(evtSpan);
        std::size_t handled = 0UZ;
        for (const StoredEvent& stored : store.ordered()) {
            if (handled >= limit || !handle(gr::property_map_view{stored.event})) {
                break;
            }
            ++handled;
        }
        store.retire(handled);
        return handled;
    } else {
        constexpr std::size_t                kMaxOrdered = detail::kMaxQueuedEvents;
        const std::size_t                    count       = std::min({limit, evtSpan.size(), kMaxOrdered});
        const auto                           timeOf      = [&](std::size_t i) { return evtSpan[i].template get_if<std::uint64_t>(gr::tag::TRIGGER_TIME.key()); };
        std::array<std::size_t, kMaxOrdered> order{};
        for (std::size_t i = 0UZ; i < count; ++i) {
            std::size_t at = i;
            for (; at > 0UZ; --at) {
                const auto mine   = timeOf(i);
                const auto prev   = timeOf(order[at - 1UZ]);
                const bool before = mine && (!prev || *mine < *prev);
                if (!before) {
                    break;
                }
                order[at] = order[at - 1UZ];
            }
            order[at] = i;
        }
        for (std::size_t k = 0UZ; k < count; ++k) {
            if (!evtSpan[order[k]].empty()) {
                std::ignore = handle(gr::property_map_view{evtSpan[order[k]]});
            }
        }
        std::ignore = evtSpan.consume(count);
        return count;
    }
}

/**
 * Hands a body the triggers of a work call in stream order: the injected events first, at the call's first sample, then
 * the tags on each sample merged into one condition. `onTags` sees every tagged sample, `handle(trigger, at)` returns
 * false when it cannot act (no output slot left) and the walk stops there. Returns the samples the call may consume:
 * those before the trigger it stopped at.
 */
template<bool kOnHost>
std::size_t forEachTrigger(EventStore& store, gr::InputSpanLike auto& evtSpan, const gr::InputSpanLike auto& inSpan, std::size_t nSamples, std::size_t eventLimit, auto&& handle, auto&& onTags) {
    bool stopped = false;
    std::ignore  = forEachEvent<kOnHost>(store, evtSpan, eventLimit, [&](const gr::property_map_view& event) {
        stopped = !handle(event, inSpan.streamIndex);
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
        if (!handle(condition, at)) {
            return at - inSpan.streamIndex;
        }
    }
    return nSamples;
}

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_EVENTSTORE_HPP
