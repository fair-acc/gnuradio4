#ifndef GNURADIO_TRIGGER_EVENTSTORE_HPP
#define GNURADIO_TRIGGER_EVENTSTORE_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <vector>

#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/trigger/MonotonicClock.hpp>

namespace gr::blocks::trigger {

struct StoredEvent {
    gr::property_map             event;
    std::uint64_t                arrived = kUnknownTime;
    std::optional<std::uint64_t> at;

    [[nodiscard]] bool dated() const noexcept { return at.has_value(); }
};

/**
 * What a block keeps of the events it has been sent.
 *
 * An event port is a bus: back-pressure on it stalls every producer sharing the ring, so a block drains its input on
 * sight and works on its own copy instead of holding the ring while it decides. Ordering is a property of that copy —
 * a bus is ordered by claim, not by time, so the store sorts by `trigger_time` and an undated event keeps its arrival
 * order.
 *
 * The store is bounded. Once full it drops its oldest entry and counts it, which is the family's rule for every queue
 * that cannot grow.
 *
 * @code
 * store.drain(evtSpan);                         // never blocks, never stalls the bus
 * for (const StoredEvent& event : store.ordered()) { ... }
 * store.retireBefore(oldest);
 * @endcode
 */
struct EventStore {
    std::size_t   capacity = 64UZ;
    std::uint32_t dropped  = 0U; // oldest-first, once the store is full

    std::deque<StoredEvent> entries;

    void clear() noexcept {
        entries.clear();
        dropped = 0U;
    }

    void drain(gr::InputSpanLike auto& evtSpan) {
        const std::uint64_t arrival = monotonicNowNs();
        for (const gr::property_map_view& incoming : evtSpan) {
            if (incoming.empty()) {
                continue;
            }
            StoredEvent stored{.event = gr::property_map{incoming}, .arrived = arrival, .at = std::nullopt};
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
        std::stable_sort(entries.begin(), entries.end(), [](const StoredEvent& lhs, const StoredEvent& rhs) {
            if (lhs.dated() != rhs.dated()) {
                return lhs.dated();
            }
            return lhs.dated() && *lhs.at < *rhs.at;
        });
    }

    [[nodiscard]] const std::deque<StoredEvent>& ordered() const noexcept { return entries; }
    [[nodiscard]] bool                           empty() const noexcept { return entries.empty(); }
    [[nodiscard]] std::size_t                    size() const noexcept { return entries.size(); }

    void retire(std::size_t count) { entries.erase(entries.begin(), entries.begin() + static_cast<std::ptrdiff_t>(std::min(count, entries.size()))); }

    [[nodiscard]] std::optional<std::uint64_t> waitedNs() const noexcept {
        if (entries.empty() || entries.front().arrived == kUnknownTime) {
            return std::nullopt;
        }
        const std::uint64_t now = monotonicNowNs();
        return now > entries.front().arrived ? now - entries.front().arrived : 0U;
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_EVENTSTORE_HPP
