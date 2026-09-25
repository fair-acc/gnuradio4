#ifndef GNURADIO_TRIGGER_SEGMENTS_HPP
#define GNURADIO_TRIGGER_SEGMENTS_HPP

#include <algorithm>
#include <cstddef>
#include <ranges>
#include <utility>
#include <vector>

#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>

namespace gr::blocks::trigger::detail {

using TriggerAt = std::pair<std::size_t, gr::property_map>;

void collectTriggers(gr::InputSpanLike auto& evtSpan, const gr::InputSpanLike auto& inSpan, std::size_t streamIndex, std::vector<TriggerAt>& into, auto&& accepts, auto&& onTag) {
    into.clear();
    for (const gr::property_map_view& event : evtSpan) {
        if (!event.empty() && accepts(event)) {
            into.emplace_back(streamIndex, gr::property_map{event});
        }
    }
    std::ignore = evtSpan.consume(evtSpan.size());

    for (const auto& tag : inSpan.rawTags()) {
        const gr::property_map_view carried{tag.map};
        const std::size_t           at = streamIndex + (tag.index - inSpan.streamIndex);
        if (accepts(carried)) {
            into.emplace_back(at, gr::property_map{carried});
        }
        onTag(carried, at);
    }
    std::ranges::stable_sort(into, {}, [](const TriggerAt& item) { return item.first; });
}

/// the span may be device-resident, so nothing here outlives the call or assumes a shared allocator
template<typename TQueue>
[[nodiscard]] std::size_t drainInto(TQueue& queue, gr::OutputSpanLike auto& outSpan) {
    const std::size_t emitted = std::min(queue.size(), outSpan.size());
    for (std::size_t i = 0UZ; i < emitted; ++i) {
        outSpan[i] = std::move(queue.front());
        queue.pop_front();
    }
    return emitted;
}

void forEachSegment(const gr::InputSpanLike auto& inSpan, std::size_t limit, auto&& onRun, auto&& onTag) {
    const std::size_t nSamples = std::min(limit, inSpan.size());
    std::size_t       consumed = 0UZ;
    for (const auto& tag : inSpan.rawTags()) {
        if (tag.index < inSpan.streamIndex) {
            continue;
        }
        const std::size_t at = tag.index - inSpan.streamIndex;
        if (at >= nSamples) {
            break;
        }
        onRun(consumed, at);
        consumed = at;
        onTag(gr::property_map_view{tag.map}, at);
    }
    onRun(consumed, nSamples);
}

} // namespace gr::blocks::trigger::detail

#endif // GNURADIO_TRIGGER_SEGMENTS_HPP
