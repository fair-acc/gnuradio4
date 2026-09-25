#ifndef GNURADIO_TRIGGER_SEGMENTCOLLECTOR_HPP
#define GNURADIO_TRIGGER_SEGMENTCOLLECTOR_HPP

#include <cstddef>
#include <deque>
#include <optional>
#include <span>
#include <vector>

#include <gnuradio-4.0/HistoryBuffer.hpp>

namespace gr::blocks::trigger {

/**
 * The samples a window needs, from before a decision to after it.
 *
 * A trigger is known only once it has happened, so the samples ahead of it have already gone past. One collector per
 * channel keeps a rolling history long enough to look back `nPre`, and holds a window open until the `nPost` samples
 * after the decision have arrived. Several windows may be open at once, in the order they were asked for.
 *
 * It is deliberately free of ports and events: a block feeds it samples and decisions, which makes it testable on
 * its own and reusable by whatever needs a segment -- one channel or many, a coincidence or a single trigger.
 *
 * @code
 * collector.setWindow(2UZ, 3UZ);
 * collector.push(samples);              // every work call
 * collector.open(decisionIndex);        // when a decision arrives
 * while (const auto segment = collector.take()) { ... }
 * @endcode
 */
template<typename T>
struct SegmentCollector {
    std::size_t nPre  = 0UZ;
    std::size_t nPost = 0UZ;

    gr::HistoryBuffer<T> history{2UZ};
    std::size_t          streamIndex = 0UZ; // samples pushed so far
    std::uint32_t        refused     = 0U;
    std::uint32_t        lost        = 0U;

    struct Window {
        std::size_t   decision;
        std::uint64_t at;
    };
    std::deque<Window> openWindows;

    void setWindow(std::size_t samplesBefore, std::size_t samplesAfter, std::size_t margin = std::dynamic_extent) {
        nPre                    = samplesBefore;
        nPost                   = samplesAfter;
        const std::size_t slack = margin == std::dynamic_extent ? nPre + nPost + 1UZ : margin;
        history                 = gr::HistoryBuffer<T>(nPre + nPost + 2UZ + slack);
        reset();
    }

    void reset() {
        streamIndex = 0UZ;
        refused     = 0U;
        lost        = 0U;
        openWindows.clear();
        for (std::size_t i = 0UZ; i < history.capacity(); ++i) {
            history.push_front(T{});
        }
    }

    void push(std::span<const T> samples) {
        for (const T& sample : samples) {
            history.push_front(sample);
            ++streamIndex;
        }
    }

    [[nodiscard]] std::size_t length() const noexcept { return nPre + nPost + 1UZ; }

    bool open(std::size_t decisionIndex, std::uint64_t at = 0U) {
        const std::size_t oldestWanted = decisionIndex >= nPre ? decisionIndex - nPre : 0UZ;
        if (streamIndex > oldestWanted + history.capacity()) {
            ++refused;
            return false;
        }
        openWindows.push_back(Window{.decision = decisionIndex, .at = at});
        return true;
    }

    [[nodiscard]] bool ready() const noexcept { return !openWindows.empty() && streamIndex >= openWindows.front().decision + nPost + 1UZ; }

    [[nodiscard]] std::size_t pending() const noexcept { return openWindows.size(); }

    [[nodiscard]] bool serviceable(const Window& window) const noexcept {
        const std::size_t oldestWanted = window.decision >= nPre ? window.decision - nPre : 0UZ;
        return streamIndex - oldestWanted <= history.capacity();
    }

    [[nodiscard]] std::optional<std::pair<std::vector<T>, std::uint64_t>> take() {
        while (ready() && !serviceable(openWindows.front())) {
            openWindows.pop_front();
            ++lost;
        }
        if (!ready()) {
            return std::nullopt;
        }
        const Window window = openWindows.front();
        openWindows.pop_front();

        std::vector<T> segment(length());
        for (std::size_t i = 0UZ; i < length(); ++i) { // history[0] is the newest sample pushed
            const auto wanted = static_cast<std::ptrdiff_t>(window.decision) - static_cast<std::ptrdiff_t>(nPre) + static_cast<std::ptrdiff_t>(i);
            if (wanted < 0) {
                segment[i] = T{}; // before the stream began
                continue;
            }
            const std::size_t back = streamIndex - 1UZ - static_cast<std::size_t>(wanted);
            segment[i]             = back < history.capacity() ? history[back] : T{};
        }
        return std::pair{std::move(segment), window.at};
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_SEGMENTCOLLECTOR_HPP
