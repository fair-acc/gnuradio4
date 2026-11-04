#ifndef GNURADIO_TRIGGER_WINDOWCOLLECTOR_HPP
#define GNURADIO_TRIGGER_WINDOWCOLLECTOR_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <ranges>
#include <span>
#include <tuple>
#include <variant>
#include <vector>

#include <gnuradio-4.0/HistoryBuffer.hpp>

namespace gr::blocks::trigger {

/**
 * The samples of every window [startsAt, closesAt) of stream positions, however many overlap, collected as they pass, so the result depends neither on how the scheduler cut the stream nor on when it is read.
 * A window may start before the current position as far as `reachBack` keeps history; one reaching further is cut where the history ends and marked partial, never padded.
 * A window without a closing position waits for `closeOldest`; one of length zero is delivered, not skipped; the windows open at once are bounded and a refusal is counted; `TInfo` travels with each window untouched.
 *
 *   WindowCollector<float> windows{.maxOpen = 8UZ}; windows.reachBack(64UZ); windows.push(inSpan); windows.open(100UZ, 200UZ); while (const auto window = windows.take()) { ... }
 */
template<typename T, typename TInfo = std::monostate>
struct WindowCollector {
    constexpr static std::size_t kNever = std::numeric_limits<std::size_t>::max();

    struct Window {
        std::vector<T> samples;
        std::size_t    startsAt = 0UZ;
        std::size_t    closesAt = kNever;
        TInfo          info{};
        bool           partial = false;
    };

    std::size_t                         maxOpen     = 16UZ;
    std::size_t                         streamIndex = 0UZ;
    std::uint32_t                       refused     = 0U;
    std::deque<Window>                  openWindows;
    std::deque<Window>                  closedWindows;
    std::optional<gr::HistoryBuffer<T>> history;

    void reset() {
        openWindows.clear();
        closedWindows.clear();
        streamIndex = 0UZ;
        refused     = 0U;
        if (history) {
            history = gr::HistoryBuffer<T>(history->capacity());
        }
    }

    /** a deeper history keeps the samples already pushed; it never shrinks */
    void reachBack(std::size_t nSamples) {
        if (nSamples == 0UZ || (history && nSamples <= history->capacity())) {
            return;
        }
        gr::HistoryBuffer<T> deeper(nSamples);
        for (std::size_t i = history ? history->size() : 0UZ; i-- > 0UZ;) {
            deeper.push_front((*history)[i]);
        }
        history = std::move(deeper);
    }

    bool open(std::size_t startsAt, std::size_t closesAt = kNever, TInfo info = {}, bool partial = false) {
        if (closesAt != kNever && closesAt <= startsAt) {
            closedWindows.push_back(Window{.samples = {}, .startsAt = startsAt, .closesAt = startsAt, .info = std::move(info), .partial = partial});
            return true;
        }
        if (openWindows.size() >= maxOpen) {
            ++refused;
            return false;
        }
        const std::size_t oldestHeld = streamIndex - (history ? history->size() : 0UZ);
        const std::size_t first      = std::max(startsAt, oldestHeld);
        const std::size_t heldEnd    = std::max(first, std::min(streamIndex, closesAt));
        Window            window{.samples = std::views::iota(first, heldEnd) | std::views::transform([&](std::size_t i) { return (*history)[streamIndex - 1UZ - i]; }) | std::ranges::to<std::vector<T>>(), .startsAt = first, .closesAt = closesAt, .info = std::move(info), .partial = partial || startsAt < oldestHeld};
        (closesAt <= streamIndex ? closedWindows : openWindows).push_back(std::move(window));
        return true;
    }

    bool openHere(std::size_t closesAt = kNever, TInfo info = {}) { return open(streamIndex, closesAt, std::move(info)); }

    [[nodiscard]] std::size_t nOpen() const noexcept { return openWindows.size(); }
    [[nodiscard]] std::size_t nClosed() const noexcept { return closedWindows.size(); }

    void push(std::span<const T> samples) {
        closeThoseEndingAt(streamIndex);
        for (const T& sample : samples) {
            for (Window& window : openWindows) {
                if (window.startsAt <= streamIndex) {
                    window.samples.push_back(sample);
                }
            }
            ++streamIndex;
            closeThoseEndingAt(streamIndex);
        }
        if (history) {
            history->push_front(samples);
        }
    }

    void closeOldest() {
        if (openWindows.empty()) {
            return;
        }
        Window& oldest  = openWindows.front();
        oldest.closesAt = std::max(oldest.startsAt, streamIndex);
        closedWindows.push_back(std::move(oldest));
        openWindows.pop_front();
    }

    void closeAll() {
        while (!openWindows.empty()) {
            closeOldest();
        }
    }

    /** also delivers a window whose closing position the caller moved to before the current one */
    [[nodiscard]] std::optional<Window> take() {
        closeThoseEndingAt(streamIndex);
        if (closedWindows.empty()) {
            return std::nullopt;
        }
        Window window = std::move(closedWindows.front());
        closedWindows.pop_front();
        return window;
    }

private:
    void closeThoseEndingAt(std::size_t position) {
        std::ignore = std::erase_if(openWindows, [&](Window& window) {
            const bool closes = window.closesAt != kNever && window.closesAt <= position;
            if (closes) {
                const std::size_t length = window.closesAt > window.startsAt ? window.closesAt - window.startsAt : 0UZ;
                window.samples.resize(std::min(window.samples.size(), length));
                closedWindows.push_back(std::move(window));
            }
            return closes;
        });
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_WINDOWCOLLECTOR_HPP
