#ifndef GNURADIO_TRIGGER_WINDOWCOLLECTOR_HPP
#define GNURADIO_TRIGGER_WINDOWCOLLECTOR_HPP

#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace gr::blocks::trigger {

/**
 * The samples a window holds while it is open, for however many windows are open at once.
 *
 * Every windowing rule -- a count, a duration, a pair of notifier events -- differs only in *when* a window opens and
 * closes. Keeping the samples is the same job in each case, and overlapping windows make it one worth doing once: with
 * `bufferCount(3, 2)` a sample belongs to two windows at the same time.
 *
 * A window is opened for the stream positions it covers rather than for "now", so a block may plan every window that
 * begins inside the span it is about to push. That is what makes the result independent of how the scheduler cut the
 * stream into work calls.
 *
 * A window may legitimately be empty, so a closed window of length zero is delivered rather than skipped: that is what
 * a quiet interval looks like, and a consumer counting windows would otherwise lose track of time. The number of
 * windows open at once is bounded and an overflow is counted, because a notifier that never closes anything would
 * otherwise grow without limit.
 *
 * @code
 * WindowCollector<float> windows{.maxOpen = 8UZ};
 * windows.open(100UZ, 200UZ); // samples 100 to 199, whenever they arrive
 * windows.push(inSpan);
 * while (const auto window = windows.take()) { ... }
 * @endcode
 */
template<typename T>
struct WindowCollector {
    constexpr static std::size_t kNever = std::numeric_limits<std::size_t>::max();

    std::size_t   maxOpen     = 16UZ;
    std::size_t   streamIndex = 0UZ;
    std::uint32_t refused     = 0U; // windows an already-full collector would not open

    struct Window {
        std::vector<T> samples;
        std::size_t    startsAt = 0UZ;
        std::size_t    closesAt = kNever;
    };

    std::deque<Window>         pending;
    std::deque<std::vector<T>> closed; // finished windows, oldest first

    void reset() noexcept {
        pending.clear();
        closed.clear();
        streamIndex = 0UZ;
        refused     = 0U;
    }

    bool open(std::size_t startsAt, std::size_t closesAt = kNever) {
        if (closesAt != kNever && closesAt <= startsAt) {
            closed.push_back({});
            return true;
        }
        if (pending.size() >= maxOpen) {
            ++refused;
            return false;
        }
        pending.push_back(Window{.samples = {}, .startsAt = startsAt, .closesAt = closesAt});
        return true;
    }

    bool openHere(std::size_t closesAt = kNever) { return open(streamIndex, closesAt); }

    [[nodiscard]] std::size_t nOpen() const noexcept { return pending.size(); }
    [[nodiscard]] std::size_t nClosed() const noexcept { return closed.size(); }

    void push(std::span<const T> samples) {
        for (const T& sample : samples) {
            closeThoseEndingAt(streamIndex);
            for (Window& window : pending) {
                if (window.startsAt <= streamIndex) {
                    window.samples.push_back(sample);
                }
            }
            ++streamIndex;
        }
        closeThoseEndingAt(streamIndex);
    }

    void closeOldest() {
        if (pending.empty()) {
            return;
        }
        closed.push_back(std::move(pending.front().samples));
        pending.pop_front();
    }

    void closeAll() {
        while (!pending.empty()) {
            closeOldest();
        }
    }

    [[nodiscard]] std::optional<std::vector<T>> take() {
        if (closed.empty()) {
            return std::nullopt;
        }
        std::vector<T> window = std::move(closed.front());
        closed.pop_front();
        return window;
    }

private:
    void closeThoseEndingAt(std::size_t position) {
        for (auto window = pending.begin(); window != pending.end();) {
            if (window->closesAt != kNever && window->closesAt <= position) {
                closed.push_back(std::move(window->samples));
                window = pending.erase(window);
            } else {
                ++window;
            }
        }
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_WINDOWCOLLECTOR_HPP
