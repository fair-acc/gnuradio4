#ifndef GNURADIO_IMCHARTMONITOR_HPP
#define GNURADIO_IMCHARTMONITOR_HPP

#include <algorithm>

#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/HistoryBuffer.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

namespace gr::testing {

template<typename T>
struct ImChartMonitor : public Block<ImChartMonitor<T>, BlockingIO<false>> {
    using ClockSourceType = std::chrono::system_clock;
    PortIn<T>   in;
    float       sample_rate = 1000.0f;
    std::string signal_name = "unknown signal";

    HistoryBuffer<T>   _historyBufferX{ 1000 };
    HistoryBuffer<T>   _historyBufferY{ 1000 };
    HistoryBuffer<Tag> _historyBufferTags{ 1000 };

    void
    start() {
        fmt::println("started sink {} aka. '{}'", this->unique_name, this->name);
        in.max_samples = 10UZ;
    }

    void
    stop() {
        fmt::println("stopped sink {} aka. '{}'", this->unique_name, this->name);
    }

    constexpr void
    processOne(const T &input) noexcept {
        in.max_samples = 2 * sample_rate / 25;
        const float Ts = 1.0f / sample_rate;
        _historyBufferX.push_back(_historyBufferX[1] + Ts);
        _historyBufferY.push_back(input);

        if (this->input_tags_present()) { // received tag
            _historyBufferTags.push_back(this->mergedInputTag());
            _historyBufferTags[1].index = 0;
            this->_mergedInputTag.map.clear(); // TODO: provide proper API for clearing tags
        } else {
            _historyBufferTags.push_back(Tag(-1, property_map()));
        }
    }

    work::Status
    draw() noexcept {
        [[maybe_unused]] const work::Status status = this->invokeWork(); // calls work(...) -> processOne(...) (all in the same thread as this 'draw()'
        const auto [xMin, xMax]                    = std::ranges::minmax_element(_historyBufferX);
        const auto [yMin, yMax]                    = std::ranges::minmax_element(_historyBufferY);
        if (_historyBufferX.empty() || *xMin == *xMax || *yMin == *yMax) {
            return status; // buffer or axes' ranges are empty -> skip drawing
        }
        fmt::println("\033[2J\033[H");
        // create reversed copies -- draw(...) expects std::ranges::input_range ->
        // TODO: change draw routine and/or write wrapper and/or provide direction option to HistoryBuffer
        std::vector<T> reversedX(_historyBufferX.rbegin(), _historyBufferX.rend());
        std::vector<T> reversedY(_historyBufferY.rbegin(), _historyBufferY.rend());
        std::vector<T> reversedTag(_historyBufferX.size());
        std::transform(_historyBufferTags.rbegin(), _historyBufferTags.rend(), _historyBufferY.rbegin(), reversedTag.begin(), [](const Tag& tag, const T& yValue) { return tag.index < 0 ? T(0) : yValue; });

        auto adjustRange = [](T min, T max) {
            min = std::min(min, T(0));
            max = std::max(max, T(0));
            const T margin = (max - min) * 0.2;
            return std::pair<double, double>{min - margin, max + margin};
        };

        auto chart = gr::graphs::ImChart<130, 28>({ { *xMin, *xMax }, adjustRange(*yMin, *yMax) });
        chart.draw(reversedX, reversedY, signal_name);
        chart.draw<gr::graphs::Style::Marker>(reversedX, reversedTag, "Tags");
        chart.draw();
        fmt::println("buffer has {} samples - status {:10} # graph range x = [{:2.2}, {:2.2}] y = [{:2.2}, {:2.2}]", _historyBufferX.size(), magic_enum::enum_name(status), *xMin, *xMax, *yMin, *yMax);
        return status;
    }
};

} // namespace gr::testing

ENABLE_REFLECTION_FOR_TEMPLATE(gr::testing::ImChartMonitor, in, sample_rate, signal_name)

#endif // GNURADIO_IMCHARTMONITOR_HPP
