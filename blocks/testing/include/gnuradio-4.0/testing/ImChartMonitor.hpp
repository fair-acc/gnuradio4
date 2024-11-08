#ifndef GNURADIO_IMCHARTMONITOR_HPP
#define GNURADIO_IMCHARTMONITOR_HPP

#include "gnuradio-4.0/BlockRegistry.hpp"
#include <algorithm>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/HistoryBuffer.hpp>
#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/algorithm/dataset/DataSetUtils.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

namespace gr::testing {

template<typename T>
requires(std::is_arithmetic_v<T> || gr::DataSetLike<T>)
struct ImChartMonitor : public Block<ImChartMonitor<T>, BlockingIO<false>, Drawable<UICategory::ChartPane, "console">> {
    using ClockSourceType = std::chrono::system_clock;
    PortIn<T>   in;
    float       sample_rate = 1000.0f;
    std::string signal_name = "unknown signal";

    GR_MAKE_REFLECTABLE(ImChartMonitor, in, sample_rate, signal_name);

    HistoryBuffer<T>   _historyBufferX{1000};
    HistoryBuffer<T>   _historyBufferY{1000};
    HistoryBuffer<Tag> _historyBufferTags{1000};

    void start() {
        fmt::println("started sink {} aka. '{}'", this->unique_name, this->name);
        in.max_samples = 10UZ;
    }

    void stop() { fmt::println("stopped sink {} aka. '{}'", this->unique_name, this->name); }

    constexpr void processOne(const T& input) noexcept {
        if constexpr (std::is_arithmetic_v<T>) {
            in.max_samples = static_cast<std::size_t>(2.f * sample_rate / 25.f);
            const T Ts     = T(1.0f) / T(sample_rate);
            _historyBufferX.push_back(_historyBufferX[1] + static_cast<T>(Ts));
        }
        _historyBufferY.push_back(input);

        if (this->inputTagsPresent()) { // received tag
            _historyBufferTags.push_back(this->mergedInputTag());
            _historyBufferTags[1].index = 0;
            this->_mergedInputTag.map.clear(); // TODO: provide proper API for clearing tags
        } else {
            _historyBufferTags.push_back(Tag(0UZ, property_map()));
        }
    }

    work::Status draw(const property_map& config = {}) noexcept {
        [[maybe_unused]] const work::Status status = this->invokeWork(); // calls work(...) -> processOne(...) (all in the same thread as this 'draw()'

        if constexpr (std::is_arithmetic_v<T>) {
            const auto [xMin, xMax] = std::ranges::minmax_element(_historyBufferX);
            const auto [yMin, yMax] = std::ranges::minmax_element(_historyBufferY);
            if (_historyBufferX.empty() || *xMin == *xMax || *yMin == *yMax) {
                return status; // buffer or axes' ranges are empty -> skip drawing
            }

            if (config.contains("reset_view")) {
                gr::graphs::resetView();
            }

            // create reversed copies -- draw(...) expects std::ranges::input_range ->
            std::vector<T> reversedX(_historyBufferX.rbegin(), _historyBufferX.rend());
            std::vector<T> reversedY(_historyBufferY.rbegin(), _historyBufferY.rend());
            std::vector<T> reversedTag(_historyBufferX.size());
            if constexpr (std::is_floating_point_v<T>) {
                std::transform(_historyBufferTags.rbegin(), _historyBufferTags.rend(), _historyBufferY.rbegin(), reversedTag.begin(), [](const Tag& tag, const T& yValue) { return tag.index < 0 ? std::numeric_limits<T>::quiet_NaN() : yValue; });
            } else {
                std::transform(_historyBufferTags.rbegin(), _historyBufferTags.rend(), _historyBufferY.rbegin(), reversedTag.begin(), [](const Tag& tag, const T& yValue) { return tag.index < 0 ? std::numeric_limits<T>::lowest() : yValue; });
            }

            auto adjustRange = [](T min, T max) {
                min            = std::min(min, T(0));
                max            = std::max(max, T(0));
                const T margin = (max - min) * static_cast<T>(0.2);
                return std::pair<double, double>{min - margin, max + margin};
            };

            auto chart = gr::graphs::ImChart<130, 28>({{*xMin, *xMax}, adjustRange(*yMin, *yMax)});
            chart.draw(reversedX, reversedY, signal_name);
            chart.draw<gr::graphs::Style::Marker>(reversedX, reversedTag, "Tags");
            chart.draw();
        } else if constexpr (gr::DataSetLike<T>) {
            if (_historyBufferY.empty()) {
                return status;
            }
            gr::dataset::draw(_historyBufferY[0], {.reset_view = config.contains("reset_view") ? graphs::ResetChartView::RESET : graphs::ResetChartView::KEEP});
        }

        return status;
    }
};

} // namespace gr::testing

inline const auto registerImChartMonitor = gr::registerBlock<gr::testing::ImChartMonitor, float, double, gr::DataSet<float>, gr::DataSet<double>>(gr::globalBlockRegistry());

#endif // GNURADIO_IMCHARTMONITOR_HPP
