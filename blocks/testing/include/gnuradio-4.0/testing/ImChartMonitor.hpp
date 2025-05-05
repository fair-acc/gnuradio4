#ifndef GNURADIO_IMCHARTMONITOR_HPP
#define GNURADIO_IMCHARTMONITOR_HPP

#include "gnuradio-4.0/BlockRegistry.hpp"
#include <algorithm>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/HistoryBuffer.hpp>
#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/algorithm/dataset/DataSetUtils.hpp>

#include <format>
#include <gnuradio-4.0/meta/formatter.hpp>

namespace gr::testing {

GR_REGISTER_BLOCK("ImChartMonitor", gr::testing::ImChartMonitor, ([T], true), [ float, double ]);
GR_REGISTER_BLOCK("ConsoleDebugSink", gr::testing::ImChartMonitor, ([T], false), [ float, double ]);

template<typename T, bool drawAsynchronously = true>
requires(std::is_arithmetic_v<T> || gr::DataSetLike<T>)
struct ImChartMonitor : Block<ImChartMonitor<T, drawAsynchronously>, std::conditional_t<drawAsynchronously, BlockingIO<false>, void>, std::conditional_t<drawAsynchronously, Drawable<UICategory::ChartPane, "console">, void>> {
    using ClockSourceType               = std::chrono::system_clock;
    using TimePoint                     = ClockSourceType::time_point;
    constexpr static bool isDataSetLike = gr::DataSetLike<T>;
    // optional shortening
    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = Annotated<U, description, Arguments...>;

    PortIn<T> in;

    A<float, "sample rate", Visible, Doc<"Sampling frequency in Hz">, Unit<"Hz">, Limits<float(0), std::numeric_limits<float>::max()>>                    sample_rate      = 1000.0f;
    A<std::string, "signal name", Visible, Doc<"human-readable identifier for the signal">>                                                               signal_name      = "unknown signal";
    A<int, "signal index", Doc<"which sub-DataSet-signal to display. -1: plot all">>                                                                      signal_index     = -1;
    A<gr::Size_t, "history length", Doc<"number of samples retained in ring buffer">>                                                                     n_history        = isDataSetLike ? 3ULL : 1000ULL;
    A<gr::Size_t, "tag history length", Doc<"number of tag entries retained in tag buffer">>                                                              n_tag_history    = 20ULL;
    A<bool, "reset view", Doc<"true: triggers a view reset">>                                                                                             reset_view       = true;
    A<bool, "plot graph", Doc<"controls whether to draw the main data graph">>                                                                            plot_graph       = true;
    A<bool, "plot timing", Doc<"controls whether to display timing info">>                                                                                plot_timing      = false;
    A<bool, "plot merged tags", Doc<"controls whether to display merged timing info">>                                                                    plot_merged_tags = false;
    A<std::uint64_t, "timeout", Unit<"ms">, Limits<std::uint64_t(0), std::numeric_limits<std::uint64_t>::max()>, Doc<"Timeout duration in milliseconds">> timeout_ms       = 40ULL;
    A<gr::Size_t, "chart width", Doc<"chart character width in terminal">>                                                                                chart_width      = 130U;
    A<gr::Size_t, "chart heigth", Doc<"chart character width in terminal">>                                                                               chart_height     = 28U;

    GR_MAKE_REFLECTABLE(ImChartMonitor, in, sample_rate, signal_name, n_history, n_tag_history, reset_view, plot_graph, plot_timing, plot_merged_tags, timeout_ms, chart_width, chart_height);

    HistoryBuffer<T> _historyBufferX{n_history};
    HistoryBuffer<T> _historyBufferY{n_history};
    HistoryBuffer<T> _historyBufferTags{n_history};
    std::size_t      _n_samples_total = 0UZ;

    struct TagInfo {
        TimePoint      timestamp;
        property_map   map;
        std::size_t    index;
        std::ptrdiff_t relIndex = 0;
        bool           merged   = false;
    };
    HistoryBuffer<TagInfo> _historyTags{n_tag_history};
    std::size_t            _tagIndex = 0UZ;

    std::source_location _location   = std::source_location::current();
    TimePoint            _lastUpdate = ClockSourceType::now();

    void settingsChanged(const property_map& /*oldSettings*/, property_map& newSettings) {
        if (newSettings.contains("n_history")) {
            _historyBufferX.resize(static_cast<std::size_t>(n_history));
            _historyBufferY.resize(static_cast<std::size_t>(n_history));
            _historyBufferTags.resize(static_cast<std::size_t>(n_history));
        }

        if (newSettings.contains("n_tag_history")) {
            _historyTags.resize(static_cast<std::size_t>(n_tag_history));
        }
    }

    void start() {
        std::println("started sink {} aka. '{}'", this->unique_name, this->name);
        in.max_samples = 10UZ;
    }

    void stop() { std::println("stopped sink {} aka. '{}'", this->unique_name, this->name); }

    constexpr void processOne(const T& input) {
        TimePoint nowStamp = ClockSourceType::now();

        _n_samples_total++;

        if constexpr (std::is_arithmetic_v<T>) {
            if constexpr (drawAsynchronously) {
                in.max_samples = static_cast<std::size_t>(2.f * sample_rate * static_cast<float>(timeout_ms) / 1000.f);
            }
            const T Ts = T(1.0f) / T(sample_rate);
            _historyBufferX.push_back(static_cast<T>(_n_samples_total) * static_cast<T>(Ts));
        }
        _historyBufferY.push_back(input);

        if (this->inputTagsPresent()) { // received tag
            _historyBufferTags.push_back(_historyBufferY.back());

            if (plot_merged_tags) {
                _historyTags.push_back(TagInfo{.timestamp = nowStamp, .map = this->_mergedInputTag.map, .index = _tagIndex++, .relIndex = 0, .merged = true});
            }
            auto tags = in.tagReader().get();
            for (const auto& [relIndex, tagMap] : tags) {
                _historyTags.push_back(TagInfo{.timestamp = nowStamp, .map = tagMap, .index = relIndex});
            }

            this->_mergedInputTag.map.clear(); // TODO: provide proper API for clearing tags
        } else {
            if constexpr (std::is_floating_point_v<T>) {
                _historyBufferTags.push_back(std::numeric_limits<T>::quiet_NaN());
            } else {
                _historyBufferTags.push_back(std::numeric_limits<T>::lowest());
            }
        }

        if (timeout_ms > 0U) {
            if (nowStamp < (_lastUpdate + std::chrono::milliseconds(timeout_ms))) {
                return;
            }
            _lastUpdate = nowStamp;
        }

        if (plot_graph) {
            plotGraph();
        }

        if (plot_timing) {
            plotTiming();
        }
    }

    work::Status draw(const property_map& config = {}, std::source_location location = std::source_location::current()) noexcept {
        reset_view                           = config.contains("reset_view");
        _location                            = location;
        [[maybe_unused]] work::Status status = work::Status::OK;
        if constexpr (drawAsynchronously) {
            status = this->invokeWork(); // calls work(...) -> processOne(...) (all in the same thread as this 'draw()'
        }
        return status;
    }

    void plotGraph() {
        if constexpr (std::is_arithmetic_v<T>) {
            const auto [xMin, xMax] = std::ranges::minmax_element(_historyBufferX);
            const auto [yMin, yMax] = std::ranges::minmax_element(_historyBufferY);
            if (_historyBufferX.empty() || *xMin == *xMax || *yMin == *yMax) {
                return; // buffer or axes' ranges are empty -> skip drawing
            }

            if (reset_view) {
                gr::graphs::resetView();
            }
            std::println("\nPlot Graph for '{}' - #samples total: {}", signal_name, _n_samples_total);

            auto adjustRange = [](T min, T max) {
                min            = std::min(min, T(0));
                max            = std::max(max, T(0));
                const T margin = (max - min) * static_cast<T>(0.2);
                return std::pair<double, double>{min - margin, max + margin};
            };

            auto chart = gr::graphs::ImChart<std::dynamic_extent, std::dynamic_extent>({{*xMin, *xMax}, adjustRange(*yMin, *yMax)}, static_cast<std::size_t>(chart_width), static_cast<std::size_t>(chart_height));
            chart.draw(_historyBufferX, _historyBufferY, signal_name);
            chart.draw<gr::graphs::Style::Marker>(_historyBufferX, _historyBufferTags, "Tags");
            chart.draw();
        } else if constexpr (isDataSetLike) {
            if (_historyBufferY.empty()) {
                return;
            }
            std::size_t signalIdx = signal_index < 0 ? std::numeric_limits<std::size_t>::max() : static_cast<std::size_t>(signal_index);
            gr::dataset::draw(_historyBufferY[0],
                {.chart_width     = static_cast<std::size_t>(chart_width),  //
                    .chart_height = static_cast<std::size_t>(chart_height), //
                    .reset_view   = reset_view ? graphs::ResetChartView::RESET : graphs::ResetChartView::KEEP},
                signalIdx, _location);
        }
    }

    void plotTiming() {
        std::println("\nPast Timing Events for '{}'", signal_name);
        for (auto const& tag : _historyTags) {
            auto isoTime = [](TimePoint time) noexcept {
                return std::format("{}", time); // ms-precision ISO time-format
            };
            std::uint64_t nsSinceEpoch = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(tag.timestamp.time_since_epoch()).count());

            std::println("{:10} {:1} {:24} {} {}", tag.index, tag.merged ? 'M' : 'T', isoTime(tag.timestamp), nsSinceEpoch, tag.map);
        }
    }
};

} // namespace gr::testing

#endif // GNURADIO_IMCHARTMONITOR_HPP
