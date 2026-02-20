#ifndef GNURADIO_IMCHARTMONITOR_HPP
#define GNURADIO_IMCHARTMONITOR_HPP

#include "gnuradio-4.0/BlockRegistry.hpp"
#include <algorithm>
#include <mutex>

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
struct ImChartMonitor : Block<ImChartMonitor<T, drawAsynchronously>, std::conditional_t<drawAsynchronously, Drawable<UICategory::Content, "console">, void>> {
    using ClockSourceType               = std::chrono::system_clock;
    using TimePoint                     = ClockSourceType::time_point;
    constexpr static bool isDataSetLike = gr::DataSetLike<T>;
    // optional shortening
    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = Annotated<U, description, Arguments...>;

    PortIn<T> in;

    A<float, "sample rate", Visible, Doc<"Sampling frequency in Hz">, Unit<"Hz">, Limits<float(0), std::numeric_limits<float>::max()>>                    sample_rate          = 1000.0f;
    A<std::string, "signal name", Visible, Doc<"human-readable identifier for the signal">>                                                               signal_name          = "unknown signal";
    A<int, "signal index", Doc<"which sub-DataSet-signal to display. -1: plot all">>                                                                      signal_index         = -1;
    A<gr::Size_t, "history length", Doc<"number of samples retained in ring buffer">>                                                                     n_history            = isDataSetLike ? 3ULL : 1000ULL;
    A<gr::Size_t, "tag history length", Doc<"number of tag entries retained in tag buffer">>                                                              n_tag_history        = 20ULL;
    A<bool, "reset view", Doc<"true: triggers a view reset">>                                                                                             reset_view           = true;
    A<bool, "plot graph", Doc<"controls whether to draw the main data graph">>                                                                            plot_graph           = true;
    A<bool, "plot timing", Doc<"controls whether to display timing info">>                                                                                plot_timing          = false;
    A<bool, "plot merged tags", Doc<"controls whether to display merged timing info">>                                                                    plot_merged_tags     = false;
    A<std::uint64_t, "timeout", Unit<"ms">, Limits<std::uint64_t(0), std::numeric_limits<std::uint64_t>::max()>, Doc<"Timeout duration in milliseconds">> timeout_ms           = 40ULL;
    A<gr::Size_t, "chart width", Doc<"chart character width in terminal">>                                                                                chart_width          = 130U;
    A<gr::Size_t, "chart heigth", Doc<"chart character width in terminal">>                                                                               chart_height         = 28U;
    A<bool, "mountain range", Doc<"enable mountain range (waterfall) visualization for DataSet inputs">>                                                  mountain_range       = false;
    A<gr::Size_t, "mountain x offset", Doc<"horizontal offset in characters per trace for mountain range">>                                               mountain_x_offset    = 2U;
    A<gr::Size_t, "mountain y offset", Doc<"vertical offset in characters per trace for mountain range">>                                                 mountain_y_offset    = 2U;
    A<gr::Size_t, "mountain color index", Doc<"color index for mountain range (max=rotating, 0=blue, 1=red, etc.)">>                                      mountain_color_index = std::numeric_limits<gr::Size_t>::max();

    GR_MAKE_REFLECTABLE(ImChartMonitor, in, sample_rate, signal_name, n_history, n_tag_history, reset_view, plot_graph, plot_timing, plot_merged_tags, timeout_ms, chart_width, chart_height, mountain_range, mountain_x_offset, mountain_y_offset, mountain_color_index);

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
    std::mutex           _drawMutex;

    void settingsChanged(const property_map& /*oldSettings*/, property_map& newSettings) {
        if (newSettings.contains("n_history")) {
            _historyBufferX.resize(static_cast<std::size_t>(n_history));
            _historyBufferY.resize(static_cast<std::size_t>(n_history));
            _historyBufferTags.resize(static_cast<std::size_t>(n_history));
        }

        if (newSettings.contains("n_tag_history")) {
            _historyTags.resize(static_cast<std::size_t>(n_tag_history));
        }

        if constexpr (drawAsynchronously && std::is_arithmetic_v<T>) {
            if (newSettings.contains("sample_rate") || newSettings.contains("timeout_ms")) {
                in.max_samples = std::max(1UZ, static_cast<std::size_t>(2.f * sample_rate * static_cast<float>(timeout_ms) / 1000.f));
            }
        }
    }

    void start() {
        std::println("started sink {} aka. '{}'", this->unique_name, this->name);
        if constexpr (drawAsynchronously && std::is_arithmetic_v<T>) {
            in.max_samples = std::max(1UZ, static_cast<std::size_t>(2.f * sample_rate * static_cast<float>(timeout_ms) / 1000.f));
        }
    }

    void stop() { std::println("stopped sink {} aka. '{}'", this->unique_name, this->name); }

    [[nodiscard]] work::Status processBulk(InputSpanLike auto& inData) noexcept {
        if (inData.empty()) {
            return work::Status::OK;
        }

        { // guarded write to local storage section
            const std::lock_guard lock(_drawMutex);
            const TimePoint       nowStamp = ClockSourceType::now();

            // collect per-sample tag positions from the input span
            auto spanTags  = inData.tags();
            using TagEntry = std::pair<std::size_t, property_map>;
            std::vector<TagEntry> localTags;
            for (const auto& [relIndex, tagMapRef] : spanTags) {
                if (relIndex >= 0 && static_cast<std::size_t>(relIndex) < inData.size()) {
                    localTags.emplace_back(static_cast<std::size_t>(relIndex), tagMapRef.get());
                }
            }

            std::size_t tagCursor = 0;
            for (std::size_t i = 0; i < inData.size(); ++i) {
                _n_samples_total++;

                if constexpr (std::is_arithmetic_v<T>) {
                    const T Ts = T(1.0f) / T(sample_rate);
                    _historyBufferX.push_back(static_cast<T>(_n_samples_total) * Ts);
                }
                _historyBufferY.push_back(inData[i]);

                if (tagCursor < localTags.size() && localTags[tagCursor].first == i) {
                    if constexpr (std::is_arithmetic_v<T>) {
                        _historyBufferTags.push_back(inData[i]);
                    }
                    _historyTags.push_back(TagInfo{.timestamp = nowStamp, .map = localTags[tagCursor].second, .index = _tagIndex++});
                    ++tagCursor;
                } else {
                    if constexpr (std::is_floating_point_v<T>) {
                        _historyBufferTags.push_back(std::numeric_limits<T>::quiet_NaN());
                    } else if constexpr (std::is_arithmetic_v<T>) {
                        _historyBufferTags.push_back(std::numeric_limits<T>::lowest());
                    }
                }
            }

            if (plot_merged_tags && this->inputTagsPresent()) {
                _historyTags.push_back(TagInfo{.timestamp = nowStamp, .map = this->_mergedInputTag.map, .index = _tagIndex++, .relIndex = 0, .merged = true});
                this->_mergedInputTag.map.clear(); // TODO: provide proper API for clearing tags
            }
        } // end lifetime of '_drawMutex' write lock-guard

        if constexpr (!drawAsynchronously) {
            draw();
        }

        return work::Status::OK;
    }

    work::Status draw(const property_map& config = {}, std::source_location location = std::source_location::current()) noexcept {
        const std::lock_guard lock(_drawMutex);
        _location = location;

        if (timeout_ms > 0U) {
            const TimePoint nowStamp = ClockSourceType::now();
            if (nowStamp < (_lastUpdate + std::chrono::milliseconds(timeout_ms))) {
                return lifecycle::isShuttingDown(this->state()) ? work::Status::DONE : work::Status::OK;
            }
            _lastUpdate = nowStamp;
        }

        const bool shouldResetView = config.contains("reset_view");
        if (plot_graph) {
            plotGraph(shouldResetView);
        }
        if (plot_timing) {
            plotTiming();
        }

        return lifecycle::isShuttingDown(this->state()) ? work::Status::DONE : work::Status::OK;
    }

    void plotGraph(bool shouldResetView) {
        if constexpr (std::is_arithmetic_v<T>) {
            const auto [xMin, xMax] = std::ranges::minmax_element(_historyBufferX);
            const auto [yMin, yMax] = std::ranges::minmax_element(_historyBufferY);
            if (_historyBufferX.empty() || *xMin == *xMax || *yMin == *yMax) {
                return; // buffer or axes' ranges are empty -> skip drawing
            }

            if (shouldResetView) {
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

            const auto& latestDataSet   = _historyBufferY[0];
            const bool  hasMultiSignals = latestDataSet.size() > 1UZ;

            if (mountain_range && (hasMultiSignals || _historyBufferY.size() > 1UZ)) {
                // mountain range mode
                dataset::MountainRangeConfig mrConfig{.chart_width = static_cast<std::size_t>(chart_width), .chart_height = static_cast<std::size_t>(chart_height), .x_offset_chars = static_cast<std::size_t>(mountain_x_offset), .y_offset_chars = static_cast<std::size_t>(mountain_y_offset), .color_index = static_cast<std::size_t>(mountain_color_index), .reset_view = shouldResetView ? graphs::ResetChartView::RESET : graphs::ResetChartView::KEEP};

                if (hasMultiSignals) {
                    // multi-signal DataSet: use signals as traces (replace on each update)
                    gr::dataset::drawMountainRange(latestDataSet, mrConfig, _location);
                } else {
                    // single-signal DataSets: accumulate in history as traces
                    std::vector<T> dataSetsVec;
                    dataSetsVec.reserve(_historyBufferY.size());
                    for (const auto& ds : _historyBufferY) {
                        dataSetsVec.push_back(ds);
                    }
                    gr::dataset::drawMountainRange(dataSetsVec, mrConfig, _location);
                }
            } else {
                // regular single-plot mode
                std::size_t signalIdx = signal_index < 0 ? std::numeric_limits<std::size_t>::max() : static_cast<std::size_t>(signal_index);
                gr::dataset::draw(latestDataSet, {.chart_width = static_cast<std::size_t>(chart_width), .chart_height = static_cast<std::size_t>(chart_height), .reset_view = shouldResetView ? graphs::ResetChartView::RESET : graphs::ResetChartView::KEEP}, signalIdx, _location);
            }
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
