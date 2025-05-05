#ifndef GNURADIO_DATASETUTILS_HPP
#define GNURADIO_DATASETUTILS_HPP

#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/Message.hpp>
#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/algorithm/dataset/DataSetHelper.hpp>

namespace gr::dataset {

struct DefaultChartConfig {
    std::size_t                chart_width  = 130UZ;
    std::size_t                chart_height = 28UZ;
    gr::graphs::ResetChartView reset_view   = gr::graphs::ResetChartView::KEEP;
};

template<DataSetLike TDataSet>
[[maybe_unused]] bool draw(const TDataSet& dataSet, const DefaultChartConfig config = {}, std::size_t signalIndex = std::numeric_limits<std::size_t>::max(), std::source_location location = std::source_location::current()) {
    using TValueType          = typename TDataSet::value_type;
    const bool plotAllSignals = signalIndex == std::numeric_limits<std::size_t>::max() || signalIndex >= dataSet.size();

    if (dataSet.signal_values.empty()                                    // check for empty data
        || dataSet.axis_values.empty() || dataSet.axis_values[0].empty() // empty axis definition
        || dataSet.signal_ranges.empty()                                 // empty min/max definition
    ) {
        return false;
    }

    std::expected<void, gr::Error> dsCheck = dataset::checkConsistency(dataSet);
    if (!dsCheck) {
        throw gr::exception(std::format("draw(const DataSet&, ...) - DataSet is not consistent - Error:\n{}", dsCheck.error().message), location);
    }

    if (config.reset_view == gr::graphs::ResetChartView::RESET) {
        gr::graphs::resetView();
    }

    auto adjustRange = [](TValueType min, TValueType max) {
        min                     = std::min(min, TValueType(0));
        max                     = std::max(max, TValueType(0));
        const TValueType margin = (max - min) * static_cast<TValueType>(0.2);
        return std::pair<double, double>{min - margin, max + margin};
    };
    assert(!dataSet.axis_values.empty());
    assert(!dataSet.signal_ranges.empty());
    const TValueType xMin = dataSet.axisValues(0UZ).front();
    const TValueType xMax = dataSet.axisValues(0UZ).back();
    TValueType       yMin = std::numeric_limits<TValueType>::max();
    TValueType       yMax = std::numeric_limits<TValueType>::lowest();

    if (plotAllSignals) {
        if constexpr (std::is_arithmetic_v<TValueType>) {
            const auto [min, max] = std::ranges::minmax_element(dataSet.signal_values);
            yMin                  = std::min(yMin, *min);
            yMax                  = std::max(yMax, *max);
        } else if constexpr (gr::meta::complex_like<TValueType>) {
            const auto [min, max] = std::ranges::minmax_element(dataSet.signal_values, //
                [](const TValueType& a, const TValueType& b) { return std::abs(a) < std::abs(b); });

            yMin = std::abs(yMin) > std::abs(*min) ? *min : yMin;
            yMax = std::abs(yMax) < std::abs(*min) ? *max : yMax;
        } else {
            static_assert(std::is_arithmetic_v<TValueType> || std::is_same_v<TValueType, std::complex<typename TValueType::value_type>>, "Unsupported type for DataSet");
        }
    } else {
        yMin = dataSet.signalRange(signalIndex).min;
        yMax = dataSet.signalRange(signalIndex).max;
    }

    auto chart        = gr::graphs::ImChart<std::dynamic_extent, std::dynamic_extent>({{xMin, xMax}, adjustRange(yMin, yMax)}, config.chart_width, config.chart_height);
    chart.axis_name_x = std::format("{} [{}]", dataSet.axisName(0UZ), dataSet.axisUnit(0UZ));
    if (plotAllSignals) {
        chart.axis_name_y = std::format("{} [{}]", dataSet.signalQuantity(0UZ), dataSet.signalUnit(0UZ));
    } else {
        chart.axis_name_y = std::format("{} [{}]", dataSet.signalQuantity(signalIndex), dataSet.signalUnit(signalIndex));
    }

    if (plotAllSignals) {
        for (std::size_t i = 0UZ; i < dataSet.size(); i++) {
            chart.draw<>(dataSet.axisValues(0UZ), dataSet.signalValues(i), dataSet.signalName(i));

            if (!dataSet.timing_events.empty() && !dataSet.timing_events[i].empty()) {
                std::vector<TValueType> tagVector(dataSet.signal_values.size());
                if constexpr (std::is_floating_point_v<TValueType>) {
                    std::ranges::fill(tagVector, std::numeric_limits<TValueType>::quiet_NaN());
                } else {
                    std::ranges::fill(tagVector, std::numeric_limits<TValueType>::lowest());
                }
                for (const auto& [index, tag] : dataSet.timing_events[i]) {
                    if (index < 0 || index >= static_cast<std::ptrdiff_t>(tagVector.size())) {
                        continue;
                    }
                    tagVector[static_cast<std::size_t>(index)] = dataSet.signal_values[static_cast<std::size_t>(index)];
                }

                chart.draw<gr::graphs::Style::Marker>(dataSet.axisValues(0UZ), tagVector, "Tags");
            }
        }
    } else { // plot single signal
        chart.draw<>(dataSet.axisValues(0UZ), dataSet.signalValues(signalIndex), dataSet.signalName(signalIndex));
    }

    chart.draw();

    return true;
}

template<typename T>
void updateMinMax(DataSet<T>& dataSet) {
    if constexpr (std::is_arithmetic_v<T>) {
        const auto [min, max]        = std::ranges::minmax_element(dataSet.signal_values);
        dataSet.signal_ranges[0].min = *min;
        dataSet.signal_ranges[0].max = *max;
    } else if constexpr (gr::meta::complex_like<T>) {
        const auto [min, max] = std::ranges::minmax_element(dataSet.signal_values, //
            [](const T& a, const T& b) { return std::abs(a) < std::abs(b); });

        dataSet.signal_ranges[0].min = *min;
        dataSet.signal_ranges[0].max = *max;
    } else {
        static_assert(std::is_arithmetic_v<T> || std::is_same_v<T, std::complex<typename T::value_type>>, "Unsupported type for DataSet");
    }
}

template<typename T, typename... TDataSets>
DataSet<T> merge(const DataSet<T>& first, const TDataSets&... others) {
    DataSet<T> mergedDataSet;

    mergedDataSet.timestamp   = first.timestamp;
    mergedDataSet.axis_names  = first.axis_names;
    mergedDataSet.axis_units  = first.axis_units;
    mergedDataSet.axis_values = first.axis_values;

    // Prepare to accumulate all other fields
    mergedDataSet.extents.emplace_back(first.signal_values.size()); // 1-dim data .> size of 1-dim data
    mergedDataSet.signal_names.reserve(sizeof...(others) + 1UZ);
    mergedDataSet.signal_units.reserve(sizeof...(others) + 1UZ);
    mergedDataSet.signal_values.reserve(first.signal_values.size() * (sizeof...(others) + 1UZ));
    mergedDataSet.signal_ranges.reserve(sizeof...(others) + 1UZ);
    mergedDataSet.meta_information.reserve(sizeof...(others) + 1UZ);
    mergedDataSet.timing_events.reserve(sizeof...(others) + 1UZ);

    // Helper lambda to add data from a single DataSet
    auto addDataSet = [&mergedDataSet](const DataSet<T>& ds, size_t dsIndex) {
        if (ds.axis_values != mergedDataSet.axis_values) {
            throw gr::exception(std::format("incompatible axis_values for DataSet {}", dsIndex));
        }
        if (ds.signal_names.size() > 1UZ || ds.signal_names.empty()) {
            throw gr::exception(std::format("incompatible signal_name.size={} for DataSet {}", ds.signal_names.size(), dsIndex));
        }

        // append signal values
        mergedDataSet.signal_values.insert(mergedDataSet.signal_values.end(), ds.signal_values.begin(), ds.signal_values.end());
        mergedDataSet.signal_ranges.push_back(ds.signal_ranges[0]);

        // append signal metadata
        mergedDataSet.signal_names.insert(mergedDataSet.signal_names.end(), ds.signal_names.begin(), ds.signal_names.end());
        mergedDataSet.signal_quantities.insert(mergedDataSet.signal_quantities.end(), ds.signal_quantities.begin(), ds.signal_quantities.end());
        mergedDataSet.signal_units.insert(mergedDataSet.signal_units.end(), ds.signal_units.begin(), ds.signal_units.end());
        mergedDataSet.meta_information.insert(mergedDataSet.meta_information.end(), ds.meta_information.begin(), ds.meta_information.end());

        // append timing events - N.B. index remain referenced to the original index of the sub-DataSet
        mergedDataSet.timing_events.push_back(ds.timing_events[0]);
    };

    std::size_t dsIndex = 0UZ;
    addDataSet(first, dsIndex++);         // add the first DataSet
    (addDataSet(others, dsIndex++), ...); // add other DataSets

    return mergedDataSet;
}

namespace generate {
enum class WaveType { Sine, Cosine };

template<typename T>
DataSet<T> waveform(WaveType waveType, size_t length, T samplingRate, T frequency, T amplitude = T(1), T offset = T(0)) {
    DataSet<T> dataSet;

    dataSet.timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count(); // Current time in nanoseconds

    dataSet.axis_names        = {"Time"};
    dataSet.axis_units        = {"s"};
    dataSet.signal_names      = {waveType == WaveType::Sine ? "Sine Wave" : "Cosine Wave"};
    dataSet.signal_quantities = {"Voltage"};
    dataSet.signal_units      = {"V"};

    dataSet.extents.emplace_back(static_cast<std::int32_t>(length));
    dataSet.axis_values.resize(1); // Only one axis (time)
    dataSet.axis_values[0].reserve(length);
    dataSet.signal_values.reserve(length);
    dataSet.signal_ranges.push_back({0, 1}); // placeholder for min/max values
    dataSet.meta_information.resize(1);
    dataSet.timing_events.resize(1); // resizing to have one set of timing events

    T dt            = T(1) / samplingRate; // time step
    T previousValue = offset * amplitude * ((waveType == WaveType::Sine) ? std::sin(T(0)) : std::cos(T(0)));
    for (size_t i = 0; i < length; ++i) {
        const T t            = T(i) * dt;
        const T phase        = T(2) * std::numbers::pi_v<T> * frequency * t;
        T       currentValue = offset + amplitude * ((waveType == WaveType::Sine) ? std::sin(phase) : std::cos(phase));
        dataSet.axis_values[0].push_back(t);
        dataSet.signal_values.push_back(currentValue);

        // check for zero crossing by seeing if the signs of previous and current values differ
        if ((previousValue < 0 && currentValue >= 0) || (previousValue > 0 && currentValue <= 0)) {
            dataSet.timing_events[0].emplace_back(static_cast<std::ptrdiff_t>(i), property_map{{"type", "Zero Crossing"}});
        }
        previousValue = currentValue;
    }

    // Update the signal ranges with actual min and max
    auto [min, max]          = std::ranges::minmax_element(dataSet.signal_values);
    dataSet.signal_ranges[0] = {*min, *max};

    return dataSet;
}

} // namespace generate

} // namespace gr::dataset

#endif // GNURADIO_DATASETUTILS_HPP
