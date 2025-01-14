#ifndef DATASETHELPER_HPP
#define DATASETHELPER_HPP

namespace gr::dataset {
namespace dim {
// N.B. explicitly and individually defined indices, rather than enum class
// since dimensionality is a priori open
inline static constexpr std::size_t X = 0UZ; /// X-axis index
inline static constexpr std::size_t Y = 1UZ; /// Y-axis index
inline static constexpr std::size_t Z = 2UZ; /// Z-axis index
} // namespace dim

enum class ProcessMode : std::uint8_t {
    InPlace = 0U, /// in-place processing
    Copy          /// copy first, then process
};

enum class MetaInfo : std::uint8_t {
    None = 0U, /// do nothing
    Apply      /// add meta-info to DataSet<T>::meta_information[] or ::timing_events[] where applicable
};

namespace detail {

template<typename T>
std::size_t checkIndexRange(const DataSet<T>& ds, std::size_t minIndex = 0UZ, std::size_t maxIndex = 0UZ, std::size_t signalIndex = 0UZ, std::source_location location = std::source_location::current()) {
    const std::size_t maxDataSetIndex = ds.axisValues(dim::X).size();
    if (maxIndex == max_size_t) { // renormalise default range
        maxIndex = maxDataSetIndex;
    }
    if (minIndex > maxIndex || minIndex >= maxDataSetIndex || maxIndex > maxDataSetIndex || signalIndex >= ds.size()) {
        throw gr::exception(fmt::format("DataSet<{}> ({}/{}: \"{}\") indices [{}, {}] out of range [0, {}]",                                    //
                                gr::meta::type_name<T>(), signalIndex, ds.size(), signalIndex < ds.size() ? ds.signalName(signalIndex) : "???", //
                                minIndex, maxIndex, maxDataSetIndex),
            location);
    }
    return maxIndex;
}

template<typename T, typename U>
[[nodiscard]] constexpr U linearInterpolate(T x0, T x1, U y0, U y1, U y) noexcept {
    if (y1 == y0) {
        return U(x0);
    }
    return x0 + (y - y0) * (x1 - x0) / (y1 - y0);
}

template<typename T>
[[nodiscard]] constexpr T sign(T positiveFactor, int val) noexcept {
    return (val >= 0) ? positiveFactor : -positiveFactor;
}

} // namespace detail

template<typename T, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] constexpr T getIndexValue(const DataSet<T>& dataSet, std::size_t dimIndex, std::size_t sampleIndex, std::size_t signalIndex = 0UZ) {
    if (dimIndex == dim::X) {
        if (dataSet.axisCount() == 0UZ || dataSet.axisValues(0UZ).size() <= sampleIndex) {
            return std::numeric_limits<TValue>::quiet_NaN();
        }
        return static_cast<T>(dataSet.axisValues(dim::X)[sampleIndex]);
    }

    if (dimIndex == dim::Y) {
        if (signalIndex >= dataSet.size()) {
            return std::numeric_limits<TValue>::quiet_NaN();
        }
        std::span<const T> vals = dataSet.signalValues(signalIndex);
        if (vals.size() <= sampleIndex) {
            return std::numeric_limits<TValue>::quiet_NaN();
        }
        return static_cast<T>(vals[sampleIndex]);
    }

    throw gr::exception(fmt::format("axis dimIndex {} is out of range [0, {}] and", dimIndex, dataSet.axisCount()));
}

template<typename T>
T getDistance(const DataSet<T>& dataSet, std::size_t dimIndex, std::size_t indexMin = 0UZ, std::size_t indexMax = max_size_t, std::size_t signalIndex = 0UZ) {
    if (indexMax == max_size_t) { // renormalise default range
        indexMax = dataSet.axisValues(dim::X).size() - 1UZ;
    }
    indexMax = detail::checkIndexRange(dataSet, indexMin, indexMax, signalIndex);

    if (dimIndex == dim::X || dimIndex == dim::Y) {
        return getIndexValue(dataSet, dimIndex, indexMax, signalIndex) - getIndexValue(dataSet, dimIndex, indexMin, signalIndex);
    }

    throw gr::exception(fmt::format("axis dimIndex {} is out of range [0, {}] and", dimIndex, dataSet.axisCount()));
}

/*!
 * \brief Interpolate in Y dimension at a given X coordinate xValue.
 *        If out of range, we clamp or return NaN.
 */
template<typename T, typename TValue = gr::meta::fundamental_base_value_type_t<T>, std::convertible_to<TValue> U>
[[nodiscard]] constexpr T getValue(const DataSet<T>& dataSet, std::size_t dimIndex, U xValue, std::size_t signalIndex = 0) {
    if (dimIndex == dim::X) {
        return xValue; // if user requests X dimension at xValue, that’s basically identity
    }

    if (dimIndex == dim::Y) { // linear interpolation in X-axis=0
        if (dataSet.axisCount() == 0) {
            return std::numeric_limits<TValue>::quiet_NaN();
        }
        const auto& xAxis = dataSet.axisValues(dim::X);
        const auto  ySpan = dataSet.signalValues(signalIndex);

        if (xAxis.empty() || ySpan.empty()) {
            return std::numeric_limits<TValue>::quiet_NaN();
        }

        // clamp
        if (xValue <= static_cast<T>(xAxis.front())) {
            return static_cast<T>(ySpan.front());
        } else if (xValue >= static_cast<T>(xAxis.back())) {
            return static_cast<T>(ySpan.back());
        }

        // binary-search for interval
        auto it = std::lower_bound(xAxis.begin(), xAxis.end(), T(xValue));
        if (it == xAxis.end()) {
            return static_cast<T>(ySpan.back());
        }
        auto idxHigh = std::distance(xAxis.begin(), it);
        if (idxHigh == 0) {
            return static_cast<T>(ySpan.front());
        }
        std::size_t iHigh = static_cast<std::size_t>(idxHigh);
        std::size_t iLow  = iHigh - 1;

        T xLow  = static_cast<T>(xAxis[iLow]);
        T xHigh = static_cast<T>(xAxis[iHigh]);
        T yLow  = static_cast<T>(ySpan[iLow]);
        T yHigh = static_cast<T>(ySpan[iHigh]);

        // linear interpolation
        T dx = xHigh - xLow;
        if (std::abs(gr::value(dx)) == T(0)) {
            return yLow;
        }
        const T t = (xValue - xLow) / dx;
        return yLow + t * (yHigh - yLow);
    }

    throw gr::exception(fmt::format("axis dimIndex {} is out of range [0, {}] and", dimIndex, dataSet.axisCount()));
}

template<typename T>
std::vector<T> getSubArrayCopy(const DataSet<T>& ds, std::size_t indexMin, std::size_t indexMax, std::size_t signalIndex = 0) {
    if (indexMax <= indexMin) {
        return {};
    }
    indexMax                  = detail::checkIndexRange(ds, indexMin, indexMax, signalIndex);
    std::span<const T> values = ds.signalValues(signalIndex);
    return std::vector<T>(values.begin() + static_cast<std::ptrdiff_t>(indexMin), values.begin() + static_cast<std::ptrdiff_t>(indexMax));
}

template<typename T, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] constexpr T tenLog10(T x) noexcept {
    if (std::abs(gr::value(x)) <= T(0)) {
        return -std::numeric_limits<TValue>::infinity();
    }
    return T(10) * gr::math::log10(x); // 10 * log10(x)
}

template<typename T, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] constexpr T decibel(T x) noexcept {
    if (std::abs(gr::value(x)) <= T(0)) {
        return -std::numeric_limits<TValue>::infinity();
    }
    return T(20) * gr::math::log10(x); // 20 * log10(x)
}

template<typename T>
[[nodiscard]] constexpr T inverseDecibel(T x) noexcept {
    return gr::math::pow(T(10), x / T(20)); // Inverse decibel => 10^(value / 20)
}

template<bool ThrowOnFailure = true, typename T>
[[maybe_unused]] constexpr bool verify(const DataSet<T>& dataSet, std::source_location location = std::source_location::current()) {
    auto handleFailure = [&](const std::string& message) -> bool {
        if constexpr (ThrowOnFailure) {
            throw gr::exception(message, location);
        }
        return false;
    };

    // axes checks
    if (dataSet.axisCount() == 0UZ) {
        return handleFailure("DataSet has zero axes.");
    }
    if (dataSet.axis_names.size() != dataSet.axisCount()) {
        return handleFailure("axis_names size does not match axisCount().");
    }
    if (dataSet.axis_units.size() != dataSet.axisCount()) {
        return handleFailure("axis_units size does not match axisCount().");
    }
    if (dataSet.axis_values.size() != dataSet.axisCount()) {
        return handleFailure("axis_values size does not match axisCount().");
    }
    for (std::size_t axis = 0; axis < dataSet.axisCount(); ++axis) {
        if (dataSet.axis_values[axis].empty()) {
            return handleFailure(fmt::format("axis_values[{}] is empty.", axis));
        }
    }

    // verify extends
    if (dataSet.extents.empty()) {
        return handleFailure("DataSet has no extents defined.");
    }
    for (std::size_t dim = 0; dim < dataSet.extents.size(); ++dim) {
        if (dataSet.extents[dim] <= 0) {
            return handleFailure(fmt::format("Extent at dimension {} is non-positive.", dim));
        }
    }

    // verify signal_names, signal_quantities, and signal_units sizes match extents[0]
    std::size_t num_signals = static_cast<std::size_t>(dataSet.extents[0]);
    if (dataSet.signal_names.size() != num_signals) {
        return handleFailure("signal_names size does not match extents[0].");
    }
    if (dataSet.signal_quantities.size() != num_signals) {
        return handleFailure("signal_quantities size does not match extents[0].");
    }
    if (dataSet.signal_units.size() != num_signals) {
        return handleFailure("signal_units size does not match extents[0].");
    }

    std::size_t expected_signal_size = std::accumulate(dataSet.extents.begin(), dataSet.extents.end(), static_cast<std::size_t>(1), std::multiplies<>());
    if (dataSet.signal_values.size() != expected_signal_size) {
        return handleFailure("signal_values size does not match the product of extents.");
    }
    if (dataSet.signal_ranges.size() != num_signals) {
        return handleFailure("signal_ranges size does not match the number of signals.");
    }
    if (dataSet.meta_information.size() != num_signals) { // Assuming meta_information per signal value
        return handleFailure("meta_information size does not match the number of signals.");
    }
    if (dataSet.timing_events.size() != num_signals) { // Assuming timing_events per signal value
        return handleFailure("timing_events size does not match the number of signals.");
    }

    return true; // all checks passed
}

} // namespace gr::dataset

#endif // DATASETHELPER_HPP
