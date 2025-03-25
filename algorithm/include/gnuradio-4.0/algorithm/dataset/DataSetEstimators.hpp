#ifndef DATASETESTIMATORS_HPP
#define DATASETESTIMATORS_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <numeric>
#include <optional>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/Message.hpp>
#include <gnuradio-4.0/meta/UncertainValue.hpp>

#include "DataSetHelper.hpp"
#include "DataSetMath.hpp"

namespace gr::dataset {

template<typename T>
struct Point {
    std::size_t index;
    T           value;
};

namespace estimators {
template<typename T>
[[nodiscard]] constexpr T computeCentreOfMass(const DataSet<T>& ds, std::size_t minIndex = 0UZ, std::size_t maxIndex = max_size, std::size_t signalIndex = 0, std::source_location location = std::source_location::current()) {
    maxIndex = detail::checkIndexRange(ds, minIndex, maxIndex, signalIndex, location);
    T com    = T(0);
    T mass   = T(0);

    for (std::size_t i = minIndex; i < maxIndex; i++) {
        const T x     = getIndexValue(ds, dim::X, i);
        const T value = getIndexValue(ds, dim::Y, i, signalIndex);
        if (gr::math::isfinite(x) && gr::math::isfinite(value)) {
            com += x * value;
            mass += value;
        }
    }
    if (gr::value(mass) == gr::value(T(0))) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    return com / mass;
}

template<std::ranges::random_access_range T, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] constexpr TValue computeFWHM(const T& data, std::size_t index) {
    using value_t = gr::meta::fundamental_base_value_type_t<TValue>; // innermost value
    if (!(index > 0UZ && index < data.size() - 1UZ)) {
        return std::numeric_limits<TValue>::quiet_NaN();
    }
    TValue      maxHalf = value_t(0.5) * gr::value(data[index]);
    std::size_t lowerLimit;
    std::size_t upperLimit;
    for (upperLimit = index; upperLimit < data.size() && data[upperLimit] > maxHalf; upperLimit++) {
        // done in condition
    }
    for (lowerLimit = index; data[lowerLimit] > maxHalf; lowerLimit--) {
        // done in condition
    }
    if (upperLimit >= data.size()) {
        return std::numeric_limits<TValue>::quiet_NaN();
    }
    return static_cast<TValue>(upperLimit - lowerLimit);
}

template<std::ranges::random_access_range T, typename TValue = typename T::value_type>
[[nodiscard]] constexpr TValue computeInterpolatedFWHM(const T& data, std::size_t index) {
    using value_t = gr::meta::fundamental_base_value_type_t<TValue>; // innermost value
    if (!(index > 0 && index < data.size() - 1)) {
        return std::numeric_limits<TValue>::quiet_NaN();
    }
    TValue      maxHalf = value_t(0.5) * data[index];
    std::size_t lowerLimit;
    std::size_t upperLimit;
    for (upperLimit = index; upperLimit < data.size() && data[upperLimit] > maxHalf; upperLimit++) {
        // done in condition
    }
    for (lowerLimit = index; data[lowerLimit] > maxHalf; lowerLimit--) {
        // done in condition
    }
    if (upperLimit >= data.size()) {
        return std::numeric_limits<TValue>::quiet_NaN();
    }
    TValue lowerRefined = detail::linearInterpolate(value_t(lowerLimit), value_t(lowerLimit + 1), data[lowerLimit], data[lowerLimit + 1], maxHalf);
    TValue upperRefined = detail::linearInterpolate(value_t(upperLimit - 1), value_t(upperLimit), data[upperLimit - 1], data[upperLimit], maxHalf);
    return upperRefined - lowerRefined;
}

template<MetaInfo mode = MetaInfo::Apply, DataSetLike D, typename T = typename std::remove_cvref_t<D>::value_type>
[[nodiscard]] constexpr std::optional<Point<T>> getMaximum(D&& dataSet, std::size_t indexMin = 0UZ, std::size_t indexMax = max_size, std::size_t signalIndex = 0UZ, std::source_location location = std::source_location::current()) {
    using TValue                  = gr::meta::fundamental_base_value_type_t<T>;
    constexpr bool isConstDataSet = std::is_const_v<std::remove_reference_t<D>>;
    indexMax                      = detail::checkIndexRange(dataSet, indexMin, indexMax, signalIndex, location);
    if (indexMin == indexMax) {
        return Point<T>{indexMin, gr::dataset::getIndexValue(dataSet, dim::Y, indexMin, signalIndex)};
    }

    int locMax = -1;
    T   maxVal = std::numeric_limits<TValue>::lowest();
    for (std::size_t i = indexMin; i < indexMax; i++) {
        if (T actual = getIndexValue(dataSet, dim::Y, i, signalIndex); gr::math::isfinite(actual) && actual > maxVal) {
            maxVal = actual;
            locMax = static_cast<int>(i);
        }
    }
    if constexpr (!isConstDataSet && mode == MetaInfo::Apply) {
        if (locMax >= 0) {
            dataSet.timing_events[signalIndex].push_back({static_cast<std::ptrdiff_t>(locMax), {{"gr:maximum", gr::value(maxVal)}}});
        }
    }

    return locMax >= 0 ? std::make_optional(Point<T>{static_cast<std::size_t>(locMax), maxVal}) : std::nullopt;
}

template<MetaInfo mode = MetaInfo::Apply, DataSetLike D, typename T = typename std::remove_cvref_t<D>::value_type>
[[nodiscard]] constexpr std::optional<Point<T>> getMinimum(D&& dataSet, std::size_t indexMin = 0UZ, std::size_t indexMax = max_size, std::size_t signalIndex = 0UZ, std::source_location location = std::source_location::current()) {
    using TValue                  = gr::meta::fundamental_base_value_type_t<T>;
    constexpr bool isConstDataSet = std::is_const_v<std::remove_reference_t<D>>;
    indexMax                      = detail::checkIndexRange(dataSet, indexMin, indexMax, signalIndex, location);
    if (indexMin == indexMax) {
        return Point<T>{indexMin, gr::dataset::getIndexValue(dataSet, dim::Y, indexMin, signalIndex)};
    }

    T   minVal = +std::numeric_limits<TValue>::max();
    int locMin = -1;
    for (std::size_t i = indexMin; i < indexMax; i++) {
        if (T actual = gr::dataset::getIndexValue(dataSet, dim::Y, i, signalIndex); gr::math::isfinite(actual)) {
            locMin = static_cast<int>(i);
            minVal = std::min(minVal, actual);
        }
    }
    if constexpr (!isConstDataSet && mode == MetaInfo::Apply) {
        if (locMin >= 0) {
            dataSet.timing_events[signalIndex].push_back({static_cast<std::ptrdiff_t>(locMin), {{"gr:minimum", gr::value(minVal)}}});
        }
    }

    return locMin >= 0 ? std::make_optional(Point<T>{static_cast<std::size_t>(locMin), minVal}) : std::nullopt;
}

template<typename T, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] constexpr T getMean(const DataSet<T>& dataSet, std::size_t indexMin = 0UZ, std::size_t indexMax = max_size, std::size_t signalIndex = 0UZ, std::source_location location = std::source_location::current()) {
    indexMax = detail::checkIndexRange(dataSet, indexMin, indexMax, signalIndex, location);

    auto signalRange  = dataSet.signalValues(signalIndex) | std::views::drop(indexMin) | std::views::take(indexMax - indexMin);
    auto finiteValues = signalRange | std::views::filter(gr::math::isfinite<T>);

    T sum   = 0;
    T count = 0;
    for (const auto& val : finiteValues) {
        sum += val;
        count += T(1);
    }

    return count > T(0) ? sum / count : std::numeric_limits<TValue>::quiet_NaN();
}

template<typename T, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] constexpr T getMedian(const DataSet<T>& dataSet, std::size_t indexMin = 0UZ, std::size_t indexMax = max_size, std::size_t signalIndex = 0UZ, std::source_location location = std::source_location::current()) {
    indexMax = detail::checkIndexRange(dataSet, indexMin, indexMax, signalIndex, location);

    std::span<const T> values = dataSet.signalValues(signalIndex).subspan(indexMin, indexMax - indexMin);

    if (values.empty()) {
        return std::numeric_limits<TValue>::quiet_NaN();
    }

    std::vector<T> data(values.begin(), values.end()); // temporary mutable copy for in-place partitioning
    auto           mid = data.begin() + static_cast<std::ptrdiff_t>(data.size()) / 2;
    std::ranges::nth_element(data, mid);

    if ((data.size() & 1UZ) == 0UZ) {
        // even-sized data, calculate the mean of the two middle elements
        auto midPrev = std::ranges::max_element(data.begin(), mid);
        return static_cast<T>(0.5) * (*midPrev + *mid);
    }

    return static_cast<T>(*mid); // odd-sized data, return the middle element
}

template<typename T>
[[nodiscard]] constexpr T getRange(const DataSet<T>& dataSet, std::size_t indexMin = 0UZ, std::size_t indexMax = max_size, std::size_t signalIndex = 0UZ, std::source_location location = std::source_location::current()) {
    indexMax = detail::checkIndexRange(dataSet, indexMin, indexMax, signalIndex, location);

    auto signalRange  = dataSet.signalValues(signalIndex) | std::views::drop(indexMin) | std::views::take(indexMax - indexMin);
    auto finiteValues = signalRange | std::views::filter(gr::math::isfinite<T>);

    auto [minIt, maxIt] = std::ranges::minmax_element(finiteValues);
    if (minIt == finiteValues.end() || maxIt == finiteValues.end()) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    return *maxIt - *minIt;
}

template<typename T>
[[nodiscard]] constexpr T getRms(const DataSet<T>& dataSet, std::size_t indexMin = 0UZ, std::size_t indexMax = max_size, std::size_t signalIndex = 0UZ, std::source_location location = std::source_location::current()) {
    indexMax = detail::checkIndexRange(dataSet, indexMin, indexMax, signalIndex, location);

    auto signalRange  = dataSet.signalValues(signalIndex) | std::views::drop(indexMin) | std::views::take(indexMax - indexMin);
    auto finiteValues = signalRange | std::views::filter(gr::math::isfinite<T>);

    if (finiteValues.empty()) {
        return T(0);
    }

    T sum   = 0;
    T sum2  = 0;
    T count = 0;
    for (const auto& val : finiteValues) {
        sum += val;
        sum2 += val * val;
        count += T(1);
    }
    T mean1 = (sum / count) * (sum / count);
    T mean2 = sum2 / count;
    return gr::math::sqrt(mean2 > mean1 ? mean2 - mean1 : mean1 - mean2); // abs for safety
}

template<typename T>
[[nodiscard]] constexpr T getIntegral(const DataSet<T>& dataSet, std::size_t indexMin = 0UZ, std::size_t indexMax = max_size, std::size_t signalIndex = 0UZ, std::source_location location = std::source_location::current()) {
    indexMax = detail::checkIndexRange(dataSet, indexMin, indexMax, signalIndex, location);

    // sign if reversed
    const T           sign_ = detail::sign(T(1), int(indexMax) - int(indexMin));
    const std::size_t start = std::min(indexMin, indexMax);
    const std::size_t stop  = std::max(indexMin, indexMax);
    if (stop <= start + 1UZ) {
        return T(0);
    }

    auto idxRange = std::views::iota(start, stop - 1);
    T    integral = std::transform_reduce(std::ranges::begin(idxRange), std::ranges::end(idxRange), T(0), std::plus<>{}, //
           [&](std::size_t i) -> T {
            T x0 = getIndexValue(dataSet, dim::X, i, signalIndex);
            T x1 = getIndexValue(dataSet, dim::X, i + 1, signalIndex);
            T y0 = getIndexValue(dataSet, dim::Y, i, signalIndex);
            T y1 = getIndexValue(dataSet, dim::Y, i + 1, signalIndex);

            T area = T(0.5) * (x1 - x0) * (y0 + y1);
            return gr::math::isfinite(area) ? area : T(0);
        });

    return static_cast<T>(sign_ * integral);
}

template<typename T, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] constexpr T getEdgeDetect(const DataSet<T>& dataSet, std::size_t indexMin = 0UZ, std::size_t indexMax = max_size, std::size_t signalIndex = 0UZ, std::source_location location = std::source_location::current()) {
    indexMax = detail::checkIndexRange(dataSet, indexMin, indexMax, signalIndex, location);

    T minVal = getMinimum(dataSet, indexMin, indexMax, signalIndex).value().value;
    T maxVal = getMaximum(dataSet, indexMin, indexMax, signalIndex).value().value;
    T range  = maxVal > minVal ? maxVal - minVal : minVal - maxVal; // abs
    if (!gr::math::isfinite(range) || range == TValue(0)) {
        return std::numeric_limits<TValue>::quiet_NaN();
    }
    // check if falling or rising
    T    startVal  = getIndexValue(dataSet, dim::Y, indexMin, signalIndex);
    T    endVal    = getIndexValue(dataSet, dim::Y, indexMax - 1, signalIndex);
    bool inverted  = (startVal > endVal);
    T    startTime = getIndexValue(dataSet, dim::X, indexMin, signalIndex);

    T halfCross = inverted ? (maxVal - TValue(0.5) * range) : (minVal + TValue(0.5) * range);
    for (std::size_t i = indexMin; i < indexMax; i++) {
        T actual = getIndexValue(dataSet, dim::Y, i, signalIndex);
        if (!gr::math::isfinite(actual)) {
            continue;
        }
        if ((!inverted && actual > halfCross) || (inverted && actual < halfCross)) {
            T xNow = getIndexValue(dataSet, dim::X, i, signalIndex);
            return xNow - startTime;
        }
    }
    return std::numeric_limits<TValue>::quiet_NaN();
}

template<typename T, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] constexpr T getDutyCycle(const DataSet<T>& dataSet, std::size_t indexMin = 0UZ, std::size_t indexMax = max_size, std::size_t signalIndex = 0UZ, std::source_location location = std::source_location::current()) {
    indexMax = detail::checkIndexRange(dataSet, indexMin, indexMax, signalIndex, location);

    T minVal = getMinimum(dataSet, indexMin, indexMax, signalIndex).value().value;
    T maxVal = getMaximum(dataSet, indexMin, indexMax, signalIndex).value().value;

    if (!gr::math::isfinite(minVal) || !gr::math::isfinite(maxVal) || minVal == maxVal) {
        return std::numeric_limits<TValue>::quiet_NaN();
    }

    auto signalRange  = dataSet.signalValues(signalIndex) | std::views::drop(indexMin) | std::views::take(indexMax - indexMin);
    auto finiteValues = signalRange | std::views::filter(gr::math::isfinite<T>);

    T       range        = maxVal > minVal ? maxVal - minVal : minVal - maxVal;
    const T thresholdMin = minVal + TValue(0.45) * range;
    const T thresholdMax = minVal + TValue(0.55) * range;
    int     countLow = 0, countHigh = 0;
    for (const auto& val : finiteValues) {
        if (val < thresholdMin) {
            countLow++;
        } else if (val > thresholdMax) {
            countHigh++;
        }
    }

    const int totalCount = countLow + countHigh;
    return totalCount > 0 ? static_cast<TValue>(countHigh) / static_cast<TValue>(totalCount) : std::numeric_limits<TValue>::quiet_NaN();
}

template<typename T, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] constexpr T getFrequencyEstimate(const DataSet<T>& dataSet, std::size_t indexMin = 0UZ, std::size_t indexMax = max_size, std::size_t signalIndex = 0UZ, std::source_location location = std::source_location::current()) {
    // getFrequencyEstimate => naive/simple approach counting edges
    indexMax = detail::checkIndexRange(dataSet, indexMin, indexMax, signalIndex, location);

    T minVal = getMinimum(dataSet, indexMin, indexMax, signalIndex).value().value;
    T maxVal = getMaximum(dataSet, indexMin, indexMax, signalIndex).value().value;
    if (!gr::math::isfinite(minVal) || !gr::math::isfinite(maxVal) || maxVal == minVal) {
        return std::numeric_limits<TValue>::quiet_NaN();
    }
    T range        = maxVal > minVal ? maxVal - minVal : minVal - maxVal; // abs
    T thresholdMin = minVal + TValue(0.45) * range;
    T thresholdMax = minVal + TValue(0.55) * range;

    T   startRisingEdge  = std::numeric_limits<TValue>::quiet_NaN();
    T   startFallingEdge = std::numeric_limits<TValue>::quiet_NaN();
    T   avgPeriod        = 0;
    int avgPeriodCount   = 0;

    // 0 => below threshold, 1 => above threshold
    float actualState = 0.f;
    for (std::size_t i = indexMin; i < indexMax; i++) {
        T actual = getIndexValue(dataSet, dim::Y, i, signalIndex);
        if (!gr::math::isfinite(actual)) {
            continue;
        }
        T x = getIndexValue(dataSet, dim::X, i, signalIndex);

        if (actualState < 0.5f) {
            // was low, check if we cross thresholdMax => rising edge
            if (actual > thresholdMax) {
                actualState = 1.0f;
                if (gr::math::isfinite(startRisingEdge)) {
                    T period        = x - startRisingEdge;
                    startRisingEdge = x;
                    avgPeriod += period;
                    avgPeriodCount++;
                } else {
                    startRisingEdge = x;
                }
            }
        } else {
            // was high, check if we cross thresholdMin => falling edge
            if (actual < thresholdMin) {
                actualState = 0.0f;
                if (gr::math::isfinite(startFallingEdge)) {
                    T period         = x - startFallingEdge;
                    startFallingEdge = x;
                    avgPeriod += period;
                    avgPeriodCount++;
                } else {
                    startFallingEdge = x;
                }
            }
        }
    }
    if (avgPeriodCount == 0) {
        return std::numeric_limits<TValue>::quiet_NaN();
    }
    return static_cast<TValue>(avgPeriodCount) / avgPeriod; // freq = # of periods / total time
}

template<std::ranges::random_access_range T, typename TValue = typename T::value_type>
[[nodiscard]] constexpr TValue interpolateGaussian(const T& data, std::size_t index) {
    using value_t = gr::meta::fundamental_base_value_type_t<TValue>;
    if (!(index > 0 && index < data.size() - 1)) {
        return static_cast<value_t>(index); // fallback
    }
    const TValue left   = data[index - 1];
    const TValue center = data[index];
    const TValue right  = data[index + 1];

    if (!gr::math::isfinite(left) || !gr::math::isfinite(right) || !gr::math::isfinite(center)) {
        return static_cast<value_t>(index);
    }

    if (gr::value(left) <= value_t(0) || gr::value(right) <= value_t(0) || gr::value(center) <= value_t(0)) {
        return static_cast<value_t>(index);
    }
    TValue val         = static_cast<value_t>(index);
    TValue numerator   = value_t(0.5f) * gr::math::log(right / left);
    TValue denominator = gr::math::log((center * center) / (left * right));
    if (gr::value(denominator) == TValue(0)) {
        return val;
    }
    return val + numerator / denominator;
}

template<typename T, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] constexpr T getLocationMaximumGaussInterpolated(const DataSet<T>& dataSet, std::size_t indexMin = 0UZ, std::size_t indexMax = max_size, std::size_t signalIndex = 0UZ, std::source_location location = std::source_location::current()) {
    // getFrequencyEstimate => naive/simple approach counting edges
    indexMax = detail::checkIndexRange(dataSet, indexMin, indexMax, signalIndex, location);

    const auto locMax_ = getMaximum(dataSet, indexMin, indexMax, signalIndex);
    if (!locMax_.has_value()) {
        return std::numeric_limits<TValue>::quiet_NaN();
    }
    const std::size_t locMax = locMax_.value().index;

    T refinedIndex = interpolateGaussian(dataSet.signalValues(signalIndex), locMax);
    T x0           = getIndexValue(dataSet, dim::X, locMax, signalIndex);
    // approximate step
    // we’ll do a naive approach: X_{locMax+1} - X_{locMax}
    if (locMax + 1UZ >= static_cast<std::size_t>(dataSet.axisValues(dim::X).size())) {
        return std::numeric_limits<TValue>::quiet_NaN();
    }
    T x1       = getIndexValue(dataSet, dim::X, locMax + 1UZ, signalIndex);
    T diff     = x1 - x0;
    T deltaBin = refinedIndex - static_cast<TValue>(locMax);
    return x0 + deltaBin * diff;
}

template<MetaInfo mode = MetaInfo::Apply, DataSetLike D, typename T = typename std::remove_cvref_t<D>::value_type, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] constexpr T getZeroCrossing(D&& dataSet, TValue threshold, std::size_t indexMin = 0UZ, std::size_t indexMax = max_size, std::size_t signalIndex = 0, std::source_location location = std::source_location::current()) {
    constexpr bool isConstDataSet = std::is_const_v<std::remove_reference_t<D>>;
    indexMax                      = detail::checkIndexRange(dataSet, indexMin, indexMax, signalIndex, location);

    T    initial  = getIndexValue(dataSet, dim::Y, 0UZ, signalIndex);
    int  index    = -1;
    T    value    = std::numeric_limits<TValue>::quiet_NaN();
    bool isRising = initial < threshold;
    if (isRising) { // rising edge
        for (std::size_t i = indexMin + 1UZ; i < indexMax; ++i) {
            T yPrev = getIndexValue(dataSet, dim::Y, i - 1UZ, signalIndex);
            T yCurr = getIndexValue(dataSet, dim::Y, i, signalIndex);
            if (gr::math::isfinite(yPrev) && gr::math::isfinite(yCurr) && yCurr >= threshold) {
                T xPrev = getIndexValue(dataSet, dim::X, i - 1UZ, signalIndex);
                T xCurr = getIndexValue(dataSet, dim::X, i, signalIndex);
                index   = static_cast<int>(i);
                value   = gr::dataset::detail::linearInterpolate(xPrev, xCurr, yPrev, yCurr, T(threshold));
                break;
            }
        }
    } else { // falling edge
        for (std::size_t i = indexMin + 1UZ; i < indexMax; ++i) {
            T yPrev = getIndexValue(dataSet, dim::Y, i - 1UZ, signalIndex);
            T yCurr = getIndexValue(dataSet, dim::Y, i, signalIndex);
            if (gr::math::isfinite(yPrev) && gr::math::isfinite(yCurr) && yCurr <= threshold) {
                T xPrev = getIndexValue(dataSet, dim::X, i - 1UZ, signalIndex);
                T xCurr = getIndexValue(dataSet, dim::X, i, signalIndex);
                index   = static_cast<int>(i);
                value   = gr::dataset::detail::linearInterpolate(xPrev, xCurr, yPrev, yCurr, T(threshold));
                break;
            }
        }
    }
    if (index < 0) {
        index = static_cast<int>(indexMax); // chose end-of-range
        value = getIndexValue(dataSet, dim::Y, indexMax - 1UZ, signalIndex);
    }

    if constexpr (!isConstDataSet && mode == MetaInfo::Apply) {
        const auto          idx = static_cast<std::size_t>(index);
        std::string         context;
        const std::uint64_t period          = 1'000'000'000;
        const std::uint64_t time            = static_cast<std::uint64_t>(gr::value(value)) * period;
        const std::uint64_t timeUncertainty = static_cast<std::uint64_t>(gr::uncertainty(value)) * period;
        property_map        data            = property_map{{gr::tag::TRIGGER_NAME.shortKey(), fmt::format("{}_EDGE_LEVEL_{}", isRising ? "RISING" : "FALLING", threshold)}, //,                        //
                              {gr::tag::TRIGGER_TIME.shortKey(), time}, {"trigger_time_error", timeUncertainty}, {gr::tag::TRIGGER_OFFSET.shortKey(), 0.f},                 //
                              {gr::tag::CONTEXT.shortKey(), context}};
        dataSet.timing_events[signalIndex].push_back({idx, std::move(data)});
    }
    return value;
}

template<typename T>
struct StepStartDetectionResult {
    std::size_t index;        ///< Index where the step starts.
    T           initialValue; ///< Initial value of the signal.
    T           minValue;     ///< Minimum value of the signal.
    T           maxValue;     ///< Maximum value of the signal.
    bool        isRising;     ///< Indicates if the detected step is rising (true) or falling (false).
};

template<MetaInfo mode = MetaInfo::Apply, DataSetLike D, typename T = typename std::remove_cvref_t<D>::value_type, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
std::optional<StepStartDetectionResult<T>> detectStepStart(D& ds, TValue threshold = TValue(0.5), std::size_t indexMin = 0UZ, std::size_t indexMax = max_size, std::size_t signalIndex = 0UZ, std::source_location location = std::source_location::current()) {
    constexpr bool isConstDataSet = std::is_const_v<std::remove_reference_t<D>>;

    std::expected<void, gr::Error> dsCheck = gr::dataset::checkConsistency(ds, "", location);
    if (!dsCheck.has_value()) {
        throw gr::exception(fmt::format("Invalid DataSet for step/pulse start detection: {}", dsCheck.error()), location);
    }
    indexMax = detail::checkIndexRange(ds, indexMin, indexMax, signalIndex, location);

    auto signal = ds.signalValues(signalIndex);
    if (signal.empty()) {
        throw gr::exception("Signal is empty.");
    }

    T initial = signal.front();
    T max_val = estimators::getMaximum(ds, indexMin, indexMax, signalIndex).value().value;
    T min_val = estimators::getMinimum(ds, indexMin, indexMax, signalIndex).value().value;
    if constexpr (!isConstDataSet && mode == MetaInfo::Apply) {
        ds.meta_information[signalIndex]["gr:minimum"] = gr::value(min_val);
        ds.meta_information[signalIndex]["gr:maximum"] = gr::value(max_val);
    }

    bool isRising  = initial < max_val;
    bool isFalling = initial > min_val;

    if (!isRising && !isFalling) {
        return std::nullopt; // no clear step detected
    }

    T thresholdCrossing = isRising ? initial + threshold * (max_val - initial) : initial - threshold * (initial - min_val);
    T triggerTime       = estimators::getZeroCrossing<MetaInfo::None>(ds, thresholdCrossing, indexMin, indexMax, signalIndex);
    if (std::isnan(gr::value(triggerTime))) {
        return std::nullopt; // failed to compute trigger time
    }

    // Find the closest index to the triggerTime
    const auto& xAxis = ds.axisValues(0);
    auto        it    = std::lower_bound(xAxis.begin(), xAxis.end(), triggerTime);
    if (it == xAxis.end()) {
        return std::nullopt; // no step detected
    }
    std::size_t index = static_cast<std::size_t>(std::distance(xAxis.begin(), it));

    // ensure the index is within bounds
    if (index >= signal.size()) {
        return std::nullopt;
    }

    if constexpr (!isConstDataSet && mode == MetaInfo::Apply) {
        std::string         context;
        const std::uint64_t period          = 1'000'000'000;
        const std::uint64_t time            = static_cast<std::uint64_t>(gr::value(xAxis[index])) * period;
        const std::uint64_t timeUncertainty = 0UZ;
        property_map        data            = property_map{{gr::tag::TRIGGER_NAME.shortKey(), fmt::format("{}_EDGE_LEVEL_{}", isRising ? "RISING" : "FALLING", threshold)}, //
                              {gr::tag::TRIGGER_TIME.shortKey(), time}, {"trigger_time_error", timeUncertainty}, {gr::tag::TRIGGER_OFFSET.shortKey(), 0.f},                 //
                              {gr::tag::CONTEXT.shortKey(), context}};
        ds.timing_events[signalIndex].push_back({index, std::move(data)});
    }
    return StepStartDetectionResult<T>{.index = index, .initialValue = initial, .minValue = min_val, .maxValue = max_val, .isRising = isRising};
}

template<MetaInfo mode = MetaInfo::Apply, DataSetLike D, typename T = typename std::remove_cvref_t<D>::value_type, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] constexpr std::optional<Point<T>> getSettingTime(D&& dataSet, std::size_t windowSize, TValue tolerance = TValue(0), std::size_t indexMin = 0UZ, std::size_t indexMax = max_size, std::size_t signalIndex = 0UZ, std::source_location location = std::source_location::current()) {
    constexpr bool isConstDataSet = std::is_const_v<std::remove_reference_t<D>>;
    indexMax                      = detail::checkIndexRange(dataSet, indexMin, indexMax, signalIndex, location);
    if (windowSize == 0UZ || (indexMax - indexMin) < windowSize) {
        return std::nullopt;
    }
    if (tolerance == TValue(0)) {
        T minVal  = getMinimum(dataSet, indexMin, indexMax, signalIndex).value().value;
        T maxVal  = getMaximum(dataSet, indexMin, indexMax, signalIndex).value().value;
        tolerance = TValue(0.02) * gr::value(gr::math::abs(maxVal - minVal));
    }

    // estimate settling time by shifting RMS window until the RMS is below the required threshold
    int settlingTime = -1;
    for (std::size_t i = indexMin; i + windowSize <= indexMax; ++i) {
        T localRMS = estimators::getRms(dataSet, i, i + windowSize, signalIndex);
        if (gr::math::abs(localRMS) <= tolerance) {
            settlingTime = static_cast<int>(i);
            break;
        }
    }
    if (settlingTime < 0) {
        return std::nullopt;
    }

    // estimate flat-top level by increasing the window as long as RMS is getting smaller
    const std::size_t settlingTimeIdx = static_cast<std::size_t>(settlingTime);
    T                 prevRMS         = estimators::getRms(dataSet, settlingTimeIdx, settlingTimeIdx + windowSize, signalIndex);
    std::size_t       maxIndex;
    for (maxIndex = settlingTimeIdx; maxIndex + windowSize < indexMax; ++maxIndex) {
        T localRMS = estimators::getRms(dataSet, settlingTimeIdx, maxIndex + windowSize, signalIndex);
        if (localRMS >= prevRMS) {
            break;
        }
        prevRMS = localRMS;
    }
    T settlingLevel = estimators::getMean(dataSet, settlingTimeIdx, maxIndex + windowSize, signalIndex);
    settlingTime += static_cast<int>(windowSize) / 2;
    if (settlingTime < 0) {
        return std::nullopt;
    }

    if constexpr (!isConstDataSet && mode == MetaInfo::Apply) {
        const auto          idx = static_cast<std::size_t>(settlingTime);
        std::string         context;
        const auto          xAxis           = dataSet.axisValues(dim::X);
        const std::uint64_t period          = 1'000'000'000;
        const std::uint64_t time            = static_cast<std::uint64_t>(gr::value(xAxis[idx])) * period;
        const std::uint64_t timeUncertainty = 0UZ;
        property_map        data            = property_map{{gr::tag::TRIGGER_NAME.shortKey(), "SETTLING_TIME"}, {"gr:settling_level", gr::value(settlingLevel)}, //
                              {gr::tag::TRIGGER_TIME.shortKey(), time}, {"trigger_time_error", timeUncertainty}, {gr::tag::TRIGGER_OFFSET.shortKey(), 0.f},      //
                              {gr::tag::CONTEXT.shortKey(), context}};
        dataSet.timing_events[signalIndex].push_back({idx, std::move(data)});
    }

    return Point<T>{static_cast<std::size_t>(settlingTime), settlingLevel};
}

template<typename T>
struct StepPulseResponseMetrics {
    bool isEstimate    = true;  ///< indicates that values are rough estimates
    bool isPulse       = false; ///< indicates if the response is a pulse.
    bool isRising      = true;  ///< indicates if the first edge is a rsiging edge
    T    triggerTime   = 0.0;   ///< time at which the trigger occurs (50% crossing).
    T    V1            = 0.0;   ///< steady-state level before the step.
    T    V2            = 0.0;   ///< steady-state level after the step.
    T    riseTime      = 0.0;   ///< rise time (e.g., 10% -> 90% of V2 - V1).
    T    peakAmplitude = 0.0;   ///< peak amplitude relative to V1.
    T    peakTime      = 0.0;   ///< time at which the peak occurs.
    T    overshoot     = 0.0;   ///< overshoot percentage.
    T    settlingTime  = 0.0;   ///< settling time within 2% of V2.
};

template<MetaInfo mode = MetaInfo::Apply, DataSetLike D, typename T = typename std::remove_cvref_t<D>::value_type>
StepPulseResponseMetrics<T> analyzeStepPulseResponse(D&& ds, std::size_t indexMin = 0UZ, std::size_t indexMax = max_size, std::size_t signalIndex = 0UZ, std::source_location location = std::source_location::current()) {
    using TValue                  = gr::meta::fundamental_base_value_type_t<T>;
    constexpr bool isConstDataSet = std::is_const_v<std::remove_reference_t<D>>;
    indexMax                      = detail::checkIndexRange(ds, indexMin, indexMax, signalIndex, location);
    if (indexMax - indexMin < 2UZ) {
        throw gr::exception("Not enough data points for step/pulse analysis.");
    }

    // Step 1: detect initial rough edge
    auto firstRoughEdgeOpt = estimators::detectStepStart<MetaInfo::None>(ds, TValue(0.5), indexMin, indexMax, signalIndex);
    if (!firstRoughEdgeOpt.has_value()) {
        throw gr::exception("no step detected in the signal.");
    }
    const StepStartDetectionResult<T>& firstRoughEdge    = firstRoughEdgeOpt.value();
    const StepStartDetectionResult<T>  firstRoughLevel20 = estimators::detectStepStart<MetaInfo::None>(ds, TValue(0.2), indexMin, indexMax, signalIndex).value();
    const StepStartDetectionResult<T>  firstRoughLevel80 = estimators::detectStepStart<MetaInfo::None>(ds, TValue(0.8), indexMin, indexMax, signalIndex).value();
    const std::size_t                  roughRiseTime     = firstRoughLevel80.index - firstRoughLevel20.index;
    const TValue                       roughMinThreshold = gr::value(TValue(0.02) * (firstRoughEdge.maxValue - firstRoughEdge.minValue));

    StepPulseResponseMetrics<T> metrics;
    metrics.triggerTime = gr::cast<T>(firstRoughEdge.index);
    metrics.riseTime    = gr::cast<T>(roughRiseTime);

    std::span<const T> signalValues = ds.signalValues(signalIndex);
    const bool         isRising     = firstRoughEdge.isRising;

    // Step 2: compute V1 using rough detected edge
    const std::size_t V1Start      = 0UZ;
    const std::size_t V1End        = firstRoughEdge.index > roughRiseTime ? firstRoughEdge.index - roughRiseTime : 4UZ; // based on 20%-80% rise-time around ~50% level
    const TValue      rmsEstimate  = std::max(gr::value(estimators::getRms(ds, V1Start, V1End, signalIndex)), roughMinThreshold);
    const TValue      meanEstimate = gr::value(estimators::getMean(ds, V1Start, V1End, signalIndex));
    const T           rampStart    = estimators::getZeroCrossing<MetaInfo::None>(ds, meanEstimate + TValue(2.5) * (isRising ? +rmsEstimate : -rmsEstimate), V1Start, firstRoughEdge.index, signalIndex);

    metrics.V1 = estimators::getMean(ds, V1Start, std::min(static_cast<std::size_t>(gr::value(rampStart)), firstRoughEdge.index), signalIndex); // should be bias-free from over- and under-shoot

    // Step 3: detect peak position
    const std::size_t peakMaxIndex = std::min(firstRoughEdge.index + 2UZ * roughRiseTime, indexMax);
    auto              peakIndex    = isRising ? estimators::getMaximum(ds, firstRoughEdge.index, peakMaxIndex) : estimators::getMinimum(ds, firstRoughEdge.index, peakMaxIndex);
    if (!peakIndex.has_value()) { // no peak detected
        metrics.peakAmplitude = std::numeric_limits<TValue>::quiet_NaN();
        metrics.peakTime      = std::numeric_limits<TValue>::quiet_NaN();
    } else {
        metrics.peakAmplitude = ds.signalValues(signalIndex)[peakIndex.value().index] - metrics.V1;
        metrics.peakTime      = ds.axisValues(dim::X)[peakIndex.value().index];

        auto settlingTimeOpt = getSettingTime(ds, 4UZ * roughRiseTime, roughMinThreshold, peakIndex.value().index, indexMax, signalIndex);
        if (settlingTimeOpt.has_value()) {
            metrics.isEstimate   = false;
            metrics.settlingTime = gr::cast<T>(settlingTimeOpt.value().index);
            metrics.V2           = gr::cast<T>(settlingTimeOpt.value().value);
        } else {
            metrics.settlingTime = std::numeric_limits<TValue>::quiet_NaN();
            metrics.V2           = firstRoughEdge.maxValue;
        }
    }

    // we have now a more precise V1 and V2 level available to minimise the bias of over- and undershoot

    // Step 4: compute bias-free overshoot

    metrics.overshoot = ((metrics.peakAmplitude) / (metrics.V2 - metrics.V1)) * TValue(100);

    // Step 5: compute bias-free trigger time and 10%->90% rise-time
    const T riseLevel   = isRising ? metrics.V1 + TValue(0.5) * (metrics.V2 - metrics.V1) : metrics.V1 - TValue(0.5) * (metrics.V1 - metrics.V2);
    const T firstEdge   = estimators::getZeroCrossing<MetaInfo::Apply>(ds, riseLevel, indexMin, indexMax, signalIndex);
    metrics.triggerTime = firstEdge;

    const T riseStart    = isRising ? metrics.V1 + TValue(0.1) * (metrics.V2 - metrics.V1) : metrics.V1 - TValue(0.1) * (metrics.V1 - metrics.V2);
    const T riseEnd      = isRising ? metrics.V1 + TValue(0.9) * (metrics.V2 - metrics.V1) : metrics.V1 - TValue(0.9) * (metrics.V1 - metrics.V2);
    const T firstLevel10 = estimators::getZeroCrossing<MetaInfo::Apply>(ds, riseStart, indexMin, indexMax, signalIndex);
    const T firstLevel90 = estimators::getZeroCrossing<MetaInfo::Apply>(ds, riseEnd, indexMin, indexMax, signalIndex);
    metrics.riseTime     = firstLevel90 - firstLevel10;

    // Step 5: Determine if it's a pulse by checking for a falling or rising edge after the step
    const T lastValue = signalValues.back();
    metrics.isPulse   = gr::value(metrics.V2 - lastValue) > TValue(3) * roughMinThreshold;

    if constexpr (!isConstDataSet && mode == MetaInfo::Apply) {
        ds.meta_information[signalIndex]["gr::detected"]       = metrics.isPulse ? "pulse" : "step";
        ds.meta_information[signalIndex]["gr::triggerTime"]    = gr::value(metrics.triggerTime);
        ds.meta_information[signalIndex]["gr::rise_time"]      = gr::value(metrics.riseTime);
        ds.meta_information[signalIndex]["gr::V1"]             = gr::value(metrics.V1);
        ds.meta_information[signalIndex]["gr::V2"]             = gr::value(metrics.V2);
        ds.meta_information[signalIndex]["gr::peak_amplitude"] = gr::value(metrics.peakAmplitude);
        ds.meta_information[signalIndex]["gr::overshoot"]      = gr::value(metrics.overshoot);
    }

    return metrics;
}

} // namespace estimators

} // namespace gr::dataset

#endif // DATASETESTIMATORS_HPP
