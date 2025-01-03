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

namespace gr::dataset {

template<typename T>
[[nodiscard]] constexpr T inverseDecibel(T x) noexcept {
    return gr::math::pow(T(10), x / T(20)); // Inverse decibel => 10^(value / 20)
}

namespace estimators {
template<typename T>
[[nodiscard]] constexpr T computeCentreOfMass(const DataSet<T>& ds, std::size_t minIndex = 0UZ, std::size_t maxIndex = 0UZ, std::size_t signalIndex = 0) {
    if (maxIndex == 0UZ) { // renormalise default range
        maxIndex = ds.axisValues(dim::X).size();
    }
    detail::checkRangeIndex(minIndex, maxIndex, ds.axisValues(dim::X).size());
    T com  = T(0);
    T mass = T(0);

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
    for (lowerLimit = index; lowerLimit >= 0 && data[lowerLimit] > maxHalf; lowerLimit--) {
        // done in condition
    }
    if (upperLimit >= data.size() || lowerLimit < 0) {
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
    for (lowerLimit = index; lowerLimit >= 0 && data[lowerLimit] > maxHalf; lowerLimit--) {
        // done in condition
    }
    if (upperLimit >= data.size() || lowerLimit < 0) {
        return std::numeric_limits<TValue>::quiet_NaN();
    }
    TValue lowerRefined = detail::linearInterpolate(value_t(lowerLimit), value_t(lowerLimit + 1), data[lowerLimit], data[lowerLimit + 1], maxHalf);
    TValue upperRefined = detail::linearInterpolate(value_t(upperLimit - 1), value_t(upperLimit), data[upperLimit - 1], data[upperLimit], maxHalf);
    return upperRefined - lowerRefined;
}

template<typename T, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] constexpr int getLocationMinimum(const DataSet<T>& dataSet, std::size_t indexMin = 0UZ, std::size_t indexMax = 0UZ, std::size_t signalIndex = 0UZ) {
    if (indexMax == 0UZ) { // renormalise default range
        indexMax = dataSet.axisValues(dim::X).size();
    }
    detail::checkRangeIndex(indexMin, indexMax, dataSet.axisValues(dim::X).size());
    int locMin = -1;
    T   minVal = std::numeric_limits<TValue>::infinity();
    for (std::size_t i = indexMin; i < indexMax; i++) {
        if (T actual = getIndexValue(dataSet, dim::Y, i, signalIndex); gr::math::isfinite(actual) && actual < minVal) {
            minVal = actual;
            locMin = static_cast<int>(i);
        }
    }
    return locMin;
}

template<typename T, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] constexpr int getLocationMaximum(const DataSet<T>& dataSet, std::size_t indexMin = 0UZ, std::size_t indexMax = 0UZ, std::size_t signalIndex = 0UZ) {
    if (indexMax == 0UZ) { // renormalise default range
        indexMax = dataSet.axisValues(dim::X).size();
    }
    detail::checkRangeIndex(indexMin, indexMax, dataSet.axisValues(dim::X).size());
    int locMax = -1;
    T   maxVal = -std::numeric_limits<TValue>::infinity();
    for (std::size_t i = indexMin; i < indexMax; i++) {
        if (T actual = getIndexValue(dataSet, dim::Y, i, signalIndex); gr::math::isfinite(actual) && actual > maxVal) {
            maxVal = actual;
            locMax = static_cast<int>(i);
        }
    }
    return locMax;
}

template<typename T, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] constexpr T getMinimum(const DataSet<T>& dataSet, std::size_t indexMin = 0UZ, std::size_t indexMax = 0UZ, std::size_t signalIndex = 0UZ) {
    if (indexMax == 0UZ) { // renormalise default range
        indexMax = dataSet.axisValues(dim::X).size();
    }
    detail::checkRangeIndex(indexMin, indexMax, dataSet.axisValues(dim::X).size());

    T    val      = std::numeric_limits<TValue>::max();
    bool foundAny = false;
    for (std::size_t i = indexMin; i < indexMax; i++) {
        if (T actual = getIndexValue(dataSet, dim::Y, i, signalIndex); gr::math::isfinite(actual)) {
            foundAny = true;
            val      = std::min(val, actual);
        }
    }
    return foundAny ? val : std::numeric_limits<TValue>::quiet_NaN();
}

template<typename T, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] constexpr T getMaximum(const DataSet<T>& dataSet, std::size_t indexMin = 0UZ, std::size_t indexMax = 0UZ, std::size_t signalIndex = 0UZ) {
    if (indexMax == 0UZ) { // renormalise default range
        indexMax = dataSet.axisValues(dim::X).size();
    }
    detail::checkRangeIndex(indexMin, indexMax, dataSet.axisValues(dim::X).size());
    T    val      = -std::numeric_limits<TValue>::min();
    bool foundAny = false;
    for (std::size_t i = indexMin; i < indexMax; i++) {
        if (T actual = getIndexValue(dataSet, dim::Y, i, signalIndex); gr::math::isfinite(actual)) {
            foundAny = true;
            val      = std::max(val, actual);
        }
    }
    return foundAny ? val : std::numeric_limits<TValue>::quiet_NaN();
}

template<typename T, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] constexpr T getMean(const DataSet<T>& dataSet, std::size_t indexMin = 0UZ, std::size_t indexMax = 0UZ, std::size_t signalIndex = 0UZ) {
    if (indexMax == 0UZ) {
        indexMax = dataSet.axisValues(dim::X).size();
    }
    detail::checkRangeIndex(indexMin, indexMax, dataSet.axisValues(dim::X).size());

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
[[nodiscard]] constexpr T getMedian(const DataSet<T>& dataSet, std::size_t indexMin = 0UZ, std::size_t indexMax = 0UZ, std::size_t signalIndex = 0UZ) {
    if (indexMax == 0UZ) { // renormalise default range
        indexMax = dataSet.axisValues(dim::X).size();
    }
    std::span<const T> values = dataSet.signalValues(signalIndex).subspan(indexMin, indexMax - indexMin);

    if (values.empty()) {
        return std::numeric_limits<TValue>::quiet_NaN();
    }

    std::vector<T> data(values.begin(), values.end()); // Temporary mutable copy for in-place partitioning
    auto           mid = data.begin() + data.size() / 2;
    std::ranges::nth_element(data, mid);

    if ((data.size() & 1UZ) == 0UZ) {
        // even-sized data, calculate the mean of the two middle elements
        auto midPrev = std::ranges::max_element(data.begin(), mid);
        return static_cast<T>(0.5) * (*midPrev + *mid);
    }

    return static_cast<T>(*mid); // odd-sized data, return the middle element
}

template<typename T>
[[nodiscard]] constexpr T getRange(const DataSet<T>& dataSet, std::size_t indexMin = 0UZ, std::size_t indexMax = 0UZ, std::size_t signalIndex = 0UZ) {
    if (indexMax == 0UZ) { // Default range
        indexMax = dataSet.axisValues(dim::X).size();
    }
    detail::checkRangeIndex(indexMin, indexMax, dataSet.axisValues(dim::X).size());

    auto signalRange  = dataSet.signalValues(signalIndex) | std::views::drop(indexMin) | std::views::take(indexMax - indexMin);
    auto finiteValues = signalRange | std::views::filter(gr::math::isfinite<T>);

    auto [minIt, maxIt] = std::ranges::minmax_element(finiteValues);
    if (minIt == finiteValues.end() || maxIt == finiteValues.end()) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    return *maxIt - *minIt;
}

template<typename T>
[[nodiscard]] constexpr T getRms(const DataSet<T>& dataSet, std::size_t indexMin = 0UZ, std::size_t indexMax = 0UZ, std::size_t signalIndex = 0UZ) {
    if (indexMax == 0UZ) {
        indexMax = dataSet.axisValues(dim::X).size();
    }
    detail::checkRangeIndex(indexMin, indexMax, dataSet.axisValues(dim::X).size());

    auto signalRange  = dataSet.signalValues(signalIndex) | std::views::drop(indexMin) | std::views::take(indexMax - indexMin);
    auto finiteValues = signalRange | std::views::filter(gr::math::isfinite<T>);

    if (finiteValues.empty()) {
        return T(0);
    }

    T sum = 0, sum2 = 0;
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
[[nodiscard]] constexpr T getIntegral(const DataSet<T>& dataSet, std::size_t indexMin = 0UZ, std::size_t indexMax = 0UZ, std::size_t signalIndex = 0UZ) {
    if (indexMax == 0UZ) { // renormalise default range
        indexMax = dataSet.axisValues(dim::X).size();
    }
    detail::checkRangeIndex(indexMin, indexMax, dataSet.axisValues(dim::X).size());

    // sign if reversed
    T           sign_    = detail::sign(T(1), int(indexMax) - int(indexMin));
    std::size_t start    = std::min(indexMin, indexMax);
    std::size_t stop     = std::max(indexMin, indexMax);
    T           integral = 0;
    for (std::size_t i = start; i < stop - 1; i++) { // compute integral via triangulation
        T x0            = getIndexValue(dataSet, dim::X, i, signalIndex);
        T x1            = getIndexValue(dataSet, dim::X, i + 1, signalIndex);
        T y0            = getIndexValue(dataSet, dim::Y, i, signalIndex);
        T y1            = getIndexValue(dataSet, dim::Y, i + 1, signalIndex);
        T localIntegral = (x1 - x0) * T(0.5) * (y0 + y1);
        if (!gr::math::isfinite(localIntegral)) {
            // skip
        } else {
            integral += localIntegral;
        }
    }
    return static_cast<T>(sign_ * integral);
}

template<typename T, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] constexpr T getEdgeDetect(const DataSet<T>& dataSet, std::size_t indexMin = 0UZ, std::size_t indexMax = 0UZ, std::size_t signalIndex = 0UZ) {
    if (indexMax == 0UZ) { // renormalise default range
        indexMax = dataSet.axisValues(dim::X).size();
    }
    if (dataSet.axisCount() == 0 || dataSet.axisValues(0).empty() || indexMin >= indexMax) {
        return std::numeric_limits<TValue>::quiet_NaN();
    }
    detail::checkRangeIndex(indexMin, indexMax, dataSet.axisValues(0).size());

    T minVal = getMinimum(dataSet, indexMin, indexMax, signalIndex);
    T maxVal = getMaximum(dataSet, indexMin, indexMax, signalIndex);
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
[[nodiscard]] constexpr T getDutyCycle(const DataSet<T>& dataSet, std::size_t indexMin = 0UZ, std::size_t indexMax = 0UZ, std::size_t signalIndex = 0UZ) {
    if (indexMax == 0UZ) {
        indexMax = dataSet.axisValues(dim::X).size();
    }
    detail::checkRangeIndex(indexMin, indexMax, dataSet.axisValues(dim::X).size());

    T minVal = getMinimum(dataSet, indexMin, indexMax, signalIndex);
    T maxVal = getMaximum(dataSet, indexMin, indexMax, signalIndex);

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
[[nodiscard]] constexpr T getFrequencyEstimate(const DataSet<T>& dataSet, std::size_t indexMin = 0UZ, std::size_t indexMax = 0UZ, std::size_t signalIndex = 0UZ) {
    // getFrequencyEstimate => naive/simple approach counting edges
    if (indexMax == 0UZ) {
        indexMax = dataSet.axisValues(dim::X).size();
    }
    detail::checkRangeIndex(indexMin, indexMax, dataSet.axisValues(dim::X).size());

    T minVal = getMinimum(dataSet, indexMin, indexMax, signalIndex);
    T maxVal = getMaximum(dataSet, indexMin, indexMax, signalIndex);
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

// getLocationMaximumGaussInterpolated => returns X-value
template<typename T, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] constexpr T getLocationMaximumGaussInterpolated(const DataSet<T>& dataSet, std::size_t indexMin = 0UZ, std::size_t indexMax = 0UZ, std::size_t signalIndex = 0UZ) {
    // getFrequencyEstimate => naive/simple approach counting edges
    if (indexMax == 0UZ) {
        indexMax = dataSet.axisValues(dim::X).size();
    }
    detail::checkRangeIndex(indexMin, indexMax, dataSet.axisValues(dim::X).size());

    const int locMax_ = getLocationMaximum(dataSet, indexMin, indexMax, signalIndex);
    if (locMax_ <= static_cast<int>(indexMin + 1) || locMax_ >= static_cast<int>(indexMax - 1) || locMax_ < 0 || indexMax == indexMin) {
        return std::numeric_limits<TValue>::quiet_NaN();
    }
    const std::size_t locMax = static_cast<std::size_t>(locMax_);

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

template<typename T, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] constexpr T getZeroCrossing(const DataSet<T>& dataSet, TValue threshold, std::size_t signalIndex = 0) {
    if (dataSet.axisCount() == 0 || dataSet.axis_values.empty() || dataSet.axis_values[0].empty()) {
        return std::numeric_limits<TValue>::quiet_NaN();
    }
    std::size_t nLength = dataSet.axis_values[0].size();
    T           initial = getIndexValue(dataSet, dim::Y, 0, signalIndex);

    if (initial < threshold) { // rising edge
        for (std::size_t i = 1; i < nLength; ++i) {
            T yPrev = getIndexValue(dataSet, dim::Y, i - 1, signalIndex);
            T yCurr = getIndexValue(dataSet, dim::Y, i, signalIndex);
            if (gr::math::isfinite(yPrev) && gr::math::isfinite(yCurr) && yCurr >= threshold) {
                T xPrev = getIndexValue(dataSet, dim::X, i - 1, signalIndex);
                T xCurr = getIndexValue(dataSet, dim::X, i, signalIndex);
                return gr::dataset::detail::linearInterpolate(xPrev, xCurr, yPrev, yCurr, T(threshold));
            }
        }
    } else if (initial > threshold) { // falling edge
        for (std::size_t i = 1; i < nLength; ++i) {
            T yPrev = getIndexValue(dataSet, dim::Y, i - 1, signalIndex);
            T yCurr = getIndexValue(dataSet, dim::Y, i, signalIndex);
            if (gr::math::isfinite(yPrev) && gr::math::isfinite(yCurr) && yCurr <= threshold) {
                T xPrev = getIndexValue(dataSet, dim::X, i - 1, signalIndex);
                T xCurr = getIndexValue(dataSet, dim::X, i, signalIndex);
                return gr::dataset::detail::linearInterpolate(xPrev, xCurr, yPrev, yCurr, T(threshold));
            }
        }
    } else { // exactly at the threshold
        return getIndexValue(dataSet, dim::X, 0, signalIndex);
    }
    return std::numeric_limits<TValue>::quiet_NaN(); // No crossing found
}

} // namespace estimators

} // namespace gr::dataset

#endif // DATASETESTIMATORS_HPP
