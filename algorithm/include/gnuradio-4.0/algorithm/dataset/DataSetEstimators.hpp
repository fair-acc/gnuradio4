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

template<typename T>
struct StepStartDetectionResult {
    std::size_t index;        ///< Index where the step starts.
    T           initialValue; ///< Initial value of the signal.
    T           minValue;     ///< Minimum value of the signal.
    T           maxValue;     ///< Maximum value of the signal.
    bool        isRising;     ///< Indicates if the detected step is rising (true) or falling (false).
};

template<typename T>
std::optional<StepStartDetectionResult<T>> detectStepStart(const DataSet<T>& ds, std::size_t signalIndex = 0UZ) {
    // Verify the DataSet
    if (!gr::dataset::verify<true>(ds)) {
        throw gr::exception("Invalid DataSet for step/pulse start detection.");
    }

    auto signal = ds.signalValues(signalIndex);
    if (signal.empty()) {
        throw gr::exception("Signal is empty.");
    }

    T initial = signal.front();
    T max_val = estimators::getMaximum(ds, signalIndex);
    T min_val = estimators::getMinimum(ds, signalIndex);

    bool isRising  = initial < max_val;
    bool isFalling = initial > min_val;

    if (!isRising && !isFalling) {
        return std::nullopt; // No clear step detected
    }

    // Determine target value based on edge type
    T target = isRising ? max_val : min_val;

    // Find the threshold crossing (50% of change)
    T halfChange = isRising ? initial + T(0.5f) * (max_val - initial) : initial - T(0.5f) * (initial - min_val);

    // Utilize existing getZeroCrossing to find the trigger time
    T triggerTime;
    try {
        triggerTime = estimators::getZeroCrossing(ds, halfChange, signalIndex);
    } catch (...) {
        return std::nullopt; // Failed to compute trigger time
    }

    // Find the closest index to the triggerTime
    const auto& xAxis = ds.axisValues(0);
    auto        it    = std::lower_bound(xAxis.begin(), xAxis.end(), triggerTime);
    if (it == xAxis.end()) {
        return std::nullopt; // No step detected
    }
    std::size_t index = std::distance(xAxis.begin(), it);

    // Ensure the index is within bounds
    if (index >= signal.size()) {
        return std::nullopt;
    }

    // Populate the result struct
    StepStartDetectionResult<T> result;
    result.index        = index;
    result.initialValue = initial;
    result.minValue     = min_val;
    result.maxValue     = max_val;
    result.isRising     = isRising;

    return result;
}

template<typename T>
struct StepPulseResponseMetrics {
    bool isPulse       = false; ///< Indicates if the response is a pulse.
    T    triggerTime   = 0.0;   ///< Time at which the trigger occurs (50% crossing).
    T    V1            = 0.0;   ///< Steady-state level before the step.
    T    V2            = 0.0;   ///< Steady-state level after the step.
    T    V3            = 0.0;   ///< Steady-state level after the pulse (if pulse).
    T    riseTime      = 0.0;   ///< Rise time (e.g., 10% -> 90% of V2 - V1).
    T    peakAmplitude = 0.0;   ///< Peak amplitude relative to V1.
    T    peakTime      = 0.0;   ///< Time at which the peak occurs.
    T    overshoot     = 0.0;   ///< Overshoot percentage.
    T    settlingTime  = 0.0;   ///< Settling time within 2% of V2.
};

template <typename T,  typename TValue = gr::meta::fundamental_base_value_type_t<T>>
StepPulseResponseMetrics<T> analyzeStepPulseResponse(const DataSet<T>& ds, std::size_t signalIndex = 0UZ) {
    StepPulseResponseMetrics<T> metrics;

    // Step 1: Smooth the signal to reduce noise
    DataSet<T> smoothed = filter::applyMovingAverage(ds, 5, signalIndex); // Window size 5 as an example
    std::span<const T> smoothedAxisValues = smoothed.axisValues(dim::X);
    std::span<const T> smoothedsignalValues = smoothed.signalValues(signalIndex);

    // Step 2: Compute the derivative
    auto derivative = computeDerivative(smoothed, signalIndex);

    // Step 3: Detect step start
    auto detectionResult = estimators::detectStepStart(smoothed, signalIndex);

    if (!detectionResult.has_value()) {
        throw gr::exception("No step detected in the signal.");
    }

    const auto& result = detectionResult.value();

    metrics.isPulse = false; // Default

    // Step 4: Compute V1 and V2 using existing getMedian
    // V1: median of pre-step region
    std::size_t V1WindowSize = smoothedsignalValues.size() / 10; // Example: first 10% as pre-step
    if (V1WindowSize == 0) {
        V1WindowSize = 10; // Minimum pre-step samples
    }
    std::size_t V1Start = (result.index >= V1WindowSize) ? result.index - V1WindowSize : 0;
    std::size_t V1End = result.index;
    metrics.V1 = estimators::getMedian(smoothed, V1Start, V1End, signalIndex);

    // V2: median of post-step region
    std::size_t V2Start = result.index;
    std::size_t V2End = V2Start + V1WindowSize;
    if (V2End > smoothedsignalValues.size()) {
        V2End = smoothedsignalValues.size();
    }
    metrics.V2 = estimators::getMedian(smoothed, V2Start, V2End, signalIndex);

    // Step 5: Determine if it's a pulse by checking for a falling or rising edge after the step
    // Compare initial value with max/min to determine edge direction
    bool isRising = result.isRising;

    if (isRising) {
        // Look for a significant drop after V2 indicating a pulse
        T postStepThreshold = metrics.V2 * TValue(0.5); // Example threshold for pulse detection
        bool pulseDetected = false;
        std::size_t pulseIndex = smoothedsignalValues.size(); // Initialize to size (no pulse)
        for (std::size_t i = V2End; i < smoothedsignalValues.size(); ++i) {
            if (smoothedsignalValues[i] < postStepThreshold) {
                pulseDetected = true;
                pulseIndex = i;
                break;
            }
        }
        if (pulseDetected) {
            metrics.isPulse = true;
            // Compute V3 as median of post-pulse region
            std::size_t V3Start = pulseIndex;
            std::size_t V3End = V3Start + V1WindowSize;
            if (V3End > smoothedsignalValues.size()) {
                V3End = smoothedsignalValues.size();
            }
            metrics.V3 = estimators::getMedian(smoothed, V3Start, V3End, signalIndex);
        }
    } else {
        // Look for a significant rise after V2 indicating a pulse
        T postStepThreshold = metrics.V2 * TValue(1.5); // Example threshold for pulse detection
        bool pulseDetected = false;
        std::size_t pulseIndex = smoothedsignalValues.size(); // Initialize to size (no pulse)
        for (std::size_t i = V2End; i < smoothedsignalValues.size(); ++i) {
            if (smoothedsignalValues[i] > postStepThreshold) {
                pulseDetected = true;
                pulseIndex = i;
                break;
            }
        }
        if (pulseDetected) {
            metrics.isPulse = true;
            // Compute V3 as median of post-pulse region
            std::size_t V3Start = pulseIndex;
            std::size_t V3End = V3Start + V1WindowSize;
            if (V3End > smoothedsignalValues.size()) {
                V3End = smoothedsignalValues.size();
            }
            metrics.V3 = estimators::getMedian(smoothed, V3Start, V3End, signalIndex);
        }
    }

    // Step 6: Compute trigger time (already obtained from detection)
    metrics.triggerTime = result.index < smoothedsignalValues.size() ? smoothedAxisValues[result.index] : 0.0;

    // Step 7: Compute rise time (10% -> 90%)
    T riseStart = isRising ? metrics.V1 + TValue(0.1) * (metrics.V2 - metrics.V1) : metrics.V1 - TValue(0.1) * (metrics.V1 - metrics.V2);
    T riseEnd = isRising ? metrics.V1 + TValue(0.9) * (metrics.V2 - metrics.V1) : metrics.V1 - TValue(0.9) * (metrics.V1 - metrics.V2);
    // Compute rise start time
    T riseStartTime;
    try {
        riseStartTime = estimators::getZeroCrossing(smoothed, riseStart, signalIndex);
    } catch (...) {
        riseStartTime = TValue(0); // Default or handle as needed
    }
    // Compute rise end time
    T riseEndTime;
    try {
        riseEndTime = estimators::getZeroCrossing(smoothed, riseEnd, signalIndex);
    } catch (...) {
        riseEndTime = TValue(0); // Default or handle as needed
    }
    metrics.riseTime = riseEndTime - riseStartTime;

    // Step 8: Compute peak amplitude and peak time
    // Find maximum value and its index
    auto maxIter = std::max_element(smoothedsignalValues.begin(), smoothedsignalValues.end(),
        [&](const T& a, const T& b) -> bool { return gr::value(a) < gr::value(b); });
    if (maxIter == smoothedsignalValues.end()) {
        throw gr::exception("Failed to find peak amplitude.");
    }
    std::size_t maxIndex = std::distance(smoothedsignalValues.begin(), maxIter);
    metrics.peakAmplitude = *maxIter - metrics.V1;
    metrics.peakTime = smoothedAxisValues[maxIndex];

    // Step 9: Compute overshoot
    metrics.overshoot = ((metrics.peakAmplitude) / (metrics.V2 - metrics.V1)) * TValue(100);

    // Step 10: Compute settling time (2% around V2)
    T settlingLower = metrics.V2 - TValue(0.02) * gr::math::abs(metrics.V2 - metrics.V1);
    T settlingUpper = metrics.V2 + TValue(0.02) * gr::math::abs(metrics.V2 - metrics.V1);
    std::size_t settlingIndex = smoothedsignalValues.size();
    for (std::size_t i = result.index; i < smoothedsignalValues.size(); ++i) {
        T val = smoothedsignalValues[i];
        if (val >= settlingLower && val <= settlingUpper) {
            // Check if all subsequent points are within the settling band
            bool settled = true;
            for (std::size_t j = i; j < smoothedsignalValues.size(); ++j) {
                T current = smoothedsignalValues[j];
                if (!(current >= settlingLower && current <= settlingUpper)) {
                    settled = false;
                    break;
                }
            }
            if (settled) {
                settlingIndex = i;
                metrics.settlingTime = smoothedAxisValues[i];
                break;
            }
        }
    }

    return metrics;
}

} // namespace estimators

} // namespace gr::dataset

#endif // DATASETESTIMATORS_HPP
