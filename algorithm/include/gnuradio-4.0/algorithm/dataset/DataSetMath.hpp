#ifndef DATASETMATH_HPP
#define DATASETMATH_HPP

#include <fmt/format.h>
#include <random>

#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/Message.hpp>
#include <gnuradio-4.0/algorithm/filter/FilterTool.hpp>
#include <gnuradio-4.0/meta/UncertainValue.hpp>

#include "DataSetHelper.hpp"

namespace gr::dataset {

enum class MathOp { ADD = 0, SUBTRACT, MULTIPLY, DIVIDE, SQR, SQRT, LOG10, DB, INV_DB, IDENTITY };

template<typename T>
[[nodiscard]] constexpr bool sameHorizontalBase(const DataSet<T>& ds1, const DataSet<T>& ds2) {
    if (ds1.axisCount() == 0 || ds2.axisCount() == 0) {
        return false;
    }
    const auto& x1 = ds1.axisValues(0);
    const auto& x2 = ds2.axisValues(0);
    if (x1.size() != x2.size()) {
        return false;
    }
    for (std::size_t i = 0; i < x1.size(); i++) {
        if (x1[i] != x2[i]) {
            return false;
        }
    }
    return true;
}

namespace detail {
template<typename T, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] constexpr T applyMathOperation(MathOp op, T y1, T y2) {
    switch (op) {
    case MathOp::ADD: return y1 + y2;
    case MathOp::SUBTRACT: return y1 - y2;
    case MathOp::MULTIPLY: return y1 * y2;
    case MathOp::DIVIDE: return (y2 == TValue(0)) ? std::numeric_limits<TValue>::quiet_NaN() : (y1 / y2);
    case MathOp::SQR: return (y1 + y2) * (y1 + y2);
    case MathOp::SQRT: return (y1 + y2) > TValue(0) ? gr::math::sqrt(y1 + y2) : std::numeric_limits<TValue>::quiet_NaN();
    case MathOp::LOG10: return tenLog10((y1 + y2));
    case MathOp::DB: return decibel((y1 + y2));
    case MathOp::INV_DB: return inverseDecibel<T>(y1);
    case MathOp::IDENTITY:
    default: return (y1 + y2);
    }
}
} // namespace detail

/*!
 * \brief mathFunction(DataSet, DataSet, MathOp) => merges or interpolates ds2 onto ds1’s base
 */
template<typename T, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] constexpr DataSet<T> mathFunction(const DataSet<T>& ds1, const DataSet<T>& ds2, MathOp op) {

    // Create new DataSet
    DataSet<T> ret;
    ret.axis_names        = ds1.axis_names;
    ret.axis_units        = ds1.axis_units;
    ret.axis_values       = ds1.axis_values; // or we do an interpolation base
    ret.layout            = ds1.layout;
    ret.extents           = ds1.extents;
    ret.signal_names      = {"mathOp"};
    ret.signal_quantities = {ds1.signal_quantities.empty() ? "quantity" : ds1.signal_quantities[0]};
    ret.signal_units      = {ds1.signal_units.empty() ? "" : ds1.signal_units[0]};
    ret.signal_ranges     = {{T(0.0), T(1.0)}};   // placeholder
    ret.meta_information  = ds1.meta_information; // or merge
    ret.timing_events     = ds1.timing_events;

    std::size_t dataCount = 0U;
    if (!ret.axis_values.empty()) {
        dataCount = ret.axis_values[0].size();
    }
    ret.signal_values.resize(dataCount);

    bool needsInterpolation = !sameHorizontalBase(ds1, ds2);

    for (std::size_t i = 0; i < dataCount; i++) {
        TValue x             = gr::value(getIndexValue(ds1, dim::X, i));
        T      Y1            = getIndexValue(ds1, dim::Y, i);
        T      Y2            = needsInterpolation ? getValue(ds2, dim::Y, x) : getIndexValue(ds2, dim::Y, i);
        T      result        = detail::applyMathOperation<T>(op, Y1, Y2);
        ret.signal_values[i] = static_cast<T>(result);
    }
    return ret;
}

template<typename T, std::convertible_to<T> U, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] constexpr DataSet<T> mathFunction(const DataSet<T>& ds, U value, MathOp op) {

    DataSet<T> ret       = ds; // copy
    auto       dataCount = ds.axis_values.empty() ? 0UZ : ds.axis_values[0].size();
    for (std::size_t i = 0; i < dataCount; i++) {
        T Y1     = getIndexValue(ds, dim::Y, i);
        T result = 0;
        switch (op) {
        case MathOp::ADD: result = Y1 + value; break;
        case MathOp::SUBTRACT: result = Y1 - value; break;
        case MathOp::MULTIPLY: result = Y1 * value; break;
        case MathOp::DIVIDE: result = (value == TValue(0)) ? std::numeric_limits<TValue>::quiet_NaN() : (Y1 / value); break;
        case MathOp::SQR: result = (Y1 + value) * (Y1 + value); break;
        case MathOp::SQRT: result = (Y1 + value) > TValue(0) ? gr::math::sqrt(Y1 + value) : std::numeric_limits<TValue>::quiet_NaN(); break;
        case MathOp::LOG10: result = tenLog10(Y1 + value); break;
        case MathOp::DB: result = decibel(Y1 + value); break;
        case MathOp::INV_DB: result = inverseDecibel(Y1); break;
        case MathOp::IDENTITY:
        default: result = Y1; break;
        }
        ret.signal_values[i] = static_cast<T>(result);
    }
    return ret;
}

template<typename T, std::convertible_to<T> U>
[[nodiscard]] constexpr DataSet<T> addFunction(const DataSet<T>& ds, U value) {
    return mathFunction(ds, value, MathOp::ADD);
}
template<typename T>
[[nodiscard]] constexpr DataSet<T> addFunction(const DataSet<T>& ds1, const DataSet<T>& ds2) {
    return mathFunction(ds1, ds2, MathOp::ADD);
}

// WIP complete other convenience and missing math functions

template<typename T>
std::vector<T> computeDerivative(const DataSet<T>& ds, std::size_t signalIndex = 0UZ) {
    auto signal = ds.signalValues(signalIndex);
    if (signal.size() < 2) {
        throw gr::exception("signal must contain at least two samples to compute derivative.");
    }

    std::vector<T> derivative;
    derivative.reserve(signal.size() - 1UZ);
    for (std::size_t i = 1UZ; i < signal.size(); ++i) {
        derivative.push_back(signal[i] - signal[i - 1UZ]);
    }
    return derivative;
}

template<ProcessMode mode = ProcessMode::Copy, typename T, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
DataSet<T> addNoise(const DataSet<T>& ds, TValue noiseLevel, std::size_t signalIndex = 0UZ, std::uint64_t seed = 0U) {
    if (noiseLevel < TValue(0)) {
        throw gr::exception(fmt::format("noiseLevel {} must be a positive number.", noiseLevel));
    }

    DataSet<T> noisy;
    if constexpr (mode == ProcessMode::Copy) { // copy
        noisy = ds;
    } else { // or move (in-place)
        noisy = std::move(ds);
    }
    const auto         signal      = ds.signalValues(signalIndex);
    auto               noisySignal = noisy.signalValues(signalIndex);
    std::random_device rd;
    std::mt19937_64    rng(seed == 0 ? rd() : seed);
    using Distribution = std::conditional_t<std::is_integral_v<TValue>, std::uniform_int_distribution<TValue>, std::uniform_real_distribution<TValue>>;

    Distribution dist(-noiseLevel, +noiseLevel);

    for (std::size_t i = 0UZ; i < signal.size(); ++i) {
        noisySignal[i] += dist(rng);
    }
    return noisy;
}

namespace filter {

template<ProcessMode mode = ProcessMode::Copy, typename T, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
DataSet<T> applyMovingAverage(const DataSet<T>& ds, std::size_t windowSize, std::size_t signalIndex = 0UZ) {
    if (windowSize == 0 || !(windowSize & 1)) {
        throw gr::exception("windowSize must be a positive odd number.");
    }

    DataSet<T> smoothed;
    if constexpr (mode == ProcessMode::Copy) { // copy
        smoothed = ds;
    } else { // or move (in-place)
        smoothed = std::move(ds);
    }
    const auto signal         = ds.signalValues(signalIndex);
    auto       smoothedSignal = smoothed.signalValues(signalIndex);

    const std::size_t halfWindow = windowSize / 2UZ;
    for (std::size_t i = 0UZ; i < signal.size(); ++i) {
        std::size_t start = (i >= halfWindow) ? i - halfWindow : 0UZ;
        std::size_t end   = std::min(i + halfWindow + 1UZ, signal.size());

        T sum             = std::accumulate(signal.begin() + static_cast<std::ptrdiff_t>(start), signal.begin() + static_cast<std::ptrdiff_t>(end), T(0));
        smoothedSignal[i] = sum / TValue(end - start);
    }
    return smoothed;
}

template<ProcessMode mode = ProcessMode::Copy, typename T, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
constexpr DataSet<T> applyMedian(const DataSet<T>& ds, std::size_t windowSize, std::size_t signalIndex = 0UZ, std::source_location location = std::source_location::current()) {
    if (windowSize == 0) {
        throw gr::exception(fmt::format("windowSize: {} must be a positive number.", windowSize), location);
    }

    DataSet<T> filtered;
    if constexpr (mode == ProcessMode::Copy) {
        filtered = ds;
    } else {
        filtered = std::move(ds);
    }

    const std::vector<T> signal{filtered.signalValues(signalIndex).begin(), filtered.signalValues(signalIndex).end()};
    auto                 filteredSignal = filtered.signalValues(signalIndex);
    const std::size_t    N              = filteredSignal.size();

    std::vector<T>    medianWindow(windowSize); // temporary mutable copy for in-place partitioning
    const std::size_t halfWindow = windowSize / 2UZ;
    for (std::size_t i = 0UZ; i < N; ++i) {
        const auto        start = static_cast<std::ptrdiff_t>(i > halfWindow ? i - halfWindow : 0UZ);
        const auto        end   = static_cast<std::ptrdiff_t>(std::min(i + halfWindow + 1UZ, N));
        const std::size_t size  = static_cast<std::size_t>(end - start);

        std::copy(signal.begin() + start, signal.begin() + end, medianWindow.begin());

        auto medianWindowView = std::span(medianWindow.data(), size);
        auto midIter          = medianWindowView.begin() + static_cast<std::ptrdiff_t>(size / 2UZ);
        std::ranges::nth_element(medianWindowView, midIter);

        if ((size & 1UZ) == 0UZ) { // even-sized window -> take average around mid-point
            auto midPrev      = std::ranges::max_element(medianWindowView.begin(), midIter);
            filteredSignal[i] = T(0.5) * (*midPrev + *midIter);
        } else { // odd-sized window -> use exact mid-point
            filteredSignal[i] = *midIter;
        }
    }

    return filtered;
}

template<ProcessMode mode = ProcessMode::Copy, typename T, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
constexpr DataSet<T> applyRms(const DataSet<T>& ds, std::size_t windowSize, std::size_t signalIndex = 0UZ, std::source_location location = std::source_location::current()) {
    if (windowSize == 0UZ) {
        throw gr::exception(fmt::format("windowSize: {} must be a positive number.", windowSize), location);
    }

    DataSet<T> filtered;
    if constexpr (mode == ProcessMode::Copy) {
        filtered = ds;
    } else {
        filtered = std::move(ds);
    }

    const std::vector<T> signal{filtered.signalValues(signalIndex).begin(), filtered.signalValues(signalIndex).end()};
    auto                 filteredSignal = filtered.signalValues(signalIndex);
    const std::size_t    N              = filteredSignal.size();

    for (std::size_t i = 0; i < N; ++i) {
        std::size_t start = (i > (windowSize / 2UZ)) ? i - (windowSize / 2UZ) : 0UZ;
        std::size_t end   = std::min(i + (windowSize / 2UZ) + 1UZ, N);
        std::size_t size  = (end - start);

        T sum  = T(0);
        T sum2 = T(0);
        for (std::size_t j = start; j < end; ++j) {
            sum += signal[j];
            sum2 += signal[j] * signal[j];
        }
        if (size > 1UZ) {
            T mean            = sum / gr::cast<TValue>(size);
            filteredSignal[i] = gr::math::sqrt(gr::math::abs(sum2 / gr::cast<TValue>(size) - mean * mean));
        } else {
            filteredSignal[i] = gr::cast<TValue>(0);
        }
    }

    return filtered;
}

template<ProcessMode mode = ProcessMode::Copy, typename T, typename TValue = gr::meta::fundamental_base_value_type_t<T>>
constexpr DataSet<T> applyPeakToPeak(const DataSet<T>& ds, std::size_t windowSize, std::size_t signalIndex = 0UZ, std::source_location location = std::source_location::current()) {
    if (windowSize == 0UZ) {
        throw gr::exception(fmt::format("windowSize: {} must be a positive number.", windowSize), location);
    }

    DataSet<T> filtered;
    if constexpr (mode == ProcessMode::Copy) {
        filtered = ds;
    } else {
        filtered = std::move(ds);
    }

    const std::vector<T> signal{filtered.signalValues(signalIndex).begin(), filtered.signalValues(signalIndex).end()};
    auto                 filteredSignal = filtered.signalValues(signalIndex);
    const std::size_t    N              = filteredSignal.size();

    const std::size_t halfWindow = windowSize / 2UZ;

    for (std::size_t i = 0; i < N; ++i) {
        std::size_t start = (i > halfWindow) ? i - halfWindow : 0UZ;
        std::size_t end   = std::min(i + halfWindow + 1UZ, N);

        auto minVal = std::numeric_limits<T>::max();
        auto maxVal = std::numeric_limits<T>::lowest();
        for (std::size_t j = start; j < end; ++j) {
            if (signal[j] < minVal) {
                minVal = signal[j];
            }
            if (signal[j] > maxVal) {
                maxVal = signal[j];
            }
        }
        filteredSignal[i] = maxVal - minVal;
    }

    return filtered;
}

template<ProcessMode mode = ProcessMode::Copy, bool symmetric = false, DataSetLike D, typename T = typename std::remove_cvref_t<D>::value_type, typename U>
DataSet<T> applyFilter(D&& dataSet, const gr::filter::FilterCoefficients<U>& coeffs, std::size_t signalIndex = max_size) {
    using TValue                  = gr::meta::fundamental_base_value_type_t<T>;
    constexpr bool isConstDataSet = std::is_const_v<std::remove_reference_t<D>>;

    static_assert(!(isConstDataSet && mode == ProcessMode::InPlace), "cannot perform in-place computation on const DataSet<T>");

    DataSet<T> smoothed;
    if constexpr (mode == ProcessMode::Copy) { // copy
        smoothed = dataSet;
    } else { // or move (in-place)
        smoothed = std::move(dataSet);
    }

    auto forwardPass = [&](std::span<T> signal) -> void {
        auto filter = gr::filter::ErrorPropagatingFilter<T>(coeffs);
        if constexpr (UncertainValueLike<TValue>) {
            std::ranges::transform(signal, signal.begin(), [&](auto& val) { return filter.processOne(val); });
        } else {
            std::ranges::transform(signal, signal.begin(), [&](auto& val) { return gr::value(filter.processOne(val)); });
        }
    };

    auto backwardPass = [&](std::span<T> signal) -> void {
        auto filter = gr::filter::ErrorPropagatingFilter<T>(coeffs);
        if constexpr (UncertainValueLike<TValue>) {
            std::ranges::transform(signal | std::views::reverse, signal.rbegin(), [&](auto& val) { return filter.processOne(val); });
        } else {
            std::ranges::transform(signal | std::views::reverse, signal.rbegin(), [&](auto& val) { return gr::value(filter.processOne(val)); });
        }
    };

    auto processSignal = [&](std::size_t sigIndex) {
        auto signal = smoothed.signalValues(sigIndex);
        if constexpr (!symmetric) {
            forwardPass(signal); // forward pass only
        } else {
            std::vector<T> copy(signal.begin(), signal.end());
            forwardPass(signal); // forward pass
            backwardPass(copy);  // reverse pass
            // combine both passes
            for (std::size_t i = 0UZ; i < signal.size(); i++) {
                signal[i] = T{0.5} * (signal[i] + copy[i]);
            }
        }
    };

    if (signalIndex != max_size) {
        processSignal(signalIndex);
    } else {
        for (std::size_t dsIndex = 0UZ; dsIndex < smoothed.size(); dsIndex++) {
            processSignal(dsIndex);
        }
    }

    return smoothed;
}

template<ProcessMode Mode = ProcessMode::Copy, typename T, typename... TFilterCoefficients>
DataSet<T> applySymmetricFilter(const DataSet<T>& ds, TFilterCoefficients&&... coeffs) {
    return applyFilter<Mode, true>(ds, 0UZ, std::forward<TFilterCoefficients>(coeffs)...);
}

} // namespace filter

} // namespace gr::dataset

#endif // DATASETMATH_HPP
