#ifndef SCHMITTTRIGGER_HPP
#define SCHMITTTRIGGER_HPP

#include <cmath>
#include <cstddef>

#include <gnuradio-4.0/HistoryBuffer.hpp>
#include <gnuradio-4.0/meta/UncertainValue.hpp>

namespace gr::trigger {

/**
 * @brief Enumeration for the interpolation methods used for sub-sample
 * precision.
 */
enum class InterpolationMethod {
    NO_INTERPOLATION           = 0,
    BASIC_LINEAR_INTERPOLATION = 1, /// basic linear interpolation
    LINEAR_INTERPOLATION       = 2, /// interpolation via linear regression over multiple samples
    POLYNOMIAL_INTERPOLATION   = 3  /// Savitzky–Golay filter-based methods
};

enum class EdgeDetection { NONE = 0, RISING = 1, FALLING = 2 };

/**
 * @brief A real-time capable digital Schmitt trigger implementation.
 *
 * @see https://en.wikipedia.org/wiki/Schmitt_trigger
 * This class processes input samples and detects rising and falling edges based on specified thresholds.
 * It supports sub-sample precision through various interpolation methods:
 *  * NO_INTERPOLATION: nomen est omen
 *  * BASIC_LINEAR_INTERPOLATION: basic linear interpolation based on the new and previous sample
 *  * LINEAR_INTERPOLATION: interpolation via linear regression over the samples between when
 *    the lower and upper threshold has been crossed and vice versa
 *  * POLYNOMIAL_INTERPOLATION: Savitzky–Golay filter-based methods
 *    https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
 *     (TODO: WIP needs Tensor<T> and SVD implementation)
 */
template<typename T, InterpolationMethod Method = InterpolationMethod::NO_INTERPOLATION, std::size_t interpolationWindow = 16UZ>
requires(std::is_arithmetic_v<T> or (UncertainValueLike<T> && std::is_arithmetic_v<meta::fundamental_base_value_type_t<T>>))
struct SchmittTrigger {
    static_assert(Method != InterpolationMethod::POLYNOMIAL_INTERPOLATION, "POLYNOMIAL_INTERPOLATION not implemented yet");
    using value_t  = meta::fundamental_base_value_type_t<T>;
    using EdgeType = UncertainValue<float>; // always float precision similar to `sample_rate` definition

    value_t _threshold{1};
    value_t _offset{0};
    value_t _upperThreshold;
    value_t _lowerThreshold;

    bool _lastState{false}; // true if above upper threshold

    HistoryBuffer<T, interpolationWindow> _historyBuffer;

    // edge timing variables
    EdgeDetection lastEdge = EdgeDetection::NONE;
    std::int32_t  lastEdgeIdx{1};    /// relative index [-std::int32_t|_MAX,1[ of the last detected edge w.r.t. current sample
    EdgeType      lastEdgeOffset{0}; /// relative sub-sample offset [0, 1[ of the last detected edge

    std::size_t accumulatedSamples = 0UZ; // number of samples accumulated once threshold has been entered

    constexpr explicit SchmittTrigger(value_t threshold = 1, value_t offset = 0) noexcept : _threshold(threshold), _offset(offset), _upperThreshold(_offset + _threshold), _lowerThreshold(_offset - _threshold) {}

    constexpr void setThreshold(value_t threshold) {
        _threshold      = threshold;
        _upperThreshold = _offset + _threshold;
        _lowerThreshold = _offset - _threshold;
    }

    constexpr void setOffset(value_t offset) {
        _offset         = offset;
        _upperThreshold = _offset + _threshold;
        _lowerThreshold = _offset - _threshold;
    }

    constexpr void reset() {
        accumulatedSamples = 0UZ;
        lastEdge           = EdgeDetection::NONE;
        lastEdgeIdx        = 1;
        lastEdgeOffset     = 0.0f;
    }

    EdgeDetection processOne(T input) {
        using enum InterpolationMethod;

        if constexpr (Method == NO_INTERPOLATION) {
            if (!_lastState && input >= _upperThreshold) { // rising edge detected
                lastEdgeIdx = 0;
                lastEdge    = EdgeDetection::RISING;
                _lastState  = true;
                return EdgeDetection::RISING;
            }

            if (_lastState && input <= _lowerThreshold) { // falling edge detected
                lastEdgeIdx = 0;
                lastEdge    = EdgeDetection::FALLING;
                _lastState  = false;
                return EdgeDetection::FALLING;
            }
            lastEdgeIdx = lastEdgeIdx > 0 ? 1 : lastEdgeIdx - 1;
            lastEdgeIdx = lastEdgeIdx > 0 ? 1 : lastEdgeIdx - 1;
            return EdgeDetection::NONE;
        }

        if constexpr (Method == BASIC_LINEAR_INTERPOLATION) {
            _historyBuffer.push_back(input);
            if (_historyBuffer.size() < 2) {
                return EdgeDetection::NONE;
            }

            const T yPrev = _historyBuffer[1];
            const T yCurr = _historyBuffer[0];

            auto computeEdgePosition = [&](const EdgeType& y1, const EdgeType& y2) -> std::pair<std::int32_t, EdgeType> {
                if (y1 == y2) {
                    return {static_cast<std::int32_t>(0), static_cast<EdgeType>(0)};
                }
                const EdgeType offset   = (EdgeType(_offset) - y1) / (y2 - y1);
                std::int32_t   intPart  = static_cast<std::int32_t>(std::floor(gr::value(offset)));
                EdgeType       fracPart = offset - static_cast<float>(intPart);
                return {intPart, fracPart};
            };

            if (!_lastState && input >= _upperThreshold) { // Rising edge detected
                lastEdge                 = EdgeDetection::RISING;
                auto [intPart, fracPart] = computeEdgePosition(yPrev, yCurr);
                lastEdgeIdx              = -1 + intPart; // edge occurred intPart samples before the current sample
                lastEdgeOffset           = fracPart;     // fractional part in [0, 1)
                _lastState               = true;
                return EdgeDetection::RISING;
            }

            if (_lastState && input <= _lowerThreshold) { // Falling edge detected
                lastEdge                 = EdgeDetection::FALLING;
                auto [intPart, fracPart] = computeEdgePosition(yPrev, yCurr);
                lastEdgeIdx              = -1 + intPart; // edge occurred intPart samples before the current sample
                lastEdgeOffset           = fracPart;     // fractional part in [0, 1)
                _lastState               = false;
                return EdgeDetection::FALLING;
            }

            lastEdgeIdx = lastEdgeIdx > 0 ? 1 : lastEdgeIdx - 1;
            lastEdgeIdx = lastEdgeIdx > 0 ? 1 : lastEdgeIdx - 1;
            return EdgeDetection::NONE;
        }

        if constexpr (Method == LINEAR_INTERPOLATION) {
            _historyBuffer.push_back(input);

            if (_historyBuffer.size() < 2) {
                return EdgeDetection::NONE;
            }

            const T yPrev = _historyBuffer[1];
            const T yCurr = _historyBuffer[0];

            if (!_lastState && yPrev <= _lowerThreshold && yCurr > _lowerThreshold) { // detected rising edge -> start accumulating samples
                accumulatedSamples = 1UZ;
            }

            if (_lastState && yPrev >= _upperThreshold && yCurr < _upperThreshold) { // detected falling edge -> start accumulating samples
                accumulatedSamples = 1UZ;
            }

            if (accumulatedSamples > 0) {
                accumulatedSamples++;

                if ((!_lastState && yCurr >= _upperThreshold) || (_lastState && yCurr <= _lowerThreshold)) { // opposite threshold has been crossed
                    EdgeDetection detectedEdge = (!_lastState && yCurr >= _upperThreshold) ? EdgeDetection::RISING : EdgeDetection::FALLING;

                    // use all accumulated samples in regression
                    size_t n        = std::min(accumulatedSamples, _historyBuffer.size());
                    auto   crossing = findCrossingIndexLinearRegression(_historyBuffer, n, _offset);

                    if (crossing) {
                        const value_t relativeIndex = gr::value(*crossing) - static_cast<value_t>(n - 1);

                        lastEdge    = detectedEdge;
                        lastEdgeIdx = static_cast<std::int32_t>(std::floor(relativeIndex));
                        if constexpr (UncertainValueLike<T>) {
                            lastEdgeOffset = T{relativeIndex - static_cast<float>(gr::value(lastEdgeIdx)), static_cast<float>(gr::uncertainty(*crossing))};
                        } else {
                            lastEdgeOffset = EdgeType{relativeIndex - static_cast<float>(lastEdgeIdx)};
                        }

                        // update state and reset accumulation
                        _lastState         = !_lastState;
                        accumulatedSamples = 0UZ;
                        return detectedEdge;
                    }
                } else {
                    // reset accumulation if threshold zone is left without crossing the opposite threshold,
                    if ((!_lastState && yCurr < _lowerThreshold) || (_lastState && yCurr > _upperThreshold)) {
                        accumulatedSamples = 0UZ;
                    }
                }
            }

            return EdgeDetection::NONE;
        }

        if constexpr (Method == POLYNOMIAL_INTERPOLATION) {
            static_assert(gr::meta::always_false<T>, "POLYNOMIAL_INTERPOLATION not implemented yet");
        }

        return EdgeDetection::NONE;
    }

    std::optional<T> findCrossingIndexLinearRegression(const auto& samples, std::size_t nSamples, value_t offset) {
        if (nSamples < 2) { // not enough samples to perform linear regression
            return std::nullopt;
        }

        const auto n_val = static_cast<value_t>(nSamples);
        // sum of squares of the first (nSamples - 1) natural numbers
        const value_t sumX2 = (n_val * (n_val - value_t(1)) * (value_t(2) * n_val - value_t(1))) / value_t(6);
        const auto    meanX = value_t(0.5f) * (n_val - value_t(1));

        value_t sumY  = 0;
        value_t sumXY = 0;
        for (std::size_t i = 0; i < nSamples; ++i) {
            const auto xi = static_cast<value_t>(nSamples - 1 - i); // rationale: reversed indexing samples
            const auto yi = gr::value(samples[i]);
            sumY += yi;
            sumXY += xi * yi;
        }

        const value_t meanY       = sumY / n_val;
        const value_t numerator   = sumXY - n_val * meanX * meanY;
        const value_t denominator = sumX2 - n_val * meanX * meanX;

        if (denominator == 0.0f) {
            return std::nullopt;
        }

        const value_t slope         = numerator / denominator; // slope of the regression line
        const value_t intercept     = meanY - slope * meanX;   // intercept point of the regression line
        const value_t crossingIndex = (offset - intercept) / slope;

        if constexpr (UncertainValueLike<T>) {   // propagation of uncertainty
            const value_t x_uncertainty = 1e-5f; // fixed uncertainty for x-axis for common ADC clock stability

            value_t varianceY = 0.0f; // variance of the mean
            for (std::size_t i = 0; i < nSamples; ++i) {
                value_t u = gr::uncertainty(samples[i]);
                varianceY += u * u;
            }
            varianceY /= (n_val * n_val);
            varianceY += slope * slope * x_uncertainty * x_uncertainty;

            // variance of slope (m) and intercept (b)
            const value_t var_m = varianceY / denominator;
            const value_t var_b = varianceY * sumX2 / (n_val * denominator);

            // covariance between slope and intercept
            value_t cov_mb = -varianceY * meanX / denominator;

            // partial derivatives for crossing index = (offset - b) / m
            // d(crossingIndex)/db = -1/m
            // d(crossingIndex)/dm = -(offset - b)/m^2
            value_t d_ci_db = -value_t(1) / slope;
            value_t d_ci_dm = -(gr::value(offset) - intercept) / (slope * slope);

            // error propagation
            value_t var_ci = (d_ci_db * d_ci_db * var_b) + (d_ci_dm * d_ci_dm * var_m) + (value_t(2) * d_ci_db * d_ci_dm * cov_mb);

            return T{crossingIndex, var_ci < 0 ? value_t(0) : std::sqrt(var_ci) /* uncertainty */};
        } else { // fundamental type ->  no uncertainty
            return std::make_optional(crossingIndex);
        }
    }
};

} // namespace gr::trigger

#endif // SCHMITTTRIGGER_HPP
