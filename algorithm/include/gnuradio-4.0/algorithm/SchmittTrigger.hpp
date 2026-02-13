#ifndef SCHMITTTRIGGER_HPP
#define SCHMITTTRIGGER_HPP

#include <cmath>
#include <cstddef>

#include <gnuradio-4.0/HistoryBuffer.hpp>
#include <gnuradio-4.0/algorithm/filter/SavitzkyGolay.hpp>
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
 */
template<typename T, InterpolationMethod Method = InterpolationMethod::NO_INTERPOLATION, std::size_t interpolationWindow = 16UZ>
requires(std::is_arithmetic_v<T> or (UncertainValueLike<T> && std::is_arithmetic_v<meta::fundamental_base_value_type_t<T>>))
struct SchmittTrigger {
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
    std::size_t _validSamples      = 0UZ; // tracks fresh samples since construction/reset (for SG window limiting)

    // -- Savitzky-Golay members (only materialised for POLYNOMIAL_INTERPOLATION) --
    static constexpr std::size_t _sgPolyOrder = 3UZ;
    using sg_compute_t                        = std::conditional_t<std::is_floating_point_v<value_t>, value_t, float>;
    std::vector<sg_compute_t> _sgCoeffs;

    constexpr explicit SchmittTrigger(value_t threshold = 1, value_t offset = 0) noexcept(Method != InterpolationMethod::POLYNOMIAL_INTERPOLATION) : _threshold(threshold), _offset(offset), _upperThreshold(_offset + _threshold), _lowerThreshold(_offset - _threshold) {
        if constexpr (Method == InterpolationMethod::POLYNOMIAL_INTERPOLATION) {
            namespace sg      = gr::algorithm::savitzky_golay;
            const auto config = sg::Config<sg_compute_t>{.derivOrder = 0UZ, .delta = sg_compute_t{1}, .alignment = sg::Alignment::Centred, .boundaryPolicy = sg::BoundaryPolicy::Reflect};
            _sgCoeffs         = sg::computeCoefficients<sg_compute_t>(interpolationWindow, std::min(_sgPolyOrder, interpolationWindow - 1UZ), config);
        }
        for (std::size_t i = 0; i < _historyBuffer.capacity(); ++i) { // prime the history buffer
            _historyBuffer.push_front(T{});
        }
    }

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
        _validSamples      = 0UZ;
        _lastState         = false;
        _historyBuffer.reset();
        for (std::size_t i = 0; i < _historyBuffer.capacity(); ++i) { // prime the history buffer
            _historyBuffer.push_front(T{});
        }
        lastEdge       = EdgeDetection::NONE;
        lastEdgeIdx    = 1;
        lastEdgeOffset = 0.0f;
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
            return EdgeDetection::NONE;
        }

        if constexpr (Method == BASIC_LINEAR_INTERPOLATION) {
            _historyBuffer.push_front(input);
            if (_historyBuffer.size() < 2) {
                return EdgeDetection::NONE;
            }

            const T yPrev = _historyBuffer[1];
            const T yCurr = _historyBuffer[0];

            auto computeEdgePosition = [&](const EdgeType& y1, const EdgeType& y2) -> std::pair<std::int32_t, EdgeType> {
                if (y1 == y2) {
                    return {0, EdgeType{0}};
                }
                const EdgeType offset      = (EdgeType(static_cast<float>(_offset)) - y1) / (y2 - y1);
                const EdgeType crossingPos = EdgeType{-1.0f} + gr::value(offset);
                std::int32_t   intPart     = static_cast<std::int32_t>(std::round(gr::value(crossingPos)));
                EdgeType       fracPart    = crossingPos - static_cast<float>(intPart);
                return {intPart, fracPart};
            };

            if (!_lastState && input >= _upperThreshold) { // Rising edge detected
                lastEdge                 = EdgeDetection::RISING;
                auto [intPart, fracPart] = computeEdgePosition(EdgeType(static_cast<float>(gr::value(yPrev))), EdgeType(static_cast<float>(gr::value(yCurr))));
                lastEdgeIdx              = intPart;
                lastEdgeOffset           = fracPart;
                _lastState               = true;
                return EdgeDetection::RISING;
            }

            if (_lastState && input <= _lowerThreshold) { // Falling edge detected
                lastEdge                 = EdgeDetection::FALLING;
                auto [intPart, fracPart] = computeEdgePosition(EdgeType(static_cast<float>(gr::value(yPrev))), EdgeType(static_cast<float>(gr::value(yCurr))));
                lastEdgeIdx              = intPart;
                lastEdgeOffset           = fracPart;
                _lastState               = false;
                return EdgeDetection::FALLING;
            }

            lastEdgeIdx = lastEdgeIdx > 0 ? 1 : lastEdgeIdx - 1;
            return EdgeDetection::NONE;
        }

        if constexpr (Method == LINEAR_INTERPOLATION) {
            _historyBuffer.push_front(input);

            if (_historyBuffer.size() < 2) {
                return EdgeDetection::NONE;
            }

            const T    yPrev     = _historyBuffer[1];
            const T    yCurr     = _historyBuffer[0];
            const bool wasInZone = accumulatedSamples > 0;

            if (!wasInZone && !_lastState && yPrev <= _lowerThreshold && yCurr > _lowerThreshold) {
                accumulatedSamples = 1UZ;
            }

            if (!wasInZone && _lastState && yPrev >= _upperThreshold && yCurr < _upperThreshold) {
                accumulatedSamples = 1UZ;
            }

            if (wasInZone) {
                accumulatedSamples++;
            }

            if (accumulatedSamples > 0) {
                if ((!_lastState && yCurr >= _upperThreshold) || (_lastState && yCurr <= _lowerThreshold)) {
                    EdgeDetection detectedEdge = (!_lastState && yCurr >= _upperThreshold) ? EdgeDetection::RISING : EdgeDetection::FALLING;

                    // use all accumulated samples in regression
                    size_t n        = std::min(std::max(accumulatedSamples, 2UZ), _historyBuffer.size());
                    auto   crossing = findCrossingIndexLinearRegression(_historyBuffer, n, _offset);

                    if (crossing) {
                        const value_t relativeIndex = gr::value(*crossing) - static_cast<value_t>(n - 1);

                        lastEdge    = detectedEdge;
                        lastEdgeIdx = static_cast<std::int32_t>(std::round(relativeIndex));
                        if constexpr (UncertainValueLike<T>) {
                            lastEdgeOffset = T{relativeIndex - static_cast<float>(gr::value(lastEdgeIdx)), static_cast<float>(gr::uncertainty(*crossing))};
                        } else {
                            lastEdgeOffset = EdgeType{static_cast<float>(relativeIndex) - static_cast<float>(lastEdgeIdx)};
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
            _historyBuffer.push_front(input);
            _validSamples++;

            if (_historyBuffer.size() < 2) {
                return EdgeDetection::NONE;
            }

            const T    yPrev     = _historyBuffer[1];
            const T    yCurr     = _historyBuffer[0];
            const bool wasInZone = accumulatedSamples > 0;

            if (!wasInZone && !_lastState && yPrev <= _lowerThreshold && yCurr > _lowerThreshold) {
                accumulatedSamples = 1UZ;
            }

            if (!wasInZone && _lastState && yPrev >= _upperThreshold && yCurr < _upperThreshold) {
                accumulatedSamples = 1UZ;
            }

            if (wasInZone) {
                accumulatedSamples++;
            }

            if (accumulatedSamples > 0) {
                if ((!_lastState && yCurr >= _upperThreshold) || (_lastState && yCurr <= _lowerThreshold)) {
                    EdgeDetection detectedEdge = (!_lastState && yCurr >= _upperThreshold) ? EdgeDetection::RISING : EdgeDetection::FALLING;

                    // Use full history window for SG fit (limited to valid fresh samples)
                    std::size_t      n = std::min({interpolationWindow, _historyBuffer.size(), _validSamples});
                    std::optional<T> crossing;

                    if (n >= interpolationWindow && n >= _sgPolyOrder + 1) {
                        crossing = findCrossingIndexSavitzkyGolay(_historyBuffer, n, _offset);
                    }

                    if (!crossing) { // fall back to linear regression
                        n        = std::min(std::max(accumulatedSamples, 2UZ), _historyBuffer.size());
                        crossing = findCrossingIndexLinearRegression(_historyBuffer, n, _offset);
                    }

                    if (crossing) {
                        // N.B. use float to avoid unsigned integer underflow for unsigned value_t
                        const float relativeIndex = static_cast<float>(gr::value(*crossing)) - static_cast<float>(n - 1);

                        lastEdge    = detectedEdge;
                        lastEdgeIdx = static_cast<std::int32_t>(std::round(relativeIndex));
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
                    // reset accumulation if threshold zone is left without crossing the opposite threshold
                    if ((!_lastState && yCurr < _lowerThreshold) || (_lastState && yCurr > _upperThreshold)) {
                        accumulatedSamples = 0UZ;
                    }
                }
            }
        }

        return EdgeDetection::NONE;
    }

    std::optional<T> findCrossingIndexLinearRegression(const auto& samples, std::size_t nSamples, value_t offset) {
        using comp_t = std::conditional_t<std::is_floating_point_v<value_t>, value_t, float>; // temporary compute type to avoid conversion losses/errors/warnings
        if (nSamples < 2) {                                                                   // not enough samples to perform linear regression
            return std::nullopt;
        }

        const comp_t n_val = static_cast<comp_t>(nSamples);
        // sum of squares of the first (nSamples - 1) natural numbers
        const comp_t sumX2 = (n_val * (n_val - comp_t{1}) * (comp_t{2} * n_val - comp_t{1})) / comp_t{6};
        const comp_t meanX = comp_t{0.5} * (n_val - comp_t{1});

        comp_t sumY  = comp_t{0};
        comp_t sumXY = comp_t{0};
        for (std::size_t i = 0; i < nSamples; ++i) {
            const comp_t xi = static_cast<comp_t>((nSamples - 1) - i); // rationale: reversed indexing samples
            const comp_t yi = static_cast<comp_t>(gr::value(samples[i]));
            sumY += yi;
            sumXY += xi * yi;
        }

        const comp_t meanY       = sumY / n_val;
        const comp_t numerator   = sumXY - n_val * meanX * meanY;
        const comp_t denominator = sumX2 - n_val * meanX * meanX;

        if (denominator == comp_t{0}) {
            return std::nullopt;
        }

        const comp_t slope         = numerator / denominator; // slope of the regression line
        const comp_t intercept     = meanY - slope * meanX;   // intercept point of the regression line
        const comp_t crossingIndex = (static_cast<comp_t>(offset) - intercept) / slope;

        if constexpr (UncertainValueLike<T>) {  // propagation of uncertainty
            const comp_t x_uncertainty = 1e-5f; // fixed uncertainty for x-axis for common ADC clock stability

            comp_t varianceY = comp_t{0}; // variance of the mean
            for (std::size_t i = 0; i < nSamples; ++i) {
                const comp_t u = static_cast<comp_t>(gr::uncertainty(samples[i]));
                varianceY += (u * u);
            }
            varianceY /= (n_val * n_val);
            varianceY += slope * slope * x_uncertainty * x_uncertainty;

            // variance of slope (m) and intercept (b)
            const comp_t var_m = varianceY / denominator;
            const comp_t var_b = varianceY * sumX2 / (n_val * denominator);

            // covariance between slope and intercept
            const comp_t cov_mb = -varianceY * meanX / denominator;

            // partial derivatives for crossing index = (offset - b) / m
            // d(crossingIndex)/db = -1/m
            // d(crossingIndex)/dm = -(offset - b)/m^2
            const comp_t d_ci_db = -comp_t{1} / slope;
            const comp_t d_ci_dm = -(static_cast<comp_t>(offset) - intercept) / (slope * slope);

            // error propagation
            const comp_t var_ci = d_ci_db * d_ci_db * var_b + d_ci_dm * d_ci_dm * var_m + comp_t{2} * d_ci_db * d_ci_dm * cov_mb;

            return T{static_cast<value_t>(crossingIndex), static_cast<float>((var_ci < comp_t{0}) ? comp_t{0} : std::sqrt(var_ci))};
        } else { // fundamental type ->  no uncertainty
            return std::make_optional(static_cast<value_t>(crossingIndex));
        }
    }

    std::optional<T> findCrossingIndexSavitzkyGolay(const auto& samples, std::size_t nSamples, value_t offset) {
        namespace sg = gr::algorithm::savitzky_golay;
        if (nSamples < 2) {
            return std::nullopt;
        }

        // a) Extract samples oldest-first, converting to sg_compute_t
        std::array<sg_compute_t, interpolationWindow> rawData{};
        std::array<sg_compute_t, interpolationWindow> smoothedData{};

        for (std::size_t i = 0; i < nSamples; ++i) {
            rawData[i] = static_cast<sg_compute_t>(gr::value(samples[nSamples - 1 - i]));
        }

        // b) Apply SG smoothing using pre-cached coefficients
        const auto sgConfig = sg::Config<sg_compute_t>{.derivOrder = 0UZ, .delta = sg_compute_t{1}, .alignment = sg::Alignment::Centred, .boundaryPolicy = sg::BoundaryPolicy::Reflect};

        sg::apply(std::span<const sg_compute_t>(rawData.data(), nSamples), std::span<sg_compute_t>(smoothedData.data(), nSamples), std::span<const sg_compute_t>(_sgCoeffs), sgConfig);

        // c) Scan for where smoothed signal crosses offset (search from newest end)
        const sg_compute_t offsetVal = static_cast<sg_compute_t>(offset);
        std::size_t        crossIdx  = 0;
        bool               found     = false;

        for (std::size_t i = nSamples - 1; i > 0; --i) {
            if ((smoothedData[i] >= offsetVal && smoothedData[i - 1] < offsetVal) || (smoothedData[i] <= offsetVal && smoothedData[i - 1] > offsetVal)) {
                crossIdx = i;
                found    = true;
                break;
            }
        }

        if (!found) {
            return std::nullopt;
        }

        // d) Linear interpolation between bracketing smoothed samples for sub-sample precision
        const sg_compute_t y1            = smoothedData[crossIdx - 1];
        const sg_compute_t y2            = smoothedData[crossIdx];
        const sg_compute_t frac          = (y2 != y1) ? (offsetVal - y1) / (y2 - y1) : sg_compute_t{0.5};
        const sg_compute_t crossingIndex = static_cast<sg_compute_t>(crossIdx - 1) + frac;

        // e) Return crossing index (0 = oldest sample, nSamples-1 = newest)
        if constexpr (UncertainValueLike<T>) {
            // f) Propagate uncertainty through SG coefficients
            //    u_smooth[n] = sqrt(sum(c_k^2 * u_k^2))
            const std::size_t W = _sgCoeffs.size();
            const auto        D = static_cast<std::ptrdiff_t>((W - 1UZ) / 2UZ);

            auto computeSmoothedUncertainty = [&](std::size_t n) -> sg_compute_t {
                sg_compute_t varSum = sg_compute_t{0};
                for (std::size_t k = 0; k < W; ++k) {
                    const auto        idx    = static_cast<std::ptrdiff_t>(n) + static_cast<std::ptrdiff_t>(k) - D;
                    const std::size_t srcIdx = sg::detail::boundaryIndex(idx, nSamples, sgConfig);
                    // rawData[srcIdx] came from samples[nSamples - 1 - srcIdx]
                    const sg_compute_t u = static_cast<sg_compute_t>(gr::uncertainty(samples[nSamples - 1 - srcIdx]));
                    const sg_compute_t c = _sgCoeffs[k];
                    varSum += c * c * u * u;
                }
                return std::sqrt(varSum);
            };

            const sg_compute_t u1 = computeSmoothedUncertainty(crossIdx - 1);
            const sg_compute_t u2 = computeSmoothedUncertainty(crossIdx);

            // Propagate through linear interpolation: crossingIndex = (crossIdx-1) + (offset - y1)/(y2 - y1)
            const sg_compute_t dy                  = y2 - y1;
            sg_compute_t       crossingUncertainty = sg_compute_t{0};
            if (dy != sg_compute_t{0}) {
                const sg_compute_t dfdy1 = (offsetVal - y2) / (dy * dy);
                const sg_compute_t dfdy2 = (y1 - offsetVal) / (dy * dy);
                crossingUncertainty      = std::sqrt(dfdy1 * dfdy1 * u1 * u1 + dfdy2 * dfdy2 * u2 * u2);
            }

            return T{static_cast<value_t>(crossingIndex), static_cast<float>(crossingUncertainty)};
        } else {
            return std::make_optional(static_cast<value_t>(crossingIndex));
        }
    }
};

} // namespace gr::trigger

#endif // SCHMITTTRIGGER_HPP
