#ifndef SCHMITTTRIGGER_HPP
#define SCHMITTTRIGGER_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <optional>
#include <span>
#include <type_traits>

#include <gnuradio-4.0/HistoryBuffer.hpp>
#include <gnuradio-4.0/algorithm/filter/SavitzkyGolay.hpp>
#include <gnuradio-4.0/meta/UncertainValue.hpp>

namespace gr::trigger {

/** how the time of a crossing is estimated: between the two samples either side of the boundary crossed (the default), or retrospectively where a fit over the samples inside the band crosses the nominal level */
enum class InterpolationMethod {
    NO_INTERPOLATION           = 0,
    BASIC_LINEAR_INTERPOLATION = 1,
    LINEAR_INTERPOLATION       = 2, /// nominal level, linear regression over the samples inside the band
    POLYNOMIAL_INTERPOLATION   = 3  /// nominal level, Savitzky–Golay fit
};

enum class EdgeDetection { NONE = 0, RISING = 1, FALLING = 2 };

/** a line's state: `unknown` until a sample reaches a boundary, after a reset and after an invalid (non-finite) sample */
enum class LineState : std::uint8_t { unknown, low, high };

/**
 * A digital Schmitt trigger: `level` with a hysteresis band of full width `hysteresis` (≥ 0).
 * low → high at x ≥ level + hysteresis/2, high → low at x ≤ level − hysteresis/2, state kept inside the band;
 * with zero hysteresis high is x ≥ level, so a plateau at the level never chatters.
 * An edge is reported only between known states: the first boundary reached after a reset settles the state silently.
 * Usage: `SchmittTrigger<float, BASIC_LINEAR_INTERPOLATION> t({.level = 0.5f, .hysteresis = 0.2f}); if (t.processOne(x) == RISING) { t.lastEdgePosition; }`
 */
template<typename T, InterpolationMethod Method = InterpolationMethod::NO_INTERPOLATION, std::size_t interpolationWindow = 16UZ>
requires(std::is_arithmetic_v<T> or (UncertainValueLike<T> && std::is_arithmetic_v<meta::fundamental_base_value_type_t<T>>))
struct SchmittTrigger {
    using value_t  = meta::fundamental_base_value_type_t<T>;
    using EdgeType = UncertainValue<float>; // always float precision similar to `sample_rate` definition
    /** doubled comparisons, so an integer hysteresis/2 does not truncate */
    using compare_t = std::conditional_t<std::is_floating_point_v<value_t>, value_t, std::int64_t>;
    static_assert(std::is_floating_point_v<value_t> || sizeof(value_t) <= 4UZ, "the doubled comparison needs an integer type twice as wide");

    struct Levels {
        value_t level{};
        value_t hysteresis{}; /// full band width, ≥ 0
    };

    value_t   _level{};
    value_t   _hysteresis{};
    LineState _state = LineState::unknown;

    HistoryBuffer<T, interpolationWindow> _historyBuffer; /// valid samples since the last reset, newest first

    EdgeDetection lastEdge = EdgeDetection::NONE;
    EdgeType      lastEdgePosition{0}; /// samples from the current sample back to the last edge's crossing, ≤ 0

    std::size_t _bandSamples  = 0UZ; /// samples inside the band since the departing boundary, the regression's span
    std::size_t _validSamples = 0UZ;

    static constexpr std::size_t _sgPolyOrder = 3UZ;
    using sg_compute_t                        = std::conditional_t<std::is_floating_point_v<value_t>, value_t, float>;
    /**
     * a view, not a container: the coefficients are configuration computed once on the host, so the trigger itself
     * stays trivially copyable and a kernel can hold it by value while reading them from device memory
     */
    std::span<const sg_compute_t> _sgCoeffs;

    constexpr explicit SchmittTrigger(Levels levels = {}) noexcept : _level(levels.level), _hysteresis(levels.hysteresis) {}

    /** the coefficients a POLYNOMIAL_INTERPOLATION trigger reads; the caller owns the storage and outlives the trigger */
    constexpr void setCoefficients(std::span<const sg_compute_t> coefficients) noexcept { _sgCoeffs = coefficients; }

    /** host-side, allocating: computes into storage the caller owns, which may come from a device-capable resource */
    template<typename TResource>
    [[nodiscard]] static std::pmr::vector<sg_compute_t> computeCoefficients(TResource* resource) {
        namespace sg      = gr::algorithm::savitzky_golay;
        const auto config = sg::Config<sg_compute_t>{.derivOrder = 0UZ, .delta = sg_compute_t{1}, .alignment = sg::Alignment::Centred, .boundaryPolicy = sg::BoundaryPolicy::Reflect};
        const auto values = sg::computeCoefficients<sg_compute_t>(interpolationWindow, std::min(_sgPolyOrder, interpolationWindow - 1UZ), config);
        return std::pmr::vector<sg_compute_t>(values.begin(), values.end(), resource);
    }

    [[nodiscard]] constexpr LineState state() const noexcept { return _state; }

    constexpr void reset() noexcept {
        _state        = LineState::unknown;
        _bandSamples  = 0UZ;
        _validSamples = 0UZ;
        _historyBuffer.reset();
        lastEdge         = EdgeDetection::NONE;
        lastEdgePosition = EdgeType{0};
    }

    [[nodiscard]] constexpr bool reachesUpper(value_t x) const noexcept { return compare_t{2} * static_cast<compare_t>(x) >= compare_t{2} * static_cast<compare_t>(_level) + static_cast<compare_t>(_hysteresis); }

    [[nodiscard]] constexpr bool reachesLower(value_t x) const noexcept {
        const compare_t doubled = compare_t{2} * static_cast<compare_t>(x);
        const compare_t level   = compare_t{2} * static_cast<compare_t>(_level);
        return _hysteresis == value_t{} ? doubled < level : doubled <= level - static_cast<compare_t>(_hysteresis);
    }

    /** the boundary a rising (or falling) edge crosses, the level itself with zero hysteresis */
    [[nodiscard]] constexpr double boundary(bool rising) const noexcept {
        const double half = static_cast<double>(_hysteresis) * 0.5;
        return static_cast<double>(_level) + (rising ? half : -half);
    }

    EdgeDetection processOne(T input) {
        using enum InterpolationMethod;
        const value_t x = gr::value(input);
        if constexpr (std::is_floating_point_v<value_t>) {
            if (!std::isfinite(x)) [[unlikely]] {
                reset();
                return EdgeDetection::NONE;
            }
        }
        if constexpr (Method != NO_INTERPOLATION) {
            _historyBuffer.push_front(input);
            ++_validSamples;
        }

        const bool upper = reachesUpper(x);
        const bool lower = reachesLower(x);
        switch (_state) {
        case LineState::unknown:
            if (upper) {
                _state = LineState::high;
            } else if (lower) {
                _state = LineState::low;
            }
            return EdgeDetection::NONE;
        case LineState::low:
            if (!upper) {
                _bandSamples = lower ? 0UZ : _bandSamples + 1UZ;
                return EdgeDetection::NONE;
            }
            _state = LineState::high;
            return recordEdge(EdgeDetection::RISING);
        case LineState::high:
            if (!lower) {
                _bandSamples = upper ? 0UZ : _bandSamples + 1UZ;
                return EdgeDetection::NONE;
            }
            _state = LineState::low;
            return recordEdge(EdgeDetection::FALLING);
        }
        return EdgeDetection::NONE;
    }

    EdgeDetection recordEdge(EdgeDetection detected) {
        using enum InterpolationMethod;
        lastEdge                  = detected;
        lastEdgePosition          = EdgeType{0};
        const std::size_t spanned = _bandSamples + 1UZ;
        _bandSamples              = 0UZ;

        if constexpr (Method == BASIC_LINEAR_INTERPOLATION) {
            if (_historyBuffer.size() >= 2UZ) {
                if constexpr (UncertainValueLike<T>) {
                    const EdgeType yPrev = asEdgeType(_historyBuffer[1]);
                    const EdgeType yCurr = asEdgeType(_historyBuffer[0]);
                    if (gr::value(yCurr) != gr::value(yPrev)) {
                        lastEdgePosition = EdgeType{-1.0f} + (EdgeType{static_cast<float>(boundary(detected == EdgeDetection::RISING))} - yPrev) / (yCurr - yPrev);
                    }
                } else { // in double: a small step on a large baseline keeps its fraction
                    const double yPrev = static_cast<double>(_historyBuffer[1]);
                    const double yCurr = static_cast<double>(_historyBuffer[0]);
                    if (yCurr != yPrev) {
                        lastEdgePosition = EdgeType{static_cast<float>(-1. + (boundary(detected == EdgeDetection::RISING) - yPrev) / (yCurr - yPrev))};
                    }
                }
            }
        } else if constexpr (Method == LINEAR_INTERPOLATION || Method == POLYNOMIAL_INTERPOLATION) {
            std::size_t      n = std::min(std::max(spanned, 2UZ), _historyBuffer.size());
            std::optional<T> crossing;
            if constexpr (Method == POLYNOMIAL_INTERPOLATION) {
                assert(!_sgCoeffs.empty() && "setCoefficients() must precede use: without them the fit degrades silently");
                const std::size_t window = std::min({interpolationWindow, _historyBuffer.size(), _validSamples});
                if (window >= interpolationWindow && window >= _sgPolyOrder + 1UZ) {
                    n        = window;
                    crossing = findCrossingIndexSavitzkyGolay(_historyBuffer, n, _level);
                }
                if (!crossing) { // too few samples for the fit
                    n = std::min(std::max(spanned, 2UZ), _historyBuffer.size());
                }
            }
            if (!crossing) {
                crossing = findCrossingIndexLinearRegression(_historyBuffer, n, _level);
            }
            if (crossing) {
                // N.B. float, so an unsigned value_t cannot underflow
                const float position = static_cast<float>(gr::value(*crossing)) - static_cast<float>(n - 1UZ);
                if constexpr (UncertainValueLike<T>) {
                    lastEdgePosition = EdgeType{position, static_cast<float>(gr::uncertainty(*crossing))};
                } else {
                    lastEdgePosition = EdgeType{position};
                }
            }
        }
        return detected;
    }

    [[nodiscard]] static constexpr EdgeType asEdgeType(const T& sample) noexcept {
        if constexpr (UncertainValueLike<T>) {
            return EdgeType{static_cast<float>(gr::value(sample)), static_cast<float>(gr::uncertainty(sample))};
        } else {
            return EdgeType{static_cast<float>(sample)};
        }
    }

    std::optional<T> findCrossingIndexLinearRegression(const auto& samples, std::size_t nSamples, value_t level) {
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
        const comp_t crossingIndex = (static_cast<comp_t>(level) - intercept) / slope;

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
            const comp_t varM = varianceY / denominator;
            const comp_t varB = varianceY * sumX2 / (n_val * denominator);

            // covariance between slope and intercept
            const comp_t covMb = -varianceY * meanX / denominator;

            // partial derivatives for crossing index = (level - b) / m
            // d(crossingIndex)/db = -1/m
            // d(crossingIndex)/dm = -(level - b)/m^2
            const comp_t dCiDb = -comp_t{1} / slope;
            const comp_t dCiDm = -(static_cast<comp_t>(level) - intercept) / (slope * slope);

            // error propagation
            const comp_t varCi = dCiDb * dCiDb * varB + dCiDm * dCiDm * varM + comp_t{2} * dCiDb * dCiDm * covMb;

            return T{static_cast<value_t>(crossingIndex), static_cast<float>((varCi < comp_t{0}) ? comp_t{0} : std::sqrt(varCi))};
        } else { // fundamental type ->  no uncertainty
            return std::make_optional(static_cast<value_t>(crossingIndex));
        }
    }

    std::optional<T> findCrossingIndexSavitzkyGolay(const auto& samples, std::size_t nSamples, value_t level) {
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

        // c) Scan for where smoothed signal crosses the level (search from newest end)
        const sg_compute_t levelValue = static_cast<sg_compute_t>(level);
        std::size_t        crossIdx   = 0;
        bool               found      = false;

        for (std::size_t i = nSamples - 1; i > 0; --i) {
            if ((smoothedData[i] >= levelValue && smoothedData[i - 1] < levelValue) || (smoothedData[i] <= levelValue && smoothedData[i - 1] > levelValue)) {
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
        const sg_compute_t frac          = (y2 != y1) ? (levelValue - y1) / (y2 - y1) : sg_compute_t{0.5};
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

            // Propagate through linear interpolation: crossingIndex = (crossIdx-1) + (level - y1)/(y2 - y1)
            const sg_compute_t dy                  = y2 - y1;
            sg_compute_t       crossingUncertainty = sg_compute_t{0};
            if (dy != sg_compute_t{0}) {
                const sg_compute_t dfdy1 = (levelValue - y2) / (dy * dy);
                const sg_compute_t dfdy2 = (y1 - levelValue) / (dy * dy);
                crossingUncertainty      = std::sqrt(dfdy1 * dfdy1 * u1 * u1 + dfdy2 * dfdy2 * u2 * u2);
            }

            return T{static_cast<value_t>(crossingIndex), static_cast<float>(crossingUncertainty)};
        } else {
            return std::make_optional(static_cast<value_t>(crossingIndex));
        }
    }
};

static_assert(std::is_trivially_copyable_v<SchmittTrigger<float, InterpolationMethod::LINEAR_INTERPOLATION, 32UZ>>, "a kernel holds the trigger by value");
static_assert(std::is_trivially_copyable_v<SchmittTrigger<float, InterpolationMethod::POLYNOMIAL_INTERPOLATION, 32UZ>>, "the coefficients are a view, so this method is device-capable too");

} // namespace gr::trigger

#endif // SCHMITTTRIGGER_HPP
