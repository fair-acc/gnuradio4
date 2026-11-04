#ifndef GNURADIO_ALGORITHM_PEAK_HPP
#define GNURADIO_ALGORITHM_PEAK_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <vector>

#include <gnuradio-4.0/meta/DeviceAnnotations.hpp>
#include <gnuradio-4.0/meta/UncertainValue.hpp>

namespace gr::algorithm::peak {

template<std::floating_point T>
struct ParabolicPeak {
    T offsetSamples;
    T value;
};

/** the vertex of the parabola through (-1, yMinus), (0, y0), (+1, yPlus); y0 itself when the three do not bend downwards */
template<std::floating_point T>
[[nodiscard]] GR_DEVICE_FN constexpr ParabolicPeak<T> parabolicPeak(T yMinus, T y0, T yPlus) noexcept {
    const T curvature = yMinus - T{2} * y0 + yPlus;
    if (curvature >= T{0}) {
        return {.offsetSamples = T{0}, .value = y0};
    }
    const T offset = (yMinus - yPlus) / (T{2} * curvature);
    return {.offsetSamples = offset, .value = y0 - (yMinus - yPlus) * offset / T{4}};
}

/** the vertex of the Gaussian through three positive samples, a parabola on their logs; empty for a sample that is not positive */
template<std::floating_point T>
[[nodiscard]] GR_DEVICE_FN std::optional<ParabolicPeak<T>> gaussianPeak(T yMinus, T y0, T yPlus) noexcept {
    if (!(yMinus > T{0} && y0 > T{0} && yPlus > T{0})) {
        return std::nullopt;
    }
    const ParabolicPeak<T> logPeak = parabolicPeak(std::log(yMinus), std::log(y0), std::log(yPlus));
    return ParabolicPeak<T>{.offsetSamples = logPeak.offsetSamples, .value = std::exp(logPeak.value)};
}

template<typename T>
[[nodiscard]] GR_DEVICE_FN constexpr bool isLocalMaximum(const T* v, std::size_t i, std::size_t n) noexcept {
    return i > 0UZ && i + 1UZ < n && v[i] > v[i - 1UZ] && v[i] > v[i + 1UZ];
}

inline constexpr std::size_t kMaxGridRank = 3UZ;

/** a score map of rank 1 to 3, row-major (the last dimension fastest): cell i of dimension d lies at origin[d] + i step[d]; a periodic dimension spans extent * step and its ends are neighbours */
template<std::floating_point T>
struct Grid {
    std::array<std::size_t, kMaxGridRank> extents{1UZ, 1UZ, 1UZ};
    std::size_t                           rank = 1UZ;
    std::array<T, kMaxGridRank>           origin{};
    std::array<T, kMaxGridRank>           step{T{1}, T{1}, T{1}};
    std::array<bool, kMaxGridRank>        periodic{};

    [[nodiscard]] GR_DEVICE_FN constexpr std::size_t stride(std::size_t d) const noexcept {
        std::size_t elements = 1UZ;
        for (std::size_t e = d + 1UZ; e < rank; ++e) {
            elements *= extents[e];
        }
        return elements;
    }

    [[nodiscard]] GR_DEVICE_FN constexpr std::size_t indexOf(std::size_t cell, std::size_t d) const noexcept { return cell / stride(d) % extents[d]; }

    /** the cell one along d in `direction` (-1 or +1), across a periodic dimension's ends; `cells` past a non-periodic one */
    [[nodiscard]] GR_DEVICE_FN constexpr std::size_t neighbour(std::size_t cell, std::size_t d, int direction, std::size_t cells) const noexcept {
        const std::size_t i     = indexOf(cell, d);
        const bool        atEnd = direction < 0 ? i == 0UZ : i + 1UZ == extents[d];
        if (atEnd && !periodic[d]) {
            return cells;
        }
        const std::size_t moved = (i + (direction < 0 ? extents[d] - 1UZ : 1UZ)) % extents[d];
        return cell - i * stride(d) + moved * stride(d);
    }

    /** the squared distance between two cells in cells, across a periodic dimension's ends where shorter */
    [[nodiscard]] GR_DEVICE_FN constexpr std::size_t squaredDistance(std::size_t a, std::size_t b) const noexcept {
        std::size_t squared = 0UZ;
        for (std::size_t d = 0UZ; d < rank; ++d) {
            const std::size_t i     = indexOf(a, d);
            const std::size_t j     = indexOf(b, d);
            const std::size_t apart = i > j ? i - j : j - i;
            const std::size_t delta = periodic[d] ? std::min(apart, extents[d] - apart) : apart;
            squared += delta * delta;
        }
        return squared;
    }
};

/**
 * true when `cell` is strictly above its 3^D - 1 neighbours and not at either end of a non-periodic dimension longer than one cell (a plateau, a flat map and a scan's edge give no peak)
 */
template<std::floating_point T>
[[nodiscard]] GR_DEVICE_FN bool isPeak(const T* values, const Grid<T>& grid, std::size_t cell, std::size_t cells) noexcept {
    std::size_t combinations = 1UZ;
    for (std::size_t d = 0UZ; d < grid.rank; ++d) {
        const std::size_t i = grid.indexOf(cell, d);
        if (!grid.periodic[d] && grid.extents[d] > 1UZ && (i == 0UZ || i + 1UZ == grid.extents[d])) {
            return false;
        }
        combinations *= 3UZ;
    }
    for (std::size_t combination = 0UZ; combination < combinations; ++combination) {
        std::size_t other  = cell;
        std::size_t digits = combination;
        for (std::size_t d = 0UZ; d < grid.rank && other < cells; ++d, digits /= 3UZ) {
            const int direction = static_cast<int>(digits % 3UZ) - 1;
            other               = direction == 0 ? other : grid.neighbour(other, d, direction, cells);
        }
        if (other < cells && other != cell && !(values[cell] > values[other])) {
            return false;
        }
    }
    return true;
}

/**
 * row {coordinates[D], score, confidence} of the peak at `cell`: per dimension a parabola through it and its two neighbours places it between cells, the coordinate is origin + index step (wrapped on a periodic dimension), the score the cell plus each parabola's rise, the confidence the score over `reference`
 */
template<std::floating_point T>
GR_DEVICE_FN void describePeak(const T* values, const Grid<T>& grid, std::size_t cell, std::size_t cells, T reference, T* row) noexcept {
    const T y0    = values[cell];
    T       score = y0;
    for (std::size_t d = 0UZ; d < grid.rank; ++d) {
        const std::size_t below  = grid.neighbour(cell, d, -1, cells);
        const std::size_t above  = grid.neighbour(cell, d, +1, cells);
        const bool        inside = below < cells && above < cells && below != cell;
        const auto        vertex = inside ? parabolicPeak(values[below], y0, values[above]) : ParabolicPeak<T>{.offsetSamples = T{0}, .value = y0};
        const T           period = static_cast<T>(grid.extents[d]) * grid.step[d];
        const T           where  = (static_cast<T>(grid.indexOf(cell, d)) + vertex.offsetSamples) * grid.step[d];
        row[d]                   = grid.origin[d] + (grid.periodic[d] ? std::fmod(std::fmod(where, period) + period, period) : where);
        score += vertex.value - y0;
    }
    row[grid.rank]       = score;
    row[grid.rank + 1UZ] = score / reference;
}

/**
 * the at most `maxPeaks` strongest cells of `peaks` (a peak's score, else -inf) as `describePeak` rows `width` apart, greedily highest first: a candidate scores at least `relative` times the reference (the larger of the map's maximum and the highest score) and lies more than `separation` (squared cells) from every accepted one
 * accepted[0] receives the count, accepted[1..] the cells
 */
template<std::floating_point T>
GR_DEVICE_FN void selectPeaks(const T* values, const T* peaks, const Grid<T>& grid, std::size_t cells, std::size_t maxPeaks, T relative, std::size_t separation, std::uint32_t* accepted, T* rows, std::size_t width) noexcept {
    T reference = -std::numeric_limits<T>::infinity();
    for (std::size_t cell = 0UZ; cell < cells; ++cell) {
        reference = values[cell] > reference ? values[cell] : reference;
        reference = peaks[cell] > reference ? peaks[cell] : reference;
    }
    std::uint32_t found = 0U;
    for (; reference > T{0} && found < maxPeaks; ++found) {
        std::size_t best = cells;
        for (std::size_t cell = 0UZ; cell < cells; ++cell) {
            bool separated = peaks[cell] >= relative * reference && (best == cells || peaks[cell] > peaks[best]);
            for (std::uint32_t a = 0U; separated && a < found; ++a) {
                separated = grid.squaredDistance(cell, accepted[1U + a]) > separation;
            }
            best = separated ? cell : best;
        }
        if (best == cells) {
            break;
        }
        accepted[1U + found] = static_cast<std::uint32_t>(best);
        describePeak(values, grid, best, cells, reference, rows + found * width);
    }
    accepted[0] = found;
}

inline constexpr float kGaussianHalfMaxToSigma = 0.849321800f;

struct PeakResult {
    float confidence           = 0.f; /**< [0,1] peak exists (ML) or prominence/sigma (classical) */
    float centre               = 0.f; /**< bin position (fractional, sub-bin interpolated) */
    float amplitude            = 0.f; /**< predicted relative amplitude (ML) or height above floor (classical) */
    float amplitudeMeasured    = 0.f; /**< raw spectrum value at centre minus noise floor (post-hoc) */
    float hwhmLeft             = 0.f;
    float hwhmRight            = 0.f;
    float w68                  = 0.f; /**< 68% energy containment width of the fitted profile */
    float w96                  = 0.f; /**< 96% energy containment width of the fitted profile */
    float w99                  = 0.f; /**< 99.7% energy containment width of the fitted profile */
    float kurtosis             = 0.f;
    float typeTag              = 0.f; /**< provenance; 0 = plain detection, model-defined otherwise */
    float prominence           = 0.f; /**< in noise sigma units */
    float isolation            = 0.f; /**< distance to nearest equal-or-higher peak (bins) */
    float positionUncertainty  = 0.f; /**< r.m.s. sigma of centre estimate (bins) */
    float widthUncertainty     = 0.f; /**< r.m.s. sigma of width estimate (relative) */
    float amplitudeUncertainty = 0.f; /**< r.m.s. sigma of amplitude estimate */

    [[nodiscard]] constexpr float fwhm() const noexcept { return hwhmLeft + hwhmRight; }

    /** the Gaussian sharing this peak's half-maximum width: a comparable scale, as the shape is generally not Gaussian */
    [[nodiscard]] constexpr float gaussianEquivalentSigma() const noexcept { return 0.5f * fwhm() * kGaussianHalfMaxToSigma; }
};

struct NoiseEstimate {
    float median;
    float robustSigma;
};

struct ContainmentWidths {
    float w68;
    float w96;
    float w99;
};

/**
 * Energy-containment widths of an asymmetric pseudo-Voigt of unit peak, integrated numerically against its analytic total energy so a truncated grid cannot bias the result.
 * Mirrors the computation an ONNX model performs in-graph; qa_PeakDetector pins the two against shared reference values.
 */
[[nodiscard]] inline ContainmentWidths containmentWidths(float sigmaLeft, float sigmaRight, float eta) noexcept {
    constexpr float kCrossEnergy   = 0.821772440f; /**< (pi/2) e^(1/2) erfc(1/sqrt2) */
    constexpr float kGaussEnergy   = 0.886226925f;
    constexpr float kLorentzEnergy = 0.785398163f;
    constexpr int   kSteps         = 2400;
    constexpr float kSpanInSigma   = 24.f; /**< reaches 99.7 % even for a pure Lorentzian */

    const float left  = std::max(sigmaLeft, 1e-3f);
    const float right = std::max(sigmaRight, 1e-3f);
    const float blend = std::clamp(eta, 0.f, 1.f);
    const float step  = kSpanInSigma / static_cast<float>(kSteps) * 0.5f * (left + right);
    const float total = (left + right) * ((1.f - blend) * (1.f - blend) * kGaussEnergy //
                                             + 2.f * blend * (1.f - blend) * kCrossEnergy + blend * blend * kLorentzEnergy);

    auto flankEnergy = [&](float offset, float sigma) {
        const float scaled = (offset / sigma) * (offset / sigma);
        const float value  = (1.f - blend) * std::exp(-0.5f * scaled) + blend / (1.f + scaled);
        return value * value;
    };

    constexpr std::array<float, 3> kThresholds{0.68f, 0.96f, 0.997f};
    std::array<float, 3>           widths{};
    std::size_t                    next       = 0UZ;
    float                          cumulative = 0.f;
    float                          previous   = 0.f;
    for (int i = 0; i < kSteps && next < kThresholds.size(); ++i) {
        const float offset = (static_cast<float>(i) + 0.5f) * step;
        previous           = cumulative / total;
        cumulative += (flankEnergy(offset, left) + flankEnergy(offset, right)) * step;
        const float reached = cumulative / total;
        while (next < kThresholds.size() && reached >= kThresholds[next]) {
            const float span     = reached - previous;
            const float fraction = span > 1e-12f ? (kThresholds[next] - previous) / span : 0.f;
            widths[next]         = 2.f * (static_cast<float>(i) + fraction) * step;
            ++next;
        }
    }
    for (std::size_t i = next; i < kThresholds.size(); ++i) {
        widths[i] = 2.f * static_cast<float>(kSteps) * step;
    }
    return {widths[0], widths[1], widths[2]};
}

/** delta = 1/2 log(w_R/w_L), the scale-free asymmetry the ML estimator is trained against; 0 when either flank is degenerate */
[[nodiscard]] inline float asymmetryOf(const PeakResult& peak) noexcept {
    if (peak.hwhmLeft <= 0.f || peak.hwhmRight <= 0.f) {
        return 0.f;
    }
    return 0.5f * std::log(peak.hwhmRight / peak.hwhmLeft);
}

template<typename T>
requires(std::floating_point<T> || gr::UncertainValueLike<T>)
struct HalfMaxWidths {
    T left; /**< from the peak to the half-maximum crossing, in samples; NaN when the data end first */
    T right;
};

/** linear half-maximum crossings either side of data[idx], the half level midway between `baseline` and the peak; compared on the value, interpolated on T, so an UncertainValue carries its (uncorrelated) uncertainty into the widths */
template<typename T>
requires(std::floating_point<T> || gr::UncertainValueLike<T>)
[[nodiscard]] constexpr HalfMaxWidths<T> halfMaxWidths(std::span<const T> data, std::size_t idx, T baseline = T{0}) noexcept {
    using V              = gr::meta::fundamental_base_value_type_t<T>;
    const T    halfMax   = baseline + (data[idx] - baseline) * V{0.5};
    const V    halfValue = gr::value(halfMax);
    const auto crossing  = [&](std::size_t i) {
        const T rise = data[i] - data[i - 1UZ];
        return gr::value(rise) == V{0} ? T{static_cast<V>(i - 1UZ)} : static_cast<V>(i - 1UZ) + (halfMax - data[i - 1UZ]) / rise;
    };
    const T nan  = T{std::numeric_limits<V>::quiet_NaN()};
    T       left = nan;
    for (std::size_t i = idx; i > 0UZ; --i) {
        if (gr::value(data[i - 1UZ]) <= halfValue) {
            left = static_cast<V>(idx) - crossing(i);
            break;
        }
    }
    T right = nan;
    for (std::size_t i = idx + 1UZ; i < data.size(); ++i) {
        if (gr::value(data[i]) <= halfValue) {
            right = crossing(i) - static_cast<V>(idx);
            break;
        }
    }
    return {left, right};
}

constexpr float kHalfNormalMadToSigma = 2.50569f;

/** reorders `values` in place */
[[nodiscard]] inline float medianOf(std::span<float> values) {
    if (values.empty()) {
        return 0.f;
    }
    auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::ranges::nth_element(values, middle);
    return *middle;
}

[[nodiscard]] inline NoiseEstimate estimateMagnitudeNoise(std::span<const float> magnitudeSpectrum) {
    if (magnitudeSpectrum.empty()) {
        return {0.f, 1.f};
    }
    std::vector<float> scratch(magnitudeSpectrum.begin(), magnitudeSpectrum.end());
    const float        median = medianOf(scratch);

    std::ranges::transform(magnitudeSpectrum, scratch.begin(), [median](float value) { return std::abs(value - median); });
    return {median, std::max(kHalfNormalMadToSigma * medianOf(scratch), 1e-10f)};
}

[[nodiscard]] inline float estimateProminence(std::span<const float> data, std::size_t peakIdx) {
    const float peakVal = data[peakIdx];
    const auto  n       = data.size();

    float leftMin = peakVal;
    for (std::size_t i = peakIdx; i > 0; --i) {
        leftMin = std::min(leftMin, data[i - 1]);
        if (data[i - 1] > peakVal) {
            break;
        }
    }

    float rightMin = peakVal;
    for (std::size_t i = peakIdx + 1; i < n; ++i) {
        rightMin = std::min(rightMin, data[i]);
        if (data[i] > peakVal) {
            break;
        }
    }

    return peakVal - std::max(leftMin, rightMin);
}

[[nodiscard]] inline float estimateIsolation(std::span<const float> data, std::size_t peakIdx) {
    const float peakVal = data[peakIdx];
    const auto  n       = data.size();
    float       minDist = static_cast<float>(n);

    for (std::size_t i = peakIdx; i > 1; --i) {
        if (data[i - 1] >= peakVal && data[i - 1] > data[i - 2]) {
            minDist = static_cast<float>(peakIdx - (i - 1));
            break;
        }
    }
    for (std::size_t i = peakIdx + 1; i < n - 1; ++i) {
        if (data[i] >= peakVal && data[i] > data[i + 1]) {
            minDist = std::min(minDist, static_cast<float>(i - peakIdx));
            break;
        }
    }
    return minDist;
}

struct UncertaintyEstimate {
    float position;
    float width;
    float amplitude;
};

[[nodiscard]] inline UncertaintyEstimate estimateUncertainty(float noiseSigma, float peakAmplitude, float peakSigma) {
    const float snr      = peakAmplitude / (noiseSigma + 1e-10f);
    const float sigmaPos = std::max(peakSigma, 1.f) / (snr + 1e-3f);
    const float relWidth = sigmaPos / (std::sqrt(2.f) * (peakSigma + 1e-10f));
    return {sigmaPos, relWidth, noiseSigma};
}

[[nodiscard]] inline float estimateExcessKurtosis(std::span<const float> data, std::size_t peakIdx, float sigma) {
    const auto        n    = data.size();
    const std::size_t win  = static_cast<std::size_t>(std::max(3.f, 3.f * sigma));
    const std::size_t wLo  = (peakIdx > win) ? peakIdx - win : 0;
    const std::size_t wHi  = std::min(peakIdx + win + 1, n);
    const std::size_t wLen = wHi - wLo;
    if (wLen < 4) {
        return 0.f;
    }

    float mean = 0.f, m2 = 0.f, m4 = 0.f;
    for (std::size_t j = wLo; j < wHi; ++j) {
        mean += data[j];
    }
    mean /= static_cast<float>(wLen);
    for (std::size_t j = wLo; j < wHi; ++j) {
        const float d = data[j] - mean;
        m2 += d * d;
        m4 += d * d * d * d;
    }
    m2 /= static_cast<float>(wLen);
    m4 /= static_cast<float>(wLen);
    const float kurt = (m2 > 1e-10f) ? m4 / (m2 * m2) : 3.f;
    return kurt - 3.f;
}

enum class SubtractionShape { Auto, Gaussian, Lorentzian, Voigt };

inline void subtractPeak(std::vector<float>& spectrum, float centre, float amplitude, float sigmaL, float sigmaR, float excessKurtosis, SubtractionShape shape) {
    const auto n = spectrum.size();

    if (shape == SubtractionShape::Auto) {
        if (excessKurtosis > 1.5f) {
            shape = SubtractionShape::Lorentzian;
        } else if (excessKurtosis > 0.5f) {
            shape = SubtractionShape::Voigt;
        } else {
            shape = SubtractionShape::Gaussian;
        }
    }

    const float radius = 6.f * std::max(sigmaL, sigmaR);
    const auto  lo     = static_cast<std::size_t>(std::max(0.f, centre - radius));
    const auto  hi     = std::min(n, static_cast<std::size_t>(centre + radius + 1));

    for (std::size_t i = lo; i < hi; ++i) {
        const float x     = static_cast<float>(i) - centre;
        const float sigma = std::max(x <= 0.f ? sigmaL : sigmaR, 0.5f);

        float value = 0.f;
        switch (shape) {
        case SubtractionShape::Auto:
        case SubtractionShape::Gaussian: value = amplitude * std::exp(-0.5f * x * x / (sigma * sigma)); break;
        case SubtractionShape::Lorentzian: value = amplitude * sigma * sigma / (sigma * sigma + x * x); break;
        case SubtractionShape::Voigt: {
            const float gaussian           = std::exp(-0.5f * x * x / (sigma * sigma));
            const float lorentzian         = sigma * sigma / (sigma * sigma + x * x);
            const float lorentzianFraction = std::clamp(excessKurtosis / 3.f, 0.f, 1.f);
            value                          = amplitude * ((1.f - lorentzianFraction) * gaussian + lorentzianFraction * lorentzian);
            break;
        }
        }
        spectrum[i] -= value;
    }
}

} // namespace gr::algorithm::peak

#endif // GNURADIO_ALGORITHM_PEAK_HPP
