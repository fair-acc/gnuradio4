#ifndef GNURADIO_PEAK_RESULT_HPP
#define GNURADIO_PEAK_RESULT_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace gr::blocks::fourier {

inline constexpr float kGaussianHalfMaxToSigma = 0.849321800f; // 1 / sqrt(2 ln 2)

struct PeakResult {
    float confidence           = 0.f; // [0,1] peak exists (ML) or prominence/sigma (classical)
    float centre               = 0.f; // bin position (fractional, sub-bin interpolated)
    float amplitude            = 0.f; // predicted relative amplitude (ML) or height above floor (classical)
    float amplitudeMeasured    = 0.f; // raw spectrum value at centre minus noise floor (post-hoc)
    float hwhmLeft             = 0.f; // left half-width at half-maximum (bins)
    float hwhmRight            = 0.f; // right half-width at half-maximum (bins)
    float w68                  = 0.f; // 68% energy containment width of the fitted profile
    float w96                  = 0.f; // 96% energy containment width of the fitted profile
    float w99                  = 0.f; // 99.7% energy containment width of the fitted profile
    float kurtosis             = 0.f; // excess kurtosis
    float typeTag              = 0.f; // provenance; 0 = plain detection, model-defined otherwise
    float prominence           = 0.f; // in noise sigma units
    float isolation            = 0.f; // distance to nearest equal-or-higher peak (bins)
    float positionUncertainty  = 0.f; // r.m.s. sigma of centre estimate (bins)
    float widthUncertainty     = 0.f; // r.m.s. sigma of width estimate (relative)
    float amplitudeUncertainty = 0.f; // r.m.s. sigma of amplitude estimate

    [[nodiscard]] constexpr float fwhm() const noexcept { return hwhmLeft + hwhmRight; }

    // the Gaussian that shares this peak's half-maximum width; the shape is generally not Gaussian,
    // so this is a comparable scale rather than a property of the peak
    [[nodiscard]] constexpr float gaussianEquivalentSigma() const noexcept { return 0.5f * fwhm() * kGaussianHalfMaxToSigma; }
};

struct NoiseEstimate {
    float median;
    float sigma; // robust noise standard deviation
};

struct ContainmentWidths {
    float w68;
    float w96;
    float w99;
};

// Energy-containment widths of an asymmetric pseudo-Voigt of unit peak, integrated numerically
// against its analytic total energy so a truncated grid cannot bias the result. This mirrors the
// computation an ONNX model performs in-graph; qa_PeakDetector pins the two against shared
// reference values so they cannot drift apart.
[[nodiscard]] inline ContainmentWidths containmentWidths(float sigmaLeft, float sigmaRight, float eta) noexcept {
    constexpr float kCrossEnergy   = 0.821772440f; // (pi/2) e^(1/2) erfc(1/sqrt2)
    constexpr float kGaussEnergy   = 0.886226925f; // sqrt(pi)/2
    constexpr float kLorentzEnergy = 0.785398163f; // pi/4
    constexpr int   kSteps         = 2400;
    constexpr float kSpanInSigma   = 24.f; // reaches 99.7 % even for a pure Lorentzian

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

// delta = 1/2 log(w_R/w_L): the scale-free asymmetry the ML estimator is trained against, published
// so consumers need not re-derive it. Scale-free, so half-maximum and sigma flanks agree. 0 when
// either flank is degenerate.
[[nodiscard]] inline float asymmetryOf(const PeakResult& peak) noexcept {
    if (peak.hwhmLeft <= 0.f || peak.hwhmRight <= 0.f) {
        return 0.f;
    }
    return 0.5f * std::log(peak.hwhmRight / peak.hwhmLeft);
}

// The event keys both detectors publish. ex06 scores them against each other, so they must agree
// key for key; keeping one definition is what stops them drifting apart. `positionScale` maps model
// bins onto input bins for a resampling detector and is 1 for one that works in input bins.
[[nodiscard]] inline gr::property_map peakEventProps(const PeakResult& peak, const NoiseEstimate& noise, float positionScale = 1.f) {
    return {
        {std::pmr::string("confidence"), gr::pmt::Value(peak.confidence)},
        // the timing_events index is an integer bin, so the fractional centre must be carried
        // separately or sub-bin accuracy is silently truncated to +-0.5 bin
        {std::pmr::string("centre"), gr::pmt::Value(peak.centre * positionScale)},
        {std::pmr::string("fwhm"), gr::pmt::Value(peak.fwhm() * positionScale)},
        {std::pmr::string("hwhm_l"), gr::pmt::Value(peak.hwhmLeft * positionScale)},
        {std::pmr::string("hwhm_r"), gr::pmt::Value(peak.hwhmRight * positionScale)},
        // info only, so a consumer can guess the shape; evaluation compares the half-maximum widths.
        // Distinct from the generator's "shape_scale", which is that shape's own parameter (sigma for
        // a Gaussian, gamma for a Lorentzian) rather than a Gaussian-equivalent one.
        {std::pmr::string("gaussian_equivalent_sigma"), gr::pmt::Value(peak.gaussianEquivalentSigma() * positionScale)},
        {std::pmr::string("asymmetry"), gr::pmt::Value(asymmetryOf(peak))},
        {std::pmr::string("amplitude"), gr::pmt::Value(peak.amplitude)},
        {std::pmr::string("amplitude_measured"), gr::pmt::Value(peak.amplitudeMeasured)},
        {std::pmr::string("prominence"), gr::pmt::Value(peak.prominence)},
        // provenance: 0 for a plain single-stage detection, otherwise model-defined (the cascade
        // writes the 1-based index of the stage that promoted the peak)
        {std::pmr::string("type_tag"), gr::pmt::Value(peak.typeTag)},
        {std::pmr::string("isolation"), gr::pmt::Value(peak.isolation * positionScale)},
        {std::pmr::string("w68"), gr::pmt::Value(peak.w68 * positionScale)},
        {std::pmr::string("w96"), gr::pmt::Value(peak.w96 * positionScale)},
        {std::pmr::string("w99"), gr::pmt::Value(peak.w99 * positionScale)},
        {std::pmr::string("kurtosis"), gr::pmt::Value(peak.kurtosis)},
        {std::pmr::string("noise_sigma"), gr::pmt::Value(noise.sigma)},
        {std::pmr::string("noise_floor"), gr::pmt::Value(noise.median)},
        {std::pmr::string("position_uncertainty"), gr::pmt::Value(peak.positionUncertainty * positionScale)},
        {std::pmr::string("width_uncertainty"), gr::pmt::Value(peak.widthUncertainty * positionScale)},
        {std::pmr::string("amplitude_uncertainty"), gr::pmt::Value(peak.amplitudeUncertainty)},
    };
}

constexpr float kHalfNormalMadToSigma = 2.50569f;

/// reorders `values` in place
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
    float snr      = peakAmplitude / (noiseSigma + 1e-10f);
    float sigmaPos = std::max(peakSigma, 1.f) / (snr + 1e-3f);
    float relWidth = sigmaPos / (std::sqrt(2.f) * (peakSigma + 1e-10f));
    return {sigmaPos, relWidth, noiseSigma};
}

} // namespace gr::blocks::fourier

#endif // GNURADIO_PEAK_RESULT_HPP
