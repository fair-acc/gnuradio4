#ifndef GNURADIO_PEAK_RESULT_HPP
#define GNURADIO_PEAK_RESULT_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace gr::blocks::fourier {

struct PeakResult {
    float confidence           = 0.f; // [0,1] peak exists (ML) or prominence/sigma (classical)
    float centre               = 0.f; // bin position (fractional, sub-bin interpolated)
    float amplitude            = 0.f; // predicted relative amplitude (ML) or height above floor (classical)
    float amplitudeMeasured    = 0.f; // raw spectrum value at centre minus noise floor (post-hoc)
    float sigmaLeft            = 0.f; // left half-width at half-maximum (bins)
    float sigmaRight           = 0.f; // right half-width at half-maximum (bins)
    float w68                  = 0.f; // 68% energy containment width
    float w96                  = 0.f; // 96% energy containment width
    float w99                  = 0.f; // 99.7% energy containment width
    float kurtosis             = 0.f; // excess kurtosis
    float prominence           = 0.f; // in noise sigma units
    float isolation            = 0.f; // distance to nearest equal-or-higher peak (bins)
    float positionUncertainty  = 0.f; // r.m.s. sigma of centre estimate (bins)
    float widthUncertainty     = 0.f; // r.m.s. sigma of width estimate (relative)
    float amplitudeUncertainty = 0.f; // r.m.s. sigma of amplitude estimate

    [[nodiscard]] constexpr float sigma() const noexcept { return 0.5f * (sigmaLeft + sigmaRight); }
};

struct NoiseEstimate {
    float median;
    float sigma; // robust noise standard deviation (1.4826 * MAD)
};

[[nodiscard]] inline NoiseEstimate estimateNoise(std::span<const float> data) {
    const std::size_t n = data.size();
    if (n == 0) {
        return {0.f, 1.f};
    }

    std::vector<float> sorted(data.begin(), data.end());
    auto               midIt = sorted.begin() + static_cast<std::ptrdiff_t>(n / 2);
    std::nth_element(sorted.begin(), midIt, sorted.end());
    float med = *midIt;

    std::vector<float> absdev(n);
    for (std::size_t i = 0; i < n; ++i) {
        absdev[i] = std::abs(data[i] - med);
    }
    auto madIt = absdev.begin() + static_cast<std::ptrdiff_t>(n / 2);
    std::nth_element(absdev.begin(), madIt, absdev.end());
    float sigma = 1.4826f * (*madIt);
    return {med, std::max(sigma, 1e-10f)};
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
