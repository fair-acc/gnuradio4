#ifndef GR_ONNX_PEAK_EXTRACTION_HPP
#define GR_ONNX_PEAK_EXTRACTION_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace gr::blocks::onnx {

struct PeakResult {
    float position   = 0.f; // bin index (fractional)
    float confidence = 0.f; // heatmap value at detection point
    float sigma      = 0.f; // width in bins
    float amplitude  = 0.f; // relative amplitude (from regression)
    float w68        = 0.f; // 68% energy containment width
    float w96        = 0.f; // 96% energy containment width
    float w99        = 0.f; // 99.7% energy containment width
    float kurtosis   = 0.f; // excess kurtosis
};

// --- Classical peak-detection utilities (from PeakDetector) -----------------

struct NoiseEstimate {
    float median;
    float sigma;
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

    // MAD
    std::vector<float> absdev(n);
    for (std::size_t i = 0; i < n; ++i) {
        absdev[i] = std::abs(data[i] - med);
    }
    auto madIt = absdev.begin() + static_cast<std::ptrdiff_t>(n / 2);
    std::nth_element(absdev.begin(), madIt, absdev.end());
    float mad = *madIt;

    float sigma = 1.4826f * mad;
    if (sigma < 1e-10f) {
        sigma = 1e-10f;
    }

    return {med, sigma};
}

[[nodiscard]] inline float estimateProminence(std::span<const float> data, std::size_t peakIdx) {
    const float peakVal = data[peakIdx];
    const auto  n       = data.size();

    // walk left to find the lowest valley before a higher peak
    float leftMin = peakVal;
    for (std::size_t i = peakIdx; i > 0; --i) {
        leftMin = std::min(leftMin, data[i - 1]);
        if (data[i - 1] > peakVal) {
            break;
        }
    }

    // walk right
    float rightMin = peakVal;
    for (std::size_t i = peakIdx + 1; i < n; ++i) {
        rightMin = std::min(rightMin, data[i]);
        if (data[i] > peakVal) {
            break;
        }
    }

    return peakVal - std::max(leftMin, rightMin);
}

struct WidthEstimate {
    float left;
    float right;
};

[[nodiscard]] inline WidthEstimate estimateWidth(std::span<const float> data, std::size_t peakIdx) {
    const float peakVal = data[peakIdx];
    const auto  n       = data.size();
    const float halfMax = peakVal * 0.5f;

    float leftWidth = 0.f;
    for (std::size_t i = peakIdx; i > 0; --i) {
        if (data[i - 1] <= halfMax) {
            float frac = (data[i] - halfMax) / (data[i] - data[i - 1] + 1e-10f);
            leftWidth  = static_cast<float>(peakIdx - i) + frac;
            break;
        }
        leftWidth = static_cast<float>(peakIdx - i + 1);
    }

    float rightWidth = 0.f;
    for (std::size_t i = peakIdx + 1; i < n; ++i) {
        if (data[i] <= halfMax) {
            float frac = (data[i - 1] - halfMax) / (data[i - 1] - data[i] + 1e-10f);
            rightWidth = static_cast<float>(i - peakIdx - 1) + frac;
            break;
        }
        rightWidth = static_cast<float>(i - peakIdx);
    }

    return {leftWidth, rightWidth};
}

// --- ONNX heatmap peak extraction (from OnnxUtils) --------------------------

// heatmap NMS + peak extraction matching Python extract_peaks_from_heatmap()
//
// heatmap: [N] confidence values
// regression: [N * R] regression channels (row-major, R channels per bin)
// R must be >= 8 for full peak info; channels: offset, amplitude, sigma, w68, w96, w99, kurtosis, excess_kurtosis
[[nodiscard]] inline std::vector<PeakResult> extractPeaks(std::span<const float> heatmap, std::span<const float> regression, std::size_t regressionChannels, float threshold, std::size_t minDistance, std::size_t maxPeaks) {

    const std::size_t n = heatmap.size();
    if (n < 2 || regression.size() < n * regressionChannels || regressionChannels < 8) {
        return {};
    }

    // find local maxima
    std::vector<std::pair<float, std::size_t>> candidates;
    if (heatmap[0] > heatmap[1] && heatmap[0] >= threshold) {
        candidates.emplace_back(heatmap[0], 0UZ);
    }
    for (std::size_t i = 1; i < n - 1; ++i) {
        if (heatmap[i] > heatmap[i - 1] && heatmap[i] > heatmap[i + 1] && heatmap[i] >= threshold) {
            candidates.emplace_back(heatmap[i], i);
        }
    }
    if (heatmap[n - 1] > heatmap[n - 2] && heatmap[n - 1] >= threshold) {
        candidates.emplace_back(heatmap[n - 1], n - 1);
    }

    // sort by confidence descending
    std::ranges::sort(candidates, [](const auto& a, const auto& b) { return a.first > b.first; });

    // NMS: suppress neighbours within minDistance
    std::vector<bool>       suppressed(n, false);
    std::vector<PeakResult> peaks;
    peaks.reserve(std::min(candidates.size(), maxPeaks));

    for (const auto& [conf, idx] : candidates) {
        if (suppressed[idx] || peaks.size() >= maxPeaks) {
            break;
        }

        const float* reg = regression.data() + idx * regressionChannels;

        float center = static_cast<float>(idx) + reg[0] * static_cast<float>(n);
        center       = std::clamp(center, 0.0f, static_cast<float>(n - 1));

        peaks.push_back({
            .position   = center,
            .confidence = conf,
            .sigma      = reg[2] * static_cast<float>(n),
            .amplitude  = reg[1],
            .w68        = reg[3] * static_cast<float>(n),
            .w96        = reg[4] * static_cast<float>(n),
            .w99        = reg[5] * static_cast<float>(n),
            .kurtosis   = reg[7] * 10.0f - 5.0f, // excess kurtosis (channel 7)
        });

        std::size_t lo = (idx > minDistance) ? idx - minDistance : 0;
        std::size_t hi = std::min(idx + minDistance + 1, n);
        for (std::size_t s = lo; s < hi; ++s) {
            suppressed[s] = true;
        }
    }

    return peaks;
}

} // namespace gr::blocks::onnx

#endif // GR_ONNX_PEAK_EXTRACTION_HPP
