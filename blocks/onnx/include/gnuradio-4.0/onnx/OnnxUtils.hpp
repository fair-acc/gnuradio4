#ifndef GR_ONNX_UTILS_HPP
#define GR_ONNX_UTILS_HPP

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

// log-MAD normalisation matching Python spectrum_to_normalized()
//
// 1. shift so min value → 1.0
// 2. log10
// 3. robust z-score using median and MAD (scaled by 1.4826)
// 4. clip to [-5, 10]
inline void normalise(std::span<const float> raw, std::span<float> out) {
    const std::size_t n = raw.size();
    if (n == 0 || out.size() < n) {
        return;
    }

    float minVal = raw[0];
    for (std::size_t i = 1; i < n; ++i) {
        minVal = std::min(minVal, raw[i]);
    }

    // shift + log10
    for (std::size_t i = 0; i < n; ++i) {
        float shifted = raw[i] - minVal + 1.0f;
        out[i]        = std::log10(shifted);
    }

    // sanitise non-finite values
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(out[i])) {
            out[i] = 0.0f;
        }
    }

    // compute median via nth_element on a copy
    std::vector<float> sorted(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(n));
    auto               midIt = sorted.begin() + static_cast<std::ptrdiff_t>(n / 2);
    std::nth_element(sorted.begin(), midIt, sorted.end());
    float median = *midIt;
    if (n % 2 == 0) {
        float lower = *std::max_element(sorted.begin(), midIt);
        median      = (lower + median) * 0.5f;
    }

    // compute MAD
    std::vector<float> absdev(n);
    for (std::size_t i = 0; i < n; ++i) {
        absdev[i] = std::abs(out[i] - median);
    }
    auto madIt = absdev.begin() + static_cast<std::ptrdiff_t>(n / 2);
    std::nth_element(absdev.begin(), madIt, absdev.end());
    float mad = *madIt;
    if (n % 2 == 0) {
        float lower = *std::max_element(absdev.begin(), madIt);
        mad         = (lower + mad) * 0.5f;
    }

    // fallback if MAD is near zero
    if (mad < 1e-10f) {
        float sum = 0.f;
        for (std::size_t i = 0; i < n; ++i) {
            sum += out[i];
        }
        float mean = sum / static_cast<float>(n);
        float var  = 0.f;
        for (std::size_t i = 0; i < n; ++i) {
            float d = out[i] - mean;
            var += d * d;
        }
        mad = std::sqrt(var / static_cast<float>(n)) + 1e-10f;
    }

    float scale = 1.0f / (1.4826f * mad + 1e-10f);
    for (std::size_t i = 0; i < n; ++i) {
        float v = (out[i] - median) * scale;
        out[i]  = std::clamp(v, -5.0f, 10.0f);
    }
}

// linear resampling from input size to output size
inline void resample(std::span<const float> input, std::span<float> output) {
    const std::size_t inSize  = input.size();
    const std::size_t outSize = output.size();
    if (inSize == 0 || outSize == 0) {
        return;
    }
    if (inSize == outSize) {
        std::copy(input.begin(), input.end(), output.begin());
        return;
    }
    for (std::size_t i = 0; i < outSize; ++i) {
        float srcIdx = static_cast<float>(i) * static_cast<float>(inSize - 1) / static_cast<float>(outSize - 1);
        auto  lo     = static_cast<std::size_t>(srcIdx);
        auto  hi     = std::min(lo + 1, inSize - 1);
        float frac   = srcIdx - static_cast<float>(lo);
        output[i]    = input[lo] * (1.0f - frac) + input[hi] * frac;
    }
}

// heatmap NMS + peak extraction matching Python extract_peaks_from_heatmap()
//
// heatmap: [N] confidence values
// regression: [N * R] regression channels (row-major, R channels per bin)
// R must be >= 8 for full peak info; channels: offset, amplitude, sigma, w68, w96, w99, kurtosis, excess_kurtosis
inline std::vector<PeakResult> extractPeaks(std::span<const float> heatmap, std::span<const float> regression, std::size_t regressionChannels, float threshold, std::size_t minDistance, std::size_t maxPeaks) {

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

#endif // GR_ONNX_UTILS_HPP
