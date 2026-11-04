#ifndef GNURADIO_PEAK_DETECTOR_HPP
#define GNURADIO_PEAK_DETECTOR_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/algorithm/peak/Peak.hpp>
#include <gnuradio-4.0/fourier/PeakEventProps.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <vector>

namespace gr::blocks::fourier {

namespace peak = gr::algorithm::peak;

// threshold defaults mirror PeakDetector's settings -- see the rationale there
[[nodiscard]] inline std::vector<peak::PeakResult> detectPeaksIterative(std::span<const float> spectrum, float noiseRejectionThreshold = 5.0f, float minAmplitude = 0.0f, float minIsolationFraction = 0.5f, std::size_t maxIterations = 10, float minProminence = 5.0f, std::size_t maxPeaks = 10, peak::SubtractionShape shape = peak::SubtractionShape::Auto) {
    const std::size_t n = spectrum.size();
    if (n < 3) {
        return {};
    }

    auto [noiseFloor, inputNoiseSigma] = peak::estimateMagnitudeNoise(spectrum);

    std::vector<float> residual(spectrum.begin(), spectrum.end());

    std::vector<peak::PeakResult> peaks;
    peaks.reserve(maxPeaks);

    for (std::size_t iter = 0; iter < maxIterations && peaks.size() < maxPeaks; ++iter) {
        const float residualFloor    = peak::estimateMagnitudeNoise(std::span<const float>(residual)).median;
        const float currentThreshold = residualFloor + noiseRejectionThreshold * inputNoiseSigma;

        std::size_t bestIdx  = 0;
        float       bestProm = 0.f;

        for (std::size_t i = 1; i < n - 1; ++i) {
            if (peak::isLocalMaximum(residual.data(), i, n) && residual[i] >= currentThreshold) {
                float prom = peak::estimateProminence(std::span<const float>(residual), i);
                if (prom > bestProm) {
                    bestProm = prom;
                    bestIdx  = i;
                }
            }
        }

        if (bestProm < minProminence * inputNoiseSigma) {
            break;
        }

        float peakAmp = residual[bestIdx] - residualFloor;
        if (peakAmp < minAmplitude) {
            break;
        }

        const float centre = static_cast<float>(bestIdx) + peak::parabolicPeak(residual[bestIdx - 1UZ], residual[bestIdx], residual[bestIdx + 1UZ]).offsetSamples;

        const auto  widths    = peak::halfMaxWidths(std::span<const float>(residual), bestIdx, residualFloor);
        const float halfLeft  = std::max(std::isnan(widths.left) ? static_cast<float>(bestIdx) : widths.left, 0.5f);
        const float halfRight = std::max(std::isnan(widths.right) ? static_cast<float>(n - 1UZ - bestIdx) : widths.right, 0.5f);
        float       sigma     = (halfLeft + halfRight) * 0.5f;

        // from the original spectrum, not the residual
        float iso = peak::estimateIsolation(spectrum, bestIdx);

        if (iso < minIsolationFraction * sigma * 2.f && !peaks.empty()) {
            // too close to an already-detected peak — skip and suppress this region
            peak::subtractPeak(residual, centre, peakAmp, halfLeft, halfRight, 0.f, shape);
            continue;
        }

        float excessKurt = peak::estimateExcessKurtosis(std::span<const float>(residual), bestIdx, sigma);

        auto [posUnc, widthUnc, ampUnc] = peak::estimateUncertainty(inputNoiseSigma, peakAmp, sigma);

        std::size_t rawIdx  = static_cast<std::size_t>(std::clamp(std::lround(centre), 0L, static_cast<long>(n - 1)));
        float       ampMeas = spectrum[rawIdx] - noiseFloor;

        // the subtraction shape is fitted through excess kurtosis, which the peaks-tensor contract
        // maps as kurt = 5*eta; reusing that mapping keeps both detectors on one width definition
        const peak::ContainmentWidths classicalWidths = peak::containmentWidths(halfLeft * peak::kGaussianHalfMaxToSigma, halfRight * peak::kGaussianHalfMaxToSigma, excessKurt / 5.f);
        peaks.push_back({
            .confidence           = bestProm / (inputNoiseSigma + 1e-10f),
            .centre               = centre,
            .amplitude            = peakAmp,
            .amplitudeMeasured    = ampMeas,
            .hwhmLeft             = halfLeft,
            .hwhmRight            = halfRight,
            .w68                  = classicalWidths.w68,
            .w96                  = classicalWidths.w96,
            .w99                  = classicalWidths.w99,
            .kurtosis             = excessKurt,
            .prominence           = bestProm / (inputNoiseSigma + 1e-10f),
            .isolation            = iso,
            .positionUncertainty  = posUnc,
            .widthUncertainty     = widthUnc,
            .amplitudeUncertainty = ampUnc,
        });

        peak::subtractPeak(residual, centre, peakAmp, halfLeft, halfRight, excessKurt, shape);
    }

    std::ranges::sort(peaks, [](const peak::PeakResult& a, const peak::PeakResult& b) { return a.centre < b.centre; });

    return peaks;
}

GR_REGISTER_BLOCK(gr::blocks::fourier::PeakDetector)

struct PeakDetector : gr::Block<PeakDetector> {
    using Description = Doc<"Classical iterative peak detector using prominence-based stripping with adaptive shape subtraction.">;

    gr::PortIn<gr::DataSet<float>>  in;
    gr::PortOut<gr::DataSet<float>> out;

    // 5 is the lowest threshold with zero false positives on SyntheticPeakSpectrum's zero-peak
    // seeds; do not lower it without re-measuring there.
    Annotated<float, "noise rejection threshold">          noise_rejection_threshold = 5.0f;
    Annotated<float, "min amplitude">                      min_amplitude             = 0.0f;
    Annotated<float, "min isolation fraction">             min_isolation             = 0.5f;
    Annotated<gr::Size_t, "max iterations">                max_iterations            = 10U;
    Annotated<float, "min prominence">                     min_prominence            = 5.0f;
    Annotated<gr::Size_t, "max peaks">                     max_peaks                 = 10U;
    Annotated<peak::SubtractionShape, "subtraction shape"> subtraction_shape         = peak::SubtractionShape::Auto;

    GR_MAKE_REFLECTABLE(PeakDetector, in, out, noise_rejection_threshold, min_amplitude, min_isolation, max_iterations, min_prominence, max_peaks, subtraction_shape);

    // not noexcept: the detection pipeline allocates and bad_alloc must propagate to the framework
    [[nodiscard]] gr::DataSet<float> processOne(gr::DataSet<float> inData) {
        if (inData.signal_values.empty()) {
            return inData;
        }

        const std::size_t nSignals = std::max(1UZ, inData.signal_names.size());
        const std::size_t n        = inData.signal_values.size() / nSignals;
        if (n == 0UZ) { // more declared signals than values: the prominence loop below would wrap
            return inData;
        }
        std::span<const float> spectrum(inData.signal_values.data(), n);

        const peak::SubtractionShape shape = subtraction_shape;

        auto detected = detectPeaksIterative(spectrum, noise_rejection_threshold, min_amplitude, min_isolation, max_iterations, min_prominence, max_peaks, shape);

        auto [noiseFloor, noiseSigma] = peak::estimateMagnitudeNoise(spectrum);

        std::vector<float> prominenceCurve(n, 0.f);
        for (std::size_t i = 1; i < n - 1; ++i) {
            if (peak::isLocalMaximum(spectrum.data(), i, n)) {
                prominenceCurve[i] = peak::estimateProminence(spectrum, i) / (noiseSigma + 1e-10f);
            }
        }

        gr::DataSet<float> output;
        output.timestamp   = inData.timestamp;
        output.axis_names  = inData.axis_names;
        output.axis_units  = inData.axis_units;
        output.axis_values = inData.axis_values;

        output.signal_names      = {"Spectrum", "Prominence"};
        output.signal_quantities = {"", ""};
        output.signal_units      = {inData.signal_units.empty() ? "" : inData.signal_units[0], "sigma"};
        output.signal_ranges     = {gr::Range<float>{0.f, 0.f}, gr::Range<float>{0.f, 0.f}};
        output.extents           = {static_cast<std::int32_t>(n)};

        output.signal_values.resize(2 * n);
        std::copy_n(spectrum.begin(), n, output.signal_values.begin());
        std::copy(prominenceCurve.begin(), prominenceCurve.end(), output.signal_values.begin() + static_cast<std::ptrdiff_t>(n));

        output.meta_information = {{}, {}};

        output.timing_events.resize(2UZ);
        auto& peakEvents = output.timing_events[0];
        peakEvents.reserve(detected.size());
        std::ranges::transform(detected, std::back_inserter(peakEvents), [&](const peak::PeakResult& p) {
            gr::property_map props = peakEventProps(p, peak::NoiseEstimate{noiseFloor, noiseSigma});
            return gr::DataSet<float>::idx_pmt_map(std::clamp<long>(std::lround(p.centre), 0L, static_cast<long>(n) - 1L), std::move(props));
        });

        return output;
    }
};

} // namespace gr::blocks::fourier

#endif // GNURADIO_PEAK_DETECTOR_HPP
