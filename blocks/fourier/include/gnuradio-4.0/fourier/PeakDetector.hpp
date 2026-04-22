#ifndef GNURADIO_PEAK_DETECTOR_HPP
#define GNURADIO_PEAK_DETECTOR_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/fourier/PeakResult.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <vector>

namespace gr::blocks::fourier {

enum class SubtractionShape { Auto, Gaussian, Lorentzian, Voigt };

struct WidthEstimate {
    float left;
    float right;
};

[[nodiscard]] inline WidthEstimate estimateWidth(std::span<const float> data, std::size_t peakIdx, float baseline = 0.f) {
    const float peakVal = data[peakIdx];
    const auto  n       = data.size();
    const float halfMax = baseline + (peakVal - baseline) * 0.5f;

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

[[nodiscard]] inline float interpolatePeakPosition(std::span<const float> data, std::size_t peakIdx) {
    if (peakIdx == 0 || peakIdx >= data.size() - 1) {
        return static_cast<float>(peakIdx);
    }
    float yL    = data[peakIdx - 1];
    float yC    = data[peakIdx];
    float yR    = data[peakIdx + 1];
    float denom = 2.f * (2.f * yC - yL - yR);
    if (std::abs(denom) < 1e-10f) {
        return static_cast<float>(peakIdx);
    }
    float offset = (yL - yR) / denom;
    return static_cast<float>(peakIdx) + std::clamp(offset, -0.5f, 0.5f);
}

[[nodiscard]] inline float estimateExcessKurtosis(std::span<const float> data, std::size_t peakIdx, float sigma) {
    const auto  n    = data.size();
    std::size_t win  = static_cast<std::size_t>(std::max(3.f, 3.f * sigma));
    std::size_t wLo  = (peakIdx > win) ? peakIdx - win : 0;
    std::size_t wHi  = std::min(peakIdx + win + 1, n);
    std::size_t wLen = wHi - wLo;
    if (wLen < 4) {
        return 0.f;
    }

    float mean = 0.f, m2 = 0.f, m4 = 0.f;
    for (std::size_t j = wLo; j < wHi; ++j) {
        mean += data[j];
    }
    mean /= static_cast<float>(wLen);
    for (std::size_t j = wLo; j < wHi; ++j) {
        float d = data[j] - mean;
        m2 += d * d;
        m4 += d * d * d * d;
    }
    m2 /= static_cast<float>(wLen);
    m4 /= static_cast<float>(wLen);
    float kurt = (m2 > 1e-10f) ? m4 / (m2 * m2) : 3.f;
    return kurt - 3.f;
}

// generate a peak shape for subtraction
inline void subtractPeak(std::vector<float>& spectrum, float centre, float amplitude, float sigmaL, float sigmaR, float excessKurtosis, SubtractionShape shape) {
    const auto n = spectrum.size();

    // auto-select shape from kurtosis
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
        float x     = static_cast<float>(i) - centre;
        float sigma = (x <= 0.f) ? sigmaL : sigmaR;
        if (sigma < 0.5f) {
            sigma = 0.5f;
        }

        float value = 0.f;
        switch (shape) {
        case SubtractionShape::Gaussian: value = amplitude * std::exp(-0.5f * x * x / (sigma * sigma)); break;
        case SubtractionShape::Lorentzian: value = amplitude * sigma * sigma / (sigma * sigma + x * x); break;
        case SubtractionShape::Voigt: {
            // pseudo-Voigt: linear mix of Gaussian and Lorentzian
            float g   = std::exp(-0.5f * x * x / (sigma * sigma));
            float l   = sigma * sigma / (sigma * sigma + x * x);
            float eta = std::clamp(excessKurtosis / 3.f, 0.f, 1.f);
            value     = amplitude * ((1.f - eta) * g + eta * l);
            break;
        }
        default: value = amplitude * std::exp(-0.5f * x * x / (sigma * sigma));
        }
        spectrum[i] -= value;
    }
}

// iterative peak stripping: detect → fit → subtract → repeat. Threshold defaults mirror
// PeakDetector::noise_rejection_threshold / min_prominence -- see the rationale there.
[[nodiscard]] inline std::vector<PeakResult> detectPeaksIterative(std::span<const float> spectrum, float noiseRejectionThreshold = 5.0f, float minAmplitude = 0.0f, float minIsolationFraction = 0.5f, std::size_t maxIterations = 10, float minProminence = 5.0f, std::size_t maxPeaks = 10, SubtractionShape shape = SubtractionShape::Auto) {

    const std::size_t n = spectrum.size();
    if (n < 3) {
        return {};
    }

    auto [noiseFloor, noiseSigma] = estimateNoise(spectrum);

    // work on a mutable copy (the residual)
    std::vector<float> residual(spectrum.begin(), spectrum.end());

    std::vector<PeakResult> peaks;
    peaks.reserve(maxPeaks);

    for (std::size_t iter = 0; iter < maxIterations && peaks.size() < maxPeaks; ++iter) {
        // re-estimate noise on residual (improves after each subtraction)
        auto [residualFloor, residualSigma] = estimateNoise(std::span<const float>(residual));
        float currentThreshold              = residualFloor + noiseRejectionThreshold * residualSigma;

        std::size_t bestIdx  = 0;
        float       bestProm = 0.f;

        for (std::size_t i = 1; i < n - 1; ++i) {
            if (residual[i] > residual[i - 1] && residual[i] > residual[i + 1] && residual[i] >= currentThreshold) {
                float prom = estimateProminence(std::span<const float>(residual), i);
                if (prom > bestProm) {
                    bestProm = prom;
                    bestIdx  = i;
                }
            }
        }

        if (bestProm < minProminence * residualSigma) {
            break;
        }

        float peakAmp = residual[bestIdx] - residualFloor;
        if (peakAmp < minAmplitude) {
            break;
        }

        // sub-bin centre
        float centre = interpolatePeakPosition(std::span<const float>(residual), bestIdx);

        // asymmetric width from residual
        auto [halfLeft, halfRight] = estimateWidth(std::span<const float>(residual), bestIdx, residualFloor);
        halfLeft                   = std::max(halfLeft, 0.5f);
        halfRight                  = std::max(halfRight, 0.5f);
        float sigma                = (halfLeft + halfRight) * 0.5f;

        // isolation (from original spectrum, not residual)
        float iso = estimateIsolation(spectrum, bestIdx);

        // min isolation check
        if (iso < minIsolationFraction * sigma * 2.f && !peaks.empty()) {
            // too close to an already-detected peak — skip and suppress this region
            subtractPeak(residual, centre, peakAmp, halfLeft, halfRight, 0.f, shape);
            continue;
        }

        float excessKurt = estimateExcessKurtosis(std::span<const float>(residual), bestIdx, sigma);

        // uncertainty estimates
        auto [posUnc, widthUnc, ampUnc] = estimateUncertainty(noiseSigma, peakAmp, sigma);

        // measured amplitude from original spectrum
        std::size_t rawIdx  = static_cast<std::size_t>(std::clamp(std::lround(centre), 0L, static_cast<long>(n - 1)));
        float       ampMeas = spectrum[rawIdx] - noiseFloor;

        peaks.push_back({
            .confidence           = bestProm / (noiseSigma + 1e-10f),
            .centre               = centre,
            .amplitude            = peakAmp,
            .amplitudeMeasured    = ampMeas,
            .sigmaLeft            = halfLeft,
            .sigmaRight           = halfRight,
            .w68                  = sigma * 2.f,
            .w96                  = sigma * 4.f,
            .w99                  = sigma * 6.f,
            .kurtosis             = excessKurt,
            .prominence           = bestProm / (noiseSigma + 1e-10f),
            .isolation            = iso,
            .positionUncertainty  = posUnc,
            .widthUncertainty     = widthUnc,
            .amplitudeUncertainty = ampUnc,
        });

        // subtract the detected peak from the residual
        subtractPeak(residual, centre, peakAmp, halfLeft, halfRight, excessKurt, shape);
    }

    // sort by centre position ascending
    std::ranges::sort(peaks, [](const PeakResult& a, const PeakResult& b) { return a.centre < b.centre; });

    return peaks;
}

GR_REGISTER_BLOCK(gr::blocks::fourier::PeakDetector)

struct PeakDetector : gr::Block<PeakDetector> {
    using Description = Doc<"Classical iterative peak detector using prominence-based stripping with adaptive shape subtraction.">;

    gr::PortIn<gr::DataSet<float>>  in;
    gr::PortOut<gr::DataSet<float>> out;

    // over N~1024 noisy bins, order statistics put ~23 excursions above 2 sigma per spectrum
    // (N*erfc(k/sqrt(2))/2) -- a 2 sigma default self-triggers on pure noise. Measured on
    // gr::testing::SyntheticPeakSpectrum's zero-peak validation seeds, false positives persist up
    // to ~4.5 sigma (edge-tapered bins run up to 3x the global noise estimate there); 5 sigma is the
    // lowest round threshold with a measured zero false positives on those seeds -- do not lower it
    // without re-measuring on a zero-peak spectrum.
    Annotated<float, "noise rejection threshold"> noise_rejection_threshold = 5.0f;
    Annotated<float, "min amplitude">             min_amplitude             = 0.0f;
    Annotated<float, "min isolation fraction">    min_isolation             = 0.5f;
    Annotated<gr::Size_t, "max iterations">       max_iterations            = 10U;
    Annotated<float, "min prominence">            min_prominence            = 5.0f;
    Annotated<gr::Size_t, "max peaks">            max_peaks                 = 10U;
    Annotated<int, "subtraction shape">           subtraction_shape         = 0; // 0=Auto, 1=Gaussian, 2=Lorentzian, 3=Voigt

    GR_MAKE_REFLECTABLE(PeakDetector, in, out, noise_rejection_threshold, min_amplitude, min_isolation, max_iterations, min_prominence, max_peaks, subtraction_shape);

    // not noexcept: the detection pipeline allocates and bad_alloc must propagate to the framework
    [[nodiscard]] gr::DataSet<float> processOne(gr::DataSet<float> inData) {
        if (inData.signal_values.empty()) {
            return inData;
        }

        const std::size_t      nSignals = std::max(1UZ, inData.signal_names.size());
        const std::size_t      n        = inData.signal_values.size() / nSignals;
        std::span<const float> spectrum(inData.signal_values.data(), n);

        auto shape = static_cast<SubtractionShape>(std::clamp(static_cast<int>(subtraction_shape), 0, 3));

        auto detected = detectPeaksIterative(spectrum, noise_rejection_threshold, min_amplitude, min_isolation, max_iterations, min_prominence, max_peaks, shape);

        auto [noiseFloor, noiseSigma] = estimateNoise(spectrum);

        // prominence curve for visualisation
        std::vector<float> prominenceCurve(n, 0.f);
        for (std::size_t i = 1; i < n - 1; ++i) {
            if (spectrum[i] > spectrum[i - 1] && spectrum[i] > spectrum[i + 1]) {
                prominenceCurve[i] = estimateProminence(spectrum, i) / (noiseSigma + 1e-10f);
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

        std::vector<gr::DataSet<float>::idx_pmt_map> peakEvents;
        peakEvents.reserve(detected.size());
        std::ranges::transform(detected, std::back_inserter(peakEvents), [&](const PeakResult& p) {
            gr::property_map props{
                {"confidence"_spmr, p.confidence},
                // fractional centre and mean sigma mirror the OnnxPeakDetector event keys, so
                // consumers can compare both detectors at sub-bin resolution
                {"centre"_spmr, p.centre},
                {"sigma"_spmr, p.sigma()},
                {"sigma_left"_spmr, p.sigmaLeft},
                {"sigma_right"_spmr, p.sigmaRight},
                {"amplitude"_spmr, p.amplitude},
                {"amplitude_measured"_spmr, p.amplitudeMeasured},
                {"prominence"_spmr, p.prominence},
                {"isolation"_spmr, p.isolation},
                {"w68"_spmr, p.w68},
                {"w96"_spmr, p.w96},
                {"w99"_spmr, p.w99},
                {"kurtosis"_spmr, p.kurtosis},
                {"noise_sigma"_spmr, noiseSigma},
                {"noise_floor"_spmr, noiseFloor},
                {"position_uncertainty"_spmr, p.positionUncertainty},
                {"width_uncertainty"_spmr, p.widthUncertainty},
                {"amplitude_uncertainty"_spmr, p.amplitudeUncertainty},
            };
            return gr::DataSet<float>::idx_pmt_map(std::lround(p.centre), std::move(props));
        });
        output.timing_events = {std::move(peakEvents), {}};

        return output;
    }
};

} // namespace gr::blocks::fourier

#endif // GNURADIO_PEAK_DETECTOR_HPP
