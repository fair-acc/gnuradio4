#ifndef GNURADIO_TESTING_SYNTHETICPEAKSPECTRUM_HPP
#define GNURADIO_TESTING_SYNTHETICPEAKSPECTRUM_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/algorithm/rng/GaussianNoise.hpp>
#include <gnuradio-4.0/algorithm/rng/Xoshiro256pp.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>
#include <string>
#include <vector>

namespace gr::testing {

GR_REGISTER_BLOCK(gr::testing::SyntheticPeakSpectrum, [T], [ float, double ]);

template<typename T>
requires std::floating_point<T>
struct SyntheticPeakSpectrum : gr::Block<SyntheticPeakSpectrum<T>, gr::Resampling<1LU, 1LU>> {
    using Description = gr::Doc<R""(Synthetic peak spectrum generator matching the Python training pipeline's peak shapes and noise model.

Produces one DataSet<T> per input tick: a spectrum with a random number of peaks (Gaussian,
Lorentzian, asymmetric Gaussian, sinc-squared, parabolic, pseudo-Voigt, or dual Gaussian), a
baseline and realistic noise, so C++ unit tests can generate reproducible test spectra without
external files. Ground-truth peak parameters are stored in timing_events for downstream
validation.)"">;

    gr::PortIn<std::uint8_t>    in;
    gr::PortOut<gr::DataSet<T>> out;

    gr::Annotated<gr::Size_t, "spectrum_size", gr::Doc<"number of frequency bins">, gr::Visible>       spectrum_size       = 1024U;
    gr::Annotated<gr::Size_t, "max_peaks", gr::Doc<"maximum number of peaks per spectrum">>            max_peaks           = 8U;
    gr::Annotated<T, "snr_min_db", gr::Unit<"dB">, gr::Doc<"minimum peak SNR">>                        snr_min_db          = T(6);
    gr::Annotated<T, "snr_max_db", gr::Unit<"dB">, gr::Doc<"maximum peak SNR">>                        snr_max_db          = T(40);
    gr::Annotated<T, "noise_level", gr::Doc<"noise standard deviation relative to signal peak">>       noise_level         = T(1);
    gr::Annotated<T, "peak_width_min_bins", gr::Doc<"minimum peak width in bins">>                     peak_width_min_bins = T(1);
    gr::Annotated<T, "peak_width_max_bins", gr::Doc<"widest physically occurring feature, in bins">>   peak_width_max_bins = T(250);
    gr::Annotated<T, "width_tail_exponent", gr::Doc<"higher concentrates width near the minimum">>     width_tail_exponent = T(4);
    gr::Annotated<T, "dc_feature_prob", gr::Doc<"probability of a DC offset feature at the low bins">> dc_feature_prob     = T(0.5);
    gr::Annotated<T, "dc_amp_max", gr::Doc<"maximum DC feature amplitude">>                            dc_amp_max          = T(1);
    gr::Annotated<T, "dc_decay_min_bins", gr::Doc<"minimum DC leakage decay constant, in bins">>       dc_decay_min_bins   = T(2);
    gr::Annotated<T, "dc_decay_max_bins", gr::Doc<"maximum DC leakage decay constant, in bins">>       dc_decay_max_bins   = T(20);
    gr::Annotated<bool, "add_coloured_noise", gr::Doc<"add low-frequency coloured noise component">>   add_coloured_noise  = true;
    gr::Annotated<bool, "add_edge_effects", gr::Doc<"add edge variance boost (FFT artefact)">>         add_edge_effects    = true;
    gr::Annotated<bool, "add_spurious_spikes", gr::Doc<"add random spurious spikes">>                  add_spurious_spikes = true;
    gr::Annotated<std::uint64_t, "seed", gr::Doc<"RNG seed for reproducibility">>                      seed                = 42ULL;

    GR_MAKE_REFLECTABLE(SyntheticPeakSpectrum, in, out, spectrum_size, max_peaks, snr_min_db, snr_max_db, noise_level, peak_width_min_bins, peak_width_max_bins, width_tail_exponent, dc_feature_prob, dc_amp_max, dc_decay_min_bins, dc_decay_max_bins, add_coloured_noise, add_edge_effects, add_spurious_spikes, seed);

    gr::rng::Xoshiro256pp _rng{seed};
    std::size_t           _sampleCount = 0;

    void start() {
        _rng         = gr::rng::Xoshiro256pp(seed);
        _sampleCount = 0;
    }

    void reset() {
        _rng         = gr::rng::Xoshiro256pp(seed);
        _sampleCount = 0;
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& newSettings) {
        if (newSettings.contains("seed")) {
            _rng = gr::rng::Xoshiro256pp(seed);
        }
    }

    [[nodiscard]] gr::work::Status processBulk(std::span<const std::uint8_t> input, std::span<gr::DataSet<T>> output) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = createSpectrum();
            ++_sampleCount;
        }
        return gr::work::Status::OK;
    }

private:
    enum class PeakShape : std::uint8_t { gaussian, asymmetricGaussian, lorentzian, sinc2, parabolic, pseudoVoigt, dualGaussian };
    static constexpr std::size_t kNumShapes       = 7;
    static constexpr T           kMaxLogAsymmetry = T(0.6931472); // ratio spans [0.5, 2]

    static constexpr std::array<const char*, kNumShapes> kShapeNames = {"gaussian", "asymmetric_gaussian", "lorentzian", "sinc2", "parabolic", "pseudo_voigt", "dual_gaussian"};

    T uniform(T lo, T hi) { return lo + (hi - lo) * _rng.uniform01<T>(); }

    std::size_t uniformInt(std::size_t lo, std::size_t hi) { return lo + static_cast<std::size_t>(_rng.uniform01<double>() * static_cast<double>(hi - lo)); }

    [[nodiscard]] gr::DataSet<T> createSpectrum() {
        const auto N = static_cast<std::size_t>(spectrum_size);

        gr::DataSet<T> ds{};
        ds.timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        ds.axis_names = {"Frequency"};
        ds.axis_units = {"bin"};
        ds.axis_values.resize(1);
        ds.axis_values[0].resize(N);
        std::iota(ds.axis_values[0].begin(), ds.axis_values[0].end(), T(0));

        ds.extents = {static_cast<std::int32_t>(N)};
        ds.layout  = gr::LayoutRight{};

        ds.signal_names      = {"Magnitude"};
        ds.signal_quantities = {"magnitude"};
        ds.signal_units      = {"a.u."};
        ds.signal_values.resize(N);
        ds.signal_ranges.resize(1);

        auto spectrum = ds.signalValues(0);

        generateBaseline(spectrum, N);
        addNoise(spectrum, N);

        const std::size_t             nPeaks = uniformInt(0, static_cast<std::size_t>(max_peaks) + 1);
        std::vector<gr::property_map> peakEvents;

        for (std::size_t p = 0; p < nPeaks; ++p) {
            const T snrDb     = uniform(snr_min_db, snr_max_db);
            const T amplitude = std::pow(T(10), snrDb / T(20));
            const T widthBins = sampleWidthBins();
            // a peak sits anywhere in the band and may be only partly visible at an edge
            const T centre = uniform(T(0), static_cast<T>(N));

            // below ~5 bins only the near-delta shapes are physically distinguishable
            const PeakShape shapeIdx = (widthBins <= T(5)) ? ((_rng.uniform01<T>() < T(0.5)) ? PeakShape::gaussian : PeakShape::sinc2) //
                                                           : static_cast<PeakShape>(uniformInt(0, kNumShapes));

            std::vector<T> solo(N, T(0));
            const T        shapeScale = addPeak(std::span<T>(solo), N, centre, widthBins, amplitude, shapeIdx);
            const auto     widths     = measureHalfMaxWidths(std::span<const T>(solo), centre);
            std::ranges::transform(spectrum, solo, spectrum.begin(), std::plus<T>{});

            peakEvents.push_back({
                {std::pmr::string("centre"), gr::pmt::Value(static_cast<float>(centre))},
                {std::pmr::string("amplitude"), gr::pmt::Value(static_cast<float>(amplitude))},
                {std::pmr::string("fwhm"), gr::pmt::Value(static_cast<float>(widths.fwhm))},
                {std::pmr::string("hwhm_l"), gr::pmt::Value(static_cast<float>(widths.hwhmLeft))},
                {std::pmr::string("hwhm_r"), gr::pmt::Value(static_cast<float>(widths.hwhmRight))},
                // the shape's own parameterisation scale: a hint for guessing the shape, NOT a
                // width and not to be used for evaluation or training
                {std::pmr::string("shape_scale"), gr::pmt::Value(static_cast<float>(shapeScale))},
                {std::pmr::string("snr_db"), gr::pmt::Value(static_cast<float>(snrDb))},
                {std::pmr::string("shape"), gr::pmt::Value(std::pmr::string(kShapeNames[static_cast<std::size_t>(shapeIdx)]))},
            });
        }

        // |X(f)| is non-negative by construction; folding also sets the noise pedestal to 0.798*sigma
        std::ranges::transform(spectrum, spectrum.begin(), [](T v) { return std::abs(v); });

        const auto [minIt, maxIt] = std::ranges::minmax_element(spectrum);
        ds.signal_ranges[0]       = {*minIt, *maxIt};

        ds.meta_information.resize(1);
        ds.meta_information[0] = {
            {std::pmr::string("peak_count"), gr::pmt::Value(static_cast<std::int32_t>(nPeaks))},
            {std::pmr::string("seed"), gr::pmt::Value(static_cast<std::int64_t>(seed.value))},
            {std::pmr::string("sample_index"), gr::pmt::Value(static_cast<std::int64_t>(_sampleCount))},
        };

        ds.timing_events.resize(1);
        for (auto& pe : peakEvents) {
            const auto centreVal = pe.value_or<float>("centre", 0.0f);
            ds.timing_events[0].emplace_back(static_cast<std::ptrdiff_t>(centreVal), std::move(pe));
        }

        return ds;
    }

    /// log-spaced width with the mass at the narrow end: broad features are physical but rare
    [[nodiscard]] T sampleWidthBins() {
        const T span = peak_width_max_bins / peak_width_min_bins;
        return peak_width_min_bins * std::pow(span, std::pow(_rng.uniform01<T>(), width_tail_exponent));
    }

    /// An offset appears as a DC component in the lowest bins (bin 0 plus spectral leakage) and
    /// decays away from DC. A spectrum carries no broad-band offset other than its noise.
    void generateBaseline(std::span<T> spectrum, std::size_t N) {
        std::ranges::fill(spectrum, T(0));
        if (_rng.uniform01<T>() >= dc_feature_prob) {
            return;
        }
        const T amplitude = uniform(T(0.1), dc_amp_max);
        const T decay     = uniform(dc_decay_min_bins, dc_decay_max_bins);
        for (std::size_t i = 0; i < N; ++i) {
            spectrum[i] = amplitude * std::exp(-static_cast<T>(i) / decay);
        }
    }

    void addNoise(std::span<T> spectrum, std::size_t N) {
        gr::rng::GaussianNoise<T> gauss(_rng);
        for (std::size_t i = 0; i < N; ++i) {
            spectrum[i] += noise_level * gauss();
        }

        if (add_coloured_noise && _rng.uniform01<T>() < T(0.5)) {
            const T                   colourScale = uniform(T(0.1), T(0.5));
            std::vector<T>            coloured(N);
            gr::rng::GaussianNoise<T> gauss2(_rng);
            for (auto& v : coloured) {
                v = gauss2();
            }
            boxBlur(coloured, N / 20);
            const T colourStd = standardDeviation(coloured);
            if (colourStd > T(1e-10)) {
                for (std::size_t i = 0; i < N; ++i) {
                    spectrum[i] += colourScale * coloured[i] / colourStd;
                }
            }
        }

        if (add_edge_effects && _rng.uniform01<T>() < T(0.7)) {
            const auto edgeWidth = N / 10;
            const T    edgeBoost = uniform(T(1.5), T(3.0));
            for (std::size_t i = 0; i < edgeWidth; ++i) {
                const T taper = edgeBoost + (T(1) - edgeBoost) * static_cast<T>(i) / static_cast<T>(edgeWidth);
                spectrum[i] *= taper;
            }
            for (std::size_t i = 0; i < edgeWidth; ++i) {
                const T taper = T(1) + (edgeBoost - T(1)) * static_cast<T>(i) / static_cast<T>(edgeWidth);
                spectrum[N - edgeWidth + i] *= taper;
            }
        }

        if (add_spurious_spikes && _rng.uniform01<T>() < T(0.3)) {
            const std::size_t nSpikes = uniformInt(1, 5);
            for (std::size_t s = 0; s < nSpikes; ++s) {
                const auto spikePos = uniformInt(0, N);
                const T    spikeAmp = uniform(T(2), T(5));
                spectrum[spikePos] += (_rng.uniform01<T>() > T(0.5) ? spikeAmp : -spikeAmp);
            }
        }
    }

    struct PeakWidths {
        T fwhm;
        T hwhmLeft;
        T hwhmRight;
    };

    // Measured on the noiseless profile of this peak alone, so it is the true half-maximum width of
    // what was actually drawn rather than a per-shape parameterisation constant. The renderers
    // consume RNG (asymmetry ratio, dual-Gaussian separation), so the peak is drawn once into a
    // scratch buffer, measured, then added — re-rendering would produce a different peak.
    [[nodiscard]] static PeakWidths measureHalfMaxWidths(std::span<const T> profile, T centre) {
        const auto n       = static_cast<std::ptrdiff_t>(profile.size());
        const auto peakIdx = std::clamp(static_cast<std::ptrdiff_t>(std::lround(centre)), std::ptrdiff_t{0}, n - 1);
        const T    peakVal = profile[static_cast<std::size_t>(peakIdx)];
        if (peakVal <= T(0)) {
            return {T(0), T(0), T(0)};
        }
        const T half = peakVal * T(0.5);

        const auto crossing = [&](std::ptrdiff_t step) -> T {
            for (std::ptrdiff_t i = peakIdx; i >= 0 && i < n; i += step) {
                if (profile[static_cast<std::size_t>(i)] <= half) {
                    const auto prev = i - step;
                    const T    a    = profile[static_cast<std::size_t>(prev)];
                    const T    b    = profile[static_cast<std::size_t>(i)];
                    const T    frac = (a > b) ? (a - half) / (a - b) : T(0); // sub-bin crossing
                    return std::abs(static_cast<T>(prev) + frac * static_cast<T>(step) - centre);
                }
            }
            return std::abs(static_cast<T>(step > 0 ? n - 1 : 0) - centre); // truncated at the edge
        };
        const T left  = crossing(-1);
        const T right = crossing(+1);
        return {left + right, left, right};
    }

    T addPeak(std::span<T> spectrum, std::size_t N, T centre, T width, T amplitude, PeakShape shape) {
        const T shapeScale = std::max(width, T(1));
        switch (shape) {
        case PeakShape::gaussian: return addGaussian(spectrum, N, centre, shapeScale, amplitude);
        case PeakShape::asymmetricGaussian: return addAsymmetricGaussian(spectrum, N, centre, shapeScale, amplitude);
        case PeakShape::lorentzian: return addLorentzian(spectrum, N, centre, shapeScale, amplitude);
        case PeakShape::sinc2: return addSinc2(spectrum, N, centre, shapeScale * T(2), amplitude);
        case PeakShape::parabolic: return addParabolic(spectrum, N, centre, shapeScale * T(1.5), amplitude);
        case PeakShape::pseudoVoigt: return addPseudoVoigt(spectrum, N, centre, shapeScale, amplitude);
        case PeakShape::dualGaussian: return addDualGaussian(spectrum, N, centre, shapeScale, amplitude);
        }
        std::unreachable();
    }

    T addGaussian(std::span<T> spectrum, std::size_t N, T centre, T sigma, T amplitude) {
        const T invTwoSigmaSq = T(1) / (T(2) * sigma * sigma);
        for (std::size_t i = 0; i < N; ++i) {
            const T d = static_cast<T>(i) - centre;
            spectrum[i] += amplitude * std::exp(-d * d * invTwoSigmaSq);
        }
        return sigma;
    }

    T addAsymmetricGaussian(std::span<T> spectrum, std::size_t N, T centre, T sigma, T amplitude) {
        // log-symmetric ratio: drawing sigmaL below sigma and sigmaR above made every asymmetric
        // peak right-wider, which a model then learns as a handedness prior
        const T ratio          = std::exp(uniform(-kMaxLogAsymmetry, kMaxLogAsymmetry));
        const T sigmaL         = sigma * std::sqrt(ratio);
        const T sigmaR         = sigma / std::sqrt(ratio);
        const T invTwoSigmaLSq = T(1) / (T(2) * sigmaL * sigmaL);
        const T invTwoSigmaRSq = T(1) / (T(2) * sigmaR * sigmaR);
        for (std::size_t i = 0; i < N; ++i) {
            const T d   = static_cast<T>(i) - centre;
            const T inv = (d <= T(0)) ? invTwoSigmaLSq : invTwoSigmaRSq;
            spectrum[i] += amplitude * std::exp(-d * d * inv);
        }
        return (sigmaL + sigmaR) / T(2);
    }

    T addLorentzian(std::span<T> spectrum, std::size_t N, T centre, T gamma, T amplitude) {
        const T gammaSq = gamma * gamma;
        for (std::size_t i = 0; i < N; ++i) {
            const T d = static_cast<T>(i) - centre;
            spectrum[i] += amplitude * gammaSq / (gammaSq + d * d);
        }
        return gamma;
    }

    T addSinc2(std::span<T> spectrum, std::size_t N, T centre, T width, T amplitude) {
        const T safeWidth = std::max(width, T(1));
        for (std::size_t i = 0; i < N; ++i) {
            const T x = std::numbers::pi_v<T> * (static_cast<T>(i) - centre) / safeWidth;
            T       s = (std::abs(x) < T(1e-10)) ? T(1) : std::sin(x) / x;
            spectrum[i] += amplitude * s * s;
        }
        return safeWidth / T(2);
    }

    T addParabolic(std::span<T> spectrum, std::size_t N, T centre, T width, T amplitude) {
        const T safeWidth = std::max(width, T(1));
        for (std::size_t i = 0; i < N; ++i) {
            const T x = (static_cast<T>(i) - centre) / safeWidth;
            spectrum[i] += amplitude * std::max(T(0), T(1) - x * x);
        }
        return safeWidth;
    }

    T addPseudoVoigt(std::span<T> spectrum, std::size_t N, T centre, T sigma, T amplitude) {
        // pseudo-Voigt: η * Lorentzian + (1-η) * Gaussian, η ∈ [0.3, 0.7]
        const T eta           = uniform(T(0.3), T(0.7));
        const T gamma         = sigma;
        const T invTwoSigmaSq = T(1) / (T(2) * sigma * sigma);
        const T gammaSq       = gamma * gamma;
        for (std::size_t i = 0; i < N; ++i) {
            const T d        = static_cast<T>(i) - centre;
            const T gaussVal = std::exp(-d * d * invTwoSigmaSq);
            const T lorenVal = gammaSq / (gammaSq + d * d);
            spectrum[i] += amplitude * (eta * lorenVal + (T(1) - eta) * gaussVal);
        }
        return sigma;
    }

    T addDualGaussian(std::span<T> spectrum, std::size_t N, T centre, T sigma, T amplitude) {
        const T sigma1 = sigma * T(0.6);
        const T sigma2 = sigma * T(0.8);
        const T sep    = sigma * uniform(T(0.3), T(1.2));
        const T ratio  = uniform(T(0.3), T(1.0));

        std::vector<T> combined(N, T(0));
        T              peak = T(0);

        const T invTwoS1Sq = T(1) / (T(2) * sigma1 * sigma1);
        const T invTwoS2Sq = T(1) / (T(2) * sigma2 * sigma2);
        for (std::size_t i = 0; i < N; ++i) {
            const T d1  = static_cast<T>(i) - (centre - sep / T(2));
            const T d2  = static_cast<T>(i) - (centre + sep / T(2));
            combined[i] = std::exp(-d1 * d1 * invTwoS1Sq) + ratio * std::exp(-d2 * d2 * invTwoS2Sq);
            peak        = std::max(peak, combined[i]);
        }
        if (peak > T(1e-12)) {
            for (std::size_t i = 0; i < N; ++i) {
                spectrum[i] += amplitude * combined[i] / peak;
            }
        }
        return (sigma1 + sigma2) / T(2);
    }

    static void boxBlur(std::vector<T>& data, std::size_t radius) {
        if (radius == 0 || data.size() < 2 * radius + 1) {
            return;
        }
        const std::size_t N = data.size();
        std::vector<T>    temp(N);
        const T           invWindow = T(1) / static_cast<T>(2 * radius + 1);
        T                 sum       = T(0);
        for (std::size_t i = 0; i < std::min(2 * radius + 1, N); ++i) {
            sum += data[i];
        }
        for (std::size_t i = 0; i < N; ++i) {
            temp[i]           = sum * invWindow;
            const auto addIdx = i + radius + 1;
            const auto remIdx = static_cast<std::ptrdiff_t>(i) - static_cast<std::ptrdiff_t>(radius);
            if (addIdx < N) {
                sum += data[addIdx];
            }
            if (remIdx >= 0) {
                sum -= data[static_cast<std::size_t>(remIdx)];
            }
        }
        data = std::move(temp);
    }

    static T standardDeviation(const std::vector<T>& data) {
        if (data.size() < 2) {
            return T(0);
        }
        const T mean     = std::accumulate(data.begin(), data.end(), T(0)) / static_cast<T>(data.size());
        T       sumSqDev = T(0);
        for (const auto& v : data) {
            const T d = v - mean;
            sumSqDev += d * d;
        }
        return std::sqrt(sumSqDev / static_cast<T>(data.size()));
    }
};

} // namespace gr::testing

#endif // GNURADIO_TESTING_SYNTHETICPEAKSPECTRUM_HPP
