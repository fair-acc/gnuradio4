#ifndef GNURADIO_TESTING_EVOLVINGPEAKSPECTRUM_HPP
#define GNURADIO_TESTING_EVOLVINGPEAKSPECTRUM_HPP

#include <gnuradio-4.0/Block.hpp>
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

GR_REGISTER_BLOCK(gr::testing::EvolvingPeakSpectrum, [T], [ float, double ]);

enum class TagMode : std::uint8_t {
    transitions,  // emit tags only on peak_start / peak_end
    everySpectrum // emit current state of all active peaks every spectrum
};

enum class PeakPhase : std::uint8_t { onset, steady, decay };

/**
 * @brief Generates spectra with peaks that appear, evolve, and disappear over time.
 *
 * Each peak follows a lifecycle: onset (ramp-up) → steady (full amplitude) → decay (ramp-down).
 * Onset and decay can be instant (1 spectrum), linear, or exponential. Peaks may also drift
 * in frequency during their lifecycle. Ground-truth parameters are emitted as timing_events
 * for downstream validation of detection latency and accuracy.
 */
template<typename T>
requires std::floating_point<T>
struct EvolvingPeakSpectrum : gr::Block<EvolvingPeakSpectrum<T>, gr::Resampling<1LU, 1LU>> {
    using Description = gr::Doc<"Spectrum generator with peaks that fade in/out and drift, for detection latency testing.">;

    gr::PortIn<std::uint8_t>    in;
    gr::PortOut<gr::DataSet<T>> out;

    gr::Annotated<gr::Size_t, "spectrum_size", gr::Doc<"number of frequency bins">, gr::Visible>           spectrum_size          = 1024U;
    gr::Annotated<T, "noise_level", gr::Doc<"noise standard deviation">>                                   noise_level            = T(1);
    gr::Annotated<std::uint64_t, "seed", gr::Doc<"RNG seed for reproducibility">>                          seed                   = 42ULL;
    gr::Annotated<gr::Size_t, "max_concurrent_peaks">                                                      max_concurrent_peaks   = 5U;
    gr::Annotated<gr::Size_t, "min_onset_spectra", gr::Doc<"1 = instant onset">>                           min_onset_spectra      = 1U;
    gr::Annotated<gr::Size_t, "max_onset_spectra">                                                         max_onset_spectra      = 30U;
    gr::Annotated<gr::Size_t, "min_steady_spectra">                                                        min_steady_spectra     = 20U;
    gr::Annotated<gr::Size_t, "max_steady_spectra">                                                        max_steady_spectra     = 80U;
    gr::Annotated<gr::Size_t, "min_decay_spectra", gr::Doc<"1 = instant decay">>                           min_decay_spectra      = 1U;
    gr::Annotated<gr::Size_t, "max_decay_spectra">                                                         max_decay_spectra      = 30U;
    gr::Annotated<T, "snr_min_db", gr::Unit<"dB">>                                                         snr_min_db             = T(6);
    gr::Annotated<T, "snr_max_db", gr::Unit<"dB">>                                                         snr_max_db             = T(40);
    gr::Annotated<T, "peak_spawn_probability", gr::Doc<"probability of spawning a new peak per spectrum">> peak_spawn_probability = T(0.1);
    gr::Annotated<T, "max_drift_rate", gr::Doc<"maximum frequency drift in bins/spectrum">>                max_drift_rate         = T(0.5);
    gr::Annotated<TagMode, "tag_mode", gr::Doc<"when to emit peak tags">>                                  tag_mode               = TagMode::everySpectrum;

    GR_MAKE_REFLECTABLE(EvolvingPeakSpectrum, in, out, spectrum_size, noise_level, seed, max_concurrent_peaks, min_onset_spectra, max_onset_spectra, min_steady_spectra, max_steady_spectra, min_decay_spectra, max_decay_spectra, snr_min_db, snr_max_db, peak_spawn_probability, max_drift_rate, tag_mode);

    enum class PeakShape : std::uint8_t { gaussian, lorentzian, asymmetricGaussian };
    static constexpr std::size_t                         kNumShapes  = 3;
    static constexpr std::array<const char*, kNumShapes> kShapeNames = {"gaussian", "lorentzian", "asymmetric_gaussian"};

    struct PeakState {
        std::uint32_t id;
        PeakPhase     phase;
        T             center;
        T             targetAmplitude;
        T             sigma;
        T             asymmetry; // ratio sigmaRight / sigmaLeft (1.0 = symmetric)
        PeakShape     shape;
        T             snrDb;
        T             driftRate; // bins per spectrum
        std::size_t   phaseCounter;
        std::size_t   onsetDuration;
        std::size_t   steadyDuration;
        std::size_t   decayDuration;

        [[nodiscard]] T currentAmplitude() const {
            T progress = T(0);
            switch (phase) {
            case PeakPhase::onset: progress = (onsetDuration <= 1) ? T(1) : static_cast<T>(phaseCounter) / static_cast<T>(onsetDuration - 1); return targetAmplitude * progress;
            case PeakPhase::steady: return targetAmplitude;
            case PeakPhase::decay: progress = (decayDuration <= 1) ? T(1) : static_cast<T>(phaseCounter) / static_cast<T>(decayDuration - 1); return targetAmplitude * (T(1) - progress);
            }
            std::unreachable();
        }
    };

    std::vector<PeakState> _activePeaks;
    std::uint32_t          _nextPeakId = 0;
    gr::rng::Xoshiro256pp  _rng{seed};                   // noise + peak parameter RNG
    gr::rng::Xoshiro256pp  _spawnRng{seed.value + 7919}; // separate RNG for spawn decisions (decoupled from noise)
    std::size_t            _sampleCount = 0;

    void start() {
        _rng         = gr::rng::Xoshiro256pp(seed);
        _spawnRng    = gr::rng::Xoshiro256pp(seed.value + 7919);
        _sampleCount = 0;
        _nextPeakId  = 0;
        _activePeaks.clear();
    }

    void reset() { start(); }

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
    T uniform(T lo, T hi) { return lo + (hi - lo) * _rng.template uniform01<T>(); }

    std::size_t uniformInt(std::size_t lo, std::size_t hi) {
        if (lo >= hi) {
            return lo;
        }
        return lo + static_cast<std::size_t>(_rng.template uniform01<double>() * static_cast<double>(hi - lo));
    }

    [[nodiscard]] gr::DataSet<T> createSpectrum() {
        const auto N = static_cast<std::size_t>(spectrum_size);

        // attempt to spawn a peak before building the spectrum (independent of noise RNG)
        maybeSpawnPeak(N);

        gr::DataSet<T> ds{};
        ds.timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        ds.axis_names = {"Frequency"};
        ds.axis_units = {"bin"};
        ds.axis_values.resize(1);
        ds.axis_values[0].resize(N);
        std::iota(ds.axis_values[0].begin(), ds.axis_values[0].end(), T(0));

        ds.extents           = {static_cast<std::int32_t>(N)};
        ds.layout            = gr::LayoutRight{};
        ds.signal_names      = {"Magnitude"};
        ds.signal_quantities = {"magnitude"};
        ds.signal_units      = {"a.u."};
        ds.signal_values.resize(N, T(0));
        ds.signal_ranges.resize(1);

        auto spectrum = ds.signalValues(0);

        addNoise(spectrum, N);

        std::vector<gr::property_map> peakEvents;
        for (const auto& peak : _activePeaks) {
            T amp = peak.currentAmplitude();
            if (amp > T(1e-6)) {
                addPeak(spectrum, N, peak.center, peak.sigma, amp, peak.shape, peak.asymmetry);
            }

            bool emitTag = (tag_mode.value == TagMode::everySpectrum) || (peak.phaseCounter == 0);
            if (emitTag) {
                std::string event = (peak.phase == PeakPhase::onset && peak.phaseCounter == 0) ? "peak_start" : (peak.phase == PeakPhase::decay && peak.phaseCounter == 0) ? "peak_decay_start" : "peak_active";
                peakEvents.push_back({
                    {std::pmr::string("event"), gr::pmt::Value(std::pmr::string(event))},
                    {std::pmr::string("peak_id"), gr::pmt::Value(static_cast<std::int32_t>(peak.id))},
                    {std::pmr::string("center"), gr::pmt::Value(static_cast<float>(peak.center))},
                    {std::pmr::string("amplitude"), gr::pmt::Value(static_cast<float>(amp))},
                    {std::pmr::string("target_amplitude"), gr::pmt::Value(static_cast<float>(peak.targetAmplitude))},
                    {std::pmr::string("sigma"), gr::pmt::Value(static_cast<float>(peak.sigma))},
                    {std::pmr::string("snr_db"), gr::pmt::Value(static_cast<float>(peak.snrDb))},
                    {std::pmr::string("shape"), gr::pmt::Value(std::pmr::string(kShapeNames[static_cast<std::size_t>(peak.shape)]))},
                    {std::pmr::string("asymmetry"), gr::pmt::Value(static_cast<float>(peak.asymmetry))},
                    {std::pmr::string("phase"), gr::pmt::Value(std::pmr::string(peak.phase == PeakPhase::onset    ? "onset"
                                                                                : peak.phase == PeakPhase::steady ? "steady"
                                                                                                                  : "decay"))},
                    {std::pmr::string("drift_rate"), gr::pmt::Value(static_cast<float>(peak.driftRate))},
                });
            }
        }

        advancePeaks();
        removeDeadPeaks();

        const auto [minIt, maxIt] = std::ranges::minmax_element(spectrum);
        ds.signal_ranges[0]       = {*minIt, *maxIt};

        ds.meta_information.resize(1);
        ds.meta_information[0] = {
            {std::pmr::string("active_peaks"), gr::pmt::Value(static_cast<std::int32_t>(_activePeaks.size()))},
            {std::pmr::string("seed"), gr::pmt::Value(static_cast<std::int64_t>(seed.value))},
            {std::pmr::string("sample_index"), gr::pmt::Value(static_cast<std::int64_t>(_sampleCount))},
        };

        ds.timing_events.resize(1);
        for (auto& pe : peakEvents) {
            const auto centerVal = pe.at(std::pmr::string("center")).template value_or<float>(0.0f);
            ds.timing_events[0].emplace_back(static_cast<std::ptrdiff_t>(centerVal), std::move(pe));
        }

        return ds;
    }

    void maybeSpawnPeak(std::size_t N) {
        if (_activePeaks.size() >= static_cast<std::size_t>(max_concurrent_peaks.value)) {
            return;
        }
        if (_spawnRng.template uniform01<T>() >= static_cast<T>(peak_spawn_probability.value)) {
            return;
        }

        const T   edgeMargin = T(0.05) * static_cast<T>(N);
        const T   center     = uniform(edgeMargin, static_cast<T>(N) - edgeMargin);
        const T   snrDb      = uniform(snr_min_db, snr_max_db);
        const T   amplitude  = std::pow(T(10), snrDb / T(20));
        const T   sigma      = uniform(T(2), T(0.1) * static_cast<T>(N));
        const T   asymmetry  = uniform(T(0.5), T(2.0));
        const T   drift      = uniform(-max_drift_rate, max_drift_rate);
        PeakShape shape      = static_cast<PeakShape>(uniformInt(0, kNumShapes));

        _activePeaks.push_back({
            .id              = _nextPeakId++,
            .phase           = PeakPhase::onset,
            .center          = center,
            .targetAmplitude = amplitude,
            .sigma           = sigma,
            .asymmetry       = (shape == PeakShape::asymmetricGaussian) ? asymmetry : T(1),
            .shape           = shape,
            .snrDb           = snrDb,
            .driftRate       = drift,
            .phaseCounter    = 0,
            .onsetDuration   = uniformInt(static_cast<std::size_t>(min_onset_spectra), static_cast<std::size_t>(max_onset_spectra) + 1),
            .steadyDuration  = uniformInt(static_cast<std::size_t>(min_steady_spectra), static_cast<std::size_t>(max_steady_spectra) + 1),
            .decayDuration   = uniformInt(static_cast<std::size_t>(min_decay_spectra), static_cast<std::size_t>(max_decay_spectra) + 1),
        });
    }

    void advancePeaks() {
        for (auto& peak : _activePeaks) {
            peak.center += peak.driftRate;
            ++peak.phaseCounter;

            switch (peak.phase) {
            case PeakPhase::onset:
                if (peak.phaseCounter >= peak.onsetDuration) {
                    peak.phase        = PeakPhase::steady;
                    peak.phaseCounter = 0;
                }
                break;
            case PeakPhase::steady:
                if (peak.phaseCounter >= peak.steadyDuration) {
                    peak.phase        = PeakPhase::decay;
                    peak.phaseCounter = 0;
                }
                break;
            case PeakPhase::decay: break;
            }
        }
    }

    void removeDeadPeaks() {
        std::erase_if(_activePeaks, [](const PeakState& p) { return p.phase == PeakPhase::decay && p.phaseCounter >= p.decayDuration; });
    }

    void addNoise(std::span<T> spectrum, std::size_t N) {
        gr::rng::GaussianNoise<T> gauss(_rng);
        for (std::size_t i = 0; i < N; ++i) {
            spectrum[i] += noise_level * gauss();
        }
    }

    void addPeak(std::span<T> spectrum, std::size_t N, T center, T sigma, T amplitude, PeakShape shape, T asymmetry) {
        switch (shape) {
        case PeakShape::gaussian: {
            const T invTwoSigmaSq = T(1) / (T(2) * sigma * sigma);
            for (std::size_t i = 0; i < N; ++i) {
                const T d = static_cast<T>(i) - center;
                spectrum[i] += amplitude * std::exp(-d * d * invTwoSigmaSq);
            }
            break;
        }
        case PeakShape::lorentzian: {
            const T gammaSq = sigma * sigma;
            for (std::size_t i = 0; i < N; ++i) {
                const T d = static_cast<T>(i) - center;
                spectrum[i] += amplitude * gammaSq / (gammaSq + d * d);
            }
            break;
        }
        case PeakShape::asymmetricGaussian: {
            const T sigmaL         = sigma / std::sqrt(asymmetry);
            const T sigmaR         = sigma * std::sqrt(asymmetry);
            const T invTwoSigmaLSq = T(1) / (T(2) * sigmaL * sigmaL);
            const T invTwoSigmaRSq = T(1) / (T(2) * sigmaR * sigmaR);
            for (std::size_t i = 0; i < N; ++i) {
                const T d   = static_cast<T>(i) - center;
                const T inv = (d <= T(0)) ? invTwoSigmaLSq : invTwoSigmaRSq;
                spectrum[i] += amplitude * std::exp(-d * d * inv);
            }
            break;
        }
        }
    }
};

} // namespace gr::testing

#endif // GNURADIO_TESTING_EVOLVINGPEAKSPECTRUM_HPP
