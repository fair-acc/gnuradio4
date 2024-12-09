#ifndef GNURADIO_TESTING_TESTSPECTRUMGENERATOR_HPP
#define GNURADIO_TESTING_TESTSPECTRUMGENERATOR_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/algorithm/rng/Xoshiro256pp.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <string>
#include <vector>

namespace gr::testing {

GR_REGISTER_BLOCK(gr::testing::TestSpectrumGenerator, [T], [ float, double ]);

/**
 * @brief Procedural beam-spectrum generator for UI testing.
 *
 * Produces one DataSet<T> per input clock tick. The spectrum contains a configurable
 * noise floor, an evolving Schottky-like peak, a sweeping line, and morse-keyed
 * interference lines — all driven by a deterministic Xoshiro256++ PRNG for reproducibility.
 * Ported from the OpenDigitizer project (fair-acc/opendigitizer).
 */
template<typename T>
requires std::floating_point<T>
struct TestSpectrumGenerator : gr::Block<TestSpectrumGenerator<T>, gr::Resampling<1LU, 1LU>> {
    using Description = gr::Doc<"Procedural beam-spectrum generator for UI testing, driven by an upstream ClockSource.">;

    gr::PortIn<std::uint8_t>    in;
    gr::PortOut<gr::DataSet<T>> out;

    gr::Annotated<gr::Size_t, "spectrum_size", gr::Doc<"number of frequency bins">, gr::Visible>                 spectrum_size    = 4096U;
    gr::Annotated<T, "center_freq", gr::Unit<"Hz">>                                                              center_freq      = T(100e6);
    gr::Annotated<T, "signal_bandwidth", gr::Unit<"Hz">, gr::Doc<"total bandwidth of the generated spectrum">>   signal_bandwidth = T(1e6);
    gr::Annotated<T, "clock_rate", gr::Unit<"Hz">, gr::Doc<"rate of incoming clock ticks for time computation">> clock_rate       = T(25);
    gr::Annotated<std::uint64_t, "seed", gr::Doc<"RNG seed for reproducibility">>                                seed             = 42ULL;

    gr::Annotated<T, "active_duration", gr::Unit<"s">, gr::Doc<"active phase duration">>    active_duration = T(10);
    gr::Annotated<T, "pause_duration", gr::Unit<"s">, gr::Doc<"noise-only pause duration">> pause_duration  = T(1);

    gr::Annotated<T, "noise_floor_db", gr::Unit<"dB">, gr::Doc<"mean noise floor level">>         noise_floor_db  = T(-80);
    gr::Annotated<T, "noise_spread_db", gr::Unit<"dB">, gr::Doc<"Gaussian sigma of noise floor">> noise_spread_db = T(0.2);

    gr::Annotated<bool, "show_schottky", gr::Visible>                                                   show_schottky       = true;
    gr::Annotated<T, "initial_peak_db", gr::Unit<"dB">>                                                 initial_peak_db     = T(6);
    gr::Annotated<T, "initial_sigma", gr::Doc<"initial peak width as fraction of spectrum">>            initial_sigma       = T(0.1);
    gr::Annotated<T, "width_ratio", gr::Doc<"ratio of initial to final peak width (narrowing factor)">> width_ratio         = T(10);
    gr::Annotated<T, "freq_shift_fraction", gr::Doc<"upward frequency shift as fraction of spectrum">>  freq_shift_fraction = T(0.05);
    gr::Annotated<T, "freq_shift_tau", gr::Unit<"s">, gr::Doc<"exponential approach time constant">>    freq_shift_tau      = T(0.33);

    gr::Annotated<bool, "show_sweep_line", gr::Visible>                                                show_sweep_line = true;
    gr::Annotated<T, "sweep_start", gr::Doc<"sweep start position as fraction of spectrum [0,1]">>     sweep_start     = T(0.05);
    gr::Annotated<T, "sweep_stop", gr::Doc<"sweep stop position as fraction of spectrum [0,1]">>       sweep_stop      = T(0.3);
    gr::Annotated<T, "sweep_period", gr::Unit<"s">, gr::Doc<"time for one full back-and-forth cycle">> sweep_period    = T(4.0);

    gr::Annotated<bool, "show_interference_lines", gr::Visible>                                                                    show_interference_lines = true;
    gr::Annotated<T, "line_amplitude_db", gr::Unit<"dB">>                                                                          line_amplitude_db       = T(12);
    gr::Annotated<T, "line_sigma", gr::Doc<"interference line width as fraction of spectrum">>                                     line_sigma              = T(0.005);
    gr::Annotated<std::string, "morse_pattern", gr::Doc<"dots/dashes/spaces for third line keying (e.g. '.... . .-.. .-.. ---')">> morse_pattern           = std::string{".... . .-.. .-.. --- ..-. .- .. .-. -.-.--"};
    gr::Annotated<T, "morse_unit_duration", gr::Unit<"s">, gr::Doc<"time unit for morse keying on third line">>                    morse_unit_duration     = T(0.2);

    gr::Annotated<gr::Size_t, "log_interval", gr::Doc<"print debug info every N clock ticks (0 = off)">> log_interval = 0U;

    GR_MAKE_REFLECTABLE(TestSpectrumGenerator, in, out, spectrum_size, center_freq, signal_bandwidth, clock_rate, seed, active_duration, pause_duration, noise_floor_db, noise_spread_db, show_schottky, initial_peak_db, initial_sigma, width_ratio, freq_shift_fraction, freq_shift_tau, show_sweep_line, sweep_start, sweep_stop, sweep_period, show_interference_lines, line_amplitude_db, line_sigma, morse_pattern, morse_unit_duration, log_interval);

    gr::rng::Xoshiro256pp     _rng{seed};
    std::size_t               _sampleCount = 0;
    std::vector<std::uint8_t> _morseKey;

    static constexpr double                  kDbPerNeper    = 10.0 / std::numbers::ln10;
    static constexpr std::array<double, 3UZ> kLinePositions = {0.12, 0.25, 0.85};

    void start() {
        _rng         = gr::rng::Xoshiro256pp(seed);
        _sampleCount = 0;
        rebuildMorseKey();
    }

    void reset() {
        _rng         = gr::rng::Xoshiro256pp(seed);
        _sampleCount = 0;
        rebuildMorseKey();
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& newSettings) {
        if (newSettings.contains("seed")) {
            _rng = gr::rng::Xoshiro256pp(seed);
        }
        if (newSettings.contains("morse_pattern")) {
            rebuildMorseKey();
        }
    }

    [[nodiscard]] gr::work::Status processBulk(std::span<const std::uint8_t> input, std::span<gr::DataSet<T>> output) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = createSpectrum(static_cast<std::size_t>(spectrum_size));
            ++_sampleCount;
        }
        return gr::work::Status::OK;
    }

    [[nodiscard]] gr::DataSet<T> createSpectrum(std::size_t N) {
        gr::DataSet<T> ds{};
        ds.timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        const auto   fMin = static_cast<double>(center_freq) - static_cast<double>(signal_bandwidth) * 0.5;
        const auto   fMax = static_cast<double>(center_freq) + static_cast<double>(signal_bandwidth) * 0.5;
        const double df   = (fMax - fMin) / static_cast<double>(N);

        ds.axis_names = {"Frequency"};
        ds.axis_units = {"Hz"};
        ds.axis_values.resize(1);
        ds.axis_values[0].resize(N);
        for (std::size_t i = 0; i < N; ++i) {
            ds.axis_values[0][i] = static_cast<T>(fMin + static_cast<double>(i) * df);
        }

        ds.extents = {static_cast<std::int32_t>(N)};
        ds.layout  = gr::LayoutRight{};

        constexpr std::size_t nSignals = 1;
        ds.signal_names                = {"Magnitude"};
        ds.signal_quantities           = {"magnitude"};
        ds.signal_units                = {"dB"};
        ds.signal_values.resize(nSignals * N);
        ds.signal_ranges.resize(nSignals);

        auto magnitudes = ds.signalValues(0);
        generateSpectrum(magnitudes, N);

        const auto [minIt, maxIt] = std::ranges::minmax_element(magnitudes);
        ds.signal_ranges[0]       = {*minIt, *maxIt};

        ds.meta_information.resize(nSignals);
        ds.meta_information[0] = {
            {std::pmr::string("sample_rate"), gr::pmt::Value(static_cast<float>(signal_bandwidth))},
            {std::pmr::string("center_frequency"), gr::pmt::Value(static_cast<float>(center_freq))},
            {std::pmr::string("output_in_db"), gr::pmt::Value(true)},
            {std::pmr::string("clock_rate"), gr::pmt::Value(static_cast<float>(clock_rate))},
        };
        ds.timing_events.resize(nSignals);
        return ds;
    }

private:
    void generateSpectrum(std::span<T> bins, std::size_t N) {
        const double cycleDur  = static_cast<double>(active_duration) + static_cast<double>(pause_duration);
        const double elapsed   = static_cast<double>(_sampleCount) / static_cast<double>(clock_rate);
        const double cycleTime = std::fmod(elapsed, cycleDur);
        const bool   active    = cycleTime < static_cast<double>(active_duration);

        const auto noiseFloor  = static_cast<double>(noise_floor_db);
        const auto noiseSpread = static_cast<double>(noise_spread_db);

        for (std::size_t i = 0; i < N; ++i) {
            bins[i] = static_cast<T>(noiseFloor + _rng.triangularM11<double>() * noiseSpread);
        }

        if (active && show_schottky) {
            addSchottkyPeak(bins, N, cycleTime);
        }

        if (show_interference_lines) {
            const double lineAmp = noiseFloor + static_cast<double>(line_amplitude_db);
            for (std::size_t j = 0; j < kLinePositions.size(); ++j) {
                if (j == 2 && !isMorseKeyOn(elapsed)) {
                    continue;
                }
                addNarrowLine(bins, N, kLinePositions[j], lineAmp);
            }
        }

        if (show_sweep_line && active) {
            const double period   = std::max(static_cast<double>(sweep_period), 0.01);
            const double phase    = std::fmod(cycleTime / period, 1.0);
            const double triangle = 1.0 - std::abs(2.0 * phase - 1.0);
            const auto   lo       = static_cast<double>(sweep_start);
            const auto   hi       = static_cast<double>(sweep_stop);
            const double sweepPos = lo + triangle * (hi - lo);
            addNarrowLine(bins, N, sweepPos, noiseFloor + static_cast<double>(line_amplitude_db) + 3.0);
        }
    }

    void addSchottkyPeak(std::span<T> bins, std::size_t N, double t) {
        const double sigmaRel      = static_cast<double>(initial_sigma) * std::pow(static_cast<double>(width_ratio), -t / static_cast<double>(active_duration));
        const double sigma         = sigmaRel * static_cast<double>(N);
        const double invTwoSigmaSq = 1.0 / (2.0 * sigma * sigma);
        const double peakDb        = static_cast<double>(noise_floor_db) + static_cast<double>(initial_peak_db) + t;
        const double shift         = static_cast<double>(freq_shift_fraction) * (1.0 - std::exp(-t / static_cast<double>(freq_shift_tau)));
        const double centerBin     = (0.5 + shift) * static_cast<double>(N);

        for (std::size_t i = 0; i < N; ++i) {
            const double dist     = static_cast<double>(i) - centerBin;
            const double signalDb = peakDb - dist * dist * invTwoSigmaSq * kDbPerNeper;
            bins[i]               = static_cast<T>(std::max(static_cast<double>(bins[i]), signalDb));
        }
    }

    void addNarrowLine(std::span<T> bins, std::size_t N, double position, double amplitudeDb) {
        const double sigma         = static_cast<double>(line_sigma) * static_cast<double>(N);
        const double centerBin     = position * static_cast<double>(N);
        const double invTwoSigmaSq = 1.0 / (2.0 * sigma * sigma);
        const auto   iMin          = static_cast<std::size_t>(std::max(0.0, centerBin - 5.0 * sigma));
        const auto   iMax          = std::min(N, static_cast<std::size_t>(centerBin + 5.0 * sigma + 1.0));

        for (std::size_t i = iMin; i < iMax; ++i) {
            const double dist     = static_cast<double>(i) - centerBin;
            const double signalDb = amplitudeDb - dist * dist * invTwoSigmaSq * kDbPerNeper;
            bins[i]               = static_cast<T>(std::max(static_cast<double>(bins[i]), signalDb));
        }
    }

    void rebuildMorseKey() {
        _morseKey.clear();
        const auto& pat = morse_pattern.value;
        for (std::size_t i = 0; i < pat.size(); ++i) {
            const char c = pat[i];
            if (c == '.') {
                _morseKey.push_back(1);
                _morseKey.push_back(0);
            } else if (c == '-') {
                _morseKey.insert(_morseKey.end(), 3, 1);
                _morseKey.push_back(0);
            } else if (c == ' ') {
                _morseKey.insert(_morseKey.end(), 2, 0);
            }
        }
        if (_morseKey.empty()) {
            _morseKey.push_back(1);
        } else {
            _morseKey.insert(_morseKey.end(), 6, 0);
        }
    }

    [[nodiscard]] bool isMorseKeyOn(double elapsedSeconds) const {
        if (_morseKey.empty()) {
            return true;
        }
        const double unitDur    = std::max(static_cast<double>(morse_unit_duration), 0.01);
        const double patternDur = static_cast<double>(_morseKey.size()) * unitDur;
        const double t          = std::fmod(elapsedSeconds, patternDur);
        const auto   idx        = static_cast<std::size_t>(t / unitDur) % _morseKey.size();
        return _morseKey[idx] != 0;
    }
};

} // namespace gr::testing

#endif // GNURADIO_TESTING_TESTSPECTRUMGENERATOR_HPP
