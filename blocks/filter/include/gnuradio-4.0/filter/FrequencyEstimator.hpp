// FrequencyEstimator.hpp
#ifndef FREQUENCY_ESTIMATOR_HPP
#define FREQUENCY_ESTIMATOR_HPP

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/HistoryBuffer.hpp>
#include <gnuradio-4.0/algorithm/filter/FilterTool.hpp>
#include <gnuradio-4.0/filter/time_domain_filter.hpp>

#include <gnuradio-4.0/algorithm/fourier/fft.hpp>
#include <gnuradio-4.0/algorithm/fourier/fft_common.hpp>
#include <gnuradio-4.0/algorithm/fourier/fftw.hpp>
#include <gnuradio-4.0/algorithm/fourier/window.hpp>

namespace gr::filter {

enum class EstimatorMethod {
    Fast,              /// Simple and fast method with 2nd-order IIR low-pass filter
    Fast_NO_LP,        /// Simple and fast method without IIR low-pass filter
    Gaussian,          /// Gaussian interpolation (Gasior & Gonzalez)
    GaussianDecimating /// Gaussian interpolation (Gasior & Gonzalez) w/ stride+decimation
};

template<typename T, EstimatorMethod algorithm = EstimatorMethod::Fast>
requires std::floating_point<T>
struct FrequencyEstimator : Block<FrequencyEstimator<T, algorithm>, void> {
    using Description = Doc<R""(@brief Frequency Estimator

This block estimates the frequency of a signal using the algorithms described in:
  [0] SimpleFast & SimpleFastWithIIR
      Mariusz Krajewski, Sergiusz Sienkowski, Wiesław Miczulski,
      "A simple and fast algorithm for measuring power system frequency",
      Measurement, Volume 201, 2022,
      https://doi.org/10.1016/j.measurement.2022.111673
  [1] Gaussian, GaussianDecimating
       M. Gasior, J.L. Gonzalez, Improving FFT frequency measurement resolution by
       parabolic and gaussian spectrum interpolation, AIP Conf. Proc. 732 (2004) 276,
       https://doi.org/10.1063/1.1831158.
)"">;
    PortIn<T>  in;
    PortOut<T> out;

    // settings
    Annotated<float, "sample rate", Doc<"signal sample rate">, Unit<"Hz">>            sample_rate{1000};
    Annotated<float, "f_expected", Doc<"expected likely frequency">, Unit<"Hz">>      f_expected{50};
    Annotated<float, "f_min", Doc<"exp. min frequency range">, Unit<"Hz">>            f_min{40};
    Annotated<float, "f_max", Doc<"exp. max/LP frequency (-1: disable)">, Unit<"Hz">> f_max{60};
    Annotated<gr::Size_t, "n periods rate", Doc<"number of periods to average over">> n_periods{4};
    Annotated<gr::Size_t, "min FFT size", Doc<"min of periods to average over">>      min_fft_size{256U};
    Annotated<T, "epsilon", Doc<"numerical error threshold">>                         epsilon{T(1e-8)};

    GR_MAKE_REFLECTABLE(FrequencyEstimator, in, out, sample_rate, f_expected, f_min, f_max, n_periods, min_fft_size, epsilon);

    // private internal state
    T          _prevFrequency{50.0};   // previous frequency value for continuity
    gr::Size_t _n_period_estimate{60}; // number of samples for estimation period according to [0]

    FilterCoefficients<T> _singleFilterSection;
    HistoryBuffer<T>      _inputHistory{32UZ};
    HistoryBuffer<T>      _outputHistory{32UZ};

    gr::algorithm::FFT<T, std::complex<T>> _fftImpl{};
    std::vector<T>                         _inData            = std::vector<T>(min_fft_size.value, T(0));
    std::vector<T>                         _window            = gr::algorithm::window::create<T>(gr::algorithm::window::Type::Hann, 32UZ);
    std::vector<std::complex<T>>           _outData           = std::vector<std::complex<T>>(min_fft_size.value, T(0));
    std::vector<T>                         _magnitudeSpectrum = std::vector<T>(min_fft_size.value / 2UZ, T(0));

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& newSettings) {
        if (newSettings.contains("Ns") || newSettings.contains("sample_rate") || newSettings.contains("f_expected") || newSettings.contains("f_min") || newSettings.contains("f_max")) {
            if (f_min < 0 || f_max >= sample_rate || f_expected < 0 || f_expected >= sample_rate) {
                throw gr::exception(fmt::format("ill-formed block parameters: f_min: {} < f_expected: {} < f_max: {} < sample_rate: {} (N.B. f_max < 0 -> disable low-pass)", f_min, f_expected, f_max, sample_rate));
            }
            initialiseFilter();
        }
    }

    void initialiseFilter() {
        // calculate filter coefficients, regarding slightly different implementation choices:
        // * IIR over FIR: reduced numerical complexity and minimises group-delay
        // * BESSEL: optimises linearity in the pass-band and in turn frequency accuracy
        _n_period_estimate = std::size_t(n_periods * (f_min > 0 ? sample_rate / std::min(f_min.value, f_expected.value) : sample_rate / f_expected.value));
        using namespace gr::filter::iir;
        _singleFilterSection = iir::designFilter<T, 0UZ>(Type::LOWPASS, FilterParameters{.order = 2UZ, .fLow = f_max, .fs = sample_rate}, Design::BESSEL);
        _inputHistory        = HistoryBuffer<T>(std::bit_ceil(_singleFilterSection.b.size()));
        _outputHistory       = HistoryBuffer<T>(std::bit_ceil(std::max(_singleFilterSection.a.size(), std::size_t(_n_period_estimate))));

        const std::size_t minFFT = std::bit_ceil(std::max(std::size_t(min_fft_size.value), _outputHistory.capacity()));
        _inData.resize(minFFT, T(0));
        _outData.resize(minFFT, std::complex<T>(T(0)));
        _window            = gr::algorithm::window::create<T>(gr::algorithm::window::Type::Hann, _outputHistory.capacity());
        _magnitudeSpectrum = std::vector<T>(min_fft_size.value / 2UZ, T(0));
    }

    void reset() {
        _prevFrequency = f_expected;
        initialiseFilter();
    }

    [[nodiscard]] T processOne(T input) noexcept {
        if constexpr (algorithm == EstimatorMethod::Fast) {
            _inputHistory.push_back(input);
            const T output = std::inner_product(_singleFilterSection.b.cbegin(), _singleFilterSection.b.cend(), _inputHistory.cbegin(), static_cast<T>(0))         // feed-forward path
                             - std::inner_product(_singleFilterSection.a.cbegin() + 1, _singleFilterSection.a.cend(), _outputHistory.cbegin(), static_cast<T>(0)); // feedback path
            _outputHistory.push_back(output);
            _prevFrequency = estimateFrequency();
        } else if constexpr (algorithm == EstimatorMethod::Fast_NO_LP) {
            _outputHistory.push_back(input);
            _prevFrequency = estimateFrequency();
        } else if constexpr (algorithm == EstimatorMethod::Gaussian) {
            _outputHistory.push_back(input);
            _prevFrequency = estimateFrequencyFFT();
        } else {
            static_assert(gr::meta::always_false<T>, "algorithm not implemented");
        }

        return _prevFrequency;
    }

private:
    T estimateFrequency() noexcept {
        if (_outputHistory.size() < _n_period_estimate) {
            return _prevFrequency; // return previous frequency during settling time
        }

        // implement reference algorithm from [0] (fast, low-group delay)
        // Step 3: compute accumulator A, B, and C
        T A = 0;
        T B = 0;
        T C = 0;

        auto         bufferStart = _outputHistory.begin();
        std::span<T> data(bufferStart, std::next(bufferStart, _n_period_estimate));
        for (std::size_t n = 1UZ; n < _n_period_estimate - 1UZ; ++n) {
            T sumNeighbours = data[n - 1] + data[n + 1];
            T denominator   = T(4) * data[n];

            // avoid division by zero or very small values
            if (std::abs(denominator) < epsilon) {
                continue;
            }

            // equations (6)
            T a_n = (sumNeighbours * sumNeighbours) / denominator;
            T b_n = data[n];

            // equations (8)
            A += a_n * a_n;
            B += b_n * b_n;
            C += T(2) * a_n * b_n;
        }

        if (B <= epsilon) {
            return _prevFrequency; // return previous frequency if B is too small
        }

        // Step 4: Calculate z and ensure it's within [-1, 1]
        T z = (C / B) - T(1);
        if (z >= T(1) || z <= -T(1)) {
            return _prevFrequency; // Return previous frequency if conditions are not met
        }

        // frequency estimate according to equations (12)
        T f = sample_rate / (T(4) * std::numbers::pi_v<T>)*std::acos(z);
        return f;
    }

    T estimateFrequencyFFT() {
        // implement reference algorithm from [1] (precise, medium group-delay, stride- and decimation compatible)
        using namespace gr::algorithm::fft;

        auto         bufferStart = _outputHistory.begin();
        std::span<T> data(bufferStart, std::next(bufferStart, _n_period_estimate));

        std::ranges::fill(_inData, T(0)); // zero padding
        // apply window function
        for (std::size_t i = 0UZ; i < data.size(); i++) {
            _inData[i] = data[i] * _window[i];
        }

        // compute FFT and magnitude spectrum
        _magnitudeSpectrum = computeMagnitudeSpectrum(_fftImpl.compute(_inData, _outData), _magnitudeSpectrum, ConfigMagnitude{.computeHalfSpectrum = true, .outputInDb = false});

        std::size_t i_min = static_cast<std::size_t>(std::floor((f_min / sample_rate) * _magnitudeSpectrum.size() * 2UZ));
        std::size_t i_max = static_cast<std::size_t>(std::ceil((f_max / sample_rate) * _magnitudeSpectrum.size() * 2UZ));

        // ensure indices are within bounds
        i_min = std::clamp(i_min, std::size_t(1), _magnitudeSpectrum.size() - 1UZ);
        i_max = std::clamp(i_max, std::size_t(1), _magnitudeSpectrum.size() - 1UZ);

        // find the index of the maximum peak in the magnitude spectrum within the [i_min, i_max] range
        auto        it_max = std::max_element(_magnitudeSpectrum.begin() + i_min, _magnitudeSpectrum.begin() + i_max);
        std::size_t k_max  = std::distance(_magnitudeSpectrum.begin(), it_max);

        // ensure the peak is not at the edges to allow interpolation with neighbours
        if (k_max == 0 || k_max >= _magnitudeSpectrum.size() - 1) {
            return _prevFrequency; // Return previous frequency if peak is at the edges
        }

        // ensure magnitudes are positive
        if (_magnitudeSpectrum[k_max - 1] <= T(0) || _magnitudeSpectrum[k_max] <= T(0) || _magnitudeSpectrum[k_max + 1] <= T(0)) {
            return _prevFrequency; // cannot compute logarithm
        }
        T log_S_km1 = std::log(_magnitudeSpectrum[k_max - 1]);
        T log_S_k   = std::log(_magnitudeSpectrum[k_max]);
        T log_S_kp1 = std::log(_magnitudeSpectrum[k_max + 1]);

        // Compute numerator and denominator
        const T denominator = 2 * log_S_k - log_S_km1 - log_S_kp1;
        if (denominator == T(0) || !std::isfinite(denominator)) {
            return _prevFrequency;
        }

        // Gaussian interpolation according to equation (21)
        const T delta_k = T(0.5) * (log_S_kp1 - log_S_km1) / denominator;

        // check if delta_k is within valid range
        if (!std::isfinite(delta_k) || std::abs(delta_k) >= T(1)) {
            return _prevFrequency;
        }

        // calculate the interpolated frequency
        T interpolated_freq_bin = static_cast<T>(k_max) + delta_k;
        T interpolated_freq     = interpolated_freq_bin * sample_rate / static_cast<T>(_inData.size());

        // update _prevFrequency and return the interpolated frequency
        _prevFrequency = interpolated_freq;
        return interpolated_freq;
    }
};

template<typename T>
using BasicFrequencyEstimator = FrequencyEstimator<T, EstimatorMethod::Fast>;

template<typename T>
using FastFrequencyEstimator = FrequencyEstimator<T, EstimatorMethod::Fast_NO_LP>;

template<typename T>
using FFTFrequencyEstimator = FrequencyEstimator<T, EstimatorMethod::Gaussian>;

template<typename T>
using DecimatingFFTFrequencyEstimator = FrequencyEstimator<T, EstimatorMethod::GaussianDecimating>;

} // namespace gr::filter

inline auto registerFrequencyEstimators = gr::registerBlock<gr::filter::BasicFrequencyEstimator, float, double>(gr::globalBlockRegistry())  //
                                          + gr::registerBlock<gr::filter::FastFrequencyEstimator, float, double>(gr::globalBlockRegistry()) //
                                          + gr::registerBlock<gr::filter::FFTFrequencyEstimator, float, double>(gr::globalBlockRegistry())  //
                                          + gr::registerBlock<gr::filter::DecimatingFFTFrequencyEstimator, float, double>(gr::globalBlockRegistry());

#endif // FREQUENCY_ESTIMATOR_HPP
