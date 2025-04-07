#ifndef FREQUENCY_ESTIMATOR_HPP
#define FREQUENCY_ESTIMATOR_HPP

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/HistoryBuffer.hpp>
#include <gnuradio-4.0/filter/time_domain_filter.hpp>

#include <gnuradio-4.0/algorithm/filter/FilterTool.hpp>
#include <gnuradio-4.0/algorithm/fourier/fft.hpp>
#include <gnuradio-4.0/algorithm/fourier/fft_common.hpp>
#include <gnuradio-4.0/algorithm/fourier/fftw.hpp>
#include <gnuradio-4.0/algorithm/fourier/window.hpp>

namespace gr::filter {

GR_REGISTER_BLOCK("gr::filter::FrequencyEstimatorTimeDomain", gr::filter::FrequencyEstimatorTimeDomain, [T], [ float, double ])
GR_REGISTER_BLOCK("gr::filter::FrequencyEstimatorTimeDomainDecimating", gr::filter::FrequencyEstimatorTimeDomain, ([T], gr::Resampling<10U>), [ float, double ])

template<typename T, typename... Args>
requires std::floating_point<T>
struct FrequencyEstimatorTimeDomain : Block<FrequencyEstimatorTimeDomain<T, Args...>, Args...> {
    using Description = Doc<R""(@brief Time Domain Frequency Estimator

This block estimates the frequency of a signal using the time-domain algorithm described in:
  [0] Mariusz Krajewski, Sergiusz Sienkowski, Wiesław Miczulski,
      "A simple and fast algorithm for measuring power system frequency",
      Measurement, Volume 201, 2022,
      https://doi.org/10.1016/j.measurement.2022.111673
)"">;
    using TParent     = Block<FrequencyEstimatorTimeDomain<T>, Args...>;

    PortIn<T>  in;
    PortOut<T> out;

    // settings
    Annotated<float, "sample rate", Doc<"signal sample rate">, Unit<"Hz">>            sample_rate{1.f};
    Annotated<float, "f_min", Doc<"exp. min frequency range">, Unit<"Hz">>            f_min{40};
    Annotated<float, "f_expected", Doc<"expected likely frequency">, Unit<"Hz">>      f_expected{50};
    Annotated<float, "f_max", Doc<"exp. max/LP frequency (-1: disable)">, Unit<"Hz">> f_max{60};
    Annotated<gr::Size_t, "n periods rate", Doc<"number of periods to average over">> n_periods{4};
    Annotated<T, "epsilon", Doc<"numerical error threshold">>                         epsilon{T(1e-8)};

    GR_MAKE_REFLECTABLE(FrequencyEstimatorTimeDomain, in, out, sample_rate, f_expected, f_min, f_max, n_periods, epsilon);

    // private internal state
    T          _prevFrequency{50.0};   // previous frequency value for continuity
    gr::Size_t _n_period_estimate{60}; // number of samples for estimation period according to [0]

    FilterCoefficients<T> _singleFilterSection;
    HistoryBuffer<T>      _inputHistory{32UZ};
    HistoryBuffer<T>      _outputHistory{32UZ};

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& newSettings) {
        if (newSettings.contains("n_periods") || newSettings.contains("sample_rate") || newSettings.contains("f_expected") || newSettings.contains("f_min") || newSettings.contains("f_max")) {
            if (f_min < 0 || f_max >= sample_rate || f_expected < 0 || f_expected >= sample_rate) {
                throw gr::exception(fmt::format("Ill-formed block parameters: f_min: {} < f_expected: {} < f_max: {} < sample_rate: {} (N.B. f_max < 0 -> disable low-pass)", f_min, f_expected, f_max, sample_rate));
            }
            initialiseFilter();
        }
    }

    void initialiseFilter() {
        // Calculate filter coefficients
        // * IIR over FIR: reduced numerical complexity and minimizes group delay
        // * BESSEL: optimizes phase linearity in the pass-band and in turn frequency accuracy
        _n_period_estimate = n_periods * static_cast<gr::Size_t>(f_min > 0 ? sample_rate / std::min(f_min.value, f_expected.value) : sample_rate / f_expected.value);
        using namespace gr::filter::iir;
        _singleFilterSection = iir::designFilter<T, 0UZ>(Type::LOWPASS, FilterParameters{.order = 2UZ, .fLow = static_cast<double>(f_max), .fs = static_cast<double>(sample_rate)}, Design::BESSEL);
        _inputHistory        = HistoryBuffer<T>(std::bit_ceil(_singleFilterSection.b.size()));
        _outputHistory       = HistoryBuffer<T>(std::bit_ceil(std::max(_singleFilterSection.a.size(), std::size_t(_n_period_estimate))));
    }

    void reset() {
        _prevFrequency = static_cast<T>(f_expected);
        initialiseFilter();
    }

    [[nodiscard]] constexpr T processOne(T input) noexcept
    requires(TParent::ResamplingControl::kIsConst)
    {
        // process input sample through the IIR filter
        _inputHistory.push_front(input);
        const T output = std::inner_product(_singleFilterSection.b.cbegin(), _singleFilterSection.b.cend(), _inputHistory.cbegin(), static_cast<T>(0))         // feed-forward
                         - std::inner_product(_singleFilterSection.a.cbegin() + 1, _singleFilterSection.a.cend(), _outputHistory.cbegin(), static_cast<T>(0)); // feed-back
        _outputHistory.push_front(output);
        _prevFrequency = estimateFrequency();
        return _prevFrequency;
    }

    [[nodiscard]] constexpr work::Status processBulk(std::span<const T> input, std::span<T> output) noexcept
    requires(not TParent::ResamplingControl::kIsConst)
    {
        const std::size_t num_chunks = input.size() / std::size_t(this->input_chunk_size);
        if (output.size() < num_chunks) {
            return work::Status::INSUFFICIENT_OUTPUT_ITEMS;
        }

        auto output_it = output.begin();
        for (std::size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
            const std::size_t  offset = chunk_idx * this->input_chunk_size;
            std::span<const T> chunk  = input.subspan(offset, this->input_chunk_size);

            for (const T& sample : chunk) {
                // process input sample through the IIR filter
                _inputHistory.push_front(sample);

                const T output_sample = std::inner_product(_singleFilterSection.b.cbegin(), _singleFilterSection.b.cend(), _inputHistory.cbegin(), static_cast<T>(0))         // feed-forward
                                        - std::inner_product(_singleFilterSection.a.cbegin() + 1, _singleFilterSection.a.cend(), _outputHistory.cbegin(), static_cast<T>(0)); // feed-back

                _outputHistory.push_front(output_sample);
            }

            _prevFrequency = estimateFrequency();
            *output_it++   = _prevFrequency;
        }

        return work::Status::OK;
    }

private:
    T estimateFrequency() noexcept {
        if (_outputHistory.size() < _n_period_estimate) {
            return _prevFrequency; // Return previous frequency during settling time
        }

        // implement reference algorithm from [0] (fast, low-group delay)
        // Step 3: compute accumulator A, B, and C
        T accA = 0;
        T accB = 0;
        T accC = 0;

        auto         bufferStart = _outputHistory.begin();
        std::span<T> data(bufferStart, std::next(bufferStart, _n_period_estimate));
        for (std::size_t n = 1UZ; n < _n_period_estimate - 1UZ; ++n) {
            T sumNeighbours = data[n - 1] + data[n + 1];
            T denominator   = T(4) * data[n];

            // Avoid division by zero or very small values
            if (std::abs(denominator) < epsilon) {
                continue;
            }

            // Equations (6)
            T a_n = (sumNeighbours * sumNeighbours) / denominator;
            T b_n = data[n];

            // Equations (8)
            accA += a_n * a_n;
            accB += b_n * b_n;
            accC += T(2) * a_n * b_n;
        }

        if (accB <= epsilon) {
            return _prevFrequency; // Return previous frequency if B is too small
        }

        // Step 4: Calculate z and ensure it's within [-1, 1]
        T z = (accC / accB) - T(1);
        if (z >= T(1) || z <= -T(1)) {
            return _prevFrequency; // Return previous frequency if conditions are not met
        }

        // Frequency estimate according to equations (12)
        T f = static_cast<T>(sample_rate) / (T(4) * std::numbers::pi_v<T>)*std::acos(z);
        return f;
    }
};

template<typename T>
using FrequencyEstimatorTimeDomainDecimating = FrequencyEstimatorTimeDomain<T, Resampling<10U>>;

GR_REGISTER_BLOCK("gr::filter::FrequencyEstimatorFrequencyDomain", gr::filter::FrequencyEstimatorFrequencyDomain, [T], [ float, double ])
GR_REGISTER_BLOCK("gr::filter::FrequencyEstimatorFrequencyDomainDecimating", gr::filter::FrequencyEstimatorFrequencyDomain, ([T], gr::Resampling<10U>), [ float, double ])

template<typename T, typename... Args>
requires std::floating_point<T>
struct FrequencyEstimatorFrequencyDomain : Block<FrequencyEstimatorFrequencyDomain<T, Args...>, Args...> {
    using Description = Doc<R""(@brief Frequency Domain Frequency Estimator

This block estimates the frequency of a signal using the frequency-domain algorithm described in:
  [0] M. Gasior, J.L. Gonzalez,
      "Improving FFT frequency measurement resolution by parabolic and gaussian spectrum interpolation",
      AIP Conf. Proc. 732 (2004) 276,
      https://doi.org/10.1063/1.1831158.
)"">;
    using TParent     = Block<FrequencyEstimatorFrequencyDomain<T, Args...>, Args...>;

    PortIn<T>  in;
    PortOut<T> out;

    // settings
    Annotated<float, "sample rate", Doc<"signal sample rate">, Unit<"Hz">>       sample_rate{1.f};
    Annotated<float, "f_min", Doc<"exp. min frequency range">, Unit<"Hz">>       f_min{40.f};
    Annotated<float, "f_expected", Doc<"expected likely frequency">, Unit<"Hz">> f_expected{50.f};
    Annotated<float, "f_max", Doc<"exp. max frequency">, Unit<"Hz">>             f_max{60.f};
    Annotated<gr::Size_t, "min FFT size", Doc<"minimum FFT size">>               min_fft_size{256U};
    Annotated<T, "epsilon", Doc<"numerical error threshold">>                    epsilon{T(1e-8)};

    GR_MAKE_REFLECTABLE(FrequencyEstimatorFrequencyDomain, in, out, sample_rate, f_expected, f_min, f_max, min_fft_size, epsilon);

    // private internal state
    T           _prevFrequency{50.0}; // previous frequency value for continuity
    std::size_t _minFFT{256UZ};       // number of min required sample (power-of-two)

    HistoryBuffer<T> _inputHistory{32UZ};

    gr::algorithm::FFT<T, std::complex<T>> _fftImpl{};
    std::vector<T>                         _inData;
    std::vector<T>                         _window;
    std::vector<std::complex<T>>           _outData;
    std::vector<T>                         _magnitudeSpectrum;

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& newSettings) {
        if (newSettings.contains("n_periods") || newSettings.contains("sample_rate") || newSettings.contains("f_expected") || newSettings.contains("f_min") || newSettings.contains("f_max") || newSettings.contains("min_fft_size")) {
            if (f_min < 0 || f_max >= sample_rate || f_expected < 0 || f_expected >= sample_rate) {
                throw gr::exception(fmt::format("Ill-formed block parameters: f_min: {} < f_expected: {} < f_max: {} < sample_rate: {}", f_min, f_expected, f_max, sample_rate));
            }
            initialiseFFT();
        }
    }

    void initialiseFFT() {
        _minFFT = std::bit_ceil(std::max(std::size_t(min_fft_size.value), std::size_t(f_min > 0 ? sample_rate / std::min(f_min.value, f_expected.value) : sample_rate / f_expected.value)));

        _inputHistory = HistoryBuffer<T>(_minFFT);
        if constexpr (not TParent::ResamplingControl::kIsConst) {
            this->input_chunk_size = static_cast<gr::Size_t>(_minFFT);
        }
        _inData.resize(_minFFT, T(0));
        _window = gr::algorithm::window::create<T>(gr::algorithm::window::Type::Hann, _minFFT);
        _outData.resize(_minFFT, std::complex<T>(T(0)));
        _magnitudeSpectrum.resize(_minFFT / 2UZ, T(0));
    }

    void reset() {
        _prevFrequency = static_cast<T>(f_expected);
        initialiseFFT();
    }

    [[nodiscard]] T processOne(T input) noexcept
    requires(TParent::ResamplingControl::kIsConst)
    {
        _inputHistory.push_front(input);
        _prevFrequency = estimateFrequencyFFT();
        return _prevFrequency;
    }

    [[nodiscard]] work::Status processBulk(std::span<const T> input, std::span<T> output) noexcept
    requires(not TParent::ResamplingControl::kIsConst)
    {
        const std::size_t num_chunks = input.size() / std::size_t(this->input_chunk_size);
        if (output.size() < num_chunks) {
            return work::Status::INSUFFICIENT_OUTPUT_ITEMS;
        }

        auto output_it = output.begin();
        for (std::size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
            const std::size_t  offset = chunk_idx * this->input_chunk_size;
            std::span<const T> chunk  = input.subspan(offset, this->input_chunk_size);

            for (const T& sample : chunk) {
                _inputHistory.push_front(sample);
            }

            _prevFrequency = estimateFrequencyFFT();
            *output_it++   = _prevFrequency;
        }

        return work::Status::OK;
    }

private:
    T estimateFrequencyFFT() {
        // implement reference algorithm from [1]
        using namespace gr::algorithm::fft;

        if (_inputHistory.size() < _minFFT) {
            return _prevFrequency; // Return previous frequency during settling time
        }

        // apply window function
        auto bufferStart      = _inputHistory.begin();
        using difference_type = typename decltype(bufferStart)::difference_type;
        std::span<T> data(bufferStart, std::next(bufferStart, static_cast<difference_type>(_minFFT)));
        for (std::size_t i = 0UZ; i < data.size(); i++) {
            _inData[i] = data[i] * _window[i];
        }

        // compute FFT and magnitude spectrum
        _magnitudeSpectrum = computeMagnitudeSpectrum(_fftImpl.compute(_inData, _outData), _magnitudeSpectrum, ConfigMagnitude{.computeHalfSpectrum = true, .outputInDb = false});

        const float scaled_size = static_cast<float>(_magnitudeSpectrum.size()) * 2.f;
        std::size_t i_min       = static_cast<std::size_t>(std::floor((f_min / sample_rate) * scaled_size));
        std::size_t i_max       = static_cast<std::size_t>(std::ceil((f_max / sample_rate) * scaled_size));

        // ensure indices are within bounds
        i_min = std::clamp(i_min, std::size_t(1), _magnitudeSpectrum.size() - 1UZ);
        i_max = std::clamp(i_max, std::size_t(1), _magnitudeSpectrum.size() - 1UZ);

        // find the index of the maximum peak in the magnitude spectrum within the [i_min, i_max] range
        const auto        it_max = std::max_element(_magnitudeSpectrum.begin() + static_cast<difference_type>(i_min), _magnitudeSpectrum.begin() + static_cast<difference_type>(i_max));
        const std::size_t k_max  = std::size_t(std::distance(_magnitudeSpectrum.begin(), it_max));

        // ensure the peak is not at the edges to allow interpolation with neighbours
        if (k_max == 0 || k_max >= _magnitudeSpectrum.size() - 1) {
            return _prevFrequency; // Return previous frequency if peak is at the edges
        }

        // ensure magnitudes are positive
        const T S_km1 = _magnitudeSpectrum[k_max - 1];
        const T S_k   = _magnitudeSpectrum[k_max];
        const T S_kp1 = _magnitudeSpectrum[k_max + 1];

        if (!std::isfinite(S_km1) || !std::isfinite(S_k) || !std::isfinite(S_kp1) //
            || S_km1 <= T(0) || S_k <= T(0) || S_kp1 <= T(0)) {
            return _prevFrequency; // cannot compute logarithm
        }

        // compute logarithms
        const T log_S_km1 = std::log(S_km1);
        const T log_S_k   = std::log(S_k);
        const T log_S_kp1 = std::log(S_kp1);

        const T denominator = 2 * log_S_k - log_S_km1 - log_S_kp1;
        if (!std::isfinite(denominator) || std::abs(denominator) < epsilon) {
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

        return interpolated_freq_bin * static_cast<T>(sample_rate) / static_cast<T>(_inData.size());
    }
};

template<typename T>
using FrequencyEstimatorFrequencyDomainDecimating = FrequencyEstimatorFrequencyDomain<T, Resampling<1U>, Stride<0U>>;

} // namespace gr::filter

#endif // FREQUENCY_ESTIMATOR_HPP
