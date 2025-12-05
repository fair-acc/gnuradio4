#ifndef FREQUENCY_ESTIMATOR_HPP
#define FREQUENCY_ESTIMATOR_HPP

#include <algorithm>
#include <cmath>
#include <execution>
#include <numbers>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/HistoryBuffer.hpp>
#include <gnuradio-4.0/filter/time_domain_filter.hpp>

#include <gnuradio-4.0/algorithm/filter/FilterTool.hpp>
#include <gnuradio-4.0/algorithm/fourier/fft.hpp>
#include <gnuradio-4.0/algorithm/fourier/fft_common.hpp>
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
    using TParent     = Block<FrequencyEstimatorTimeDomain<T, Args...>, Args...>;

    PortIn<T>  in;
    PortOut<T> out;

    // settings
    Annotated<float, "sample rate", Doc<"signal sample rate">, Unit<"Hz">>            sample_rate{1e3f};
    Annotated<float, "f_min", Doc<"exp. min frequency range">, Unit<"Hz">>            f_min{40.f};
    Annotated<float, "f_expected", Doc<"expected likely frequency">, Unit<"Hz">>      f_expected{50.f};
    Annotated<float, "f_max", Doc<"exp. max/LP frequency (-1: disable)">, Unit<"Hz">> f_max{60.f};
    Annotated<gr::Size_t, "n periods rate", Doc<"number of periods to average over">> n_periods{4U};
    Annotated<T, "epsilon", Doc<"numerical error threshold">>                         epsilon{T(1e-8)};

    GR_MAKE_REFLECTABLE(FrequencyEstimatorTimeDomain, in, out, sample_rate, f_expected, f_min, f_max, n_periods, epsilon);

    // private internal state
    T          _prevFrequency{T(50)};   // previous frequency value for continuity
    gr::Size_t _n_period_estimate{60U}; // number of samples for estimation period according to [0]

    FilterCoefficients<T> _singleFilterSection;
    HistoryBuffer<T>      _inputHistory{32UZ};
    HistoryBuffer<T>      _outputHistory{32UZ};

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& newSettings) {
        if (newSettings.contains("n_periods") || newSettings.contains("sample_rate") || newSettings.contains("f_expected") || newSettings.contains("f_min") || newSettings.contains("f_max")) {
            if (f_min < 0.f || f_max >= sample_rate / 2.f || f_expected < 0.f || f_expected >= sample_rate / 2.f) {
                throw gr::exception(std::format("Ill-formed block parameters: f_min: {} < f_expected: {} < f_max: {} < sample_rate/2: {} (Nyquist limit, N.B. f_max < 0 -> disable low-pass)", f_min, f_expected, f_max, sample_rate / 2.f));
            }
            initialiseFilter();
        }
    }

    void initialiseFilter() {
        // Calculate filter coefficients
        // * IIR over FIR: reduced numerical complexity and minimizes group delay
        // * BESSEL: optimizes phase linearity in the pass-band and in turn frequency accuracy
        _n_period_estimate = n_periods * static_cast<gr::Size_t>(f_min > 0.f ? sample_rate / std::min(f_min.value, f_expected.value) : sample_rate / f_expected.value);
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
        const T output = std::inner_product(_singleFilterSection.b.cbegin(), _singleFilterSection.b.cend(), _inputHistory.cbegin(), T(0))         // feed-forward
                         - std::inner_product(_singleFilterSection.a.cbegin() + 1, _singleFilterSection.a.cend(), _outputHistory.cbegin(), T(0)); // feed-back
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

                const T output_sample = std::inner_product(_singleFilterSection.b.cbegin(), _singleFilterSection.b.cend(), _inputHistory.cbegin(), T(0))         // feed-forward
                                        - std::inner_product(_singleFilterSection.a.cbegin() + 1, _singleFilterSection.a.cend(), _outputHistory.cbegin(), T(0)); // feed-back

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
        T accA = T(0);
        T accB = T(0);
        T accC = T(0);

        auto         bufferStart = _outputHistory.begin();
        std::span<T> data(bufferStart, std::next(bufferStart, static_cast<std::ptrdiff_t>(_n_period_estimate)));
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
        if (z >= T(1) || z <= T(-1)) {
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
    Annotated<float, "sample rate", Doc<"signal sample rate">, Unit<"Hz">>       sample_rate{1e3f};
    Annotated<float, "f_min", Doc<"exp. min frequency range">, Unit<"Hz">>       f_min{40.f};
    Annotated<float, "f_expected", Doc<"expected likely frequency">, Unit<"Hz">> f_expected{50.f};
    Annotated<float, "f_max", Doc<"exp. max frequency">, Unit<"Hz">>             f_max{60.f};
    Annotated<gr::Size_t, "min FFT size", Doc<"minimum FFT size">>               min_fft_size{256U};
    Annotated<T, "epsilon", Doc<"numerical error threshold">>                    epsilon{T(1e-8)};

    GR_MAKE_REFLECTABLE(FrequencyEstimatorFrequencyDomain, in, out, sample_rate, f_expected, f_min, f_max, min_fft_size, epsilon);

    // private internal state
    T           _prevFrequency{T(50)}; // previous frequency value for continuity
    std::size_t _minFFT{256UZ};        // number of min required sample (power-of-two)

    HistoryBuffer<T> _inputHistory{32UZ};

    gr::algorithm::FFT<T, std::complex<T>> _fftImpl{};
    std::vector<T>                         _inData;
    std::vector<T>                         _window;
    std::vector<std::complex<T>>           _outData;
    std::vector<T>                         _magnitudeSpectrum;

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& newSettings) {
        if (newSettings.contains("sample_rate") || newSettings.contains("f_expected") || newSettings.contains("f_min") || newSettings.contains("f_max") || newSettings.contains("min_fft_size")) {
            if (f_min < 0.f || f_max >= sample_rate / 2.f || f_expected < 0.f || f_expected >= sample_rate / 2.f) {
                throw gr::exception(std::format("Ill-formed block parameters: f_min: {} < f_expected: {} < f_max: {} < sample_rate/2: {} (Nyquist limit)", f_min, f_expected, f_max, sample_rate / 2.f));
            }
            initialiseFFT();
        }
    }

    void initialiseFFT() {
        _minFFT = std::bit_ceil(std::max(std::size_t(min_fft_size.value), std::size_t(f_min > 0.f ? sample_rate / std::min(f_min.value, f_expected.value) : sample_rate / f_expected.value)));

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
        // implement reference algorithm from [0]
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
        i_min = std::clamp(i_min, 1UZ, _magnitudeSpectrum.size() - 1UZ);
        i_max = std::clamp(i_max, 1UZ, _magnitudeSpectrum.size() - 1UZ);

        // find the index of the maximum peak in the magnitude spectrum within the [i_min, i_max] range
        auto              searchRange = std::ranges::subrange(_magnitudeSpectrum.begin() + static_cast<std::ptrdiff_t>(i_min), _magnitudeSpectrum.begin() + static_cast<std::ptrdiff_t>(i_max));
        const auto        it_max      = std::ranges::max_element(searchRange);
        const std::size_t k_max       = static_cast<std::size_t>(std::distance(_magnitudeSpectrum.begin(), it_max));

        // ensure the peak is not at the edges to allow interpolation with neighbours
        if (k_max == 0UZ || k_max >= _magnitudeSpectrum.size() - 1UZ) {
            return _prevFrequency; // Return previous frequency if peak is at the edges
        }

        // ensure magnitudes are positive
        const T S_km1 = _magnitudeSpectrum[k_max - 1];
        const T S_k   = _magnitudeSpectrum[k_max];
        const T S_kp1 = _magnitudeSpectrum[k_max + 1];

        if (!std::isfinite(S_km1) || !std::isfinite(S_k) || !std::isfinite(S_kp1) || S_km1 <= T(0) || S_k <= T(0) || S_kp1 <= T(0)) {
            return _prevFrequency; // cannot compute logarithm
        }

        // compute logarithms
        const T log_S_km1 = std::log(S_km1);
        const T log_S_k   = std::log(S_k);
        const T log_S_kp1 = std::log(S_kp1);

        const T denominator = T(2) * log_S_k - log_S_km1 - log_S_kp1;
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

// ============================================================================
// IQDemodulator - Digital Vector Detector / Lock-In Amplifier
// ============================================================================

enum class PhaseUnit : int { Radians = 0, Degrees = 1 };

enum class DerivativeMethod : int {
    SymmetricDifference = 0, /// [-1, 0, +1] kernel
    SavitzkyGolay5      = 1, /// 5-point SG derivative
    SavitzkyGolay7      = 2  /// 7-point SG derivative
};

template<DerivativeMethod method = DerivativeMethod::SymmetricDifference, bool isConst = false>
struct Derivative {
    static constexpr DerivativeMethod kMethod  = method;
    static constexpr bool             kIsConst = isConst;
};

namespace detail {
template<typename T>
struct is_derivative : std::false_type {};

template<DerivativeMethod M, bool C>
struct is_derivative<Derivative<M, C>> : std::true_type {};

template<typename T>
inline constexpr bool is_derivative_v = is_derivative<T>::value;
} // namespace detail

GR_REGISTER_BLOCK("gr::filter::IQDemodulator", gr::filter::IQDemodulator, ([T], gr::Resampling<1024U, 1U, false>), [ float, double ])

template<typename T, typename... Args>
requires std::floating_point<T>
struct IQDemodulator : Block<IQDemodulator<T, Args...>, Args...> {
    using Description = Doc<R""(@brief Digital vector detector for coherent signal demodulation (lock-in amplifier).

Signal model:
  ref(t)  = A_r · cos(ωt) + DC (ADC)-- reference oscillator (DDS)
  resp(t) = A_x · cos(ωt + φ) + DC  -- response signal with phase shift φ

Signal chain:
  1. HP filter (DC blocking):     y[n] = α·(y[n-1] + x[n] - x[n-1]),  α = exp(-2π·f_hp/f_s)
  2. Quadrature via derivative:   Q = d/dt{ref},  |H(ω)| varies by method
  3. Lock-in mixer:               I = resp·ref,   Q_mix = resp·Q
  4. LP filter (averaging):       y[n] = y[n-1] + α·(x[n] - y[n-1]),  α = 1 - exp(-2π·f_lp/f_s)

Output extraction (after LP averaging):
  amplitude = √(P_x / P_r)                          -- ratio of response to reference power
  phase     = atan2(Q_mix, I · √(P_d/P_r))          -- with automatic gain compensation
  frequency = solved iteratively from |H(ω)| = √(P_d/P_r) using method-specific G(ω)

Derivative methods (G(ω) = |H(ω)|/sin(ω), used for frequency estimation):
  SymmetricDifference: h = [-1, 0, +1],        G(ω) = 2,                          delay = 1
  SavitzkyGolay5:      h = [-2,-1,0,+1,+2]/10, G(ω) = 0.8·cos(ω) + 0.2,           delay = 2
  SavitzkyGolay7:      h = [-3..+3]/28,        G(ω) = (6cos²ω + 2cosω - 1)/7,     delay = 3

Limitations:
  - Carrier frequency must satisfy: f_hp < f_carrier < f_s/2 (Nyquist)
  - Settling time ≈ 5/(2π·f_lp) samples after transient

Typical application: RF cavity field measurement at 0.1–5 MHz carriers, 62.5 MHz ADC rate.
)"">;

    using TParent           = Block<IQDemodulator<T, Args...>, Args...>;
    using ArgumentsTypeList = typename TParent::ArgumentsTypeList;
    using DerivativeControl = typename ArgumentsTypeList::template find_or_default<detail::is_derivative, Derivative<DerivativeMethod::SymmetricDifference, false>>;

    PortIn<T>  ref;       // reference/DDS signal
    PortIn<T>  resp;      // response signal (e.g., cavity pickup)
    PortOut<T> amplitude; // A_resp / A_ref (dimensionless)
    PortOut<T> phase;     // phase of response relative to reference
    PortOut<T> frequency; // estimated carrier frequency

    // settings
    Annotated<float, "sample_rate", Doc<"input sample rate">, Unit<"Hz">>                 sample_rate{62.5e6f};
    Annotated<float, "f_high_pass", Doc<"high-pass cutoff for DC blocking">, Unit<"Hz">>  f_high_pass{100.f};
    Annotated<float, "f_low_pass", Doc<"low-pass cutoff for averaging">, Unit<"Hz">>      f_low_pass{10000.f};
    Annotated<PhaseUnit, "phase_unit", Doc<"output phase unit">>                          phase_unit{PhaseUnit::Radians};
    Annotated<bool, "invert_phase", Doc<"invert phase sign (swap lead/lag)">>             invert_phase{false};
    Annotated<DerivativeMethod, "derivative_method", Doc<"quadrature generation method">> derivative_method{DerivativeControl::kMethod};
    Annotated<T, "epsilon", Doc<"numerical threshold for division safety">>               epsilon{T(1e-12)};

    GR_MAKE_REFLECTABLE(IQDemodulator, ref, resp, amplitude, phase, frequency, sample_rate, f_high_pass, f_low_pass, phase_unit, invert_phase, derivative_method, epsilon);

    // filter state variables
    T _hp_ref_state{}, _hp_ref_prev{};
    T _hp_resp_state{}, _hp_resp_prev{};
    T _lp_I{}, _lp_Q{}, _lp_Pr{}, _lp_Pd{}, _lp_Px{};
    T _alpha_hp{}, _alpha_lp{};

    // delay line for derivative and time alignment
    HistoryBuffer<T> _ref_history{8UZ};
    HistoryBuffer<T> _resp_history{8UZ};

    std::vector<T> _derivative_kernel; // time-reversed for direct inner_product
    std::size_t    _derivative_delay{};
    T              _derivative_gain_factor{T(2)}; // DC gain: |H(ω)|/sin(ω) as ω→0

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& newSettings) {
        if constexpr (DerivativeControl::kIsConst) {
            if (newSettings.contains("derivative_method")) {
                throw gr::exception("derivative_method is compile-time fixed and cannot be changed at runtime");
            }
        }
        if (newSettings.contains("sample_rate") || newSettings.contains("f_high_pass") || newSettings.contains("f_low_pass") || newSettings.contains("derivative_method")) {
            if (f_high_pass <= 0.f || f_low_pass <= 0.f || f_high_pass >= f_low_pass || f_low_pass >= sample_rate / 2.f) {
                throw gr::exception(std::format("invalid filter frequencies: 0 < f_hp({}) < f_lp({}) < fs/2({})", f_high_pass.value, f_low_pass.value, sample_rate / 2.f));
            }
            initialiseFilters();
        }
    }

    void initialiseFilters() {
        const T fs = static_cast<T>(sample_rate);

        // compute filter coefficients
        _alpha_hp = std::exp(T(-2) * std::numbers::pi_v<T> * static_cast<T>(f_high_pass) / fs);
        _alpha_lp = T(1) - std::exp(T(-2) * std::numbers::pi_v<T> * static_cast<T>(f_low_pass) / fs);

        // reset filter states
        _hp_ref_state = _hp_ref_prev = T(0);
        _hp_resp_state = _hp_resp_prev = T(0);
        _lp_I = _lp_Q = _lp_Pr = _lp_Pd = _lp_Px = T(0);

        // configure derivative kernel (stored time-reversed for direct inner_product)
        // gain factor = lim_{ω→0} |H(ω)|/sin(ω), used for frequency estimation
        switch (derivative_method) {
        case DerivativeMethod::SymmetricDifference:
            // H(ω) = 2j·sin(ω), |H(ω)| = 2·sin(ω)
            _derivative_kernel      = {T(1), T(0), T(-1)};
            _derivative_delay       = 1UZ;
            _derivative_gain_factor = T(2);
            break;
        case DerivativeMethod::SavitzkyGolay5:
            // H(ω) = j·(0.4·sin(2ω) + 0.2·sin(ω)), |H(ω)| ≈ sin(ω)·(0.8·cos(ω) + 0.2) ≈ sin(ω) at DC
            _derivative_kernel      = {T(0.2), T(0.1), T(0), T(-0.1), T(-0.2)};
            _derivative_delay       = 2UZ;
            _derivative_gain_factor = T(1);
            break;
        case DerivativeMethod::SavitzkyGolay7:
            // H(ω) = j·(6·sin(3ω) + 4·sin(2ω) + 2·sin(ω))/28, |H(ω)| ≈ sin(ω) at DC
            _derivative_kernel      = {T(3) / T(28), T(2) / T(28), T(1) / T(28), T(0), T(-1) / T(28), T(-2) / T(28), T(-3) / T(28)};
            _derivative_delay       = 3UZ;
            _derivative_gain_factor = T(1);
            break;
        }

        const std::size_t histSize = std::bit_ceil(_derivative_kernel.size() + 1UZ);
        _ref_history               = HistoryBuffer<T>(histSize);
        _resp_history              = HistoryBuffer<T>(histSize);
    }

    void reset() { initialiseFilters(); }
    void start() { initialiseFilters(); }

    [[nodiscard]] work::Status processBulk(std::span<const T> inRef, std::span<const T> inResp, std::span<T> outAmp, std::span<T> outPhase, std::span<T> outFreq) noexcept {
        const auto chunkSize = static_cast<std::size_t>(this->input_chunk_size);
        const auto nChunks   = std::min({inRef.size(), inResp.size()}) / chunkSize;
        const auto nOutputs  = std::min({outAmp.size(), outPhase.size(), outFreq.size()});

        if (nOutputs < nChunks) {
            return work::Status::INSUFFICIENT_OUTPUT_ITEMS;
        }

        // HP filter: y[n] = α·(y[n-1] + x[n] - x[n-1])
        auto processHP = [alpha_hp = _alpha_hp](T x, T& state, T& x_prev) noexcept -> T {
            state  = alpha_hp * (state + x - x_prev);
            x_prev = x;
            return state;
        };

        // LP filter: y[n] = y[n-1] + α·(x[n] - y[n-1])
        auto processLP = [alpha_lp = _alpha_lp](T x, T& state) noexcept -> T {
            state += alpha_lp * (x - state);
            return state;
        };

        for (std::size_t chunk = 0; chunk < nChunks; ++chunk) {
            const auto refChunk  = inRef.subspan(chunk * chunkSize, chunkSize);
            const auto respChunk = inResp.subspan(chunk * chunkSize, chunkSize);

            for (std::size_t i = 0; i < chunkSize; ++i) {
                // step 1: high-pass filter inputs (DC blocking)
                const T r_hp = processHP(refChunk[i], _hp_ref_state, _hp_ref_prev);
                const T x_hp = processHP(respChunk[i], _hp_resp_state, _hp_resp_prev);

                // push to history buffers (most recent at front)
                _ref_history.push_front(r_hp);
                _resp_history.push_front(x_hp);

                // step 2: compute derivative via kernel convolution (kernel is pre-reversed)
                T r_Q = T(0);
                if (_ref_history.size() >= _derivative_kernel.size()) {
                    r_Q = std::transform_reduce(std::execution::unseq, _derivative_kernel.cbegin(), _derivative_kernel.cend(), _ref_history.cbegin(), T(0), std::plus<>{}, std::multiplies<>{});
                }

                // extract time-aligned reference and response (at derivative center)
                const T r_I = (_ref_history.size() > _derivative_delay) ? _ref_history[_derivative_delay] : T(0);
                const T x_I = (_resp_history.size() > _derivative_delay) ? _resp_history[_derivative_delay] : T(0);

                // step 3: lock-in products (time-aligned at derivative center)
                const T I_raw  = x_I * r_I;
                const T Q_raw  = x_I * r_Q;
                const T Pr_raw = r_I * r_I;
                const T Pd_raw = r_Q * r_Q;
                const T Px_raw = x_I * x_I;

                // step 4: low-pass filter all products
                processLP(I_raw, _lp_I);
                processLP(Q_raw, _lp_Q);
                processLP(Pr_raw, _lp_Pr);
                processLP(Pd_raw, _lp_Pd);
                processLP(Px_raw, _lp_Px);
            }

            // step 5: extract amplitude, phase, frequency from filtered values
            const T I  = _lp_I;
            const T Q  = _lp_Q;
            const T Pr = _lp_Pr;
            const T Pd = _lp_Pd;
            const T Px = _lp_Px;

            // amplitude ratio: √(P_x / P_r)
            const T amp = (Pr > epsilon && Px > epsilon) ? std::sqrt(Px / Pr) : T(0);

            // frequency from derivative power ratio with iterative gain correction
            // solve: G(ω)·sin(ω) = √(P_d/P_r) where G(ω) is method-specific
            T freq = T(0);
            if (Pr > epsilon && Pd > epsilon) {
                const T measured_ratio = std::sqrt(Pd / Pr);
                const T fs             = static_cast<T>(sample_rate);

                // compute frequency-dependent gain: G(ω) = |H(ω)|/sin(ω)
                auto computeGainFactor = [this](T omega) -> T {
                    const T c = std::cos(omega);
                    switch (derivative_method.value) {
                    case DerivativeMethod::SymmetricDifference: return T(2);                               // |H(ω)| = 2·sin(ω)
                    case DerivativeMethod::SavitzkyGolay5: return T(0.8) * c + T(0.2);                     // |H(ω)| = sin(ω)·(0.8c + 0.2)
                    case DerivativeMethod::SavitzkyGolay7: return (T(6) * c * c + T(2) * c - T(1)) / T(7); // |H(ω)| = sin(ω)·(6c²+2c-1)/7
                    default: return _derivative_gain_factor;
                    }
                };

                // initial estimate using DC gain factor
                T omega = std::asin(std::clamp(measured_ratio / _derivative_gain_factor, T(-1), T(1)));

                // iterative refinement with Aitken acceleration for faster convergence
                T omega0 = omega, omega1, omega2;
                for (int iter = 0; iter < 3; ++iter) {
                    // three standard iterations
                    omega1         = std::asin(std::clamp(measured_ratio / computeGainFactor(omega0), T(-1), T(1)));
                    omega2         = std::asin(std::clamp(measured_ratio / computeGainFactor(omega1), T(-1), T(1)));
                    const T omega3 = std::asin(std::clamp(measured_ratio / computeGainFactor(omega2), T(-1), T(1)));

                    // Aitken's delta-squared acceleration
                    const T denom = omega3 - T(2) * omega2 + omega1;
                    if (std::abs(denom) > epsilon) {
                        const T delta = omega2 - omega1;
                        omega0        = omega1 - (delta * delta) / denom;
                    } else {
                        omega0 = omega3;
                    }
                }
                omega = omega0;
                freq  = omega * fs / (T(2) * std::numbers::pi_v<T>);
            }

            // phase: atan2(Q, I·√(P_d/P_r)) with gain compensation
            T ph = T(0);
            if (Pr > epsilon && Pd > epsilon && (std::abs(I) > epsilon || std::abs(Q) > epsilon)) {
                const T sqrt_ratio = std::sqrt(Pd / Pr);
                ph                 = std::atan2(Q, I * sqrt_ratio);
            }

            // apply phase sign convention
            if (invert_phase) {
                ph = -ph;
            }

            // convert to degrees if requested
            if (phase_unit == PhaseUnit::Degrees) {
                ph *= T(180) / std::numbers::pi_v<T>;
            }

            outAmp[chunk]   = amp;
            outPhase[chunk] = ph;
            outFreq[chunk]  = freq;
        }

        return work::Status::OK;
    }
};

template<typename T>
using IQDemodulatorDecimating = IQDemodulator<T, Resampling<1024U, 1U, false>>;

template<typename T, DerivativeMethod M>
using IQDemodulatorFixed = IQDemodulator<T, Resampling<1024U, 1U, false>, Derivative<M, true>>;

} // namespace gr::filter

#endif // FREQUENCY_ESTIMATOR_HPP
