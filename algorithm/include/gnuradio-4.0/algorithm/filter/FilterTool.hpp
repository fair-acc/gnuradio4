#ifndef GNURADIO_FILTERTOOL_HPP
#define GNURADIO_FILTERTOOL_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <execution>
#include <format>
#include <iterator>
#include <numbers>
#include <numeric>
#include <ranges>
#include <unordered_set>
#include <vector>

#ifdef __GNUC__
#pragma GCC diagnostic push // ignore warning of external libraries that from this lib-context we do not have any control over
#ifndef __clang__
#pragma GCC diagnostic ignored "-Wuseless-cast"
#endif
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif
#include <magic_enum.hpp>
#include <magic_enum_utility.hpp>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include <gnuradio-4.0/HistoryBuffer.hpp>
#include <gnuradio-4.0/Message.hpp>
#include <gnuradio-4.0/algorithm/fourier/window.hpp>
#include <gnuradio-4.0/meta/UncertainValue.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>

// this mocks the execution policy until Emscripten's libc++ does support this (Clang already does)
#if not defined(__GLIBCXX__) && (defined(__EMSCRIPTEN__) || defined(__clang__))

namespace std {

namespace execution {
class mock_execution_policy {};

inline constexpr mock_execution_policy seq{};
inline constexpr mock_execution_policy unseq{};
inline constexpr mock_execution_policy par{};
} // namespace execution

template<typename InputIt1, typename InputIt2, typename T, typename BinaryOp1, typename BinaryOp2>
inline T transform_reduce(auto, InputIt1 first1, InputIt1 last1, InputIt2 first2, T init, BinaryOp1 binary_op1, BinaryOp2 binary_op2) {
    return std::transform_reduce(first1, last1, first2, init, binary_op1, binary_op2);
}

} // namespace std
#endif

namespace gr::filter {

enum class Frequency {
    Hertz,        /// frequency in cycles per second. Standard unit for frequency.
    RadianPerSec, /// angular frequency, indicating the rate of change per second in radians.
    Normalised    /// frequency as a fraction of the sampling rate, typically in the range [0, 0.5].
};

enum class Type { LOWPASS, HIGHPASS, BANDPASS, BANDSTOP };

struct FilterParameters {
    std::size_t order{4UZ};                                      /// default filter order
    double      fLow{std::numeric_limits<double>::quiet_NaN()};  /// Lower cutoff frequency [Hertz].
    double      fHigh{std::numeric_limits<double>::quiet_NaN()}; /// Upper cutoff frequency [Hertz].
    double      gain{1.0};                                       /// required total filter gain
    double      rippleDb{0.1};                                   /// Maximum allowed ripple in the pass-band [dB].
    double      attenuationDb{40};                               /// Minimum required attenuation in the stop-band [dB].
    double      beta{1.6};                                       /// default beta for Kaiser-type windowing
    double      fs{std::numeric_limits<double>::quiet_NaN()};    /// Sampling frequency for digital filters [Hertz].
};

/**
 * @brief Filter coefficients of a digital transfer function H(z) = B(z)/A(z) in the z-domain with:
 *   B(z) = b[0] + b[1]·z^{-1} + b[2]·z^{-2} + …
 *   A(z) = a[0] + a[1]·z^{-1} + a[2]·z^{-2} + …
 *
 * The difference equation representing the filter is:
 * y[n] = b[0]·x[n] + b[1]·x[n-1] + … - (a[1]·y[n-1] + a[2]·y[n-2] + …)
 *
 * @note Typically, a[0] is 1 for causal systems and a{
 */
template<typename T>
struct FilterCoefficients {
    using value_type = T;
    std::vector<T> b{};                  /// numerator coefficients
    std::vector<T> a{static_cast<T>(1)}; /// denominator coefficients
};

template<typename T>
concept HasFilterCoefficients = requires(T t) {
    typename T::value_type;
    { t.a } -> std::convertible_to<std::vector<typename T::value_type>&>;
    { t.b } -> std::convertible_to<std::vector<typename T::value_type>&>;
};

static_assert(HasFilterCoefficients<FilterCoefficients<double>>);
static_assert(HasFilterCoefficients<FilterCoefficients<float>>);

enum class Form {
    DF_I,  /// direct form I: preferred for fixed-point arithmetics (e.g. no overflow)
    DF_II, /// direct form II: preferred for floating-point arithmetics (less operations)
    DF_I_TRANSPOSED,
    DF_II_TRANSPOSED
};

namespace detail {

template<typename T, std::size_t bufferSize = std::dynamic_extent, typename TBaseType = meta::fundamental_base_value_type_t<T>>
struct Section;

template<typename T, std::size_t bufferSize, Form form = std::is_floating_point_v<T> ? Form::DF_II : Form::DF_I, auto execPolicy = std::execution::unseq>
[[nodiscard]] inline constexpr T computeFilter(const T& input, Section<T, bufferSize>& section) noexcept {
    const auto& a             = section.a;
    const auto& b             = section.b;
    auto&       inputHistory  = section.inputHistory;
    auto&       outputHistory = section.outputHistory;
    if constexpr (form == Form::DF_I) {
        // y[n] = b[0]·x[n]   + b[1]·x[n-1] + … + b[N]·x[n-N]
        //      - a[1]·y[n-1] - a[2]·y[n-2] - … - a[M]·y[n-M]
        inputHistory.push_front(input);
        T output = std::transform_reduce(execPolicy, b.cbegin(), b.cend(), inputHistory.cbegin(), static_cast<T>(0), std::plus<>(), std::multiplies<>())                // feed-forward path
                   - std::transform_reduce(execPolicy, std::next(a.cbegin()), a.cend(), outputHistory.cbegin(), static_cast<T>(0), std::plus<>(), std::multiplies<>()); // feedback path
        outputHistory.push_front(output);
        return output;
    } else if constexpr (form == Form::DF_II) {
        // w[n] = x[n] - a[1]·w[n-1] - a[2]·w[n-2] - … - a[M]·w[n-M]
        // y[n] =        b[0]·w[n]   + b[1]·w[n-1] + … + b[N]·w[n-N]
        if (a.size() > 1) {
            const T w = input - std::transform_reduce(execPolicy, std::next(a.cbegin()), a.cend(), inputHistory.cbegin(), T{0}, std::plus<>(), std::multiplies<>());
            inputHistory.push_front(w);
            return std::transform_reduce(execPolicy, b.cbegin(), b.cend(), inputHistory.cbegin(), T{0}, std::plus<>(), std::multiplies<>());
        } else {
            inputHistory.push_front(input);
            return std::transform_reduce(execPolicy, b.cbegin(), b.cend(), inputHistory.cbegin(), T{0}, std::plus<>(), std::multiplies<>());
        }
    } else if constexpr (form == Form::DF_I_TRANSPOSED) {
        // w_1[n] = x[n] - a[1]·w_2[n-1] - a[2]·w_2[n-2] - … - a[M]·w_2[n-M]
        // y[n]   = b[0]·w_2[n] + b[1]·w_2[n-1] + … + b[N]·w_2[n-N]
        T v0 = input - std::transform_reduce(execPolicy, std::next(a.cbegin()), a.cend(), outputHistory.cbegin(), static_cast<T>(0), std::plus<>(), std::multiplies<>());
        outputHistory.push_front(v0);
        return std::transform_reduce(execPolicy, b.cbegin(), b.cend(), outputHistory.cbegin(), T{0}, std::plus<>(), std::multiplies<>());
    } else if constexpr (form == Form::DF_II_TRANSPOSED) {
        // y[n] = b_0·f[n] + \sum_(k=1)^N(b_k·f[n−k] − a_k·y[n−k])
        T output = b[0] * input                                                                                                                                       //
                   + std::transform_reduce(execPolicy, std::next(b.cbegin()), b.cend(), inputHistory.cbegin(), static_cast<T>(0), std::plus<>(), std::multiplies<>()) //
                   - std::transform_reduce(execPolicy, std::next(a.cbegin()), a.cend(), outputHistory.cbegin(), static_cast<T>(0), std::plus<>(), std::multiplies<>());
        inputHistory.push_front(input);
        outputHistory.push_front(output);
        return output;
    } else {
        static_assert(gr::meta::always_false<T>, "should not reach here");
    }
}

template<typename T, std::size_t bufferSize, Form form = std::is_floating_point_v<T> ? Form::DF_II : Form::DF_I, auto execPolicy = std::execution::unseq>
inline constexpr std::vector<T> computeImpulseResponse(Section<T, bufferSize>& section, std::size_t length) {
    std::vector<T> impulseResponse(length, T(0));
    T              input = T(1); // impulse response: first input is 1
    for (std::size_t i = 0; i < length; ++i) {
        impulseResponse[i] = computeFilter<T, bufferSize, form, execPolicy>(input, section);
        input              = T(0); // subsequent inputs are 0
    }
    section.reset(T(0));
    return impulseResponse;
}

template<arithmetic_or_complex_like T>
inline constexpr std::vector<T> computeAutoCorrelation(const std::vector<T>& impulseResponse) {
    const std::size_t length = impulseResponse.size();
    std::vector<T>    autoCorrelation(length, T(0));
    for (std::size_t lag = 0; lag < length; ++lag) {
        for (std::size_t i = 0; i < length - lag; ++i) {
            autoCorrelation[lag] += impulseResponse[i] * impulseResponse[i + lag];
        }
    }
    return autoCorrelation;
}

template<typename T, std::size_t bufferSize, typename TBaseType>
struct Section : public FilterCoefficients<TBaseType> {
    // note: bufferSize as upper maximum, since most IIR filter sections will have to be much smaller (for numerical stability reasons)
    HistoryBuffer<T, bufferSize> inputHistory{};
    HistoryBuffer<T, bufferSize> outputHistory{};
    std::vector<T>               autoCorrelation{}; // w.r.t. impulse response, computed for the combined feed-forward and -feedback filter length only

    explicit Section(const FilterCoefficients<TBaseType>& section)
    requires(bufferSize == std::dynamic_extent)
        : FilterCoefficients<TBaseType>(section), inputHistory(section.b.size()), outputHistory(section.a.size()) {
        auto impulseResponse = computeImpulseResponse(*this, section.a.size() + section.b.size());
        autoCorrelation      = computeAutoCorrelation(impulseResponse);
    }

    explicit Section(const FilterCoefficients<TBaseType>& section)
    requires(bufferSize != std::dynamic_extent)
        : FilterCoefficients<TBaseType>(section) {
        auto impulseResponse = computeImpulseResponse(*this, section.a.size() + section.b.size());
        autoCorrelation      = computeAutoCorrelation(impulseResponse);
    }

    inline constexpr void reset(T defaultValue = T()) {
        inputHistory.reset(defaultValue);
        outputHistory.reset(defaultValue);
    }
};

} // namespace detail

/**
 * @brief: Infinite-Impulse-Response (IIR) as well as Finite-Impulse-Response (FIR) filter based on a single or set of biquad filter coefficients.
 *
 * usage example:
 * Filter<double> myFilter(filterSections);
 * double outputSample = myFilter.processOne(inputSample);
 */
template<typename T, std::size_t bufferSize = std::dynamic_extent, Form form = std::is_floating_point_v<meta::fundamental_base_value_type_t<T>> ? Form::DF_II : Form::DF_I, auto execPolicy = std::execution::unseq>
struct Filter;

template<typename T, std::size_t bufferSize, Form form, auto execPolicy>
requires(std::is_arithmetic_v<T>)
struct Filter<T, bufferSize, form, execPolicy> {
    using TBaseType = meta::fundamental_base_value_type_t<T>;
    alignas(64UZ) std::vector<detail::Section<T, bufferSize>> _sectionsMeanValue;

    constexpr Filter() noexcept { _sectionsMeanValue.emplace_back(FilterCoefficients<TBaseType>{.b = {1}, .a = {1}}); }

    template<typename... TFilterCoefficients>
    explicit Filter(TFilterCoefficients&&... filterSections) noexcept {
        std::vector<FilterCoefficients<TBaseType>> filterSections_{std::forward<TFilterCoefficients>(filterSections)...};
        _sectionsMeanValue.reserve(filterSections_.size());
        for (const auto& section : filterSections_) {
            _sectionsMeanValue.emplace_back(section);
        }
    }

    constexpr void reset(T defaultValue = T()) {
        std::for_each(_sectionsMeanValue.begin(), _sectionsMeanValue.end(), [&defaultValue](auto& section) { section.reset(defaultValue); });
    }

    [[nodiscard]] inline constexpr T processOne(T inputSample) noexcept {
        return std::accumulate(_sectionsMeanValue.begin(), _sectionsMeanValue.end(), inputSample, [](T acc, auto& section) { return detail::computeFilter<T, bufferSize, form, execPolicy>(acc, section); });
    }
};

/**
 * @brief: Infinite-Impulse-Response (IIR) as well as Finite-Impulse-Response (FIR) filter based on a single or set of biquad filter coefficients.
 *
 * This includes the computation of the propagation of uncertainty according to:
 * (σ_y[0])² = ∑_{i=0}^{M} (b[i])²·(σ_x[i])² + ∑_{j=1}^{N} ∑_{k=1}^{N} a[j]·a[k]·R_{yy}[|j-k|]·σ_y[j]·σ_y[k]
 * with R_{yy} being the auto-correlation function of the impulse response as an estimate of the covariance matrix.
 *
 * usage example:
 * Filter<UncertainValue<double>> myFilter(filterSections);
 * UncertainValue<double> outputSample = myFilter.processOne({inputSample, noise});
 * double mean   = gr::value(outputSample);
 * double stddev = gr::uncertainty(outputSample);
 */
template<typename T, std::size_t bufferSize, Form form, auto execPolicy>
requires(std::is_arithmetic_v<meta::fundamental_base_value_type_t<T>>)
struct Filter<UncertainValue<T>, bufferSize, form, execPolicy> {
    using TBaseType = meta::fundamental_base_value_type_t<T>;
    alignas(64UZ) std::vector<detail::Section<TBaseType, bufferSize>> _sectionsMeanValue;
    alignas(64UZ) std::vector<detail::Section<TBaseType, bufferSize>> _sectionsSquareUncertaintyValue;

    [[nodiscard]] inline constexpr TBaseType propagateError(const TBaseType& inputUncertainty, detail::Section<TBaseType, bufferSize>& section) noexcept {
        const auto& a                       = section.a;
        const auto& b                       = section.b;
        auto&       inputHistory            = section.inputHistory;
        auto&       outputHistory           = section.outputHistory;
        const auto& autocorrelationFunction = section.autoCorrelation;

        // Feed-forward path (uncorrelated uncertainties)
        inputHistory.push_front(inputUncertainty * inputUncertainty);
        TBaseType feedForwardUncertainty = std::transform_reduce(execPolicy, b.cbegin(), b.cend(), inputHistory.cbegin(), static_cast<TBaseType>(0), //
            std::plus<>(), [](TBaseType bVal, TBaseType sigma2) { return bVal * bVal * sigma2; });

        if (a.size() <= 1 || autocorrelationFunction.empty()) {
            outputHistory.push_front(feedForwardUncertainty);
            return feedForwardUncertainty;
        }

        // Feedback path (correlated uncertainties)
        TBaseType feedbackUncertainty = 0;
        for (std::size_t j = 1; j < a.size(); ++j) {
            for (std::size_t k = 1; k < a.size(); ++k) {
                int       jk{std::abs(static_cast<int>(j) - static_cast<int>(k))};
                TBaseType correlationFactor = autocorrelationFunction[static_cast<std::size_t>(jk)]; // w/o causality (i.e. causality j - k < 0 -> autoC = 0.0), this is a conservative estimate, to be checked
                feedbackUncertainty += a[j] * a[k] * correlationFactor * std::sqrt(outputHistory[j - 1]) * std::sqrt(outputHistory[k - 1]);
            }
        }

        TBaseType totalUncertainty = feedForwardUncertainty + feedbackUncertainty;
        outputHistory.push_front(totalUncertainty);
        return totalUncertainty;
    }

    constexpr Filter() noexcept { _sectionsMeanValue.emplace_back(FilterCoefficients<TBaseType>{.b = {1}, .a = {1}}); }

    template<typename... TFilterCoefficients>
    explicit Filter(TFilterCoefficients&&... filterSections) noexcept {
        std::vector<FilterCoefficients<TBaseType>> filterSections_{std::forward<TFilterCoefficients>(filterSections)...};
        _sectionsMeanValue.reserve(filterSections_.size());
        _sectionsSquareUncertaintyValue.reserve(filterSections_.size());
        for (const auto& section : filterSections_) {
            _sectionsMeanValue.emplace_back(section);
            _sectionsSquareUncertaintyValue.emplace_back(section);
        }
    }

    constexpr void reset(UncertainValue<T> defaultValue = T()) {
        std::for_each(_sectionsMeanValue.begin(), _sectionsMeanValue.end(), [&defaultValue](auto& section) { section.reset(gr::value(defaultValue)); });
        std::for_each(_sectionsSquareUncertaintyValue.begin(), _sectionsSquareUncertaintyValue.end(), [&defaultValue](auto& section) { section.reset(gr::uncertainty(defaultValue) * gr::uncertainty(defaultValue)); });
    }

    [[nodiscard]] inline constexpr UncertainValue<T> processOne(UncertainValue<T> inputSample) noexcept {
        TBaseType value       = std::accumulate(_sectionsMeanValue.begin(), _sectionsMeanValue.end(), gr::value(inputSample), [](TBaseType acc, auto& section) { return detail::computeFilter<TBaseType, bufferSize, form, execPolicy>(acc, section); });
        TBaseType uncertainty = std::accumulate(_sectionsSquareUncertaintyValue.begin(), _sectionsSquareUncertaintyValue.end(), gr::uncertainty(inputSample), [this](TBaseType acc, auto& section) { return propagateError(acc, section); });
        return {value, std::sqrt(uncertainty)};
    }
};

template<typename T, std::size_t bufferSize = std::dynamic_extent, Form form = std::is_floating_point_v<T> ? Form::DF_II : Form::DF_I, auto execPolicy = std::execution::unseq, typename TBaseType = meta::fundamental_base_value_type_t<T>>
struct ErrorPropagatingFilter {
    Filter<T, bufferSize, form, execPolicy>         filterMean;
    Filter<TBaseType, bufferSize, form, execPolicy> filterSquared;

    constexpr ErrorPropagatingFilter() noexcept {
        filterMean    = Filter<T, bufferSize, form, execPolicy>(FilterCoefficients<TBaseType>{.b = {1}, .a = {1}});
        filterSquared = Filter<TBaseType, bufferSize, form, execPolicy>(FilterCoefficients<TBaseType>{.b = {1}, .a = {1}});
    }

    template<typename... TFilterCoefficients>
    explicit ErrorPropagatingFilter(TFilterCoefficients&&... filterSections) : filterMean(filterSections...), filterSquared(filterSections...) {}

    constexpr void reset(T defaultValue = T()) {
        filterMean.reset(defaultValue);
        filterSquared.reset(gr::value(defaultValue) * gr::value(defaultValue));
    }

    gr::UncertainValue<TBaseType> processOne(const T& inputSample) {
        const T         mean     = filterMean.processOne(inputSample);
        const TBaseType meanVal  = gr::value(mean);
        const TBaseType meanUnc  = gr::uncertainty(mean);
        const TBaseType inputVal = gr::value(inputSample);
        const TBaseType square   = filterSquared.processOne(inputVal * inputVal);

        if constexpr (UncertainValueLike<T>) {
            return {meanVal, std::sqrt(std::abs(square - meanVal * meanVal) + meanUnc * meanUnc)};
        } else {
            return {meanVal, std::sqrt(std::abs(square - meanVal * meanVal))};
        }
    }
};

template<typename... Coeffs>
[[nodiscard]] inline constexpr std::size_t countFilterCoefficients(const Coeffs&... coeffs) {
    std::size_t count   = 0;
    auto        counter = [&count](const auto& filter) mutable {
        if constexpr (std::is_same_v<std::decay_t<decltype(filter)>, FilterCoefficients<typename std::decay_t<decltype(filter)>::value_type>>) {
            // It's a single FilterCoefficients object
            count += filter.b.size() + filter.a.size() - 1;
        } else {
            // It's a std::vector of FilterCoefficients objects
            for (const auto& fc : filter) {
                count += fc.b.size() + fc.a.size() - 1;
            }
        }
    };
    (counter(coeffs), ...);
    return count;
}

enum class ResponseType { Magnitude, MagnitudeDB, Phase, PhaseDegrees };

template<Frequency frequencyType, ResponseType responseType, std::floating_point T, typename... TFilterCoefficients>
[[nodiscard]] inline T calculateResponse(T normalisedDigitalFrequency, TFilterCoefficients... filterCoefficients) {
    using C = std::complex<T>;
    static_assert(frequencyType == Frequency::Normalised, "Frequency::Hertz not applicable for digital filters");
    std::vector<FilterCoefficients<T>> filterCoefficients_{std::forward<TFilterCoefficients>(filterCoefficients)...};

    // e^(i*omega) term for the frequency
    const std::complex<T> iOmega   = std::polar(static_cast<T>(1), frequencyType == Frequency::RadianPerSec ? normalisedDigitalFrequency : (static_cast<T>(2) * std::numbers::pi_v<T> * normalisedDigitalFrequency));
    T                     response = responseType == ResponseType::Magnitude ? static_cast<T>(1) : static_cast<T>(0);

    for (const auto& filter : filterCoefficients_) {
        // calculates numerator and denominator of the transfer function H(z) = B(z)/A(z)
        const auto power       = [iOmega, n = 0UZ](C acc, T coefficient) mutable { return acc + coefficient * static_cast<C>(std::pow(iOmega, -static_cast<int>(n++))); };
        C          numerator   = std::accumulate(filter.b.begin(), filter.b.end(), C(0), power);
        C          denominator = std::accumulate(filter.a.begin(), filter.a.end(), C(0), power);

        if constexpr (responseType == ResponseType::Magnitude) {
            response *= std::abs(numerator / denominator);
        } else if constexpr (responseType == ResponseType::MagnitudeDB) {
            response += static_cast<T>(20) * std::log10(std::abs(numerator / denominator));
        } else if constexpr (responseType == ResponseType::Phase) {
            response += (std::arg(numerator) - std::arg(denominator));
        } else if constexpr (responseType == ResponseType::PhaseDegrees) {
            response += (std::arg(numerator) - std::arg(denominator)) * 180. / std::numbers::pi;
        }
    }

    if constexpr (responseType == ResponseType::Phase) {
        return std::fmod(response + std::numbers::pi, 2 * std::numbers::pi) - std::numbers::pi; // [-pi, +pi]
    } else if constexpr (responseType == ResponseType::PhaseDegrees) {
        return std::fmod(response + 180.0, 360.) - 180.0; // [-180°, +180°]
    } else {
        return response;
    }
}

template<typename T>
[[nodiscard]] inline constexpr std::pair<bool, T> normaliseFilterCoefficients(FilterCoefficients<T>& coefficients, T normalisedFrequency, T targetGain = static_cast<T>(1)) {
    const T magnitude = calculateResponse<Frequency::Normalised, ResponseType::Magnitude>(normalisedFrequency, coefficients);
    if (magnitude == 0) {
        return {false, magnitude};
    }
    std::ranges::transform(coefficients.b, coefficients.b.begin(), [magnitude, targetGain](T coeff) { return coeff * targetGain / magnitude; });
    return {true, magnitude};
}

namespace iir {

enum class Design {
    BUTTERWORTH = 0, /// Maximally flat pass-band magnitude response without ripples, commonly used for its smooth response.
    BESSEL      = 1, /// Linear phase response with a gentle roll-off, ideal for preserving time-domain wave shapes.
    CHEBYSHEV1  = 2, /// Equi-ripple pass-band with a steeper roll-off than Butterworth, at the expense of pass-band ripple.
    CHEBYSHEV2  = 3  /// Equi-ripple stop-band with a steeper roll-off, but with ripple only in the stop-band.
};

/**
 * @brief Transfer function H(s) (analog) or H(z) (digitialy) represented in terms of its poles and zeros, with
 *   H(s) = gain·(s - zero[0])·(s - zero[1])· … / ((s - pole[0])·(s - pole[1])· …), or
 *   H(z) = gain·(1 - zero[0]·z^{-1})·(1 - zero[1]·z^{-1})· … / ((1 - pole[0]·z^{-1})·(1 - pole[1]·z^{-1})· …)
 *
 * @note poles and zeros are complex numbers!
 */
struct PoleZeroLocations {
    using value_type = double;
    std::vector<std::complex<double>> poles{};   /// locations in the s- or z-plane where the transfer function goes to infinity.
    std::vector<std::complex<double>> zeros{};   /// locations in the s- or z-plane where the transfer function becomes zero.
    double                            gain{1.0}; /// scaling factor applied to the transfer function
};

template<typename T>
concept HasPoleZeroLocations = requires(T t) {
    typename T::value_type;
    { t.zeros } -> std::convertible_to<std::vector<std::complex<typename T::value_type>>&>;
    { t.poles } -> std::convertible_to<std::vector<std::complex<typename T::value_type>>&>;
};

static_assert(HasPoleZeroLocations<PoleZeroLocations>);

template<Frequency frequencyType, ResponseType responseType>
[[nodiscard]] inline constexpr double calculateResponse(double frequency, const PoleZeroLocations& value) {
    using C = std::complex<double>;
    static_assert(frequencyType != Frequency::Normalised, "Frequency::Normalised not applicable for analog filters");

    // s -> i·ω
    const C    iOmega{0, frequencyType == Frequency::RadianPerSec ? frequency : (2. * std::numbers::pi * frequency)};
    const auto product_over_range = [&iOmega](const auto& range) { return std::accumulate(range.begin(), range.end(), C{1.0}, [&iOmega](const C& acc, const C& val) { return acc * (iOmega - val); }); };
    if constexpr (responseType == ResponseType::Magnitude) {
        return value.gain * std::abs(product_over_range(value.zeros) / product_over_range(value.poles));
    } else if constexpr (responseType == ResponseType::MagnitudeDB) {
        return 20.0 * std::log10(std::abs(value.gain * std::abs(product_over_range(value.zeros) / product_over_range(value.poles))));
    } else if (responseType == ResponseType::Phase) {
        return (std::arg(value.gain * std::abs(product_over_range(value.zeros))) - std::arg(product_over_range(value.poles)));
    } else if (responseType == ResponseType::PhaseDegrees) {
        return (std::arg(value.gain * std::abs(product_over_range(value.zeros))) - std::arg(product_over_range(value.poles))) * 180. / std::numbers::pi;
    }
}

[[nodiscard]] inline constexpr PoleZeroLocations calculateFilterButterworth(std::size_t order) {
    // Butterworth design criteria: https://en.wikipedia.org/wiki/Butterworth_filter
    // place poles equally spaced along the lhs unit half-circle starting with
    // the real-pole at -1 if needed and then continue adding complex conjugate pairs
    PoleZeroLocations ret;
    ret.poles.reserve(order);
    if (order % 2 != 0) {
        ret.poles.emplace_back(-1.0); // real pole for odd orders
    }

    for (std::size_t i = 0UZ; i < order / 2; ++i) {
        double                     theta = std::numbers::pi * (1.0 - static_cast<double>(i * 2 + 1 + order % 2) / (2.0 * static_cast<double>(order)));
        const std::complex<double> pole  = std::polar(1.0, theta);
        ret.poles.emplace_back(pole.real(), +pole.imag());
        ret.poles.emplace_back(pole.real(), -pole.imag()); // conjugate pair, for numerical precision
    }

    return ret;
}

[[nodiscard]] inline constexpr PoleZeroLocations calculateFilterBessel(std::size_t order) {
    // pole location data: Steve Winder, "Filter Design," Newnes Press, 1998.
    using C = std::complex<double>;
    switch (order) {
    case 0: [[fallthrough]];
    case 1: return {.poles = {C{-1.0000}}, .gain = 1.0};
    case 2: return {.poles = {C{-1.1030, 0.6368}, C{-1.1030, -0.6368}}, .gain = 1.6221};
    case 3: return {.poles = {C{-1.0509}, C{-1.3270, 1.0025}, C{-1.3270, -1.0025}}, .gain = 2.9067};
    case 4: return {.poles = {C{-1.3596, 0.4071}, C{-1.3596, -0.4071}, C{-0.9877, 1.2476}, C{-0.9877, -1.2476}}, .gain = 5.1002};
    case 5: return {.poles = {C{-1.3851}, C{-0.9606, 1.4756}, C{-0.9606, -1.4756}, C{-1.5069, 0.7201}, C{-1.5069, -0.7201}}, .gain = 11.9773};
    case 6: return {.poles = {C{-1.5735, 0.3213}, C{-1.5735, -0.3213}, C{-1.3836, 0.9727}, C{-1.3836, -0.9727}, C{-0.9318, 1.6640}, C{-0.9318, -1.6640}}, .gain = 26.8334};
    case 7: return {.poles = {C{-1.6130}, C{-1.3797, 0.5896}, C{-1.3797, -0.5896}, C{-1.1397, 1.1923}, C{-1.1397, -1.1923}, C{-0.9104, 1.8375}, C{-0.9104, -1.8375}}, .gain = 41.5419};
    case 8: return {.poles = {C{-1.7627, 0.2737}, C{-1.7627, -0.2737}, C{-0.8955, 2.0044}, C{-0.8955, -2.0044}, C{-1.3780, 0.8253}, C{-1.3780, -0.8253}, C{-1.6419, 1.3926}, C{-1.6419, -1.3926}}, .gain = 183.3982};
    case 9: return {.poles = {C{-1.8081}, C{-1.6532, 0.5126}, C{-1.6532, -0.5126}, C{-1.16532, 1.0319}, C{-1.16532, -1.0319}, C{-1.3683, 1.5685}, C{-1.3683, -1.5685}, C{-0.8788, 2.1509}, C{-0.8788, -2.1509}}, .gain = 306.9539};
    case 10: return {.poles = {C{-1.9335, 0.2424}, C{-1.9335, -0.2424}, C{-0.8684, 2.2996}, C{-0.8684, -2.2996}, C{-1.8478, 0.7295}, C{-1.8478, -0.7295}, C{-1.6669, 1.2248}, C{-1.6669, -1.2248}, C{-1.3649, 1.7388}, C{-1.3649, -1.7388}}, .gain = 1893.1098};
    default: throw std::out_of_range("supported orders are 1 to 10");
    }
}

[[nodiscard]] inline constexpr PoleZeroLocations calculateFilterChebyshevType1(std::size_t order, double rippleDb = 0.1) {
    // Cauer, W. "The realization of impedances of prescribed frequency dependence." Annalen der Physik 401.2 (1930): 157-229.
    // see also: https://en.wikipedia.org/wiki/Chebyshev_filter

    // calculate the epsilon value based on the pass-band ripple
    const double epsilon = std::sqrt(std::pow(10, rippleDb / 10.0) - 1);
    // calculate the shifter value based on epsilon and the filter order
    const double shifter = std::asinh(1 / epsilon) / static_cast<double>(order);

    PoleZeroLocations value;
    for (std::size_t k = 0; k < order; ++k) {
        double angle = std::numbers::pi * (static_cast<double>(k) + 0.5) / static_cast<double>(order);
        value.poles.emplace_back(-std::sinh(shifter) * std::sin(angle), std::cosh(shifter) * std::cos(angle));
    }
    value.gain = 1.0 / calculateResponse<Frequency::RadianPerSec, ResponseType::Magnitude>(0.0, value);
    return value;
}

[[nodiscard]] inline PoleZeroLocations calculateFilterChebyshevType2(std::size_t numPoles, double stopBandDb = 40) {
    // Cauer, W. "The realization of impedances of prescribed frequency dependence." Annalen der Physik 401.2 (1930): 157-229.
    // see also: https://en.wikipedia.org/wiki/Chebyshev_filter#Poles_and_zeroes_2
    const double epsilon = 1.0 / std::sqrt(std::pow(10, stopBandDb / 10.0) - 1);
    const double v0      = std::asinh(1.0 / epsilon) / static_cast<double>(numPoles);
    const double sinh_v0 = -std::sinh(v0);
    const double cosh_v0 = std::cosh(v0);

    PoleZeroLocations ret;
    for (std::size_t k = 1UZ; k < numPoles; k += 2) {
        const double theta = 0.5 * (static_cast<double>(k) - static_cast<double>(numPoles)) / static_cast<double>(numPoles);
        const double a     = sinh_v0 * std::cos(std::numbers::pi * theta);
        const double b     = cosh_v0 * std::sin(std::numbers::pi * theta);
        const double d2    = a * a + b * b;

        ret.poles.emplace_back(a / d2, +b / d2);
        ret.poles.emplace_back(a / d2, -b / d2); // add conjugate pole pair

        const double im = 1.0 / std::cos(0.5 * std::numbers::pi * static_cast<double>(k) / static_cast<double>(numPoles));
        ret.zeros.emplace_back(0, +im);
        ret.zeros.emplace_back(0, -im); // add conjugate zero pair
    }

    if (numPoles & 1) {
        ret.poles.emplace_back(1.0 / sinh_v0, 0); // add single real-valued pole
        // a zero at infinity is implicitly represented and doesn't need to be stored in 'zeros' vector.
    }
    ret.gain = 1.0 / calculateResponse<Frequency::RadianPerSec, ResponseType::Magnitude>(0.0, ret);
    return ret;
}

[[nodiscard]] inline constexpr PoleZeroLocations analogToDigitalTransform(const PoleZeroLocations& analogPoleZeros, double samplingRate) {
    // see: https://en.wikipedia.org/wiki/Bilinear_transform#Discrete-time_approximation
    const double twoFs             = 2. * samplingRate;
    auto         bilinearTransform = [twoFs](std::complex<double> s) {
        return (twoFs + s) / (twoFs - s); // using Tustin's default method
    };

    PoleZeroLocations digitalPoleZeros;
    digitalPoleZeros.poles.reserve(analogPoleZeros.poles.size());
    digitalPoleZeros.zeros.reserve(analogPoleZeros.zeros.size());

    std::transform(analogPoleZeros.poles.begin(), analogPoleZeros.poles.end(), std::back_inserter(digitalPoleZeros.poles), bilinearTransform);
    std::transform(analogPoleZeros.zeros.begin(), analogPoleZeros.zeros.end(), std::back_inserter(digitalPoleZeros.zeros), bilinearTransform);
    digitalPoleZeros.gain = analogPoleZeros.gain;

    return digitalPoleZeros;
}

namespace details {

template<std::floating_point T>
std::vector<std::complex<T>> sortComplexWithConjugates(const std::vector<std::complex<T>>& numbers) {
    constexpr T epsilon = 1e-10;

    // Separate the numbers into three groups: positive imaginary, negative imaginary, and real-only
    std::vector<std::complex<T>> positiveImag, negativeImag, realOnly;
    for (const auto& num : numbers) {
        if (num.imag() > epsilon) {
            positiveImag.push_back(num);
        } else if (num.imag() < -epsilon) {
            negativeImag.push_back(num);
        } else {
            realOnly.push_back(num);
        }
    }

    // Sort each group based on the real part
    auto realComparator = [](const std::complex<T>& a, const std::complex<T>& b) { return a.real() < b.real(); };
    std::sort(positiveImag.begin(), positiveImag.end(), realComparator);
    std::sort(negativeImag.begin(), negativeImag.end(), realComparator);
    std::sort(realOnly.begin(), realOnly.end(), realComparator);

    // Merge the two groups of complex numbers
    std::vector<std::complex<T>> sorted;
    for (size_t i = 0; i < negativeImag.size(); ++i) {
        sorted.push_back(negativeImag[i]);
        if (i < positiveImag.size()) {
            sorted.push_back(positiveImag[i]);
        }
    }

    // If there are any remaining numbers in the positiveImag group, append them
    for (size_t i = negativeImag.size(); i < positiveImag.size(); ++i) {
        sorted.push_back(positiveImag[i]);
    }

    // Append the real-only numbers at the end
    sorted.insert(sorted.end(), realOnly.begin(), realOnly.end());

    return sorted;
}

} // namespace details

template<typename T, std::ranges::input_range Range>
[[nodiscard]] std::vector<T> expandRootsToPolynomial(Range&& roots, std::size_t desiredOrder) {
    if (roots.empty()) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference" // gcc13 false positive
        std::vector<T> coefficients(desiredOrder + 1UZ, static_cast<T>(0));
        coefficients[0UZ] = static_cast<T>(1);
#pragma GCC diagnostic pop
        return coefficients;
    }

    constexpr T    epsilon      = static_cast<T>(1e-10);
    std::vector<T> coefficients = {static_cast<T>(1)}; // Starts with "x^0" coefficient (1.0)

    auto convolve = [](const std::vector<T>& a, const std::vector<T>& b) {
        std::vector<T> result(a.size() + b.size() - 1, static_cast<T>(0));
        for (std::size_t i = 0UZ; i < a.size(); ++i) {
            for (std::size_t j = 0UZ; j < b.size(); ++j) {
                result[i + j] += a[i] * b[j];
            }
        }
        return result;
    };

    for (size_t i = 0; i < roots.size(); ++i) {
        const auto& root = roots[i];

        if (static_cast<T>(std::abs(root.imag())) > epsilon) { // complex root
            if (i + 1 >= roots.size()) {
                throw std::runtime_error(std::format("Unmatched complex root at i={}: {}", i, root)); // Unpaired complex root.
            }

            // ensure the next root is the conjugate pair.
            const auto& next_root = roots[i + 1];
            if (static_cast<T>(std::abs(root.real() - next_root.real())) > epsilon || static_cast<T>(std::abs(root.imag() + next_root.imag())) > epsilon) {
                std::string fmt = "Complex roots {} vs. {} are not conjugate pairs.\nroots:\n{}"; // workaround for missing std::runtime_format (C++26)
                throw std::runtime_error(std::vformat(fmt, std::make_format_args(root, next_root, roots)));
            }

            // use the quadratic factor for the complex root and its conjugate.
            coefficients = convolve(coefficients, {static_cast<T>(1), -static_cast<T>(2 * root.real()), static_cast<T>(std::norm(root))});
            ++i; // skip the next root since it's the conjugate pair.
        } else { // real root
            coefficients = convolve(coefficients, {static_cast<T>(1), -static_cast<T>(root.real())});
        }
    }

    return coefficients;
}

[[nodiscard]] inline constexpr PoleZeroLocations lowPassProtoToLowPass(const PoleZeroLocations& lowPassProto, FilterParameters parameters) {
    PoleZeroLocations lowPass = lowPassProto;

    std::ranges::for_each(lowPass.poles, [fLow = parameters.fLow](std::complex<double>& pole) { pole *= 2. * std::numbers::pi * fLow; });
    std::ranges::for_each(lowPass.zeros, [fLow = parameters.fLow](std::complex<double>& zero) { zero *= 2. * std::numbers::pi * fLow; });

    lowPass.gain = parameters.gain * lowPassProto.gain / calculateResponse<Frequency::RadianPerSec, ResponseType::Magnitude>(0., lowPass); // normalise at DC
    return lowPass;
}

[[nodiscard]] inline constexpr PoleZeroLocations lowPassProtoToHighPass(const PoleZeroLocations& lowPassProto, FilterParameters parameters) {
    PoleZeroLocations highPass = lowPassProto;

    std::ranges::for_each(highPass.poles, [freqHigh = parameters.fHigh](auto& pole) { pole = 2. * std::numbers::pi * freqHigh / pole; });
    if (highPass.zeros.empty()) {
        highPass.zeros.resize(lowPassProto.poles.size()); // zeros at infinity (represented as explicit zeros here)
    } else {
        std::ranges::for_each(highPass.zeros, [freqHigh = parameters.fHigh](auto& zeros) { zeros = 2. * std::numbers::pi * freqHigh / zeros; });
        if (lowPassProto.poles.size() > highPass.zeros.size()) {
            const std::size_t missingZeros = lowPassProto.poles.size() - highPass.zeros.size();
            highPass.zeros.resize(highPass.zeros.size() + missingZeros); // zeros at infinity (represented as explicit zeros here)
        }
    }

    const double normFreq = std::isfinite(parameters.fs) ? parameters.fs : 10 * parameters.fHigh;
    highPass.gain         = parameters.gain * lowPassProto.gain / calculateResponse<Frequency::RadianPerSec, ResponseType::Magnitude>(std::numbers::pi * normFreq, highPass); // gain == 1.0 at Nyquist
    return highPass;
}

[[nodiscard]] inline constexpr PoleZeroLocations lowPassProtoToBandPass(const PoleZeroLocations& lowPassProto, FilterParameters parameters) {
    constexpr double epsilon   = 1e-10;
    const double     omega0    = 2. * std::numbers::pi * std::sqrt(parameters.fLow * parameters.fHigh); // centre frequency
    const double     bandWidth = 2. * std::numbers::pi * std::abs(parameters.fHigh - parameters.fLow);
    const double     Q         = omega0 / bandWidth; // quality factor

    const auto transform = [Q, omega0](const std::complex<double>& s) -> std::pair<std::complex<double>, std::complex<double>> {
        // https://en.wikipedia.org/wiki/Prototype_filter
        // using Zobel transform: s' <- Q(s/ω₀ + ω₀/s)
        // -> 0 = s² - (ω₀/Q)s'·s + ω₀² -> p := -(ω₀/Q)/2·s', q := ω₀²  -> s = -½p ± √[(½p)² - q]
        // s =  ½ω₀/Q·s' ± √[¼(ω₀/Q)²s'² - ω₀²]
        // s =  ½(ω₀/Q·s' ± 2·ω₀√[¼s'²/Q² - 4·Q²])
        // The transformation creates two poles for each original pole
        const std::complex<double> discriminant = 2.0 * omega0 * std::sqrt(s * s / (4.0 * Q * Q) - 1.0);
        const std::complex<double> base         = (omega0 / Q) * s;

        const std::complex<double> s1 = 0.5 * (base + discriminant);
        const std::complex<double> s2 = 0.5 * (base - discriminant);

        return {s1, s2};
    };

    PoleZeroLocations bandPass;
    bandPass.poles.reserve(2UZ * lowPassProto.poles.size());
    bandPass.zeros.reserve(2UZ * lowPassProto.zeros.size());

    for (const auto& pole : lowPassProto.poles) {
        auto [bpPole1, bpPole2] = transform(pole);
        bandPass.poles.push_back(bpPole1);
        bandPass.poles.push_back(bpPole2);
    }

    // In LP to BP transformation, zeros at the origin in the LPF become poles at infinity (or at the cutoff frequencies) in the BPF.
    // additionally, for each zero in LPF, we add two zeros at +/- ω₀ in BPF.
    for (const auto& zero : lowPassProto.zeros) {
        if (std::norm(zero) < epsilon) {
            // for zeros at the origin, we add zeros at +/- omega0 for the BPF.
            bandPass.zeros.emplace_back(0., +omega0);
            bandPass.zeros.emplace_back(0., -omega0);
        } else {
            auto [bpZero1, bpZero2] = transform(zero);
            bandPass.zeros.push_back(bpZero1);
            bandPass.zeros.push_back(bpZero2);
        }
    }

    if (lowPassProto.poles.size() > lowPassProto.zeros.size()) {
        // need to place additional zeros - implicit zeros at infinity in the LPF become zeros at ω₀ in the BPF
        const std::size_t additionalZeros = lowPassProto.poles.size() - lowPassProto.zeros.size();
        for (std::size_t i = 0UZ; i < additionalZeros; ++i) {
            bandPass.zeros.emplace_back(0., 0.);
        }
    }

    bandPass.gain = parameters.gain / calculateResponse<Frequency::RadianPerSec, ResponseType::Magnitude>(omega0, bandPass); // normalise at centre frequency ω₀
    return bandPass;
}

[[nodiscard]] inline constexpr PoleZeroLocations lowPassProtoToBandStop(const PoleZeroLocations& lowPassProto, FilterParameters parameters) {
    constexpr double epsilon   = 1e-10;
    const double     omega0    = 2. * std::numbers::pi * std::sqrt(parameters.fLow * parameters.fHigh); // centre frequency
    const double     bandWidth = 2. * std::numbers::pi * std::abs(parameters.fHigh - parameters.fLow);
    const double     Q         = omega0 / bandWidth; // quality factor

    auto transform = [omega0, Q](const std::complex<double>& s) -> std::pair<std::complex<double>, std::complex<double>> {
        // https://en.wikipedia.org/wiki/Prototype_filter
        // using Zobel transform: 1/s' <- Q(s/ω₀ + ω₀/s)
        // -> 0 = s² - ω₀/(Q·s')·s + ω₀² -> p := -ω₀/(Q·s'), q := ω₀²  -> s = -½p ± √[(½p)² - q]
        // s =  ½ω₀/(Q·s') ± √[¼(ω₀²/(Q²·s'²) - ω₀²]
        // s =  ½ω₀/(Q·s') ± ½ω₀·√[1/(Q·s')² - 4]
        // The transformation creates two poles for each original pole
        std::complex<double> discriminant = 0.5 * omega0 * std::sqrt(1.0 / (Q * Q * s * s) - 4.0);
        std::complex<double> base         = 0.5 * omega0 / (Q * s);

        std::complex<double> s1 = base + discriminant;
        std::complex<double> s2 = base - discriminant;

        return {s1, s2};
    };

    PoleZeroLocations bandStop;
    bandStop.poles.reserve(2UZ * lowPassProto.poles.size());
    bandStop.zeros.reserve(2UZ * lowPassProto.zeros.size());
    for (const auto& pole : lowPassProto.poles) {
        auto [bpPole1, bpPole2] = transform(pole);
        bandStop.poles.push_back(bpPole1);
        bandStop.poles.push_back(bpPole2);
    }

    // In LP to BP transformation, zeros at the origin in the LPF become poles at infinity (or at the cutoff frequencies) in the BPF.
    // Additionally, for each zero in LPF, we add two zeros at +/- omega0 in BPF.
    for (const auto& zero : lowPassProto.zeros) {
        if (std::norm(zero) < epsilon) {
            // For zeros at the origin, we add zeros at +/- omega0 for the BPF.
            bandStop.zeros.emplace_back(0., +omega0);
            bandStop.zeros.emplace_back(0., -omega0);
        } else {
            auto [bpZero1, bpZero2] = transform(zero);
            bandStop.zeros.push_back(bpZero1);
            bandStop.zeros.push_back(bpZero2);
        }
    }

    // if there are implicit zeros at infinity in the LPF, these become zeros at ω₀ in the BPF.
    const std::size_t additionalZeros = lowPassProto.poles.size() - lowPassProto.zeros.size();
    for (std::size_t i = 0UZ; i < additionalZeros; ++i) {
        bandStop.zeros.emplace_back(0., +omega0);
        bandStop.zeros.emplace_back(0., -omega0);
    }

    bandStop.gain = parameters.gain / calculateResponse<Frequency::RadianPerSec, ResponseType::Magnitude>(0., bandStop); // normalise at DC
    return bandStop;
}

[[nodiscard]] inline constexpr PoleZeroLocations designAnalogFilter(const Type filterType, FilterParameters params, const Design filterDesign = Design::BUTTERWORTH) {
    // step 1: continuous-time analog prototype
    PoleZeroLocations analogPoleZeros;
    switch (filterDesign) { // only the Elliptic and inverse Chebyshev (Type II) filters have zeros.
    case Design::BUTTERWORTH: analogPoleZeros = calculateFilterButterworth(params.order); break;
    case Design::CHEBYSHEV1: analogPoleZeros = calculateFilterChebyshevType1(params.order, params.rippleDb); break;
    case Design::CHEBYSHEV2: analogPoleZeros = calculateFilterChebyshevType2(params.order, params.attenuationDb); break;
    case Design::BESSEL: analogPoleZeros = calculateFilterBessel(params.order); break;
    }

    // step 2: frequency transformation (s to analog filter with desired cutoff frequencies)
    if (filterType != Type::HIGHPASS && !std::isfinite(params.fLow)) {
        throw std::invalid_argument("FilterParameters::fLow is NaN -> please set");
    }
    if (filterType != Type::LOWPASS && !std::isfinite(params.fHigh)) {
        throw std::invalid_argument("FilterParameters::fHigh is NaN -> please set");
    }

    switch (filterType) {
    case Type::BANDPASS: analogPoleZeros = lowPassProtoToBandPass(analogPoleZeros, params); break;
    case Type::BANDSTOP: analogPoleZeros = lowPassProtoToBandStop(analogPoleZeros, params); break;
    case Type::HIGHPASS: analogPoleZeros = lowPassProtoToHighPass(analogPoleZeros, params); break;
    case Type::LOWPASS: analogPoleZeros = lowPassProtoToLowPass(analogPoleZeros, params); break;
    }
    return analogPoleZeros;
}

template<std::floating_point T, std::size_t maxSectionSize = std::is_same_v<std::remove_cvref_t<T>, double> ? 4UZ : 2UZ /* == 2 -> aka. 'biquads' */>
requires((maxSectionSize & 1) == 0) // to handle complex conjugate pole-zero pairs
[[nodiscard]] inline constexpr auto designFilter(const Type filterType, FilterParameters params, const Design filterDesign = Design::BUTTERWORTH) {
    PoleZeroLocations analogPoleZeros = designAnalogFilter(filterType, params, filterDesign);

    T referenceFrequency;
    switch (filterType) {
    case Type::BANDPASS: referenceFrequency = std::sqrt(static_cast<T>(params.fLow * params.fHigh)); break;
    case Type::BANDSTOP: referenceFrequency = static_cast<T>(0); break;
    case Type::HIGHPASS: referenceFrequency = static_cast<T>(0.49 * params.fs); break;
    case Type::LOWPASS: referenceFrequency = static_cast<T>(0); break;
    }

    if (!std::isfinite(params.fs)) {
        throw std::invalid_argument("FilterParameters::fs is NaN -> please set");
    }
    // Step 3: discretisation using bi-linear transform (s -> z)
    PoleZeroLocations digitalPoleZeros = analogToDigitalTransform(analogPoleZeros, params.fs);
    // sort poles and zeros because the low-pass-prototype to high-/band-/stop transformation may create additional poles/zeros
    // that are not necessary sorted according to match complex-conjugate pairs
    digitalPoleZeros.poles = details::sortComplexWithConjugates(digitalPoleZeros.poles);
    digitalPoleZeros.zeros = details::sortComplexWithConjugates(digitalPoleZeros.zeros);

    if constexpr (maxSectionSize == 0) {
        // Step 4: Coefficient Calculation and Gain Adjustment
        FilterCoefficients<T> singleSection;
        singleSection.a = expandRootsToPolynomial<T>(digitalPoleZeros.poles, params.order); // 'a' coefficients (denominator)
        singleSection.b = expandRootsToPolynomial<T>(digitalPoleZeros.zeros, params.order); // 'b' coefficients (numerator)

        const auto [ok, actualGain] = normaliseFilterCoefficients(singleSection, referenceFrequency / static_cast<T>(params.fs), static_cast<T>(params.gain));
        if (ok) {
            return singleSection;
        }
        throw std::invalid_argument(std::format("({}, {}, {}) gain correction {} for target gain {} too small for fs = {} f = [{},{}]", //
            magic_enum::enum_name(filterDesign), magic_enum::enum_name(filterType), params.order, actualGain, params.gain, params.fs, params.fLow, params.fHigh));
    } else { // Step 4: convert the poles and zeros into biquad sections.
        constexpr double epsilon = 1e-10;

        std::vector<FilterCoefficients<T>> sections;
        while (!digitalPoleZeros.poles.empty()) {
            // extract and remove poles from the front
            const std::size_t            numPoles = static_cast<T>(std::abs(digitalPoleZeros.poles.front().imag())) > static_cast<T>(epsilon) ? std::min(maxSectionSize, digitalPoleZeros.poles.size()) : 1UZ;
            std::vector<std::complex<T>> sectionPoles(digitalPoleZeros.poles.begin(), digitalPoleZeros.poles.begin() + static_cast<ptrdiff_t>(numPoles));
            digitalPoleZeros.poles.erase(digitalPoleZeros.poles.begin(), digitalPoleZeros.poles.begin() + static_cast<ptrdiff_t>(numPoles));

            // extract and remove zeros from the front
            const std::size_t            numZeros = std::min(maxSectionSize, digitalPoleZeros.zeros.size());
            std::vector<std::complex<T>> sectionZeros;
            if (!digitalPoleZeros.zeros.empty()) { // ensure we have zeros to pair with the pole
                sectionZeros.assign(digitalPoleZeros.zeros.begin(), digitalPoleZeros.zeros.begin() + static_cast<ptrdiff_t>(numZeros));
                digitalPoleZeros.zeros.erase(digitalPoleZeros.zeros.begin(), digitalPoleZeros.zeros.begin() + static_cast<ptrdiff_t>(numZeros));
            }

            // calculate the coefficients from these poles and zeros
            FilterCoefficients<T> section;
            section.a = expandRootsToPolynomial<T>(sectionPoles, numPoles);
            section.b = expandRootsToPolynomial<T>(sectionZeros, numPoles); // Ensure it's the same order as the denominator

            // Adjust the gain for each section if necessary
            const auto [ok, actualGain] = normaliseFilterCoefficients(section, referenceFrequency / static_cast<T>(params.fs), static_cast<T>(params.gain));
            if (ok) {
                sections.push_back(section);
            } else {
                throw std::invalid_argument(std::format("({}, {}, {}) biquad gain correction {} for target gain {} too small for fs = {} f = [{},{}]", //
                    magic_enum::enum_name(filterDesign), magic_enum::enum_name(filterType), params.order, actualGain, params.gain, params.fs, params.fLow, params.fHigh));
            }
        }
        return sections;
    }
}

template<std::floating_point T>
FilterCoefficients<T> designResonatorDigital(T samplingRate, T frequency, T poleRadius, std::source_location location = std::source_location::current()) {
    if (samplingRate <= T(0) || frequency <= T(0) || frequency >= samplingRate / T(2)) {
        throw gr::exception(std::format("samplingRateHz {} and frequencyHz {} must be > 0.", samplingRate, frequency), location);
    }
    if (poleRadius <= T(0) || poleRadius >= T(1)) {
        throw gr::exception(std::format("poleRadius {} must be in (0,1).", poleRadius), location);
    }

    const T theta = T(2) * std::numbers::pi_v<T> * frequency / samplingRate;
    const T a1    = -T(2) * poleRadius * gr::math::cos(theta);                            // a1 = -2⋅r⋅cos(θ)
    const T a2    = poleRadius * poleRadius;                                              // a2 = r²
    return FilterCoefficients<T>{.b = {T(1) + a1 + a2, T(0), T(0)}, .a = {T(1), a1, a2}}; // b0 = 1 + a1 + a2 (DC normalisation)
}

template<std::floating_point T>
FilterCoefficients<T> designResonatorPhysical(T samplingRate, T frequency, T zetaDamping, std::source_location location = std::source_location::current()) {
    if (frequency <= T(0)) {
        throw gr::exception(std::format("frequency {} must be > 0.", frequency), location);
    }
    if (zetaDamping < T(0)) {
        throw gr::exception(std::format("zetaDamping {} must be >= 0.", zetaDamping), location);
    }
    T rApprox = std::exp(-zetaDamping * T(2) * std::numbers::pi_v<T> * frequency / samplingRate); // approximation: r ~ exp(-ζ * omega0 / fs))
    return designResonatorDigital<T>(samplingRate, frequency, std::min(rApprox, T(0.9999)), location);
}

template<std::floating_point T>
FilterCoefficients<T> designResonatorRF(T samplingRateHz, T frequency, T Q, std::source_location location = std::source_location::current()) {
    if (frequency <= 0.0) {
        throw gr::exception(std::format("frequency {} must be > 0.", frequency), location);
    }
    if (Q <= 0.0) {
        throw gr::exception(std::format("Q {} must be > 0.", Q), location);
    }

    const T BW      = frequency / Q;                                            // approximates 3dB BW
    T       rApprox = std::exp(-std::numbers::pi_v<T> * (BW / samplingRateHz)); // approximation: discrete r => exp(-pi * BW/fs):
    return designResonatorDigital<T>(samplingRateHz, frequency, std::min(rApprox, T(0.9999)), location);
}

} // namespace iir

namespace fir {

template<std::floating_point T>
[[nodiscard]] inline constexpr FilterCoefficients<T> generateCoefficients(std::size_t N, gr::algorithm::window::Type window, T fc, T beta = static_cast<T>(1.6)) {
    const T    M    = static_cast<T>(N - 1) / static_cast<T>(2);
    const auto sinc = [](T x, T a = std::numbers::pi_v<T>) noexcept -> T { return x == static_cast<T>(0) ? static_cast<T>(1) : std::sin(a * x) / (a * x); };

    std::vector<T> coefficients(N);
    gr::algorithm::window::create(coefficients, window, beta);

    std::size_t index = 0; // Index variable to keep track of the current index
    std::ranges::transform(coefficients, coefficients.begin(), [&index, M, fc, &sinc](T coeff) { return coeff * static_cast<T>(2) * fc * sinc(static_cast<T>(2) * fc * (static_cast<T>(index++) - M)); });

    return {coefficients};
}

/**
 * @brief Kaiser-window based formula to estimate the number of FIR taps
 *
 * @param attenuationStopBand attenuation in the stopband (in dB),
 * @param transitionWidth Δω is the width of the transition band (normalised to radians/sample).
 * @return always returns odd-number of required taps
 */
[[nodiscard]] inline constexpr std::size_t estimateNumberOfTapsKaiser(double attenuationStopBand, double transitionWidth) {
    auto N = static_cast<std::size_t>(std::ceil((attenuationStopBand - 8.0) / (2.285 * transitionWidth)));
    if (N % 2 == 0) {
        N++;
    }
    return N;
}

[[nodiscard]] inline constexpr double estimateRequiredTransitionWidth(const Type filterType, FilterParameters params) {
    // assumption: increase #taps to achieve the required filter corner frequencies and pass-band attenuation targets
    const double minWidthFromOrder  = 0.1 / static_cast<double>(params.order); // N.B. Butterworth approximation: 20 dB attenuation per decade and order
    double       minTransitionWidth = minWidthFromOrder;
    switch (filterType) {
    case Type::LOWPASS: return std::min(minTransitionWidth, std::min(std::abs(params.fLow / params.fs), std::abs(0.5 - params.fLow / params.fs))); // attenuation at Nyquist and corner frequencies
    case Type::HIGHPASS: return std::min(minTransitionWidth, std::abs(params.fHigh / params.fs));                                                  // attenuation at DC
    case Type::BANDPASS: return std::min(minTransitionWidth, std::min(std::abs(params.fLow / params.fs), std::abs(0.5 - params.fHigh / params.fs)));
    case Type::BANDSTOP: return std::min(minTransitionWidth, std::min(std::abs(0.5 - params.fHigh / params.fs), std::min(params.fLow, 0.5 * std::abs(params.fHigh - params.fLow)) / params.fs));
    }
    return 0.;
}

template<std::floating_point T>
[[nodiscard]] inline constexpr FilterCoefficients<T> designFilter(const Type filterType, FilterParameters params, gr::algorithm::window::Type window = algorithm::window::Type::Kaiser) {
    const std::size_t N = estimateNumberOfTapsKaiser(params.attenuationDb, 2. * std::numbers::pi * estimateRequiredTransitionWidth(filterType, params));

    // design high-pass FIR filter using the window method
    switch (filterType) {
    case Type::LOWPASS: {
        auto lowPassCoefficients    = generateCoefficients<T>(N, window, static_cast<T>(params.fLow / params.fs), static_cast<T>(params.beta));
        const auto [ok, actualGain] = normaliseFilterCoefficients(lowPassCoefficients, static_cast<T>(0), static_cast<T>(params.gain));
        if (ok) {
            return lowPassCoefficients;
        }
        throw std::invalid_argument(std::format("({}, {}, {}) gain correction {} for target gain {} too small for fs = {} f = [{},{}]", //
            magic_enum::enum_name(filterType), params.order, magic_enum::enum_name(window), actualGain, params.gain, params.fs, params.fLow, params.fHigh));
    }
    case Type::HIGHPASS: {
        // generate low-pass filter coefficients with "mirror" frequency (f_s/2 - f_c)
        auto highPassCoefficients = generateCoefficients<T>(N, window, static_cast<T>((0.5 - params.fHigh / params.fs)), static_cast<T>(params.beta));

        for (std::size_t n = 0UZ; n < N; ++n) {
            // apply spectral inversion by multiplying each coefficient with (-1)^n
            highPassCoefficients.b[n] *= (n % 2 == 0 ? 1 : -1);
        }

        const auto [ok, actualGain] = normaliseFilterCoefficients<T>(highPassCoefficients, static_cast<T>(0.48), static_cast<T>(params.gain));
        if (ok) {
            return highPassCoefficients;
        }
        throw std::invalid_argument(std::format("({}, {}, {}) gain correction {} for target gain {} too small for fs = {} f = [{},{}]", //
            magic_enum::enum_name(filterType), params.order, magic_enum::enum_name(window), actualGain, params.gain, params.fs, params.fLow, params.fHigh));
    }
    case Type::BANDPASS: {
        auto bandPassCoefficients = generateCoefficients<T>(N, window, static_cast<T>(params.fLow / params.fs), static_cast<T>(params.beta));
        auto highPassCoefficients = generateCoefficients<T>(N, window, static_cast<T>(params.fHigh / params.fs), static_cast<T>(params.beta));

        std::transform(bandPassCoefficients.b.begin(), bandPassCoefficients.b.end(), highPassCoefficients.b.begin(), bandPassCoefficients.b.begin(), std::minus<>());

        const auto [ok, actualGain] = normaliseFilterCoefficients(bandPassCoefficients, static_cast<T>(sqrt(params.fHigh * params.fLow) / params.fs), static_cast<T>(params.gain));
        if (ok) {
            return bandPassCoefficients;
        }
        throw std::invalid_argument(std::format("({}, {}, {}) gain correction {} for target gain {} too small for fs = {} f = [{},{}]", //
            magic_enum::enum_name(filterType), params.order, magic_enum::enum_name(window), actualGain, params.gain, params.fs, params.fLow, params.fHigh));
    }
    case Type::BANDSTOP: {
        auto       bandStopCoefficients = generateCoefficients<T>(N, window, static_cast<T>(params.fLow / params.fs), static_cast<T>(params.beta));
        const auto highPassCoefficients = generateCoefficients<T>(N, window, static_cast<T>(params.fHigh / params.fs), static_cast<T>(params.beta));

        for (std::size_t n = 0; n < N; ++n) {
            bandStopCoefficients.b[n] -= highPassCoefficients.b[n];

            if (N % 2 != 0 && n == (N - 1) / 2) { // adjust the centre tap if N is odd. even N are not handled here
                bandStopCoefficients.b[n] = 1 - bandStopCoefficients.b[n];
            }
        }

        const auto [ok, actualGain] = normaliseFilterCoefficients(bandStopCoefficients, static_cast<T>(0), static_cast<T>(params.gain));
        if (ok) {
            return bandStopCoefficients;
        }
        throw std::invalid_argument(std::format("({}, {}, {}) gain correction {} for target gain {} too small for fs = {} f = [{},{}]", //
            magic_enum::enum_name(filterType), params.order, magic_enum::enum_name(window), actualGain, params.gain, params.fs, params.fLow, params.fHigh));
    }
    }
    throw std::runtime_error("unexpectedly reached this end");
}

} // namespace fir
} // namespace gr::filter

#endif // GNURADIO_FILTERTOOL_HPP
