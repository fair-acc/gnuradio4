#ifndef GNURADIO_UNCERTAINVALUE_HPP
#define GNURADIO_UNCERTAINVALUE_HPP

#include <atomic>
#include <complex>
#include <concepts>
#include <cstdint>
#include <numbers>
#include <optional>
#include <type_traits>

#include <gnuradio-4.0/meta/utils.hpp>

namespace gr::meta {

/**
 *
 * @brief Propagation of Uncertainties
 *
 * original idea by: Evan Manning, "Uncertainty Propagation in C++", NASA Jet Propulsion Laboratory,
 * C/C++ Users Journal Volume 14, Number 3, March, 1996
 * http://www.pennelynn.com/Documents/CUJ/HTML/14.03/MANNING/MANNING.HTM
 *
 * implements +,-,*,/ operators for basic arithmetic and complex types, for details see:
 * https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulae
 * This implements only propagation of errors between variables that are uncorrelated.
 * A more rigorous treatment would require the calculation and propagation of the
 * corresponding covariance matrix which is out of scope of this implementation.
 */

template<typename T>
concept arithmetic_or_complex_like = std::is_arithmetic_v<T> || complex_like<T>;

template<arithmetic_or_complex_like T>
struct UncertainValue {
    using value_type = T;

    T value       = static_cast<T>(0); /// mean value
    T uncertainty = static_cast<T>(0); /// uncorrelated standard deviation

    // Default constructor
    constexpr UncertainValue() noexcept = default;

    constexpr UncertainValue(T value_, T uncertainty_) noexcept : value(value_), uncertainty(uncertainty_) {}

    explicit(false) constexpr UncertainValue(T value_) noexcept : value(value_), uncertainty(static_cast<T>(0)) {}

    constexpr UncertainValue(const UncertainValue &) noexcept = default;
    constexpr UncertainValue(UncertainValue &&) noexcept      = default;
    constexpr UncertainValue &
    operator=(const UncertainValue &) noexcept
            = default;
    constexpr UncertainValue &
    operator=(UncertainValue &&) noexcept
            = default;
    ~UncertainValue() = default;

    constexpr UncertainValue &
    operator=(const T &other) noexcept {
        value       = other;
        uncertainty = static_cast<T>(0);
        return *this;
    }

    constexpr UncertainValue &
    operator=(T &&other) noexcept {
        value       = std::move(other);
        uncertainty = static_cast<T>(0);
        return *this;
    }

    auto operator<=>(UncertainValue const&) const = default;
};

template<typename T>
UncertainValue(T, T) -> UncertainValue<T>;

template<typename T>
concept UncertainValueLike = requires(T t) {
    typename T::value_type;                                           // UncertainValue must have a nested value_type
    { t.uncertainty } -> std::convertible_to<typename T::value_type>; // Must have an 'uncertainty' field
};

namespace detail {
template<typename T>
struct UncertainValueTypeType {
    using type = T;
};

template<typename T>
struct UncertainValueTypeType<UncertainValue<T>> {
    using type = T;
};
} // namespace detail

template<typename T>
using UncertainValueType_t = detail::UncertainValueTypeType<T>::type;

/********************** some basic math operation definitions *********************************/

template<typename T, typename U>
    requires UncertainValueLike<T> || UncertainValueLike<U>
[[nodiscard]] inline constexpr auto
operator+(const T &lhs, const U &rhs) noexcept {
    using ValueTypeT = UncertainValueType_t<T>;
    using ValueTypeU = UncertainValueType_t<U>;
    if constexpr (UncertainValueLike<T> && UncertainValueLike<U>) {
        using ResultType = decltype(lhs.value + rhs.value);
        if constexpr (complex_like<ValueTypeT> || complex_like<ValueTypeU>) {
            // we are dealing with complex numbers -> use the standard uncorrelated calculation.
            ResultType newUncertainty = { std::hypot(std::real(lhs.uncertainty), std::real(rhs.uncertainty)), std::hypot(std::imag(lhs.uncertainty), std::imag(rhs.uncertainty)) };
            return UncertainValue<ResultType>{ lhs.value + rhs.value, newUncertainty };
        } else {
            // both ValueType[T,U] are arithmetic uncertainties
            return UncertainValue<ResultType>{ lhs.value + rhs.value, std::hypot(lhs.uncertainty, rhs.uncertainty) };
        }
    } else if constexpr (UncertainValueLike<T> && arithmetic_or_complex_like<ValueTypeU>) {
        return T{ lhs.value + rhs, lhs.uncertainty };
    } else if constexpr (arithmetic_or_complex_like<ValueTypeT> && UncertainValueLike<U>) {
        return U{ lhs + rhs.value, rhs.uncertainty };
    } else {
        static_assert(std::is_arithmetic_v<ValueTypeT> && std::is_arithmetic_v<ValueTypeU>);
        return lhs + rhs; // unlikely to be called due to default '+' definition
    }
}

template<UncertainValueLike T, typename U>
inline constexpr T &
operator+=(T &lhs, const U &rhs) noexcept {
    lhs = lhs + rhs;
    return lhs;
}

template<typename T, typename U>
    requires UncertainValueLike<T> || UncertainValueLike<U>
[[nodiscard]] inline constexpr auto
operator-(const T &lhs, const U &rhs) noexcept {
    using ValueTypeT = UncertainValueType_t<T>;
    using ValueTypeU = UncertainValueType_t<U>;
    if constexpr (UncertainValueLike<T> && UncertainValueLike<U>) {
        using ResultType = decltype(lhs.value - rhs.value);
        if constexpr (complex_like<ValueTypeT> || complex_like<ValueTypeU>) {
            // we are dealing with complex numbers -> use the standard uncorrelated calculation.
            ResultType newUncertainty = { std::hypot(std::real(lhs.uncertainty), std::real(rhs.uncertainty)), std::hypot(std::imag(lhs.uncertainty), std::imag(rhs.uncertainty)) };
            return UncertainValue<ResultType>{ lhs.value - rhs.value, newUncertainty };
        } else {
            // both ValueType[T,U] are arithmetic uncertainties
            return UncertainValue<ResultType>{ lhs.value - rhs.value, std::hypot(lhs.uncertainty, rhs.uncertainty) };
        }
    } else if constexpr (UncertainValueLike<T> && arithmetic_or_complex_like<ValueTypeU>) {
        return T{ lhs.value - rhs, lhs.uncertainty };
    } else if constexpr (arithmetic_or_complex_like<ValueTypeT> && UncertainValueLike<U>) {
        return U{ lhs - rhs.value, rhs.uncertainty };
    } else {
        static_assert(std::is_arithmetic_v<ValueTypeT> && std::is_arithmetic_v<ValueTypeU>);
        return lhs - rhs; // unlikely to be called due to default '-' definition
    }
}

template<UncertainValueLike T, typename U>
inline constexpr T &
operator-=(T &lhs, const U &rhs) noexcept {
    lhs = lhs - rhs;
    return lhs;
}

template<typename T, typename U>
    requires UncertainValueLike<T> || UncertainValueLike<U>
[[nodiscard]] inline constexpr auto
operator*(const T &lhs, const U &rhs) noexcept {
    using ValueTypeT = UncertainValueType_t<T>;
    using ValueTypeU = UncertainValueType_t<U>;
    if constexpr (UncertainValueLike<T> && UncertainValueLike<U>) {
        using ResultType = decltype(lhs.value * rhs.value);
        if constexpr (complex_like<ValueTypeT> || complex_like<ValueTypeU>) {
            // we are dealing with complex numbers -> use standard uncorrelated calculation
            ResultType newUncertainty = { std::hypot(std::real(lhs.value) * std::real(rhs.uncertainty), std::real(rhs.value) * std::real(lhs.uncertainty)),
                                          std::hypot(std::imag(lhs.value) * std::imag(rhs.uncertainty), std::imag(rhs.value) * std::imag(lhs.uncertainty)) };
            return UncertainValue<ResultType>{ lhs.value * rhs.value, newUncertainty };
        } else {
            // both ValueType[T,U] are arithmetic uncertainties
            auto combinedUncertainty = std::hypot(lhs.value * rhs.uncertainty, rhs.value * lhs.uncertainty);
            return UncertainValue<ResultType>{ lhs.value * rhs.value, combinedUncertainty };
        }
    } else if constexpr (UncertainValueLike<T> && arithmetic_or_complex_like<ValueTypeU>) {
        return T{ lhs.value * rhs, lhs.uncertainty * rhs };
    } else if constexpr (arithmetic_or_complex_like<ValueTypeT> && UncertainValueLike<U>) {
        return U{ lhs * rhs.value, lhs * rhs.uncertainty };
    } else {
        static_assert(std::is_arithmetic_v<ValueTypeT> && std::is_arithmetic_v<ValueTypeU>);
        return lhs * rhs; // unlikely to be called due to default '*' definition
    }
}

template<UncertainValueLike T, typename U>
inline constexpr T &
operator*=(T &lhs, const U &rhs) noexcept {
    lhs = lhs * rhs;
    return lhs;
}

template<typename T, typename U>
    requires UncertainValueLike<T> || UncertainValueLike<U>
[[nodiscard]] inline constexpr auto
operator/(const T &lhs, const U &rhs) noexcept {
    using ValueTypeT = UncertainValueType_t<T>;
    using ValueTypeU = UncertainValueType_t<U>;
    if constexpr (UncertainValueLike<T> && UncertainValueLike<U>) {
        using ResultType = decltype(lhs.value * rhs.value);
        if constexpr (complex_like<ValueTypeT> || complex_like<ValueTypeU>) {
            // we are dealing with complex numbers -> use standard uncorrelated calculation
            ResultType newUncertainty;
            if constexpr (std::is_arithmetic_v<ValueTypeT> && complex_like<ValueTypeU>) {
                // LHS is real, RHS is complex
                newUncertainty = { std::sqrt(std::pow(lhs.uncertainty / std::real(rhs.value), 2)), std::sqrt(std::pow(std::imag(rhs.uncertainty) * lhs.value / std::norm(rhs.value), 2)) };
            } else if constexpr (complex_like<ValueTypeT> && std::is_arithmetic_v<ValueTypeU>) {
                // LHS is complex, RHS is real
                newUncertainty = { std::hypot(std::real(lhs.uncertainty) / rhs.value, rhs.uncertainty * std::real(lhs.value) / std::pow(rhs.value, 2)),
                                   std::sqrt(std::pow(std::imag(lhs.uncertainty) / rhs.value, 2)) };
            } else {
                newUncertainty = { std::hypot(std::real(lhs.uncertainty) / std::real(rhs.value), std::real(rhs.uncertainty) * std::real(lhs.value) / std::norm(rhs.value)),
                                   std::hypot(std::imag(lhs.uncertainty) / std::imag(rhs.value), std::imag(rhs.uncertainty) * std::imag(lhs.value) / std::norm(rhs.value)) };
            }

            return UncertainValue<ResultType>{ lhs.value / rhs.value, newUncertainty };
        } else {
            // both ValueType[T,U] are arithmetic uncertainties
            auto combinedUncertainty = std::hypot(lhs.uncertainty / rhs.value, rhs.uncertainty * lhs.value / std::pow(rhs.value, 2));
            return UncertainValue<ResultType>{ lhs.value / rhs.value, combinedUncertainty };
        }
    } else if constexpr (UncertainValueLike<T> && arithmetic_or_complex_like<ValueTypeU>) {
        return T{ lhs.value / rhs, lhs.uncertainty / std::abs(rhs) };
    } else if constexpr (arithmetic_or_complex_like<ValueTypeT> && UncertainValueLike<U>) {
        auto rhsMagSquared = std::norm(rhs.value);
        return U{ lhs / rhs.value, rhs.uncertainty * std::abs(lhs) / rhsMagSquared };
    } else {
        static_assert(std::is_arithmetic_v<ValueTypeT> && std::is_arithmetic_v<ValueTypeU>);
        return lhs / rhs; // unlikely to be called due to default '/' definition
    }
}

template<UncertainValueLike T, typename U>
inline constexpr T &
operator/=(T &lhs, const U &rhs) noexcept {
    lhs = lhs / rhs;
    return lhs;
}

} // namespace gr::meta

namespace std { // std:: basic math overloads

template<typename T>
    requires std::is_floating_point_v<T>
[[nodiscard]] inline constexpr T
hypot(const std::complex<T> &a, const std::complex<T> &b) noexcept { // doesn't exist in the STL for some reason
    return std::sqrt(std::real(a * std::conj(a) + b * std::conj(b))); // c*c⃰ == is always real valued
}

template<gr::meta::UncertainValueLike T, typename ValueTypeT = gr::meta::UncertainValueType_t<T>>
[[nodiscard]] inline constexpr auto
pow(const T &base, std::floating_point auto exponent) noexcept {
    if (base.value == 0.0) [[unlikely]] {
        if (exponent == 0) [[unlikely]] {
            return T{ 1, 0 };
        } else {
            return T{ 0, 0 };
        }
    }

    ValueTypeT newValue = std::pow(base.value, exponent);
    if constexpr (gr::meta::complex_like<ValueTypeT>) {
        auto val = exponent / base.value * newValue;
        return T{ newValue, std::sqrt(val * std::conj(val)) * base.uncertainty };
    } else {
        return T{ newValue, std::abs(newValue * exponent * base.uncertainty / base.value) };
    }
}

template<gr::meta::UncertainValueLike T, gr::meta::UncertainValueLike U, typename ValueTypeT = gr::meta::UncertainValueType_t<T>, typename ValueTypeU = gr::meta::UncertainValueType_t<T>>
[[nodiscard]] inline constexpr auto
pow(const T &base, const U &exponent) noexcept {
    static_assert(std::is_same_v<ValueTypeT, ValueTypeU>, "base and exponent must be of the same type");

    if (base.value == 0.0) [[unlikely]] {
        if (exponent.value == static_cast<ValueTypeU>(0)) [[unlikely]] {
            return T{ 1, 0 };
        } else {
            return T{ 0, 0 };
        }
    }

    ValueTypeT newValue = std::pow(base.value, exponent.value);
    if constexpr (gr::meta::complex_like<ValueTypeT>) {
        return T{ newValue, std::hypot(exponent.value / base.value * newValue * base.uncertainty, std::log(base.value) * newValue * exponent.uncertainty) };
    } else {
        return T{ newValue, std::abs(newValue) * std::hypot(exponent.value / base.value * base.uncertainty, std::log(base.value) * exponent.uncertainty) };
    }
}

template<gr::meta::UncertainValueLike T>
[[nodiscard]] inline constexpr auto
sqrt(const T &value) noexcept {
    return std::pow(value, 0.5);
}

template<gr::meta::UncertainValueLike T, typename ValueTypeT = gr::meta::UncertainValueType_t<T>>
[[nodiscard]] inline constexpr auto
sin(const T &x) noexcept {
    return T{ std::sin(x.value), std::abs(std::cos(x.value) * x.uncertainty) };
}

template<gr::meta::UncertainValueLike T, typename ValueTypeT = gr::meta::UncertainValueType_t<T>>
[[nodiscard]] inline constexpr auto
cos(const T &x) noexcept {
    return T{ std::cos(x.value), std::abs(std::sin(x.value) * x.uncertainty) };
}

template<gr::meta::UncertainValueLike T, typename ValueTypeT = gr::meta::UncertainValueType_t<T>>
[[nodiscard]] inline constexpr auto
exp(const T &x) noexcept {
    if constexpr (gr::meta::complex_like<ValueTypeT>) {
        return std::pow(gr::meta::UncertainValue<ValueTypeT>{ std::numbers::e_v<typename ValueTypeT::value_type>, static_cast<ValueTypeT>(0) }, x);
    } else {
        return std::pow(gr::meta::UncertainValue<ValueTypeT>{ std::numbers::e_v<ValueTypeT>, static_cast<ValueTypeT>(0) }, x);
    }
}

} // namespace std

#endif // GNURADIO_UNCERTAINVALUE_HPP