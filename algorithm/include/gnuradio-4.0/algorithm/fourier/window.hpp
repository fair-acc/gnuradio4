
#ifndef GRAPH_PROTOTYPE_ALGORITHM_WINDOW_HPP
#define GRAPH_PROTOTYPE_ALGORITHM_WINDOW_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <ranges>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include <gnuradio-4.0/meta/utils.hpp>

namespace gr::algorithm::window {

/**
 * Implementation of window function (also known as an apodization function or tapering function).
 * See Wikipedia for more info: https://en.wikipedia.org/wiki/Window_function
 */
enum class Type : int { None, Rectangular, Hamming, Hann, HannExp, Blackman, Nuttall, BlackmanHarris, BlackmanNuttall, FlatTop, Exponential, Kaiser };
using enum Type;
inline static constexpr std::array<Type, 12>     TypeList{ None, Rectangular, Hamming, Hann, HannExp, Blackman, Nuttall, BlackmanHarris, BlackmanNuttall, FlatTop, Exponential, Kaiser };
inline static constexpr gr::meta::fixed_string TypeNames = "[None, Rectangular, Hamming, Hann, HannExp, Blackman, Nuttall, BlackmanHarris, BlackmanNuttall, FlatTop, Exponential, Kaiser]";

constexpr std::string_view
to_string(Type window) noexcept {
    switch (window) {
    case None: return "None";
    case Rectangular: return "Rectangular";
    case Hamming: return "Hamming";
    case Hann: return "Hann";
    case HannExp: return "HannExp";
    case Blackman: return "Blackman";
    case Nuttall: return "Nuttall";
    case BlackmanHarris: return "BlackmanHarris";
    case BlackmanNuttall: return "BlackmanNuttall";
    case FlatTop: return "FlatTop";
    case Exponential: return "Exponential";
    case Kaiser: return "Kaiser";
    default: return "Unknown";
    }
}

constexpr Type
parse(std::string_view name) {
    constexpr auto toLower = [](std::string_view sv) {
        std::string lowerStr(sv);
        std::ranges::transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(), ::tolower);
        return lowerStr;
    };

    for (const auto &type : TypeList) {
        if (toLower(to_string(type)) == toLower(name)) {
            return type;
        }
    }

    throw std::invalid_argument(fmt::format("unknown window type '{}'", name));
}

namespace detail {
template<typename T>
    requires std::is_floating_point_v<T>
constexpr T
bessel_i0(const T x) noexcept {
    T       sum    = 1;
    T       term   = 1;
    int     k      = 1;

    const T x_half = x / 2;

    do {
        term *= (x_half / static_cast<T>(k));
        sum += term * term;
        ++k;
    } while (term * term > sum * std::numeric_limits<T>::epsilon());

    return sum;
}
} // namespace detail

/**
 * @brief Creates in-place a window function (mathematically aka. 'apodisation function') of a specified type and size.
 *
 * This function generates various window functions used in digital signal processing.
 * See Wikipedia for more info: https://en.wikipedia.org/wiki/Window_function
 *
 * @tparam T The floating-point type to use for the window function values.
 * @param windowFunction The type of window function to create.
 * @param Container std::vector containing the values of the window function.
 */
template<gr::meta::array_or_vector_type ContainerType, typename T = ContainerType::value_type>
    requires std::is_floating_point_v<T>
void
create(ContainerType &container, Type windowFunction, const T beta = static_cast<T>(1.6)) {
    constexpr T       pi2 = 2 * std::numbers::pi_v<T>;
    const std::size_t n   = container.size();
    if (n == 0) {
        return;
    }

    using enum Type;
    switch (windowFunction) {
    case None:
    case Rectangular: {
        std::ranges::fill(container, 1);
        return;
    }
    case Hamming: {
        // formula: w(n) = 0.54 - 0.46 * cos((2 * pi * n) / (N - 1))
        // reference: Hamming, R. W. (1977). Digital filters. Prentice-Hall.
        const T a = pi2 / static_cast<T>(n);
        std::ranges::transform(std::views::iota(0UL, n), container.begin(), [a](const auto i) { return static_cast<T>(0.53836) - static_cast<T>(0.46164) * std::cos(a * static_cast<T>(i)); });
        return;
    }
    case Hann: {
        // formula: w(n) = 0.5 - 0.5 * cos((2 * pi * n) / (N - 1))
        // reference: von Hann, J. (1901). Über den Durchgang einer elektrischen Welle längs der Erdoberfläche. Elektrische Nachrichtentechnik, 17, 421-424.
        const T a = pi2 / static_cast<T>(n - 1);
        std::ranges::transform(std::views::iota(0UL, n), container.begin(), [a](const auto i) { return static_cast<T>(.5) - static_cast<T>(.5) * std::cos(a * static_cast<T>(i)); });
        return;
    }
    case HannExp: {
        const T a = pi2 / static_cast<T>(n - 1);
        std::ranges::transform(std::views::iota(0UL, n), container.begin(), [a](const auto i) { return std::pow(std::sin(a * static_cast<T>(i)), static_cast<T>(2.)); });
        return;
    }
    case Blackman: {
        // formula: w(n) = 0.42 - 0.5 * cos((2 * pi * n) / (N - 1)) + 0.08 * cos((4 * pi * n) / (N - 1))
        // reference: Blackman, R. B., & Tukey, J. W. (1958). The measurement of power spectra from the point of view of communications engineering. Dover Publications.
        const T a = pi2 / static_cast<T>(n - 1);
        std::ranges::transform(std::views::iota(0UL, n), container.begin(), [a](const auto i) {
            const T ai = a * static_cast<T>(i);
            return static_cast<T>(0.42) - static_cast<T>(0.5) * std::cos(ai) + static_cast<T>(0.08) * std::cos(static_cast<T>(2.) * ai);
        });
        return;
    }
    case Nuttall: {
        // Formula: w(n) = a0 - a1 * cos((2 * pi * n) / (N - 1)) + a2 * cos((4 * pi * n) / (N - 1)) - a3 * cos((6 * pi * n) / (N - 1))
        // Reference: Nuttall, A. (1981). Some Windows with Very Good Sidelobe Behavior. IEEE Transactions on Acoustics, Speech, and Signal Processing, 29(1), 84-91.
        const T a = pi2 / static_cast<T>(n - 1);
        std::ranges::transform(std::views::iota(0UL, n), container.begin(), [a](const auto i) {
            constexpr std::array<T, 4> coeff = { static_cast<T>(0.355768), static_cast<T>(0.487396), static_cast<T>(0.144232), static_cast<T>(0.012604) };
            const T                    ai    = a * static_cast<T>(i);
            return coeff[0] - coeff[1] * std::cos(ai) + coeff[2] * std::cos(2 * ai) - coeff[3] * std::cos(3 * ai);
        });
        return;
    }
    case BlackmanHarris: {
        // formula: w(n) = a0 - a1 * cos((2 * pi * n) / (N - 1)) + a2 * cos((4 * pi * n) / (N - 1)) - a3 * cos((6 * pi * n) / (N - 1))
        // reference: Harris, F. J. (1978). On the use of windows for harmonic analysis with the discrete Fourier transform. Proceedings of the IEEE, 66(1), 51-83.
        const T a = pi2 / static_cast<T>(n - 1);
        std::ranges::transform(std::views::iota(0UL, n), container.begin(), [a](const auto i) {
            constexpr std::array<T, 4> coeff = { static_cast<T>(0.35875), static_cast<T>(0.48829), static_cast<T>(0.14128), static_cast<T>(0.01168) };
            const T                    ai    = a * static_cast<T>(i);
            return coeff[0] - coeff[1] * std::cos(ai) + coeff[2] * std::cos(2 * ai) - coeff[3] * std::cos(3 * ai);
        });
        return;
    }
    case BlackmanNuttall: {
        // formula: w(n) = 0.3635819 - 0.4891775 * cos(2*pi*n/(N-1)) + 0.1365995 * cos(4*pi*n/(N-1)) - 0.0106411 * cos(6*pi*n/(N-1))
        // reference: Generalized from Nuttall, A. (1981) and Harris, F. J. (1978).
        const T a = pi2 / static_cast<T>(n - 1);
        std::ranges::transform(std::views::iota(0UL, n), container.begin(), [a](const auto i) {
            const T ai = a * static_cast<T>(i);
            return static_cast<T>(0.3635819) - static_cast<T>(0.4891775) * std::cos(ai) + static_cast<T>(0.1365995) * std::cos(static_cast<T>(2.) * ai)
                 - static_cast<T>(0.0106411) * std::cos(static_cast<T>(3.) * ai);
        });
        return;
    }
    case FlatTop: {
        // formula: w(n) = a0 - a1 * cos((2 * pi * n) / (N - 1)) + a2 * cos((4 * pi * n) / (N - 1)) - a3 * cos((6 * pi * n) / (N - 1)) + a4 * cos((8 * pi * n) / (N - 1))
        // reference: D'Antona, G., & Ferrero, A. (2006). Digital Signal Processing for Measurement Systems: Theory and Applications. Springer.
        const T a = pi2 / static_cast<T>(n - 1);
        std::ranges::transform(std::views::iota(0UL, n), container.begin(), [a](const auto i) {
            constexpr std::array<T, 5> coeff = { static_cast<T>(1.0), static_cast<T>(1.93), static_cast<T>(1.29), static_cast<T>(0.388), static_cast<T>(0.032) };
            const T                    ai    = a * static_cast<T>(i);
            return coeff[0] - coeff[1] * std::cos(ai) + coeff[2] * std::cos(2 * ai) - coeff[3] * std::cos(3 * ai) + coeff[4] * std::cos(4 * ai);
        });
        return;
    }
    case Exponential: {
        // formula: w(n) = exp(n/a)
        const T exp0 = std::exp(static_cast<T>(0.));
        const T a    = static_cast<T>(3.) * static_cast<T>(n);
        std::ranges::transform(std::views::iota(0UL, n), container.begin(), [a, exp0](const auto i) { return std::exp(static_cast<T>(i) / a) / exp0; });
        return;
    }
    case Kaiser: {
        // formula: w(n) = I0(beta * sqrt(1 - ((2*n/(N-1)) - 1)^2)) / I0(beta)
        // reference: J. F. Kaiser and R. W. Schafer. On the use of the i0-sinh window for spectrum analysis. IEEE Transactions on Acoustics, Speech, and Signal Processing, 28(1):105–107, 1980.
        if (beta < 0) throw std::invalid_argument("beta must be non-negative");
        if (n <= 1) throw std::invalid_argument("n must be larger than one");

        const T factor = static_cast<T>(1) / static_cast<T>(n - 1);
        const T i0Beta = detail::bessel_i0(static_cast<T>(beta)); // Compute the zeroth order modified Bessel function of the first kind for beta
        std::ranges::transform(std::views::iota(0UL, n), container.begin(), [beta, factor, i0Beta](const auto i) {
            const T term = (static_cast<T>(2 * i) * factor) - static_cast<T>(1);
            return detail::bessel_i0(static_cast<T>(beta) * std::sqrt(std::abs(static_cast<T>(1) - term * term))) / i0Beta;
        });
        return;
    }
    }
}

/**
 * @brief Creates a new window function (mathematically aka. 'apodisation function') of a specified type and size.
 *
 * This function generates various window functions used in digital signal processing.
 * See Wikipedia for more info: https://en.wikipedia.org/wiki/Window_function
 *
 * @tparam T The floating-point type to use for the window function values.
 * @param windowFunction The type of window function to create.
 * @param n The size of the window function.
 * @return A std::vector<float> containing the values of the window function.
 */
template<typename T = float>
    requires std::is_floating_point_v<T>
[[nodiscard]] auto
create(Type windowFunction, const std::size_t n, const T beta = static_cast<T>(1.6)) {
    std::vector<T> container(n);
    create(container, windowFunction, beta);
    return container;
}

// this is to speed-up typical instantiations
template void
create<std::vector<float>>(std::vector<float> &container, Type windowFunction, float beta);
template void
create<std::vector<double>>(std::vector<double> &container, Type windowFunction, double beta);

} // namespace gr::algorithm::window

#endif // GRAPH_PROTOTYPE_ALGORITHM_WINDOW_HPP
