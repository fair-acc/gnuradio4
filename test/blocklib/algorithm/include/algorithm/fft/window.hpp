
#ifndef GRAPH_PROTOTYPE_ALGORITHM_WINDOW_HPP
#define GRAPH_PROTOTYPE_ALGORITHM_WINDOW_HPP

#include <algorithm>
#include <cmath>
#include <numbers>
#include <ranges>
#include <vector>

namespace gr::algorithm {

/*
 * Implementation of window function (also known as an apodization function or tapering function).
 * See Wikipedia for more info: https://en.wikipedia.org/wiki/Window_function
 */

enum class WindowFunction : int { None, Rectangular, Hamming, Hann, HannExp, Blackman, Nuttall, BlackmanHarris, BlackmanNuttall, FlatTop, Exponential };

// Only float or double are allowed because FFT supports only float or double precisions.
template<typename T>
    requires(std::is_floating_point_v<T>)
std::vector<T>
createWindowFunction(WindowFunction func, const std::size_t n) {
    constexpr T pi  = std::numbers::pi_v<T>;
    constexpr T pi2 = static_cast<T>(2.) * pi;
    constexpr T c1  = 1.;
    constexpr T c2  = 2.;
    constexpr T c3  = 3.;
    constexpr T c4  = 4.;

    switch (func) {
    case WindowFunction::None: {
        return {};
    }
    case WindowFunction::Rectangular: {
        return std::vector<T>(n, T(1.));
    }
    case WindowFunction::Hamming: {
        std::vector<T> res(n);
        const T        a = pi2 / static_cast<T>(n);
        std::ranges::transform(std::views::iota(std::size_t(0), n), std::ranges::begin(res), [a](const auto i) { return T(0.53836) - T(0.46164) * std::cos(a * static_cast<T>(i)); });
        return res;
    }
    case WindowFunction::Hann: {
        std::vector<T> res(n);
        const T        a = pi2 / static_cast<T>(n - 1);
        std::ranges::transform(std::views::iota(std::size_t(0), n), std::ranges::begin(res), [a](const auto i) { return T(0.5) - T(0.5) * std::cos(a * static_cast<T>(i)); });
        return res;
    }
    case WindowFunction::HannExp: {
        std::vector<T> res(n);
        const T        a = pi2 / static_cast<T>(n - 1);
        std::ranges::transform(std::views::iota(std::size_t(0), n), std::ranges::begin(res), [a](const auto i) { return std::pow(std::sin(a * static_cast<T>(i)), c2); });
        return res;
    }
    case WindowFunction::Blackman: {
        std::vector<T> res(n);
        const T        a = pi2 / static_cast<T>(n - 1);
        std::ranges::transform(std::views::iota(std::size_t(0), n), std::ranges::begin(res), [a](const auto i) {
            const T ai = a * static_cast<T>(i);
            return T(0.42) - T(0.5) * std::cos(ai) + T(0.08) * std::cos(c2 * ai);
        });
        return res;
    }
    case WindowFunction::Nuttall: {
        std::vector<T> res(n);
        const T        a = pi2 / static_cast<T>(n - 1);
        std::ranges::transform(std::views::iota(std::size_t(0), n), std::ranges::begin(res), [a](const auto i) {
            const T ai = a * static_cast<T>(i);
            return T(0.355768) - T(0.487396) * std::cos(ai) + T(0.144232) * std::cos(c2 * ai) - T(0.012604) * std::cos(c3 * ai);
        });
        return res;
    }
    case WindowFunction::BlackmanHarris: {
        std::vector<T> res(n);
        const T        a = pi2 / static_cast<T>(n - 1);
        std::ranges::transform(std::views::iota(std::size_t(0), n), std::ranges::begin(res), [a](const auto i) {
            const T ai = a * static_cast<T>(i);
            return T(0.35875) - T(0.48829) * std::cos(ai) + T(0.14128) * std::cos(c2 * ai) - T(0.01168) * std::cos(c3 * ai);
        });
        return res;
    }
    case WindowFunction::BlackmanNuttall: {
        std::vector<T> res(n);
        const T        a = pi2 / static_cast<T>(n - 1);
        std::ranges::transform(std::views::iota(std::size_t(0), n), std::ranges::begin(res), [a](const auto i) {
            const T ai = a * static_cast<T>(i);
            return T(0.3635819) - T(0.4891775) * std::cos(ai) + T(0.1365995) * std::cos(c2 * ai) - T(0.0106411) * std::cos(c3 * ai);
        });
        return res;
    }
    case WindowFunction::FlatTop: {
        std::vector<T> res(n);
        const T        a = pi2 / static_cast<T>(n - 1);
        std::ranges::transform(std::views::iota(std::size_t(0), n), std::ranges::begin(res), [a](const auto i) {
            const T ai = a * static_cast<T>(i);
            return c1 - T(1.93) * std::cos(ai) + T(1.29) * std::cos(c2 * ai) - T(0.388) * std::cos(c3 * ai) + T(0.032) * std::cos(c4 * ai);
        });
        return res;
    }
    case WindowFunction::Exponential: {
        std::vector<T> res(n);
        const T        exp0 = std::exp(T(0.));
        const T        a    = c3 * static_cast<T>(n);
        std::ranges::transform(std::views::iota(std::size_t(0), n), std::ranges::begin(res), [a, exp0](const auto i) { return std::exp(static_cast<T>(i) / a) / exp0; });
        return res;
    }
    default: {
        return {};
    }
    }
}
} // namespace gr::algorithm

#endif // GRAPH_PROTOTYPE_ALGORITHM_WINDOW_HPP
