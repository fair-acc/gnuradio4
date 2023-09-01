
#ifndef GRAPH_PROTOTYPE_WINDOW_HPP
#define GRAPH_PROTOTYPE_WINDOW_HPP

#include <cmath>
#include <numbers>
#include <vector>

namespace gr::blocks::fft {

template<typename T>
concept FloatOrDoubleType = std::is_same_v<T, float> || std::is_same_v<T, double>;

// Implementation of window function (also known as an apodization function or tapering function).
// See Wikipedia for more info: https://en.wikipedia.org/wiki/Window_function

enum class WindowFunction : int { None, Rectangular, Hamming, Hann, HannExp, Blackman, Nuttall, BlackmanHarris, BlackmanNuttall, FlatTop, Exponential };

// Only float or double are allowed because FFT supports only float or double precisions.
template<FloatOrDoubleType T>
std::vector<T>
createWindowFunction(WindowFunction func, const std::size_t n) {
    constexpr T    pi   = std::numbers::pi_v<T>;
    constexpr T    pi2  = static_cast<T>(2.) * pi;
    const T        exp0 = std::exp(static_cast<T>(0.));
    constexpr T    c1   = 1.;
    constexpr T    c2   = 2.;
    constexpr T    c3   = 3.;
    constexpr T    c4   = 4.;
    std::vector<T> res(n);
    switch (func) {
    case WindowFunction::None: {
        return {};
    }
    case WindowFunction::Rectangular: {
        for (std::size_t i = 0; i < n; i++) res[i] = 1.;
        return res;
    }
    case WindowFunction::Hamming: {
        const T a = pi2 / static_cast<T>(n);
        for (std::size_t i = 0; i < n; i++) res[i] = static_cast<T>(0.53836) - static_cast<T>(0.46164) * std::cos(a * static_cast<T>(i));
        return res;
    }
    case WindowFunction::Hann: {
        const T a = pi2 / static_cast<T>(n - 1);
        for (std::size_t i = 0; i < n; i++) res[i] = static_cast<T>(0.5) - static_cast<T>(0.5) * std::cos(a * static_cast<T>(i));
        return res;
    }
    case WindowFunction::HannExp: {
        const T a = pi2 / static_cast<T>(n - 1);
        for (std::size_t i = 0; i < n; i++) res[i] = std::pow(std::sin(a * static_cast<T>(i)), c2);
        return res;
    }
    case WindowFunction::Blackman: {
        const T a = pi2 / static_cast<T>(n - 1);
        for (std::size_t i = 0; i < n; i++) {
            const T ai = a * static_cast<T>(i);
            res[i]     = static_cast<T>(0.42) - static_cast<T>(0.5) * std::cos(ai) + static_cast<T>(0.08) * std::cos(c2 * ai);
        }
        return res;
    }
    case WindowFunction::Nuttall: {
        const T a = pi2 / static_cast<T>(n - 1);
        for (std::size_t i = 0; i < n; i++) {
            const T ai = a * static_cast<T>(i);
            res[i]     = static_cast<T>(0.355768) - static_cast<T>(0.487396) * std::cos(ai) + static_cast<T>(0.144232) * std::cos(c2 * ai) - static_cast<T>(0.012604) * std::cos(c3 * ai);
        }
        return res;
    }
    case WindowFunction::BlackmanHarris: {
        const T a = pi2 / static_cast<T>(n - 1);
        for (std::size_t i = 0; i < n; i++) {
            const T ai = a * static_cast<T>(i);
            res[i]     = static_cast<T>(0.35875) - static_cast<T>(0.48829) * std::cos(ai) + static_cast<T>(0.14128) * std::cos(c2 * ai) - static_cast<T>(0.01168) * std::cos(c3 * ai);
        }
        return res;
    }
    case WindowFunction::BlackmanNuttall: {
        const T a = pi2 / static_cast<T>(n - 1);
        for (std::size_t i = 0; i < n; i++) {
            const T ai = a * static_cast<T>(i);
            res[i]     = static_cast<T>(0.3635819) - static_cast<T>(0.4891775) * std::cos(ai) + static_cast<T>(0.1365995) * std::cos(c2 * ai) - static_cast<T>(0.0106411) * std::cos(c3 * ai);
        }
        return res;
    }
    case WindowFunction::FlatTop: {
        const T a = pi2 / static_cast<T>(n - 1);
        for (std::size_t i = 0; i < n; i++) {
            const T ai = a * static_cast<T>(i);
            res[i]     = c1 - static_cast<T>(1.93) * std::cos(ai) + static_cast<T>(1.29) * std::cos(c2 * ai) - static_cast<T>(0.388) * std::cos(c3 * ai) + static_cast<T>(0.032) * std::cos(c4 * ai);
        }
        return res;
    }
    case WindowFunction::Exponential: {
        const T a = c3 * static_cast<T>(n);
        for (std::size_t i = 0; i < n; i++) res[i] = std::exp(static_cast<T>(i) / a) / exp0;
        return res;
    }
    default: {
        return {};
    }
    }
}
} // namespace gr::blocks::fft

#endif // GRAPH_PROTOTYPE_WINDOW_HPP
