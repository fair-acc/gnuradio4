#ifndef GNURADIO_FORMATTER_HPP
#define GNURADIO_FORMATTER_HPP

#include <complex>
#include <fmt/format.h>

#include "UncertainValue.hpp"

template<typename T>
struct fmt::formatter<std::complex<T>> {
    char presentation = 'g'; // default format

    template<typename ParseContext>
    constexpr auto
    parse(ParseContext &ctx) {
        auto it = ctx.begin(), end = ctx.end();
        if (it != end && (*it == 'f' || *it == 'F' || *it == 'e' || *it == 'E' || *it == 'g' || *it == 'G')) {
            presentation = *it++;
        }
        if (it != end && *it != '}') {
            throw fmt::format_error("invalid format");
        }
        return it;
    }

    template<typename FormatContext>
    constexpr auto
    format(const std::complex<T> &value, FormatContext &ctx) const {
        // format according to: https://fmt.dev/papers/p2197r0.html#examples
        const auto imag = value.imag();
        switch (presentation) {
        case 'e':
            if (imag == 0) {
                return fmt::format_to(ctx.out(), "{:e}", value.real());
            }
            return fmt::format_to(ctx.out(), "({:e}{:+e}i)", value.real(), imag);
        case 'E':
            if (imag == 0) {
                return fmt::format_to(ctx.out(), "{:E}", value.real());
            }
            return fmt::format_to(ctx.out(), "({:E}{:+E}i)", value.real(), imag);
        case 'f':
            if (imag == 0) {
                return fmt::format_to(ctx.out(), "{:f}", value.real());
            }
            return fmt::format_to(ctx.out(), "({:f}{:+f}i)", value.real(), imag);
        case 'F':
            if (imag == 0) {
                return fmt::format_to(ctx.out(), "{:F}", value.real());
            }
            return fmt::format_to(ctx.out(), "({:F}{:+F}i)", value.real(), imag);
        case 'G':
            if (imag == 0) {
                return fmt::format_to(ctx.out(), "{:G}", value.real());
            }
            return fmt::format_to(ctx.out(), "({:G}{:+G}i)", value.real(), imag);
        case 'g':
        default:
            if (imag == 0) {
                return fmt::format_to(ctx.out(), "{:g}", value.real());
            }
            return fmt::format_to(ctx.out(), "({:g}{:+g}i)", value.real(), imag);
        }
    }
};

// simplified formatter for UncertainValue
template<gr::meta::arithmetic_or_complex_like T>
struct fmt::formatter<gr::meta::UncertainValue<T>> {
    template<typename ParseContext>
    constexpr auto parse(ParseContext &ctx) { return ctx.begin(); }

    template<typename FormatContext>
    constexpr auto
    format(const gr::meta::UncertainValue<T>&  value, FormatContext &ctx) const {
        if constexpr (gr::meta::complex_like<T>) {
            return fmt::format_to(ctx.out(), "({} ± {})", value.value, value.uncertainty);
        } else {
            return fmt::format_to(ctx.out(), "({:G} ± {:G})", value.value, value.uncertainty);
        }
    }
};

#endif // GNURADIO_FORMATTER_HPP
