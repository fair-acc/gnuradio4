#ifndef GNURADIO_FORMATTER_HPP
#define GNURADIO_FORMATTER_HPP

#include <complex>
#include <expected>
#include <fmt/format.h>
#include <gnuradio-4.0/meta/UncertainValue.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <source_location>
#include <vector>

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
template<gr::arithmetic_or_complex_like T>
struct fmt::formatter<gr::UncertainValue<T>> {
    template<typename ParseContext>
    constexpr auto
    parse(ParseContext &ctx) {
        return ctx.begin();
    }

    template<typename FormatContext>
    constexpr auto
    format(const gr::UncertainValue<T> &value, FormatContext &ctx) const {
        if constexpr (gr::meta::complex_like<T>) {
            return fmt::format_to(ctx.out(), "({} ± {})", value.value, value.uncertainty);
        } else {
            return fmt::format_to(ctx.out(), "({:G} ± {:G})", value.value, value.uncertainty);
        }
    }
};

template<>
struct fmt::formatter<gr::property_map> {
    template<typename ParseContext>
    constexpr auto
    parse(ParseContext &ctx) {
        return ctx.begin();
    }

    template<typename FormatContext>
    constexpr auto
    format(const gr::property_map &value, FormatContext &ctx) const {
        return fmt::format_to(ctx.out(), "{{ {} }}", fmt::join(value, ", "));
    }
};

template<>
struct fmt::formatter<std::vector<bool>> {
    char presentation = 'c';

    constexpr auto
    parse(format_parse_context &ctx) -> decltype(ctx.begin()) {
        auto it = ctx.begin(), end = ctx.end();
        if (it != end && (*it == 's' || *it == 'c')) presentation = *it++;
        if (it != end && *it != '}') throw fmt::format_error("invalid format");
        return it;
    }

    template<typename FormatContext>
    auto
    format(const std::vector<bool> &v, FormatContext &ctx) const -> decltype(ctx.out()) {
        auto   sep = (presentation == 'c' ? ", " : " ");
        size_t len = v.size();
        fmt::format_to(ctx.out(), "[");
        for (size_t i = 0; i < len; ++i) {
            if (i > 0) {
                fmt::format_to(ctx.out(), "{}", sep);
            }
            fmt::format_to(ctx.out(), "{}", v[i] ? "true" : "false");
        }
        fmt::format_to(ctx.out(), "]");
        return ctx.out();
    }
};

template<>
struct fmt::formatter<std::source_location> {
    constexpr auto
    parse(format_parse_context &ctx) -> decltype(ctx.begin()) {
        return ctx.begin();
    }

    // Formats the source_location, using 'f' for file and 'l' for line
    template<typename FormatContext>
    auto
    format(const std::source_location &loc, FormatContext &ctx) const -> decltype(ctx.out()) {
        // Example format: "file:line"
        return fmt::format_to(ctx.out(), "{}:{}", loc.file_name(), loc.line());
    }
};

template<typename Value, typename Error>
struct fmt::formatter<std::expected<Value, Error>> {
    constexpr auto
    parse(format_parse_context &ctx) -> decltype(ctx.begin()) {
        return ctx.begin();
    }

    // Formats the source_location, using 'f' for file and 'l' for line
    template<typename FormatContext>
    auto
    format(const std::expected<Value, Error> &ret, FormatContext &ctx) const -> decltype(ctx.out()) {
        if (ret.has_value()) {
            return fmt::format_to(ctx.out(), "<std::expected-value: {}>", ret.value());
        } else {
            return fmt::format_to(ctx.out(), "<std::unexpected: {}>", ret.error());
        }
    }
};

#endif // GNURADIO_FORMATTER_HPP
