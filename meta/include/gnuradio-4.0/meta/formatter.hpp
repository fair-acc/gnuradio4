#ifndef GNURADIO_FORMATTER_HPP
#define GNURADIO_FORMATTER_HPP

#include <chrono>
#include <complex>
#include <concepts>
#include <expected>
#include <format>
#include <source_location>
#include <vector>

#if defined(__GNUC__) && !defined(__clang__) && !defined(__EMSCRIPTEN__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuseless-cast"
#endif
#include <magic_enum.hpp>

#include <gnuradio-4.0/meta/UncertainValue.hpp>
#include <gnuradio-4.0/meta/utils.hpp>
#if defined(__GNUC__) && !defined(__clang__) && !defined(__EMSCRIPTEN__)
#pragma GCC diagnostic pop
#endif

namespace gr {
namespace time {
template<typename Clock, typename Duration>
[[nodiscard]] inline std::string getIsoTime(std::chrono::time_point<Clock, Duration> timePoint) noexcept {
    const auto secs = std::chrono::time_point_cast<std::chrono::seconds>(timePoint);
    const auto ms   = std::chrono::duration_cast<std::chrono::milliseconds>(timePoint - secs).count();
#if defined(_WIN32)
    // In windows the colon (:) characer is reserved.  Using _ instead.
    return std::format("{:%Y-%m-%dT%H_%M_%S}.{:03}", secs, ms); // ms-precision ISO time-format
#else
    return std::format("{:%Y-%m-%dT%H:%M:%S}.{:03}", secs, ms); // ms-precision ISO time-format
#endif
}

[[nodiscard]] inline std::string getIsoTime() noexcept { return getIsoTime(std::chrono::system_clock::now()); }
} // namespace time

#ifndef STD_FORMATTER_RANGES
#define STD_FORMATTER_RANGES
template<std::ranges::input_range R>
requires std::formattable<std::ranges::range_value_t<R>, char>
std::string join(const R& range, std::string_view sep = ", ") {
    std::string out;
    auto        it  = std::ranges::begin(range);
    const auto  end = std::ranges::end(range);
    if (it != end) {
        out += std::format("{}", *it);
        while (++it != end) {
            out += std::format("{}{}", sep, *it);
        }
    }
    return out;
}
#endif

template<typename T>
constexpr auto ptr(const T* p) {
    return std::format("{:#x}", reinterpret_cast<std::uintptr_t>(p));
}
} // namespace gr

#ifndef STD_FORMATTER_SOURCE_LOCATION
#define STD_FORMATTER_SOURCE_LOCATION
template<>
struct std::formatter<std::source_location, char> {
    char presentation = 's';

    constexpr auto parse(std::format_parse_context& ctx) {
        auto it = ctx.begin(), end = ctx.end();
        if (it != end && (*it == 's' || *it == 'f' || *it == 't')) {
            presentation = *it++;
        }
        if (it != end && *it != '}') {
            throw std::format_error("invalid format specifier for source_location");
        }
        return it;
    }

    template<typename FormatContext>
    auto format(const std::source_location& loc, FormatContext& ctx) const {
        switch (presentation) {
        case 's': return std::format_to(ctx.out(), "{}", loc.file_name());
        case 't': return std::format_to(ctx.out(), "{}:{}", loc.file_name(), loc.line());
        case 'f':
        default: return std::format_to(ctx.out(), "{}:{} in {}", loc.file_name(), loc.line(), loc.function_name());
        }
    }
};
#endif

#ifndef STD_FORMATTER_COMPLEX
#define STD_FORMATTER_COMPLEX
template<typename T>
struct std::formatter<std::complex<T>, char> {
    char presentation = 'g'; // default format

    template<typename ParseContext>
    constexpr auto parse(ParseContext& ctx) {
        auto it = ctx.begin(), end = ctx.end();
        if (it != end && (*it == 'f' || *it == 'F' || *it == 'e' || *it == 'E' || *it == 'g' || *it == 'G')) {
            presentation = *it++;
        }
        if (it != end && *it != '}') {
            throw std::format_error("invalid format");
        }
        return it;
    }

    template<typename FormatContext>
    constexpr auto format(const std::complex<T>& value, FormatContext& ctx) const {
        const auto imag = value.imag();
        switch (presentation) {
        case 'e':
            if (imag == 0) {
                return std::format_to(ctx.out(), "{:e}", value.real());
            }
            return std::format_to(ctx.out(), "({:e}{:+e}i)", value.real(), imag);
        case 'E':
            if (imag == 0) {
                return std::format_to(ctx.out(), "{:E}", value.real());
            }
            return std::format_to(ctx.out(), "({:E}{:+E}i)", value.real(), imag);
        case 'f':
            if (imag == 0) {
                return std::format_to(ctx.out(), "{:f}", value.real());
            }
            return std::format_to(ctx.out(), "({:f}{:+f}i)", value.real(), imag);
        case 'F':
            if (imag == 0) {
                return std::format_to(ctx.out(), "{:F}", value.real());
            }
            return std::format_to(ctx.out(), "({:F}{:+F}i)", value.real(), imag);
        case 'G':
            if (imag == 0) {
                return std::format_to(ctx.out(), "{:G}", value.real());
            }
            return std::format_to(ctx.out(), "({:G}{:+G}i)", value.real(), imag);
        case 'g':
        default:
            if (imag == 0) {
                return std::format_to(ctx.out(), "{:g}", value.real());
            }
            return std::format_to(ctx.out(), "({:g}{:+g}i)", value.real(), imag);
        }
    }
};
#endif

// simplified formatter for UncertainValue
template<gr::arithmetic_or_complex_like T>
struct std::formatter<gr::UncertainValue<T>> {
    formatter<T> value_formatter;

    constexpr auto parse(format_parse_context& ctx) { return value_formatter.parse(ctx); }

    template<typename FormatContext>
    auto format(const gr::UncertainValue<T>& uv, FormatContext& ctx) const {
        auto out = ctx.out();
        out      = std::format_to(out, "(");
        out      = value_formatter.format(uv.value, ctx);
        out      = std::format_to(out, " ± ");
        out      = value_formatter.format(uv.uncertainty, ctx);
        out      = std::format_to(out, ")");
        return out;
    }
};

namespace gr {
template<gr::UncertainValueLike T>
std::ostream& operator<<(std::ostream& os, const T& v) {
    return os << std::format("{}", v);
}
} // namespace gr

// DataSet - Range formatter

namespace gr {
template<typename T>
struct Range;
}

template<typename T>
struct std::formatter<gr::Range<T>> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template<typename FormatContext>
    auto format(const gr::Range<T>& range, FormatContext& ctx) const {
        return std::format_to(ctx.out(), "[min: {}, max: {}]", range.min, range.max);
    }
};

namespace gr {
template<typename T>
std::ostream& operator<<(std::ostream& os, const gr::Range<T>& v) {
    return os << std::format("{}", v);
}
} // namespace gr

// pmt formatter

namespace gr {

template<typename R>
concept FormattableRange = std::ranges::range<R> && !gr::meta::string_like<std::remove_cvref_t<R>> && !std::is_array_v<std::remove_cvref_t<R>> && std::formattable<std::ranges::range_value_t<R>, char>;

template<typename OutputIt, typename Container>
constexpr auto format_join(OutputIt out, const Container& container, std::string_view separator = ", ") {
    auto it = container.begin();
    if (it != container.end()) {
        out = std::format_to(out, "{}", *it); // format first element
        ++it;
    }

    for (; it != container.end(); ++it) {
        out = std::format_to(out, "{}", separator); // insert separator
        out = std::format_to(out, "{}", *it);       // format remaining element
    }

    return out;
}

template<typename Container>
constexpr std::string join(const Container& container, std::string_view separator = ", ") {
    std::ostringstream ss;
    auto               out = std::ostream_iterator<char>(ss);
    format_join(out, container, separator);
    return ss.str();
}

} // namespace gr

#ifndef STD_FORMATTER_VECTOR_BOOL
#define STD_FORMATTER_VECTOR_BOOL
template<>
struct std::formatter<std::vector<bool>> {
    char presentation = 'c';

    constexpr auto parse(std::format_parse_context& ctx) -> decltype(ctx.begin()) {
        auto it = ctx.begin(), end = ctx.end();
        if (it != end && (*it == 's' || *it == 'c')) {
            presentation = *it++;
        }
        if (it != end && *it != '}') {
            throw std::format_error("invalid format");
        }
        return it;
    }

    template<typename FormatContext>
    auto format(const std::vector<bool>& v, FormatContext& ctx) const noexcept -> decltype(ctx.out()) {
        auto   sep = (presentation == 'c' ? ", " : " ");
        size_t len = v.size();
        std::format_to(ctx.out(), "[");
        for (size_t i = 0; i < len; ++i) {
            if (i > 0) {
                std::format_to(ctx.out(), "{}", sep);
            }
            std::format_to(ctx.out(), "{}", v[i] ? "true" : "false");
        }
        std::format_to(ctx.out(), "]");
        return ctx.out();
    }
};
#endif

#ifndef STD_FORMATTER_PAIR
#define STD_FORMATTER_PAIR
template<typename T1, typename T2>
struct std::formatter<std::pair<T1, T2>, char> {
    constexpr auto parse(std::format_parse_context& ctx) { return ctx.begin(); }

    template<typename FormatContext>
    auto format(const std::pair<T1, T2>& p, FormatContext& ctx) const {
        return std::format_to(ctx.out(), "({}, {})", p.first, p.second);
    }
};
#endif

#ifndef STD_FORMATTER_RANGE
#define STD_FORMATTER_RANGE
template<gr::FormattableRange R>
struct std::formatter<R, char> {
    char separator = ',';

    constexpr auto parse(std::format_parse_context& ctx) { return ctx.begin(); }

    template<typename FormatContext>
    auto format(const R& range, FormatContext& ctx) const {
        auto out = ctx.out();
        std::format_to(out, "[");
        bool first = true;

        for (const auto& val : range) {
            if (!first) {
                std::format_to(out, "{} ", separator);
            } else {
                first = false;
            }
            std::format_to(out, "{}", val);
        }

        return std::format_to(out, "]");
    }
};

template<typename T, std::size_t N>
requires(!std::same_as<T, char>)
struct std::formatter<T[N], char> {
    std::formatter<T> elemFmt;
    std::string       elemSpec;
    char              separator = ',';

    constexpr auto parse(std::format_parse_context& ctx) {
        auto it = ctx.begin();
        if (it != ctx.end() && *it != '}') {
            if (*it != ':') {
                separator = *it++;
            }
            if (it != ctx.end() && *it == ':') {
                ++it;
                auto spec_start = it;
                while (it != ctx.end() && *it != '}') {
                    ++it;
                }
                elemSpec = std::string(spec_start, it);
            }
        }

        if (it == ctx.end() || *it != '}') {
            throw std::format_error("invalid format specifier for C-style array");
        }

        return it + 1;
    }

    template<typename FormatContext>
    auto format(const T (&arr)[N], FormatContext& ctx) const {
        auto out = ctx.out();
        std::format_to(out, "[");
        for (std::size_t i = 0; i < N; ++i) {
            if (i > 0) {
                std::format_to(out, "{} ", separator);
            }
            const auto fmt = std::string("{:") + elemSpec + "}";
            std::vformat_to(out, fmt, std::make_format_args(arr[i]));
        }
        return std::format_to(out, "]");
    }
};
#endif

#ifndef STD_FORMATTER_EXPECTED
#define STD_FORMATTER_EXPECTED
template<typename Value, typename Error>

struct std::formatter<std::expected<Value, Error>> {
    constexpr auto parse(format_parse_context& ctx) const noexcept -> decltype(ctx.begin()) { return ctx.begin(); }

    template<typename FormatContext>
    auto format(const std::expected<Value, Error>& ret, FormatContext& ctx) const -> decltype(ctx.out()) {
        if (ret.has_value()) {
            return std::format_to(ctx.out(), "<std::expected-value: {}>", ret.value());
        } else {
            return std::format_to(ctx.out(), "<std::unexpected: {}>", ret.error());
        }
    }
};
#endif

#ifndef STD_FORMATTER_EXCEPTION
#define STD_FORMATTER_EXCEPTION
template<typename T>
requires std::derived_from<T, std::exception>
struct std::formatter<T, char> {
    constexpr auto parse(std::format_parse_context& ctx) { return ctx.begin(); }

    template<typename FormatContext>
    auto format(const T& e, FormatContext& ctx) const {
        return std::format_to(ctx.out(), "{}", e.what());
    }
};
#endif

#ifndef STD_FORMATTER_ENUM
#define STD_FORMATTER_ENUM
template<typename E>
requires std::is_enum_v<E>
struct std::formatter<E, char> {
    std::formatter<std::string_view, char> _strFormatter;

    constexpr auto parse(std::format_parse_context& ctx) { return _strFormatter.parse(ctx); }

    template<typename FormatContext>
    auto format(E e, FormatContext& ctx) const {
        if (auto name = magic_enum::enum_name(e); !name.empty()) {
            return _strFormatter.format(name, ctx); // delegate string formatting
        } else {
            return std::format_to(ctx.out(), "{}", std::to_underlying(e)); // fallback to underlying type
        }
    }
};
#endif

#endif // GNURADIO_FORMATTER_HPP
