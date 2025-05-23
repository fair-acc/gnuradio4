#ifndef GNURADIO_FORMATTER_HPP
#define GNURADIO_FORMATTER_HPP

#include <chrono>
#include <complex>
#include <concepts>
#include <expected>
#include <source_location>
#include <vector>

#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/meta/UncertainValue.hpp>

namespace gr {
namespace time {
[[nodiscard]] inline std::string getIsoTime(std::chrono::system_clock::time_point timePoint = std::chrono::system_clock::now()) noexcept {
    const auto secs = std::chrono::time_point_cast<std::chrono::seconds>(timePoint);
    const auto ms   = std::chrono::duration_cast<std::chrono::milliseconds>(timePoint - secs).count();
    return std::format("{:%Y-%m-%dT%H:%M:%S}.{:06}", secs, ms); // ms-precision ISO time-format
}
} // namespace time

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

template<typename T>
constexpr auto ptr(const T* p) {
    return std::format("{:#x}", reinterpret_cast<std::uintptr_t>(p));
}
} // namespace gr

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
concept FormattableRange = std::ranges::range<R> && !std::same_as<std::remove_cvref_t<R>, std::string> && !std::same_as<std::remove_cvref_t<R>, std::string_view> && !std::is_array_v<std::remove_cvref_t<R>> && std::formattable<std::ranges::range_value_t<R>, char>;

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

template<>
struct std::formatter<pmtv::map_t::value_type> {
    constexpr auto parse(std::format_parse_context& ctx) const noexcept -> decltype(ctx.begin()) { return ctx.begin(); }

    template<typename FormatContext>
    auto format(const pmtv::map_t::value_type& kv, FormatContext& ctx) const noexcept {
        return std::format_to(ctx.out(), "{}: {}", kv.first, kv.second);
    }
};

template<pmtv::IsPmt T>
struct std::formatter<T> { // alternate pmtv formatter optimised for compile-time not runtime
    constexpr auto parse(std::format_parse_context& ctx) const noexcept -> decltype(ctx.begin()) { return ctx.begin(); }

    template<typename FormatContext>
    auto format(const T& value, FormatContext& ctx) const noexcept {
        // if the std::visit dispatch is too expensive then maybe manually loop-unroll this
        return std::visit([&ctx](const auto& format_arg) { return format_value(format_arg, ctx); }, value);
    }

private:
    template<typename FormatContext, typename U>
    static auto format_value(const U& arg, FormatContext& ctx) -> decltype(std::format_to(ctx.out(), "")) {
        if constexpr (pmtv::Scalar<U> || pmtv::Complex<U>) {
            return std::format_to(ctx.out(), "{}", arg);
        } else if constexpr (std::same_as<U, std::string>) {
            return std::format_to(ctx.out(), "{}", arg);
        } else if constexpr (pmtv::UniformVector<U> || pmtv::UniformStringVector<U>) { // format vector
            std::format_to(ctx.out(), "[");
            gr::format_join(ctx.out(), arg, ", ");
            return std::format_to(ctx.out(), "]");
        } else if constexpr (std::same_as<U, std::vector<pmtv::pmt>>) { // format vector of pmts
            std::format_to(ctx.out(), "[");
            gr::format_join(ctx.out(), arg, ", ");
            return std::format_to(ctx.out(), "]");
        } else if constexpr (pmtv::PmtMap<U>) { // format map
            std::format_to(ctx.out(), "{{ ");
            for (auto it = arg.begin(); it != arg.end(); ++it) {
                format_value(it->first, ctx); // Format key
                std::format_to(ctx.out(), ": ");
                format_value(it->second, ctx); // Format value
                if (std::next(it) != arg.end()) {
                    std::format_to(ctx.out(), ", ");
                }
            }
            return std::format_to(ctx.out(), " }}");
        } else if constexpr (requires { std::visit([](const auto&) {}, arg); }) {
            return std::visit([&](const auto& value) { return format_value(value, ctx); }, arg);
        } else if constexpr (std::same_as<std::monostate, U>) {
            return std::format_to(ctx.out(), "null");
        } else {
            return std::format_to(ctx.out(), "unknown type {}", gr::meta::type_name<U>());
        }
    }
};

template<>
struct std::formatter<pmtv::map_t> {
    constexpr auto parse(std::format_parse_context& ctx) const noexcept -> decltype(ctx.begin()) { return ctx.begin(); }

    template<typename FormatContext>
    constexpr auto format(const pmtv::map_t& value, FormatContext& ctx) const noexcept {
        std::format_to(ctx.out(), "{{ ");
        gr::format_join(ctx.out(), value, ", ");
        return std::format_to(ctx.out(), " }}");
    }
};

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

template<typename T1, typename T2>
struct std::formatter<std::pair<T1, T2>, char> {
    constexpr auto parse(std::format_parse_context& ctx) { return ctx.begin(); }

    template<typename FormatContext>
    auto format(const std::pair<T1, T2>& p, FormatContext& ctx) const {
        return std::format_to(ctx.out(), "({}, {})", p.first, p.second);
    }
};

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

template<typename T>
requires std::derived_from<T, std::exception>
struct std::formatter<T, char> {
    constexpr auto parse(std::format_parse_context& ctx) { return ctx.begin(); }

    template<typename FormatContext>
    auto format(const T& e, FormatContext& ctx) const {
        return std::format_to(ctx.out(), "{}", e.what());
    }
};

template<typename E>
requires std::is_enum_v<E>
struct std::formatter<E, char> {
    constexpr auto parse(std::format_parse_context& ctx) { return ctx.begin(); }

    template<typename FormatContext>
    auto format(E e, FormatContext& ctx) const {
        if (auto name = magic_enum::enum_name(e); !name.empty()) {
            return std::format_to(ctx.out(), "{}", name);
        } else {
            return std::format_to(ctx.out(), "{}", static_cast<std::underlying_type_t<E>>(e));
        }
    }
};

#endif // GNURADIO_FORMATTER_HPP
