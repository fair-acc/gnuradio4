#ifndef GNURADIO_FORMATTER_HPP
#define GNURADIO_FORMATTER_HPP

#include <chrono>
#include <fmt/chrono.h>
#include <fmt/format.h>
#include <fmt/std.h>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/meta/UncertainValue.hpp>
#include <source_location>
#include <vector>

namespace gr {
namespace time {
[[nodiscard]] inline std::string getIsoTime() noexcept {
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    return fmt::format("{:%Y-%m-%dT%H:%M:%S}.{:06}",               // ms-precision ISO time-format
        fmt::localtime(std::chrono::system_clock::to_time_t(now)), //
        std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1'000);
}
} // namespace time
} // namespace gr

// simplified formatter for UncertainValue
template<gr::arithmetic_or_complex_like T>
struct fmt::formatter<gr::UncertainValue<T>> {
    constexpr auto parse(fmt::format_parse_context& ctx) const noexcept -> decltype(ctx.begin()) { return ctx.begin(); }

    template<typename FormatContext>
    constexpr auto format(const gr::UncertainValue<T>& value, FormatContext& ctx) const noexcept {
        if constexpr (gr::meta::complex_like<T>) {
            return fmt::format_to(ctx.out(), "({} ± {})", value.value, value.uncertainty);
        } else {
            return fmt::format_to(ctx.out(), "({:G} ± {:G})", value.value, value.uncertainty);
        }
    }
};

// pmt formatter

namespace gr {

template<typename OutputIt, typename Container, typename Separator>
constexpr auto format_join(OutputIt out, const Container& container, const Separator& separator) {
    auto it = container.begin();
    if (it != container.end()) {
        out = fmt::format_to(out, "{}", *it); // format first element
        ++it;
    }

    for (; it != container.end(); ++it) {
        out = fmt::format_to(out, "{}", separator); // insert separator
        out = fmt::format_to(out, "{}", *it);       // format remaining element
    }

    return out;
}

template<typename Container, typename Separator>
constexpr std::string join(const Container& container, const Separator& separator) {
    std::ostringstream ss;
    auto               out = std::ostream_iterator<char>(ss);
    format_join(out, container, separator);
    return ss.str();
}

} // namespace gr

template<>
struct fmt::formatter<pmtv::map_t::value_type> {
    constexpr auto parse(fmt::format_parse_context& ctx) const noexcept -> decltype(ctx.begin()) { return ctx.begin(); }

    template<typename FormatContext>
    auto format(const pmtv::map_t::value_type& kv, FormatContext& ctx) const noexcept {
        return fmt::format_to(ctx.out(), "{}: {}", kv.first, kv.second);
    }
};

template<pmtv::IsPmt T>
struct fmt::formatter<T> { // alternate pmtv formatter optimised for compile-time not runtime
    constexpr auto parse(fmt::format_parse_context& ctx) const noexcept -> decltype(ctx.begin()) { return ctx.begin(); }

    template<typename FormatContext>
    auto format(const T& value, FormatContext& ctx) const noexcept {
        // if the std::visit dispatch is too expensive then maybe manually loop-unroll this
        return std::visit([&ctx](const auto& format_arg) { return format_value(format_arg, ctx); }, value);
    }

private:
    template<typename FormatContext, typename U>
    static auto format_value(const U& arg, FormatContext& ctx) -> decltype(fmt::format_to(ctx.out(), "")) {
        if constexpr (pmtv::Scalar<U> || pmtv::Complex<U>) {
            return fmt::format_to(ctx.out(), "{}", arg);
        } else if constexpr (std::same_as<U, std::string>) {
            return fmt::format_to(ctx.out(), "{}", arg);
        } else if constexpr (pmtv::UniformVector<U> || pmtv::UniformStringVector<U>) { // format vector
            fmt::format_to(ctx.out(), "[");
            gr::format_join(ctx.out(), arg, ", ");
            return fmt::format_to(ctx.out(), "]");
        } else if constexpr (std::same_as<U, std::vector<pmtv::pmt>>) { // format vector of pmts
            fmt::format_to(ctx.out(), "[");
            gr::format_join(ctx.out(), arg, ", ");
            return fmt::format_to(ctx.out(), "]");
        } else if constexpr (pmtv::PmtMap<U>) { // format map
            fmt::format_to(ctx.out(), "{{ ");
            for (auto it = arg.begin(); it != arg.end(); ++it) {
                format_value(it->first, ctx); // Format key
                fmt::format_to(ctx.out(), ": ");
                format_value(it->second, ctx); // Format value
                if (std::next(it) != arg.end()) {
                    fmt::format_to(ctx.out(), ", ");
                }
            }
            return fmt::format_to(ctx.out(), " }}");
        } else if constexpr (std::same_as<std::monostate, U>) {
            return fmt::format_to(ctx.out(), "null");
        } else {
            return fmt::format_to(ctx.out(), "unknown type {}", typeid(U).name());
        }
    }
};

template<>
struct fmt::formatter<pmtv::map_t> {
    constexpr auto parse(fmt::format_parse_context& ctx) const noexcept -> decltype(ctx.begin()) { return ctx.begin(); }

    template<typename FormatContext>
    constexpr auto format(const pmtv::map_t& value, FormatContext& ctx) const noexcept {
        fmt::format_to(ctx.out(), "{{ ");
        gr::format_join(ctx.out(), value, ", ");
        return fmt::format_to(ctx.out(), " }}");
    }
};

template<>
struct fmt::formatter<std::vector<bool>> {
    char presentation = 'c';

    constexpr auto parse(fmt::format_parse_context& ctx) -> decltype(ctx.begin()) {
        auto it = ctx.begin(), end = ctx.end();
        if (it != end && (*it == 's' || *it == 'c')) {
            presentation = *it++;
        }
        if (it != end && *it != '}') {
            throw fmt::format_error("invalid format");
        }
        return it;
    }

    template<typename FormatContext>
    auto format(const std::vector<bool>& v, FormatContext& ctx) const noexcept -> decltype(ctx.out()) {
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

#endif // GNURADIO_FORMATTER_HPP
