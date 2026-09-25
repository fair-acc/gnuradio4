#ifndef GNURADIO_COLOUR_HPP
#define GNURADIO_COLOUR_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <format>
#include <optional>
#include <string>
#include <string_view>

#include <gnuradio-4.0/meta/utils.hpp>

namespace gr::utf8::color {

struct Colour {
    std::uint8_t   r{0U}, g{0U}, b{0U};
    constexpr auto operator<=>(const Colour&) const = default;
};
static_assert(sizeof(Colour) == 3UZ);

template<std::floating_point T = float>
struct ColourHSV {
    T              h, s, v; // h: [0,360), s,v: [0,1]
    constexpr auto operator<=>(const ColourHSV&) const = default;
};

template<std::uint8_t R, std::uint8_t G, std::uint8_t B, bool isForegroundColour = true>
[[nodiscard]] inline constexpr std::string_view makeAnsi() {
    static constexpr auto ansi_string = gr::meta::fixed_string("\x1b[") + gr::meta::fixed_string((isForegroundColour ? "38" : "48")) + gr::meta::fixed_string(";2;") // assures compile-time storage
                                        + gr::meta::fixed_string_from_number<R> + gr::meta::fixed_string(";")                                                        //
                                        + gr::meta::fixed_string_from_number<G> + gr::meta::fixed_string(";")                                                        //
                                        + gr::meta::fixed_string_from_number<B> + gr::meta::fixed_string("m");
    return std::string_view{ansi_string.data(), ansi_string.size};
}

[[nodiscard]] inline constexpr std::string makeAnsi(const Colour& c, bool isForegroundColour = true) { return std::format("\x1b[{};2;{};{};{}m", isForegroundColour ? "38" : "48", c.r, c.g, c.b); }

constexpr std::string_view RESET = "\x1b[0m";

template<std::floating_point T>
[[nodiscard]] constexpr Colour interpolateRGB(const Colour& c1, const Colour& c2, T t) noexcept {
    t = std::clamp(t, T(0), T(1));
    return {static_cast<std::uint8_t>(c1.r + t * (c2.r - c1.r)), static_cast<std::uint8_t>(c1.g + t * (c2.g - c1.g)), static_cast<std::uint8_t>(c1.b + t * (c2.b - c1.b))};
}

template<std::floating_point T>
[[nodiscard]] constexpr ColourHSV<T> rgbToHSV(const Colour& c) noexcept {
    T r = c.r / T(255);
    T g = c.g / T(255);
    T b = c.b / T(255);

    T cmax = std::max({r, g, b});
    T cmin = std::min({r, g, b});
    T diff = cmax - cmin;

    T h = 0;
    if (diff > 0) {
        if (cmax == r) {
            T q = (g - b) / diff; // in (-inf, +inf)
            while (q < T(0)) {
                q += T(6);
            }
            while (q >= T(6)) {
                q -= T(6);
            }
            h = T(60) * q; // [0,360)
        } else if (cmax == g) {
            h = T(60) * ((b - r) / diff + T(2)); // [0,360) after wrap
        } else {
            h = T(60) * ((r - g) / diff + T(4)); // [0,360) after wrap
        }
        while (h < T(0)) {
            h += T(360);
        }
        while (h >= T(360)) {
            h -= T(360);
        }
    }

    return {.h = h, .s = ((cmax > 0) ? (diff / cmax) : 0), .v = cmax};
}

template<std::floating_point T>
[[nodiscard]] constexpr Colour hsvToRGB(const ColourHSV<T>& hsv) noexcept {
    constexpr static auto abs   = [](T x) constexpr noexcept { return x < T(0) ? -x : x; }; // not yet constexpr in all [std]libc++ implementations
    constexpr static auto clamp = [](T x) constexpr noexcept { return x < T(0) ? T(0) : (x > T(1) ? T(1) : x); };
    constexpr static auto fract = [](T x) constexpr noexcept { return x - static_cast<T>(static_cast<long long>(x)); };
    constexpr static auto trunc = [](T v) constexpr noexcept { return static_cast<std::uint8_t>(clamp(v) * T(255)); };

    T h = hsv.h; // normalise hue to [0,360)
    while (h < T(0)) {
        h += T(360);
    }
    while (h >= T(360)) {
        h -= T(360);
    }

    const T s  = clamp(hsv.s);
    const T v  = clamp(hsv.v);
    const T hn = h / T(360); // hue in [0,1)
    auto    p  = [&](T n) constexpr noexcept { return clamp(abs(fract(hn + n) * T(6) - T(3)) - T(1)); };

    const T r = v * ((T(1) - s) + s * p(T(0)));
    const T g = v * ((T(1) - s) + s * p(T(2) / T(3)));
    const T b = v * ((T(1) - s) + s * p(T(1) / T(3)));
    return {trunc(r), trunc(g), trunc(b)};
}

template<std::floating_point T>
[[nodiscard]] constexpr Colour interpolateHSV(const Colour& c1, const Colour& c2, T t) noexcept {
    t            = std::clamp<T>(t, 0, 1);
    ColourHSV h1 = rgbToHSV<T>(c1);
    ColourHSV h2 = rgbToHSV<T>(c2);

    // hue interpolation (shortest path)
    T hdiff = h2.h - h1.h;
    if (hdiff > 180) {
        hdiff -= 360;
    } else if (hdiff < -180) {
        hdiff += 360;
    }
    return hsvToRGB<T>({std::fmod(h1.h + t * hdiff + T(360), T(360)), h1.s + t * (h2.s - h1.s), h1.v + t * (h2.v - h1.v)});
}

[[nodiscard]] inline constexpr std::string toHexRGB(Colour c) {
    std::array<char, 8UZ> buf{'#', 0, 0, 0, 0, 0, 0, 0}; // #RRGGBB (uppercase)
    auto                  to2 = [](unsigned v, char* out) {
        static constexpr char kHex[] = "0123456789ABCDEF";
        out[0]                       = kHex[(v >> 4) & 0xF];
        out[1]                       = kHex[v & 0xF];
    };
    to2(c.r, &buf[1]);
    to2(c.g, &buf[3]);
    to2(c.b, &buf[5]);
    return std::string(buf.data(), 7UZ);
}

[[nodiscard]] inline constexpr std::optional<Colour> parseHexRGB(std::string_view s) {
    if (s.size() != 7 || s[0] != '#') {
        return std::nullopt;
    }
    auto hex2 = [](char hi, char lo) -> std::optional<std::uint8_t> {
        auto cvt = [](char ch) -> int {
            if ('0' <= ch && ch <= '9') {
                return ch - '0';
            }
            if ('a' <= ch && ch <= 'f') {
                return ch - 'a' + 10;
            }
            if ('A' <= ch && ch <= 'F') {
                return ch - 'A' + 10;
            }
            return -1;
        };
        int h = cvt(hi), l = cvt(lo);
        if (h < 0 || l < 0) {
            return std::nullopt;
        }
        return static_cast<std::uint8_t>((h << 4) | l);
    };
    auto r = hex2(s[1], s[2]);
    if (!r) {
        return std::nullopt;
    }
    auto g = hex2(s[3], s[4]);
    if (!g) {
        return std::nullopt;
    }
    auto b = hex2(s[5], s[6]);
    if (!b) {
        return std::nullopt;
    }
    return Colour{*r, *g, *b};
}

} // namespace gr::utf8::color

namespace std {

template<class Char>
struct formatter<gr::utf8::color::Colour, Char> {
    constexpr auto parse(basic_format_parse_context<Char>& pc) { return pc.begin(); }
    template<class Ctx>
    constexpr auto format(const gr::utf8::color::Colour& c, Ctx& ctx) const {
        return std::format_to(ctx.out(), "({},{},{})", static_cast<unsigned>(c.r), static_cast<unsigned>(c.g), static_cast<unsigned>(c.b));
    }
};

template<std::floating_point T, class Char>
struct formatter<gr::utf8::color::ColourHSV<T>, Char> {
    std::formatter<T, Char> spec{};
    constexpr auto          parse(basic_format_parse_context<Char>& pc) { return spec.parse(pc); }
    template<class Ctx>
    constexpr auto format(const gr::utf8::color::ColourHSV<T>& c, Ctx& ctx) const {
        auto out = std::format_to(ctx.out(), "(");
        out      = spec.format(c.h, ctx);
        out      = std::format_to(out, ",");
        ctx.advance_to(out);
        out = spec.format(c.s, ctx);
        out = std::format_to(out, ",");
        ctx.advance_to(out);
        out = spec.format(c.v, ctx);
        return std::format_to(out, ")");
    }
};

} // namespace std

#endif // GNURADIO_COLOUR_HPP
