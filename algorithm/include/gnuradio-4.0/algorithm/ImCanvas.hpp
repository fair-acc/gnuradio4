#ifndef GNURADIO_IMCANVAS_HPP
#define GNURADIO_IMCANVAS_HPP

#include <gnuradio-4.0/meta/utils.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace gr::utf8 {

namespace color {

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
    static constexpr auto ansi_string = gr::fixed_string("\x1b[") + gr::fixed_string((isForegroundColour ? "38" : "48")) + gr::fixed_string(";2;") // assures compile-time storage
                                        + gr::meta::fixed_string_from_number<R> + gr::fixed_string(";")                                            //
                                        + gr::meta::fixed_string_from_number<G> + gr::fixed_string(";")                                            //
                                        + gr::meta::fixed_string_from_number<B> + gr::fixed_string("m");
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

[[nodiscard]] inline property_map toPropertyMap(Colour c) {
    property_map m;
    m["colour"] = toHexRGB(c);
    return m;
}
[[nodiscard]] inline Colour fromPropertyMap(Colour, const property_map& m) {
    Colour out{};
    if (auto it = m.find("colour"); it != m.end()) {
        if (auto pstr = std::get_if<std::string>(&it->second)) {
            if (auto c = parseHexRGB(*pstr)) {
                out = *c; // else default {}
            }
        }
    }
    return out;
}

namespace palette {
struct EGA {
    static constexpr Colour Black         = {0x00, 0x00, 0x00};
    static constexpr Colour Blue          = {0x00, 0x00, 0xAA};
    static constexpr Colour Green         = {0x00, 0xAA, 0x00};
    static constexpr Colour Cyan          = {0x00, 0xAA, 0xAA};
    static constexpr Colour Red           = {0xAA, 0x00, 0x00};
    static constexpr Colour Magenta       = {0xAA, 0x00, 0xAA};
    static constexpr Colour Brown         = {0xAA, 0x55, 0x00}; // Dark yellow/orange
    static constexpr Colour Yellow        = Brown;              // Dark yellow/orange
    static constexpr Colour Orange        = Brown;              // Dark yellow/orange
    static constexpr Colour White         = {0xAA, 0xAA, 0xAA}; // Light grey
    static constexpr Colour LightGrey     = White;              // Light grey
    static constexpr Colour BrightBlack   = {0x55, 0x55, 0x55}; // Dark grey
    static constexpr Colour DarkGrey      = BrightBlack;        // Dark grey
    static constexpr Colour BrightBlue    = {0x55, 0x55, 0xFF};
    static constexpr Colour BrightGreen   = {0x55, 0xFF, 0x55};
    static constexpr Colour BrightCyan    = {0x55, 0xFF, 0xFF};
    static constexpr Colour BrightRed     = {0xFF, 0x55, 0x55};
    static constexpr Colour BrightMagenta = {0xFF, 0x55, 0xFF};
    static constexpr Colour BrightYellow  = {0xFF, 0xFF, 0x55};
    static constexpr Colour BrightWhite   = {0xFF, 0xFF, 0xFF};

    static constexpr std::array<Colour, 16UZ> all = {Black, Blue, Green, Cyan, Red, Magenta, Brown, White, BrightBlack, BrightBlue, BrightGreen, BrightCyan, BrightRed, BrightMagenta, BrightYellow, BrightWhite};
};

using Default = EGA;
} // namespace palette

} // namespace color

struct Style {
    color::Colour fg{};
    color::Colour bg{};
    std::uint16_t fgSet : 1     = 0U;
    std::uint16_t bgSet : 1     = 0U;
    std::uint16_t bold : 1      = 0U;
    std::uint16_t faint : 1     = 0U;
    std::uint16_t italic : 1    = 0U;
    std::uint16_t underline : 1 = 0U;
    std::uint16_t blinkSlow : 1 = 0U;
    std::uint16_t blinkFast : 1 = 0U;
    std::uint16_t inverse : 1   = 0U;
    std::uint16_t strike : 1    = 0U;
    std::uint16_t vertical : 1  = 0U;
    std::uint16_t _pad : 5      = 0U;

    constexpr auto operator<=>(const Style&) const = default;

    constexpr bool        isSet() const noexcept { return fgSet || bgSet || bold || faint || italic || underline || blinkSlow || blinkFast || inverse || strike; }
    constexpr std::string toAnsi() const {
        if (!isSet()) {
            return "";
        }

        std::string out;
        out.reserve(64);

        // style attributes
        if (bold) {
            out += "\x1b[1m";
        }
        if (faint) {
            out += "\x1b[2m";
        }
        if (italic) {
            out += "\x1b[3m";
        }
        if (underline) {
            out += "\x1b[4m";
        }
        if (blinkSlow) {
            out += "\x1b[5m";
        }
        if (blinkFast) {
            out += "\x1b[6m";
        }
        if (inverse) {
            out += "\x1b[7m";
        }
        if (strike) {
            out += "\x1b[9m";
        }

        // colours
        if (fgSet) {
            out += color::makeAnsi(fg, true);
        }
        if (bgSet) {
            out += color::makeAnsi(bg, false);
        }

        return out;
    }
};
static_assert(sizeof(Style) == 8UZ);

template<bool hsvInterpolation = true, std::floating_point T, typename Range>
requires std::ranges::random_access_range<Range> && std::ranges::sized_range<Range> && std::same_as<std::remove_cvref_t<std::ranges::range_value_t<Range>>, Style>
[[nodiscard]] static constexpr Style interpolateStyles(Range&& styles, T interpolationFactor) {
    if (std::ranges::empty(styles)) {
        return Style{};
    }

    if (std::ranges::size(styles) == 1UZ) {
        return std::ranges::begin(styles)[0UZ]; // no interpolation needed
    }

    constexpr auto interpolateColor = [](const auto& fromColour, const auto& toColour, T factor) {
        if constexpr (hsvInterpolation) {
            return color::interpolateHSV(fromColour, toColour, std::clamp<T>(factor, 0, 1));
        } else {
            return color::interpolateRGB(fromColour, toColour, std::clamp<T>(factor, 0, 1));
        }
    };

    interpolationFactor                = std::clamp<T>(interpolationFactor, 0, 1);
    const std::ptrdiff_t numSegments   = std::ptrdiff_t(std::ranges::size(styles)) - 1;
    const T              segmentLength = T(1) / T(numSegments);
    const std::ptrdiff_t segmentIndex  = std::min<std::ptrdiff_t>(static_cast<std::ptrdiff_t>(interpolationFactor / segmentLength), numSegments - 1);

    const T segmentStart     = T(segmentIndex) * segmentLength;
    const T localInterpolate = std::clamp<T>((interpolationFactor - segmentStart) / segmentLength, 0, 1);

    const Style& from = std::ranges::begin(styles)[segmentIndex];
    const Style& to   = std::ranges::begin(styles)[segmentIndex + 1];

    Style result = from;
    if (from.fgSet && to.fgSet) {
        result.fg    = interpolateColor(from.fg, to.fg, localInterpolate);
        result.fgSet = 1U;
    } else if (to.fgSet && localInterpolate > T(0.5)) {
        result.fg    = to.fg;
        result.fgSet = to.fgSet;
    }

    if (from.bgSet && to.bgSet) {
        result.bg    = interpolateColor(from.bg, to.bg, localInterpolate);
        result.bgSet = 1U;
    } else if (to.bgSet && localInterpolate > T(0.5)) {
        result.bg    = to.bg;
        result.bgSet = to.bgSet;
    }

    return result;
}

[[nodiscard]] inline property_map toPropertyMap(const Style& s) {
    property_map m;
    if (s.fgSet) {
        m["fg"] = toHexRGB(s.fg);
    }
    if (s.bgSet) {
        m["bg"] = toHexRGB(s.bg);
    }

    if (s.bold) {
        m["bold"] = true;
    }
    if (s.italic) {
        m["italic"] = true;
    }
    if (s.underline) {
        m["underline"] = true;
    }
    if (s.blinkSlow) {
        m["blinkSlow"] = true;
    }
    if (s.blinkFast) {
        m["blinkFast"] = true;
    }
    if (s.inverse) {
        m["inverse"] = true;
    }
    if (s.strike) {
        m["strike"] = true;
    }

    // store *_Set only if true (flat keys)
    if (s.fgSet) {
        m["fgSet"] = true;
    }
    if (s.bgSet) {
        m["bgSet"] = true;
    }

    return m;
}
[[nodiscard]] inline Style fromPropertyMap(Style, const property_map& m) {
    Style s{};
    if (auto it = m.find("fgSet"); it != m.end()) {
        if (auto pb = std::get_if<bool>(&it->second)) {
            s.fgSet = *pb;
        }
    }

    if (auto it = m.find("bgSet"); it != m.end()) {
        if (auto pb = std::get_if<bool>(&it->second)) {
            s.bgSet = *pb;
        }
    }

    if (auto it = m.find("fg"); it != m.end()) {
        if (auto pstr = std::get_if<std::string>(&it->second)) {
            if (auto c = color::parseHexRGB(*pstr)) {
                s.fg    = *c;
                s.fgSet = true;
            }
        }
    }

    if (auto it = m.find("bg"); it != m.end()) {
        if (auto pstr = std::get_if<std::string>(&it->second)) {
            if (auto c = color::parseHexRGB(*pstr)) {
                s.bg    = *c;
                s.bgSet = true;
            }
        }
    }

    auto asBool = [&](std::string_view k) -> bool {
        if (auto it = m.find(std::string{k}); it != m.end()) {
            if (auto pb = std::get_if<bool>(&it->second)) {
                return *pb;
            }
        }
        return false;
    };
    s.bold      = asBool("bold");
    s.italic    = asBool("italic");
    s.underline = asBool("underline");
    s.blinkFast = asBool("blinkFast");
    s.blinkSlow = asBool("blinkSlow");
    s.inverse   = asBool("inverse");
    s.strike    = asBool("strike");
    return s;
}

[[nodiscard]] inline constexpr std::uint64_t pack_edge(std::uint32_t src, std::uint32_t dst) noexcept { return (static_cast<std::uint64_t>(src) << 32U) | static_cast<std::uint64_t>(dst); }
[[nodiscard]] inline constexpr std::uint32_t edge_src(std::uint64_t e) noexcept { return std::uint32_t(e >> 32); }
[[nodiscard]] inline constexpr std::uint32_t edge_dst(std::uint64_t e) noexcept { return std::uint32_t(e & 0xFFFF'FFFFu); }
[[nodiscard]] inline constexpr bool          shares_endpoint(std::uint64_t a, std::uint64_t b) noexcept {
    if (!a || !b) {
        return false;
    }
    return edge_src(a) == edge_src(b) || edge_dst(a) == edge_dst(b);
}

enum class Direction : std::uint8_t {
    North     = (1 << 0),
    NorthEast = (1 << 1),
    East      = (1 << 2),
    SouthEast = (1 << 3),
    South     = (1 << 4),
    SouthWest = (1 << 5),
    West      = (1 << 6),
    NorthWest = (1 << 7),
    None      = 0b00000000,
};

[[nodiscard]] inline constexpr Direction   operator|(Direction a, Direction b) noexcept { return static_cast<Direction>(std::to_underlying(a) | std::to_underlying(b)); }
[[nodiscard]] inline constexpr Direction   operator&(Direction a, Direction b) noexcept { return static_cast<Direction>(std::to_underlying(a) & std::to_underlying(b)); }
[[nodiscard]] inline constexpr Direction   operator~(Direction d) noexcept { return static_cast<Direction>(~std::to_underlying(d)); }
[[nodiscard]] inline constexpr bool        has(std::uint8_t mask, std::uint8_t flag) noexcept { return static_cast<bool>(mask & flag); }
[[nodiscard]] inline constexpr std::size_t index(Direction d) noexcept { return (d == Direction::None) ? 8UZ : std::size_t(std::countr_zero(std::to_underlying(d))); }
[[nodiscard]] inline constexpr Direction   oppositeDirection(Direction d) noexcept {
    using enum Direction;
    switch (d) {
    case North: return South;
    case South: return North;
    case East: return West;
    case West: return East;
    case NorthEast: return SouthWest;
    case SouthWest: return NorthEast;
    case SouthEast: return NorthWest;
    case NorthWest: return SouthEast;
    default: return None;
    }
}
[[nodiscard]] inline constexpr int angleDegree(Direction to, Direction from = Direction::North) noexcept {
    static constexpr std::array<int, 9> dirAngles{0, 45, 90, 135, 180, 225, 270, 315, 0};
    return (dirAngles[index(to)] - dirAngles[index(from)] + 180) % 360 - 180;
}
[[nodiscard]] inline constexpr bool isDiagonalDirection(Direction a) noexcept {
    using enum Direction;
    return a == NorthEast || a == SouthWest || a == NorthWest || a == SouthEast;
}
[[nodiscard]] inline constexpr Direction directionFromDelta(int dx, int dy) noexcept {
    using enum Direction;
    static constexpr std::array<std::array<Direction, 3>, 3> table = {{{{NorthWest, North, NorthEast}}, {{West, None, East}}, {{SouthWest, South, SouthEast}}}};
    return table[static_cast<std::size_t>(std::clamp(dy, -1, +1)) + 1UZ][static_cast<std::size_t>(std::clamp(dx, -1, +1)) + 1UZ];
}

[[nodiscard]] inline property_map toPropertyMap(Direction d) {
    property_map m;
    m["direction"] = std::string(magic_enum::enum_name(d));
    return m;
}
[[nodiscard]] inline Direction fromPropertyMap(Direction /*tag*/, const property_map& m) {
    if (auto it = m.find("direction"); it != m.end()) {
        if (auto pstr = std::get_if<std::string>(&it->second)) {
            if (auto e = magic_enum::enum_cast<Direction>(*pstr)) {
                return *e;
            }
        }
    }
    return Direction::None; // default on missing/invalid
}

[[nodiscard]] inline constexpr std::size_t decodeLength(std::string_view s) {
    if (s.empty()) {
        return 0UZ;
    }
    const auto b0 = static_cast<unsigned char>(s[0]);
    if (b0 <= 0x7F) {
        return 1UZ;
    }
    if ((b0 & 0xE0) == 0xC0 && s.size() >= 2) {
        return 2UZ;
    }
    if ((b0 & 0xF0) == 0xE0 && s.size() >= 3) {
        return 3UZ;
    }
    if ((b0 & 0xF8) == 0xF0 && s.size() >= 4) {
        return 4UZ;
    }
    return 0;
}

[[nodiscard]] inline constexpr std::size_t length(std::string_view s) noexcept {
    std::size_t count = 0, i = 0;
    while (i < s.size()) {
        const std::size_t n = decodeLength(s.substr(i));
        if (n == 0) {
            ++i;
        } // invalid starter or truncated sequence → skip one byte
        else {
            ++count;
            i += n;
        }
    }
    return count;
}

namespace braille {
inline constexpr static char32_t                                       base = 0x2800; // Braille range: U+2800 to U+28FF encoded as UTF-8: E2 A0 80 to E2 A3 BF
inline constexpr static std::array<std::array<std::uint8_t, 2UZ>, 4UZ> dotNum{{{1, 4}, {2, 5}, {3, 6}, {7, 8}}};
[[nodiscard]] inline constexpr std::uint8_t                            bitFor(std::uint8_t r, std::uint8_t c) noexcept {
    assert(r < 4 && c < 2);
    return static_cast<std::uint8_t>(1u << (dotNum[r][c] - 1u));
}

[[nodiscard]] inline constexpr std::optional<std::uint8_t> maskFromUtf8(std::string_view s) {
    if (s.size() != 3UZ) {
        return std::nullopt;
    }
    if (static_cast<unsigned char>(s[0]) != 0xE2) {
        return std::nullopt;
    }
    const auto b1 = static_cast<unsigned char>(s[1]);
    const auto b2 = static_cast<unsigned char>(s[2]);
    if (b1 < 0xA0 || b1 > 0xA3) {
        return std::nullopt;
    }
    if ((b1 == 0xA0 && b2 < 0x80) || (b1 == 0xA3 && b2 > 0xBF)) {
        return std::nullopt;
    }

    // reconstruct the Unicode code point
    char32_t cp = static_cast<char32_t>(0x2800 + ((b1 - 0xA0) * 64) + (b2 & 0x3F));
    return std::uint8_t(cp - base);
}

[[nodiscard]] inline constexpr std::string utf8FromMask(std::uint8_t mask) {
    char32_t    codePoint = base + mask;
    std::string out;
    out.push_back(static_cast<char>(0xE0 | ((codePoint >> 12) & 0x0F)));
    out.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
    return out;
}
} // namespace braille

namespace quad {
inline constexpr static std::array<const char*, 16UZ> kQuadrant{" ", "▖", "▗", "▄", "▘", "▌", "▚", "▙", "▝", "▞", "▐", "▟", "▀", "▛", "▜", "█"};

[[nodiscard]] inline constexpr std::uint8_t                bit(std::uint8_t r, std::uint8_t c) noexcept { return static_cast<std::uint8_t>((r == 0 && c == 0) ? (1u << 2) : (r == 0 && c == 1) ? (1u << 3) : (r == 1 && c == 0) ? (1u << 0) : (1u << 1)); }
[[nodiscard]] inline constexpr std::string_view            glyph(std::uint8_t m) noexcept { return kQuadrant[m & 0xF]; }
[[nodiscard]] inline constexpr std::optional<std::uint8_t> maskFromUtf8(std::string_view s) {
    for (std::size_t i = 0UZ; i < kQuadrant.size(); ++i) {
        if (s == kQuadrant[i]) {
            return static_cast<std::uint8_t>(i);
        }
    }
    return std::nullopt;
}
} // namespace quad

namespace detail {
constexpr inline static std::array<const char*, 9UZ> kBarsH{" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"};
constexpr inline static std::array<const char*, 9UZ> kBarsV{" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"};
// constexpr inline static std::array<const char*, 9UZ> kMarker{"X", "O", "★", "+", "■", "◎", "○", "□", "▢"};

[[nodiscard]] inline static constexpr std::string_view mapBoxSymbols(std::uint8_t m, bool isConnected = true) {
    using namespace std;
    constexpr std::uint8_t NORTH      = std::to_underlying(Direction::North);
    constexpr std::uint8_t SOUTH      = std::to_underlying(Direction::South);
    constexpr std::uint8_t EAST       = std::to_underlying(Direction::East);
    constexpr std::uint8_t WEST       = std::to_underlying(Direction::West);
    constexpr std::uint8_t NORTH_EAST = std::to_underlying(Direction::NorthEast);
    constexpr std::uint8_t SOUTH_EAST = std::to_underlying(Direction::SouthEast);
    constexpr std::uint8_t NORTH_WEST = std::to_underlying(Direction::NorthWest);
    constexpr std::uint8_t SOUTH_WEST = std::to_underlying(Direction::SouthWest);

    switch (m) { // NOSONAR number of switch statements is necessary
    case 0U: return " ";
    case NORTH:
    case SOUTH:
    case NORTH | SOUTH: return "│";
    case EAST:
    case WEST:
    case EAST | WEST:
        return "─";
        // alt frame characters:  '┌┐└┘' or '╭╮╰╯'
    case EAST | SOUTH: return "╭";
    case SOUTH | WEST: return "╮";
    case WEST | NORTH: return "╯";
    case NORTH | EAST: return "╰";
    case NORTH | EAST | SOUTH: return "├";
    case EAST | SOUTH | WEST: return "┬";
    case SOUTH | WEST | NORTH: return "┤";
    case WEST | NORTH | EAST: return "┴";
    case NORTH | EAST | SOUTH | WEST: return isConnected ? "┼" : "⌒";
    case NORTH_EAST:
    case SOUTH_WEST: return "╱";
    case NORTH_EAST | SOUTH_WEST: return "╱";
    case NORTH_WEST:
    case SOUTH_EAST: return "╲";
    case NORTH_WEST | SOUTH_EAST: return "╲";
    case NORTH_EAST | SOUTH_WEST | NORTH_WEST | SOUTH_EAST:
        return isConnected ? "╳" : "○"; // all diagonals
        // special cases, couldn't find a better character than '•' or '·'
    case EAST | NORTH_WEST: return "•";
    case EAST | NORTH_EAST: return "•";
    case WEST | NORTH_EAST: return "•";
    case WEST | NORTH_WEST: return "•";
    case SOUTH | NORTH_WEST: return "•";
    case SOUTH | NORTH_EAST: return "•";
    case NORTH | NORTH_EAST: return "•";
    case NORTH | NORTH_WEST: return "•";

    case EAST | SOUTH_WEST: return "•";
    case EAST | SOUTH_EAST: return "•";
    case WEST | SOUTH_EAST: return "•";
    case WEST | SOUTH_WEST: return "•";
    case SOUTH | SOUTH_WEST: return "•";
    case SOUTH | SOUTH_EAST: return "•";
    case NORTH | SOUTH_EAST: return "•";
    case NORTH | SOUTH_WEST: return "•";

    case NORTH_WEST | WEST | SOUTH_EAST | EAST: return isConnected ? "⊗" : "○";
    case SOUTH_WEST | WEST | NORTH_EAST | EAST: return isConnected ? "⊗" : "○";
    case SOUTH_WEST | NORTH | NORTH_EAST | SOUTH: return isConnected ? "⊗" : "○";

    default: return "ム"; // mu, nothing matches
    }
}
} // namespace detail

template<gr::arithmetic_or_complex_like T = std::size_t>
struct Point {
    using value_type                    = T;
    constexpr static T invalid_position = std::numeric_limits<T>::max();
    T                  x                = invalid_position;
    T                  y                = invalid_position;

    template<gr::arithmetic_or_complex_like U>
    [[nodiscard]] constexpr Point<U> get() const noexcept {
        return {static_cast<U>(x), static_cast<U>(y)};
    }

    template<gr::arithmetic_or_complex_like U>
    constexpr bool operator==(const Point<U>& other) const {
        return x == static_cast<T>(other.x) && y == static_cast<T>(other.y);
    }

    static Point undefined() noexcept { return {invalid_position, invalid_position}; }

    friend constexpr Point operator+(Point a, Point b) noexcept { return Point<T>{a.x + b.x, a.y + b.y}; }
    friend constexpr Point operator-(Point a, Point b) noexcept
    requires(std::is_signed_v<T>)
    {
        return {a.x - b.x, a.y - b.y};
    }

    constexpr Point& operator+=(const Point& o) noexcept {
        x += o.x;
        y += o.y;
        return *this;
    }
    constexpr Point& operator-=(const Point& o) noexcept
    requires(std::is_signed_v<T>)
    {
        x -= o.x;
        y -= o.y;
        return *this;
    }
    constexpr Point& operator*=(T s) noexcept {
        x *= s;
        y *= s;
        return *this;
    }
    constexpr Point& operator/=(T s) noexcept {
        x /= s;
        y /= s;
        return *this;
    }
    [[nodiscard]] friend constexpr Point operator+(Point a, T s) noexcept { return {a.x + s, a.y + s}; }
    [[nodiscard]] friend constexpr Point operator+(T s, Point a) noexcept { return a + s; }
    [[nodiscard]] friend constexpr Point operator-(Point a, T s) noexcept
    requires(std::is_signed_v<T>)
    {
        return {a.x - s, a.y - s};
    }
    [[nodiscard]] friend constexpr Point operator-(T s, Point a) noexcept
    requires(std::is_signed_v<T>)
    {
        return {s - a.x, s - a.y};
    }
    [[nodiscard]] friend constexpr Point operator*(Point a, T s) noexcept { return {a.x * s, a.y * s}; }
    [[nodiscard]] friend constexpr Point operator*(T s, Point a) noexcept { return a * s; }
    [[nodiscard]] friend constexpr Point operator/(Point a, T s) noexcept { return {a.x / s, a.y / s}; }
};

template<class X, class Y>
Point(X, Y) -> Point<std::common_type_t<X, Y>>;

template<typename R>
concept PointRangeLike = std::ranges::contiguous_range<R> && requires(R r) { typename std::ranges::range_value_t<R>::value_type; };

template<gr::arithmetic_or_complex_like T>
constexpr T manhattanNorm(Point<T> p1, Point<T> p2 = {0, 0}) noexcept {
    const T dx = std::max(p1.x, p2.x) - std::min(p1.x, p2.x);
    const T dy = std::max(p1.y, p2.y) - std::min(p1.y, p2.y);
    return dx + dy;
}

template<gr::arithmetic_or_complex_like T>
constexpr T chebyshevNorm(Point<T> p1, Point<T> p2 = {0, 0}) noexcept {
    const T dx = p1.x > p2.x ? p1.x - p2.x : p2.x - p1.x;
    const T dy = p1.y > p2.y ? p1.y - p2.y : p2.y - p1.y;
    return std::max(dx, dy);
}

template<gr::arithmetic_or_complex_like T>
[[nodiscard]] constexpr T euclideanNorm(Point<T> a, Point<T> b = {0, 0}) noexcept {
    using U    = std::common_type_t<T, long double>;
    const U dx = static_cast<U>(a.x) - static_cast<U>(b.x);
    const U dy = static_cast<U>(a.y) - static_cast<U>(b.y);

    constexpr auto root = [](U x, U y) noexcept -> T {
        if consteval {
            const U z = x * x + y * y;
            if (z <= U(0)) {
                return T(0);
            }
            U s = z;                        // Newton–Raphson sqrt
            for (int i = 0; i != 32; ++i) { // fixed iters; constexpr-friendly
                s = (s + z / s) / U(2);
            }
            return static_cast<T>(s);
        } else {
            return static_cast<T>(std::hypot<T>(static_cast<T>(x), static_cast<T>(y)));
        }
    };

    return static_cast<T>(root(dx, dy));
}

template<gr::arithmetic_or_complex_like T>
constexpr Point<T> min(Point<T> p1, Point<T> p2 = {Point<T>::invalid_position, Point<T>::invalid_position}) {
    return {std::min(p1.x, p2.x), std::min(p1.y, p2.y)};
}

template<gr::arithmetic_or_complex_like T>
constexpr Point<T> max(Point<T> p1, Point<T> p2 = {0, 0}) {
    return {std::max(p1.x, p2.x), std::max(p1.y, p2.y)};
}

template<class T>
[[nodiscard]] inline property_map toPropertyMap(const Point<T>& p) {
    gr::property_map       m;
    std::vector<pmtv::pmt> arr(2UZ);
    if constexpr (std::is_integral_v<T>) {
        arr[0UZ] = static_cast<std::int64_t>(p.x);
        arr[1UZ] = static_cast<std::int64_t>(p.y);
    } else {
        arr[0UZ] = static_cast<double>(p.x);
        arr[1UZ] = static_cast<double>(p.y);
    }
    m["point"] = std::move(arr);
    return m;
}

template<class T>
[[nodiscard]] inline Point<T> fromPropertyMap(Point<T> /*tag*/, const property_map& m) {
    Point<T> p{};
    if (auto it = m.find("point"); it != m.end()) {
        if (auto parr = std::get_if<std::vector<pmtv::pmt>>(&it->second)) {
            if (parr->size() >= 2) {
                auto get = [](const pmtv::pmt& pr) -> std::optional<long double> {
                    if (auto pi = std::get_if<std::int64_t>(&pr)) {
                        return static_cast<long double>(*pi);
                    }
                    if (auto pd = std::get_if<double>(&pr)) {
                        return static_cast<long double>(*pd);
                    }
                    return std::nullopt;
                };
                if (auto x = get((*parr)[0])) {
                    p.x = static_cast<T>(*x);
                }
                if (auto y = get((*parr)[1])) {
                    p.y = static_cast<T>(*y);
                }
            }
        }
    }
    return p;
}

enum class ZeroPosition { TopLeft, BottomLeft }; /// controls the position of the (0,0) coordinate

template<std::size_t nCols = std::dynamic_extent, std::size_t nRows = std::dynamic_extent, ZeroPosition zeroPosition = ZeroPosition::TopLeft, class Allocator = std::allocator<char>>
class ImCanvas {
    static constexpr bool is_static = (nCols != std::dynamic_extent) && (nRows != std::dynamic_extent);
    template<typename T>
    using RebindAlloc = std::allocator_traits<Allocator>::template rebind_alloc<T>;

    template<class T>
    using StorageT = std::conditional_t<is_static, std::array<T, nCols * nRows>, std::vector<T, RebindAlloc<T>>>;

    std::size_t _width{nCols == std::dynamic_extent ? 0UZ : nCols};
    std::size_t _height{nRows == std::dynamic_extent ? 0UZ : nRows};

public:
    StorageT<std::string>   _grapheme{}; // future upgrade: replace std::string with constexpr/consteval compatible storage
    StorageT<Style>         _style{};
    StorageT<std::uint8_t>  _mask{};
    StorageT<bool>          _blocked{};
    StorageT<std::uint64_t> _ownE1{};
    StorageT<std::uint64_t> _ownE2{};

    constexpr ImCanvas()
    requires(is_static)
    {
        clear();
    }

    constexpr ImCanvas(std::size_t width_ = nCols, std::size_t height_ = nRows, const Allocator& alloc = Allocator{})
    requires(!is_static)
        : _width{width_}, _height{height_}, _grapheme(width_ * height_, " ", RebindAlloc<std::string>(alloc)), //
          _style(width_ * height_, Style{}, RebindAlloc<Style>(alloc)),                                        //
          _mask(width_ * height_, 0U, RebindAlloc<std::uint8_t>(alloc)),                                       //
          _blocked(width_ * height_, false, RebindAlloc<bool>(alloc)),                                         //
          _ownE1(width_ * height_, 0ULL, RebindAlloc<std::uint64_t>(alloc)), _ownE2(width_ * height_, 0ULL, RebindAlloc<std::uint64_t>(alloc)) {}

    constexpr void clear(const Style& s = {}) {
        std::fill(_grapheme.begin(), _grapheme.end(), " ");
        std::fill(_style.begin(), _style.end(), s);
        std::fill(_mask.begin(), _mask.end(), 0U);
        std::fill(_blocked.begin(), _blocked.end(), false);
        std::fill(_ownE1.begin(), _ownE1.end(), 0ULL);
        std::fill(_ownE2.begin(), _ownE2.end(), 0ULL);
    }

    [[nodiscard]] constexpr std::size_t height() const noexcept {
        if constexpr (nRows == std::dynamic_extent) {
            return _height;
        } else {
            return nRows;
        }
    }
    [[nodiscard]] constexpr std::size_t width() const noexcept {
        if constexpr (nCols == std::dynamic_extent) {
            return _width;
        } else {
            return nCols;
        }
    }
    [[nodiscard]] constexpr Point<std::size_t> size() const noexcept { return {width(), height()}; }

    [[nodiscard]] constexpr std::size_t index(std::size_t x, std::size_t y) const noexcept {
        if constexpr (zeroPosition == ZeroPosition::BottomLeft) {
            return (_height - 1 - y) * _width + x;
        } else {
            return y * _width + x;
        }
    }

    [[nodiscard]] constexpr std::size_t index(Point<std::size_t> p) const noexcept {
        if constexpr (zeroPosition == ZeroPosition::BottomLeft) {
            return (_height - 1 - p.y) * _width + p.x;
        } else {
            return p.y * _width + p.x;
        }
    }

    [[nodiscard]] constexpr bool isInBounds(Point<std::size_t> p) const noexcept { return p.x < _width && p.y < _height; }

    constexpr void put(Point<std::size_t> p, std::string_view txt, const Style& s = {}) {
        if (!isInBounds(p) || txt.empty()) {
            return;
        }

        while (!txt.empty()) {
            auto len = decodeLength(txt);
            if (len == 0) {
                break;
            }

            _grapheme[index(p)] = std::string(txt.substr(0, len));
            _style[index(p)]    = s;

            if (s.vertical) {
                if (++p.y >= _height) {
                    break;
                }
            } else {
                if (++p.x >= _width) {
                    break;
                }
            }
            txt.remove_prefix(len);
        }
    }

    constexpr void hLine(Point<std::size_t> from, Point<std::size_t> to, const Style& st = {}) {
        auto [x0, x1] = std::minmax(from.x, to.x);
        for (std::size_t x = x0; x < x1; ++x) {
            addMask({x, from.y}, Direction::East, gr::meta::invalid_index, st);
            addMask({x + 1UZ, from.y}, Direction::West, gr::meta::invalid_index, st);
        }
    }

    constexpr void vLine(Point<std::size_t> from, Point<std::size_t> to, const Style& st = {}) {
        auto [y0, y1] = std::minmax(from.y, to.y);
        for (std::size_t y = y0; y < y1; ++y) {
            if constexpr (zeroPosition == ZeroPosition::BottomLeft) {
                addMask({from.x, y + 1UZ}, Direction::South, gr::meta::invalid_index, st);
                addMask({from.x, y}, Direction::North, gr::meta::invalid_index, st);
            } else {
                addMask({from.x, y}, Direction::South, gr::meta::invalid_index, st);
                addMask({from.x, y + 1UZ}, Direction::North, gr::meta::invalid_index, st);
            }
        }
    }

    template<bool hsvInterpolation = true, typename T>
    constexpr void hBar(Point<T> from, T toX, std::initializer_list<Style> styles, T referenceLength = T(0)) {
        const auto [x0, x1] = std::minmax(from.x, toX);
        const auto y        = static_cast<std::size_t>(from.y);

        if (!isInBounds({0, y}) || styles.size() == 0) {
            return;
        }

        const std::vector<Style> styleVec(styles);
        referenceLength      = referenceLength > T(0) ? referenceLength : (x1 - x0);
        const auto startCell = static_cast<std::size_t>(std::max(T(0), x0));
        const auto endCell   = static_cast<std::size_t>(std::min(T(_width - 1), x1));

        for (auto x = startCell; x <= endCell && x < _width; ++x) {
            const T cellStart = x == startCell ? (x0 - T(startCell)) : T(0);
            const T cellEnd   = x == endCell ? (x1 - T(endCell)) : T(1);
            const T coverage  = cellEnd - cellStart;

            using detail::kBarsH;
            const auto idx = index({x, y});
            _style[idx]    = interpolateStyles<hsvInterpolation>(styleVec, std::clamp<T>(((T(x) - x0) + std::midpoint(cellStart, cellEnd)) / referenceLength, 0, 1));
            _grapheme[idx] = _style[idx].fgSet == 1U ? kBarsH[std::clamp<std::size_t>(static_cast<std::size_t>(coverage * T(kBarsH.size())), 0UZ, kBarsH.size() - 1UZ)] : " ";
        }
    }

    template<bool hsvInterpolation = true, std::floating_point T>
    constexpr void hBar(Point<T> from, T length, const Style& style, T referenceLength = T(0)) {
        hBar<hsvInterpolation>({from.x, from.y}, {from.x + length, from.y}, {style}, referenceLength);
    }

    template<bool hsvInterpolation = true, typename T>
    constexpr void vBar(Point<T> from, T toY, std::initializer_list<Style> styles, T referenceLength = T(0)) {
        const auto [y0, y1] = std::minmax(from.y, toY);
        const auto x        = static_cast<std::size_t>(from.x);

        if (!isInBounds({x, 0})) {
            return;
        }

        std::vector<Style> styleVec(styles);
        if (styleVec.empty()) {
            return;
        }

        const auto startCell = static_cast<std::size_t>(std::max(T{0}, y0));
        const auto endCell   = static_cast<std::size_t>(std::min(T(_height - 1), y1));
        referenceLength      = referenceLength > T{0} ? referenceLength : (y1 - y0);

        for (auto y = startCell; y <= endCell && y < _height; ++y) {
            const T cellStart = y == startCell ? (y0 - T(startCell)) : T{0};
            const T cellEnd   = y == endCell ? (y1 - T(endCell)) : T{1};
            const T coverage  = cellEnd - cellStart;

            using detail::kBarsV;
            const auto idx = index({x, y});
            _style[idx]    = interpolateStyles<hsvInterpolation>(styleVec, std::clamp<T>((y1 - T(y) - std::midpoint(cellStart, cellEnd)) / referenceLength, 0, 1));
            _grapheme[idx] = _style[idx].fgSet == 1U ? kBarsV[std::clamp<std::size_t>(static_cast<std::size_t>(coverage * T(kBarsV.size())), 0UZ, kBarsV.size() - 1UZ)] : " ";
        }
    }

    template<std::floating_point T = double>
    constexpr void put(Point<T> p, const Style& s = {}, bool overwrite = false) {
        if (p.x < T(0) || p.y < T(0) || p.x >= T(_width) || p.y >= T(_height)) {
            return;
        }

        const Point<std::size_t> gp{static_cast<std::size_t>(p.x), static_cast<std::size_t>(p.y)};
        if (!isInBounds(gp)) {
            return;
        }

        const std::size_t idx = index(gp);
        if (!overwrite && !_grapheme[idx].empty() && _grapheme[idx] != " ") {
            return;
        }

        const T fx = p.x - T(gp.x);
        T       fy = p.y - T(gp.y);
        if constexpr (zeroPosition == ZeroPosition::BottomLeft) {
            fy = T(1) - fy;
        }

        const std::uint8_t col = (fx < T(0.5)) ? 0u : 1u;
        const auto         row = static_cast<std::uint8_t>(std::min<T>(3, fy * T(4)));

        std::uint8_t mask = braille::maskFromUtf8(_grapheme[idx]).value_or(0U);
        mask              = static_cast<std::uint8_t>(mask | braille::bitFor(row, col));
        _grapheme[idx]    = braille::utf8FromMask(mask);
        _style[idx]       = s;
    }

    template<bool hsvInterpolation = true, typename T = double>
    constexpr void put(std::span<const Point<T>> points, std::span<const Style> styles = {}, bool overwrite = false) {
        if (points.empty()) {
            return;
        }

        if (points.size() == 1) {
            put(points[0], styles.empty() ? Style{} : styles[0], overwrite);
            return;
        }

        std::vector<T> distances(points.size());
        distances[0] = T(0);
        for (std::size_t i = 1; i < points.size(); ++i) {
            distances[i] = distances[i - 1] + euclideanNorm(points[i], points[i - 1]);
        }

        const T totalLength = distances.back();
        if (totalLength < T(0.001)) {
            put(points[0], styles.empty() ? Style{} : styles[0], overwrite);
            return;
        }

        auto getStyleAt = [&styles, totalLength](T d) -> Style {
            if (styles.empty()) {
                return {};
            }
            if (styles.size() == 1) {
                return styles.front();
            }
            const T t = std::clamp(d / totalLength, T(0), T(1));
            return interpolateStyles<hsvInterpolation>(styles, t);
        };

        for (std::size_t seg = 0; seg + 1 < points.size(); ++seg) {
            const auto& p0     = points[seg];
            const auto& p1     = points[seg + 1];
            const T     segLen = distances[seg + 1] - distances[seg];

            if (segLen < T(0.001)) {
                put(p0, getStyleAt(distances[seg]), overwrite);
                continue;
            }

            const std::size_t steps    = static_cast<std::size_t>(chebyshevNorm(p1, p0) * T(4)) + 1UZ;
            const T           invSteps = T(1) / T(steps);
            const auto        delta    = p1 - p0;

            for (std::size_t i = 0; i <= steps; ++i) {
                const T    t  = T(i) * invSteps;
                const auto pt = p0 + delta * t;
                const T    d  = distances[seg] + t * segLen;
                put(pt, getStyleAt(d), overwrite);
            }
        }
    }

    template<bool hsvInterpolation = true, std::ranges::contiguous_range SR>
    requires std::same_as<std::remove_cvref_t<std::ranges::range_value_t<SR>>, Style>
    constexpr void put(PointRangeLike auto&& points, SR&& styles, bool overwrite = false) {
        using T = std::remove_cvref_t<std::ranges::range_value_t<decltype(points)>>::value_type;
        put<hsvInterpolation, T>(std::span{points}, std::span{styles}, overwrite);
    }

    template<bool hsvInterpolation = true, PointRangeLike PR>
    constexpr void put(PR&& points, const Style& style, bool overwrite = false) {
        using T            = std::remove_cvref_t<std::ranges::range_value_t<decltype(points)>>::value_type;
        const Style one[1] = {style};
        put<hsvInterpolation, T>(std::span{points}, std::span{one}, overwrite);
    }

    template<bool hsvInterpolation = true, std::ranges::contiguous_range SR>
    requires std::same_as<std::remove_cvref_t<std::ranges::range_value_t<SR>>, Style>
    constexpr void put(PointRangeLike auto&& points, bool overwrite = false) {
        using T = std::remove_cvref_t<std::ranges::range_value_t<decltype(points)>>::value_type;
        put<hsvInterpolation, T>(std::span{points}, std::span<const Style>{}, overwrite);
    }

    template<bool hsvInterpolation = true, class T = double>
    constexpr void put(std::initializer_list<std::type_identity_t<Point<T>>> points, const Style& style, bool overwrite = false) {
        const Style one[1] = {style};
        put<hsvInterpolation, T>(std::span{points}, std::span{one}, overwrite);
    }

    template<bool hsvInterpolation = true, class T = double>
    constexpr void put(std::initializer_list<std::type_identity_t<Point<T>>> points, bool overwrite = false) {
        put<hsvInterpolation, T>(std::span{points}, std::span<const Style>{}, overwrite);
    }

    template<std::floating_point T>
    constexpr void putArea(Point<T> p, const Style& s = {}, bool overwrite = false) {
        if (p.x < T(0) || p.y < T(0) || p.x >= T(_width) || p.y >= T(_height)) {
            return;
        }

        const Point<std::size_t> gp{static_cast<std::size_t>(p.x), static_cast<std::size_t>(p.y)};
        if (!isInBounds(gp)) {
            return;
        }

        const std::size_t idx = index(gp);
        if (!overwrite && !_grapheme[idx].empty() && _grapheme[idx] != " ") {
            if (!quad::maskFromUtf8(_grapheme[idx]).has_value()) {
                return;
            }
        }

        const T fx = p.x - T(gp.x);
        T       fy = p.y - T(gp.y);
        if constexpr (zeroPosition == ZeroPosition::BottomLeft) {
            fy = T(1) - fy;
        }

        const std::uint8_t c = (fx < T(0.5)) ? 0u : 1u;
        const std::uint8_t r = (fy < T(0.5)) ? 0u : 1u;

        std::uint8_t m = quad::maskFromUtf8(_grapheme[idx]).value_or(0U);
        m |= quad::bit(r, c);
        _grapheme[idx] = std::string(quad::glyph(m));
        _style[idx]    = s;
    }

    template<bool hsvInterpolation = true, class T = double>
    constexpr void putArea(std::initializer_list<std::type_identity_t<Point<T>>> points, const Style& style = {}, bool overwrite = false) {
        putArea<hsvInterpolation, T>(std::span{points}, std::array<Style, 1UZ>{style}, overwrite);
    }

    template<class T>
    requires std::floating_point<T>
    constexpr void putArea(std::initializer_list<T> xy, const Style& s = {}, bool overwrite = false) {
        if (xy.size() != 2) {
            return;
        }
        auto it = xy.begin();
        putArea(Point<T>{*it, *(it + 1)}, s, overwrite);
    }

    constexpr void addMask(Point<std::size_t> p, Direction direction, std::uint64_t edgeId = gr::meta::invalid_index, Style st = {}) {
        if (!isInBounds(p)) {
            return;
        }

        const std::size_t idx = index(p);
        if (_blocked[idx]) {
            return;
        }

        std::uint8_t        m = std::to_underlying(direction);
        const std::uint64_t e = (edgeId == gr::meta::invalid_index) ? 0ull : std::uint64_t(edgeId);

        using enum Direction;
        const std::uint8_t diag_bits = std::to_underlying(NorthEast) | std::to_underlying(NorthWest) | std::to_underlying(SouthEast) | std::to_underlying(SouthWest);
        const bool         has_diag  = has(m, diag_bits);

        bool wants_h = has(m, std::to_underlying(East | West)) || has_diag;
        bool wants_v = has(m, std::to_underlying(North | South)) || has_diag;

        // KISS: first edge claims ownership
        if (wants_h && _ownE1[idx] == 0ull) {
            _ownE1[idx] = e;
        }
        if (wants_v && _ownE2[idx] == 0ull) {
            _ownE2[idx] = e;
        }

        _mask[idx] |= m;

        // collect all unique edges actually present in this cell (≤4)
        std::array<std::uint64_t, 4UZ> ids{};
        std::size_t                    n = 0UZ;

        auto add_unique = [&](std::uint64_t v) constexpr {
            if (v == 0ULL) {
                return;
            }
            for (std::size_t i = 0; i < n; ++i) {
                if (ids.at(i) == v) {
                    return;
                }
            }
            ids.at(n++) = v;
        };

        add_unique(_ownE1[idx]);
        add_unique(_ownE2[idx]);
        add_unique(e);

        // choose the same "first" as std::set iteration would: the smallest id
        std::uint64_t first = 0ULL;
        if (n) {
            first = ids[0UZ];
            for (std::size_t i = 1UZ; i < n; ++i) {
                if (ids.at(i) < first) {
                    first = ids.at(i);
                }
            }
        }

        // connectivity: every other unique edge must share an endpoint with 'first'
        bool connected = true;
        for (std::size_t i = 0; i < n; ++i) {
            const auto id = ids.at(i);
            if (id != first && !shares_endpoint(first, id)) {
                connected = false;
                break;
            }
        }

        _grapheme[idx] = detail::mapBoxSymbols(_mask[idx], connected);
        _style[idx]    = st;
    }

    constexpr bool isBlocked(Point<std::size_t> p) const { return isInBounds(p) ? _blocked[index(p)] : true; }

    constexpr void blockRect(Point<std::size_t> topLeft, Point<std::size_t> bottomRight) {
        bottomRight = min(bottomRight, {_width - 1U, _height - 1U});

        auto [x0, x1] = std::minmax(topLeft.x, bottomRight.x);
        auto [y0, y1] = std::minmax(topLeft.y, bottomRight.y);

        for (std::size_t y = y0; y <= y1; ++y) {
            for (std::size_t x = x0; x <= x1; ++x) {
                _blocked[index({x, y})] = true;
            }
        }
    }

    template<class Dest = std::string>
    Dest toString(Dest data = {}, bool withAnsiColors = false) const {
        data.clear();
        data.reserve(_height * (_width * 4U + 32U));

        Style lastStyle{};
        bool  needReset = false;

        for (std::size_t row = 0; row < _height; ++row) {
            for (std::size_t x = 0; x < _width; ++x) {
                std::size_t y = (zeroPosition == ZeroPosition::BottomLeft) ? (_height - 1 - row) : row;

                const auto  idx          = index(x, y);
                const auto& currentStyle = _style[idx];

                // emit ANSI codes only on style change
                if (withAnsiColors && currentStyle != lastStyle) {
                    if (needReset) {
                        data += color::RESET;
                    }
                    std::string ansi = currentStyle.toAnsi();
                    if (!ansi.empty()) {
                        data += ansi;
                        needReset = true;
                    } else {
                        needReset = false;
                    }
                    lastStyle = currentStyle;
                }

                const auto& g = _grapheme[idx];
                data.append(g.empty() ? " " : g);
            }
            data.push_back('\n');
        }

        if (withAnsiColors && needReset) {
            data += color::RESET;
        }

        return data;
    }
};

template<class T>
struct isImCanvas : std::false_type {};
template<std::size_t C, std::size_t R, ZeroPosition Z, class A>
struct isImCanvas<gr::utf8::ImCanvas<C, R, Z, A>> : std::true_type {};

template<class T>
concept ImCanvasLike = isImCanvas<std::remove_cvref_t<T>>::value;

} // namespace gr::utf8

namespace std {
template<gr::arithmetic_or_complex_like T, typename CharT>
struct formatter<gr::utf8::Point<T>, CharT> {
    std::formatter<T, CharT> _spec;

    constexpr auto parse(std::basic_format_parse_context<CharT>& ctx) { return _spec.parse(ctx); }

    template<typename FormatContext>
    constexpr auto format(const gr::utf8::Point<T>& p, FormatContext& ctx) const {
        auto out = std::format_to(ctx.out(), "(");
        out      = _spec.format(p.x, ctx);
        out      = std::format_to(out, ",");
        ctx.advance_to(out); // keep ctx/out in sync
        out = _spec.format(p.y, ctx);
        return std::format_to(out, ")");
    }
};

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

#endif // GNURADIO_IMCANVAS_HPP
