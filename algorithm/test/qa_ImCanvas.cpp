#include <boost/ut.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>

namespace boost::ut { // unit test helper for pod struct comparable but not stream-insertable objects

template<class L, class R>
requires(std::equality_comparable_with<L, R> && !(boost::ut::type_traits::is_stream_insertable_v<L> && boost::ut::type_traits::is_stream_insertable_v<R>))
constexpr bool eq(const L& lhs, const R& rhs) noexcept {
    return lhs == rhs;
}
} // namespace boost::ut

#include <gnuradio-4.0/algorithm/ImCanvas.hpp>

using namespace boost::ut;
using namespace gr::utf8;

[[nodiscard]] std::string stripAnsiMetaInfo(std::string_view str) {
    std::string result;
    bool        inEscape = false;
    for (char c : str) {
        if (c == '\x1b') {
            inEscape = true;
        } else if (inEscape && c == 'm') {
            inEscape = false;
        } else if (!inEscape) {
            result += c;
        }
    }
    return result;
}

const suite<"utf8::color"> colorSuite = [] {
    using namespace std::string_literals;

    "RGB interpolation"_test = [] {
        constexpr color::Colour black{0, 0, 0};
        constexpr color::Colour white{255, 255, 255};
        constexpr auto          mid = color::interpolateRGB(black, white, 0.5f);

        expect(ge(mid.r, 127U) and le(mid.r, 128U));
        expect(ge(mid.g, 127U) and le(mid.g, 128U));
        expect(ge(mid.b, 127U) and le(mid.b, 128U));

        static_assert(color::interpolateRGB(black, white, 0.0f) == black);
        static_assert(color::interpolateRGB(black, white, 1.0f) == white);
        expect(eq(color::interpolateRGB(black, white, 0.0f), black));
        expect(eq(color::interpolateRGB(black, white, 1.0f), white));
    };

    "HSV conversion round-trip"_test = [] {
        constexpr color::Colour red{255, 0, 0};
        constexpr auto          hsv = color::rgbToHSV<float>(red);
        constexpr auto          rgb = color::hsvToRGB(hsv);

        expect(le(std::abs(static_cast<int>(rgb.r) - 255), 1));
        expect(eq(rgb.g, 0U));
        expect(eq(rgb.b, 0U));

        expect(hsv.h >= -1.0f and hsv.h <= 1.0f);
        expect(ge(hsv.s, 0.99f));
        expect(ge(hsv.v, 0.99f));
    };

    "HSV interpolation shortest path"_test = [] {
        constexpr color::Colour red{255, 0, 0};
        constexpr color::Colour cyan{0, 255, 255};
        auto                    mid = color::interpolateHSV(red, cyan, 0.5f);

        expect(lt(static_cast<int>(mid.r), static_cast<int>(red.r))) << std::format("should be bluish (not greenish): {:02x} {:02x} {:02x}", mid.r, mid.g, mid.b);
    };

    "ANSI color codes"_test = [] {
        using namespace std::string_view_literals;
        using namespace std::string_literals;

        constexpr std::string_view staticAnsi = color::makeAnsi<255, 128, 64>();
        expect(eq(staticAnsi, "\x1b[38;2;255;128;64m"sv));

        std::string dynamicAnsi = color::makeAnsi({255, 128, 64}, false);
        expect(eq(dynamicAnsi, "\x1b[48;2;255;128;64m"s));
    };

    "EGA palette"_test = [] {
        using Pal = color::palette::Default;
        static_assert(Pal::all.size() == 16UZ);
        static_assert(Pal::Black == color::Colour{0x00, 0x00, 0x00});
        static_assert(Pal::BrightWhite == color::Colour{0xFF, 0xFF, 0xFF});

        expect(Pal::all[0] == Pal::Black);
        expect(Pal::all[15] == Pal::BrightWhite);
    };

    "formatter.colour.tuple"_test = [] {
        static_assert(sizeof(color::Colour) == 3UZ);
        const color::Colour c1{255, 0, 128};
        const color::Colour c2{0, 255, 0};

        expect(eq(std::format("{}", c1), "(255,0,128)"s));
        expect(eq(std::format("{}", c2), "(0,255,0)"s));
    };

    "formatter.colourhsv.tuple"_test = [] {
        const color::ColourHSV<float>  f{300.0f, 0.5f, 0.8f};
        const color::ColourHSV<double> d{45.0, 1.0, 0.25};

        expect(eq(std::format("{:.2f}", f), "(300.00,0.50,0.80)"s));
        expect(eq(std::format("{:.3f}", d), "(45.000,1.000,0.250)"s));
    };

    "Colour_to_from_hex_roundtrip"_test = [] {
        color::Colour in{255, 210, 0};
        auto          m = toPropertyMap(in);
        expect(m.contains("colour")) << "flat key present";
        auto out = fromPropertyMap(color::Colour{}, m);
        expect(in.r == out.r && in.g == out.g && in.b == out.b) << "bit-for-bit RGB";
        expect(eq(m["colour"].value_or(std::string()), std::string("#FFD200")));
    };

    "Colour_from_missing_defaults"_test = [] {
        gr::property_map m; // empty
        auto             out = fromPropertyMap(color::Colour{}, m);
        expect(out.r == 0U && out.g == 0U && out.b == 0U);
    };
};

const suite<"utf8::Style"> styleSuite = [] {
    "Style construction and comparison"_test = [] {
        constexpr Style s1{};
        constexpr Style s2{.fg = {255, 0, 0}, .fgSet = 1, .bold = 1};

        static_assert(s1 != s2);
        static_assert(sizeof(Style) == 8UZ);

        expect(!s1.isSet());
        expect(s2.isSet());
    };

    "Style to ANSI"_test = [] {
        Style s{.fg = {255, 0, 0}, .bg = {0, 0, 255}, .fgSet = 1, .bgSet = 1, .bold = 1, .underline = 1};
        auto  ansi = s.toAnsi();

        expect(ansi.find("\x1b[1m") != std::string::npos) << "bold";
        expect(ansi.find("\x1b[4m") != std::string::npos) << "underline";
        expect(ansi.find("38;2;255;0;0") != std::string::npos) << "red fg";
        expect(ansi.find("48;2;0;0;255") != std::string::npos) << "blue bg";
    };

    "Style interpolation"_test = [] {
        constexpr std::array<Style, 3UZ> styles{
            Style{.fg = {255, 0, 0}, .fgSet = 1}, // red
            Style{.fg = {0, 255, 0}, .fgSet = 1}, // green
            Style{.fg = {0, 0, 255}, .fgSet = 1}  // blue
        };

        constexpr Style s0 = interpolateStyles<false>(styles, 0.0f);
        constexpr Style s1 = interpolateStyles<false>(styles, 0.5f);
        constexpr Style s2 = interpolateStyles<false>(styles, 1.0f);

        expect(eq(s0.fg, color::Colour{255, 0, 0}));
        expect(eq(static_cast<int>(s1.fg.g), 255)); // midpoint (green) between red and blue
        expect(eq(s2.fg, color::Colour{0, 0, 255}));

        static_assert(s0.fg == color::Colour{255, 0, 0});
        static_assert(s1.fg == color::Colour{0, 255, 0});
        static_assert(s2.fg == color::Colour{0, 0, 255});
    };

    "Style_to_from_roundtrip_only_set_keys"_test = [] {
        Style s{.fg = {10, 20, 30}, .fgSet = 1, .bold = 1, .underline = 1};
        auto  m = toPropertyMap(s);

        expect(m.contains("fg"));
        expect(!m.contains("bg"));
        expect(m.contains("fgSet") && !m.contains("bgSet"));
        expect(m.contains("bold") && m.contains("underline"));
        expect(!m.contains("italic") && !m.contains("strike"));

        auto s2 = fromPropertyMap(Style{}, m);
        expect(s2.fgSet) << "fgSet restored";
        expect(eq(s2.fg.r, 10_u) && eq(s2.fg.g, 20_u) && eq(s2.fg.b, 30_u));
        expect(s2.bold && s2.underline);
        expect(!s2.italic && !s2.strike && !s2.blinkSlow && !s2.blinkFast && !s2.inverse);
        expect(!s2.bgSet);
    };
};

const suite<"utf8::Direction"> directionSuite = [] {
    "Direction operations"_test = [] {
        using enum Direction;

        static_assert((North | South) == static_cast<Direction>(0b00010001));
        static_assert(oppositeDirection(North) == South);
        static_assert(oppositeDirection(NorthEast) == SouthWest);
        static_assert(isDiagonalDirection(NorthEast));
        static_assert(!isDiagonalDirection(North));

        expect(eq(directionFromDelta(1, 0), East));
        expect(eq(directionFromDelta(-1, -1), NorthWest));
        expect(eq(directionFromDelta(0, 0), None));
    };

    "Angle calculations"_test = [] {
        using enum Direction;

        expect(eq(angleDegree(East, North), 90));
        expect(eq(angleDegree(South, North), 180) or eq(angleDegree(South, North), -180));
        expect(eq(angleDegree(West, North), -90));
        expect(eq(angleDegree(North, North), 0));
    };

    "Direction has() function"_test = [] {
        std::uint8_t mask = std::to_underlying(Direction::North) | std::to_underlying(Direction::East);

        expect(has(mask, std::to_underlying(Direction::North)));
        expect(has(mask, std::to_underlying(Direction::East)));
        expect(!has(mask, std::to_underlying(Direction::South)));
    };

    "Direction_string_roundtrip"_test = [] {
        for (auto d : magic_enum::enum_values<Direction>()) {
            auto m  = toPropertyMap(d);
            auto d2 = fromPropertyMap(Direction{}, m);
            expect(d == d2);
        }
        // missing -> default
        gr::property_map m;
        expect(Direction::None == fromPropertyMap(Direction{}, m));
    };
};

const suite<"utf8::braille"> brailleSuite = [] {
    "Braille bit mapping"_test = [] {
        // Test dot numbering (1-8 in standard Braille order)
        expect(eq(braille::bitFor(0, 0), 0b00000001U)) << "dot 1";
        expect(eq(braille::bitFor(1, 0), 0b00000010U)) << "dot 2";
        expect(eq(braille::bitFor(2, 0), 0b00000100U)) << "dot 3";
        expect(eq(braille::bitFor(0, 1), 0b00001000U)) << "dot 4";
        expect(eq(braille::bitFor(1, 1), 0b00010000U)) << "dot 5";
        expect(eq(braille::bitFor(2, 1), 0b00100000U)) << "dot 6";
        expect(eq(braille::bitFor(3, 0), 0b01000000U)) << "dot 7";
        expect(eq(braille::bitFor(3, 1), 0b10000000U)) << "dot 8";
    };

    "Braille UTF-8 round-trip"_test = [] {
        "test empty pattern"_test = [] {
            auto utf8_0 = braille::utf8FromMask(0x00);
            expect(eq(utf8_0.size(), 3UZ));
            expect(eq(braille::maskFromUtf8(utf8_0), 0x00U));
        };

        "test full pattern"_test = [] {
            auto utf8_ff = braille::utf8FromMask(0xFF);
            expect(braille::maskFromUtf8(utf8_ff) == 0xFFU);
        };

        "test specific patterns"_test = [] {
            for (std::uint8_t mask : std::vector<std::uint8_t>{0x01U, 0x42U, 0xA5U}) {
                auto utf8 = braille::utf8FromMask(mask);
                expect(braille::maskFromUtf8(utf8).value() == mask);
            }
        };
    };

    "Braille invalid input"_test = [] {
        expect(!braille::maskFromUtf8("ab"));           // Too short
        expect(!braille::maskFromUtf8("abcd"));         // Too long
        expect(!braille::maskFromUtf8("\xE0\xA0\x80")); // Wrong prefix
    };
};

const suite<"utf8::quad"> quadSuite = [] {
    using namespace std::string_view_literals;

    "Quad bit mapping"_test = [] {
        expect(quad::bit(0, 0) == 0b0100U) << "top-left";
        expect(quad::bit(0, 1) == 0b1000U) << "top-right";
        expect(quad::bit(1, 0) == 0b0001U) << "bottom-left";
        expect(quad::bit(1, 1) == 0b0010U) << "bottom-right";
    };

    "Quad glyph lookup"_test = [] {
        expect(quad::glyph(0x0) == " "sv);
        expect(quad::glyph(0xF) == "█"sv);
        expect(quad::glyph(0x1) == "▖"sv);
        expect(quad::glyph(0x2) == "▗"sv);
    };

    "Quad UTF-8 round-trip"_test = [] {
        for (std::uint8_t m = 0; m < 16; ++m) {
            auto glyph   = quad::glyph(m);
            auto decoded = quad::maskFromUtf8(glyph);
            expect(decoded.has_value());
            expect(decoded.value() == m);
        }
    };
};

const suite<"Point<T>"> pointSuite = [] {
    "Point arithmetic"_test = [] {
        constexpr Point p1{3.0, 4.0};
        constexpr Point p2{1.0, 2.0};

        constexpr Point sum    = p1 + p2;
        constexpr Point diff   = p1 - p2;
        constexpr Point scaled = p1 * 2.0;

        static_assert(sum == Point{4.0, 6.0});
        static_assert(diff == Point{2.0, 2.0});
        static_assert(scaled == Point{6.0, 8.0});

        Point p3 = p1;
        p3 += p2;
        expect(eq(p3, sum));
    };

    "Point norms"_test = [] {
        constexpr Point p1{3.0, 4.0};
        constexpr Point p2{0.0, 0.0};

        expect(eq(manhattanNorm(p1, p2), 7.0));
        expect(eq(chebyshevNorm(p1, p2), 4.0));
        expect(lt(std::abs(euclideanNorm(p1, p2) - 5.0), 0.001));

        static_assert(manhattanNorm(p1, p2) == 7.0);
        static_assert(chebyshevNorm(p1, p2) == 4.0);
        static_assert((euclideanNorm(p1, p2) - 5.0) < 0.001);
    };

    "Point deduction guide"_test = [] {
        constexpr Point p1{3, 4};    // Should deduce Point<int>
        constexpr Point p2{3.0, 4};  // Should deduce Point<double>
        constexpr Point p3{3, 4.0f}; // Should deduce Point<float>

        static_assert(std::same_as<decltype(p1)::value_type, int>);
        static_assert(std::same_as<decltype(p2)::value_type, double>);
        static_assert(std::same_as<decltype(p3)::value_type, float>);
    };

    "Point formatter"_test = [] {
        using namespace std::string_view_literals;
        using namespace std::string_literals;
        constexpr Point p{3.14, 2.71};
        expect(eq(std::format("{}", p), "(3.14,2.71)"s));
        expect(eq(std::format("{:.1f}", p), "(3.1,2.7)"s));
        ;
    };

    "Point_int_roundtrip"_test = [] {
        Point<int> p{42, -7};
        auto       m = toPropertyMap(p);
        expect(m.contains("point"));
        auto out = fromPropertyMap(Point<int>{}, m);
        expect(eq(out.x, 42)) << out.x;
        expect(eq(out.y, -7)) << out.y;
    };

    "Point_double_roundtrip"_test = [] {
        Point<double> p{3.5, -0.25};
        auto          m   = toPropertyMap(p);
        auto          out = fromPropertyMap(Point<double>{}, m);
        expect(approx(out.x, 3.5, 1e-3));
        expect(approx(out.y, -0.25, 1e-3));
    };

    "visual sanity for local runs"_test = [] {
        color::Colour c{0xFF, 0xD2, 0x00};
        auto          m = toPropertyMap(c);
        std::println("Colour hex: {}", m["colour"].value_or(std::string()));
    };
};

const suite<"ImCanvas"> canvasSuite = [] {
    "Canvas construction"_test = [] {
        "static canvas"_test = [] {
            ImCanvas<10UZ, 5UZ> static_cv;
            expect(static_cv.width() == 10UZ);
            expect(static_cv.height() == 5UZ);
            static_assert(static_cv.width() == 10UZ);
            static_assert(static_cv.height() == 5UZ);
        };

        "dynamic canvas"_test = [] {
            ImCanvas<> dynamic_cv(80, 25);
            expect(dynamic_cv.width() == 80UZ);
            expect(dynamic_cv.height() == 25UZ);
        };

        "compile-time operations"_test = [] {
            constexpr auto makeCanvas = []() consteval {
                ImCanvas<5, 3> cv;

                // text
                cv.put({0, 0}, "Hi", {});

                // points
                cv.put({1.02, 2.3});
                cv.put({{1.02, 2.3}, {2.02, 3.3}});

                // area
                cv.putArea({1.02, 2.3});

                // frame/masks
                cv.addMask({3, 0}, Direction::East);

                return cv._grapheme[0];
            };
            static_assert(makeCanvas() == "H");
        };
    };

    "Text placement"_test = [](bool colored) {
        ImCanvas<20, 5> cv;
        Style           s{.fg = {255, 0, 0}, .fgSet = 1};

        cv.put({0, 0}, "Test", s);
        cv.put({0, 2}, "UTF-8: ™€", {});

        auto output = cv.toString({}, colored);
        auto plain  = stripAnsiMetaInfo(output);

        expect(plain.starts_with("Test"));
        expect(plain.find("UTF-8: ™€") != std::string::npos);

        if (colored) {
            expect(output.find("\x1b[") != std::string::npos);
        }
    } | std::vector{false, true};

    "Braille points"_test = [] {
        ImCanvas<5, 5> cv;

        // Draw points to form all dots in one cell
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 2; ++c) {
                cv.put(Point{c * 0.5, r * 0.25}, {});
            }
        }

        auto mask = braille::maskFromUtf8(cv._grapheme[0]);
        expect(eq(static_cast<int>(mask.value()), 1));
    };

    "Quad areas"_test = [] {
        auto result = [] constexpr {
            ImCanvas<5, 5> cv;

            // fill all quadrants
            cv.putArea({0.25, 0.25});
            cv.putArea({0.75, 0.25});
            cv.putArea({0.25, 0.75});
            cv.putArea({0.75, 0.75});
            return cv._grapheme[0];
        };

        static_assert(result() == "█");
    };

    "Bar drawing"_test = [](bool colored) {
        ImCanvas<30, 10> cv;

        Style red{.fg = {255, 0, 0}, .fgSet = 1};
        Style blue{.fg = {0, 0, 255}, .fgSet = 1};

        cv.hBar<false>({5.0, 2.0}, 25.0, {red, blue});
        cv.hBar<true>({5.0, 4.0}, 25.0, {red, blue});
        cv.vBar({10.0, 8.0}, 6.0, {blue, red});

        auto output = cv.toString({}, colored);

        // check bars exist
        using namespace std::string_literals;
        for (std::size_t x = 5; x < 25; ++x) {
            expect(not eq(cv._grapheme[cv.index({x, 2})], " "s));
            expect(not eq(cv._grapheme[cv.index({x, 4})], " "s));
        }
    } | std::vector{false, true};

    "Line drawing"_test = [] {
        ImCanvas<20, 10> cv;

        std::vector<Point<double>> line = {{2.0, 2.0}, {10.0, 8.0}};
        cv.put(line, Style{.fg = {0, 255, 0}, .fgSet = 1});

        // Check some points were drawn
        bool hasPoints = false;
        for (const auto& g : cv._grapheme) {
            if (g != " " && !g.empty()) {
                hasPoints = true;
                break;
            }
        }
        expect(hasPoints);
    };

    "Box drawing and crossing"_test = [] {
        using namespace std::string_literals;

        ImCanvas<30, 20> cv;

        const auto e1 = pack_edge(1, 100);
        const auto e2 = pack_edge(2, 200);

        // Horizontal line
        for (std::size_t x = 5; x < 25; ++x) {
            cv.addMask({x, 10}, Direction::East | Direction::West, e1, {});
        }

        // Vertical line crossing (connected)
        for (std::size_t y = 5; y < 15; ++y) {
            cv.addMask({15, y}, Direction::North | Direction::South, e1, {});
        }

        // Check crossing point has proper character
        auto idx = cv.index({15, 10});
        expect(cv._grapheme[idx] == "┼"s);

        // Non-connected diagonal crossing
        for (std::size_t i = 0; i < 8; ++i) {
            cv.addMask({20 + i, 5 + i}, Direction::NorthWest | Direction::SouthEast, e1, {});
            cv.addMask({27 - i, 5 + i}, Direction::NorthEast | Direction::SouthWest, e2, {});
        }

        // Check diagonal crossing (should be ○ for non-connected)
        auto crossIdx = cv.index({23, 8}); // Approximate crossing point
        expect(cv._mask[crossIdx] != 0U);
    };

    using namespace boost::ut;
    using namespace std::string_literals;
    using namespace std::string_view_literals;

    "Visual demo"_test = [] {
        auto render = []<class F>(std::string_view label, F&& func) {
            for (bool ansi : {true, false}) { // ANSI first, then plain
                std::println("=== {} [{}] ===", label, ansi ? "ANSI" : "plain");
                auto canvas = std::forward<F>(func)();
                std::println("{}", canvas.toString({}, ansi));
            }
        };

        render("title", [] {
            ImCanvas<40UZ, 1UZ> cv;
            cv.put({0, 0}, "ImCanvas Demo", {Style{.fg = {255, 255, 255}, .fgSet = 1, .bold = 1}});
            return cv;
        });

        render("partial bars", [] {
            ImCanvas<40UZ, 1UZ> canvas;
            std::size_t         partialBarOffset = 1UZ;
            constexpr Style     red{.fg = {255, 0, 0}, .fgSet = 1};
            constexpr Style     yellow{.fg = {255, 210, 0}, .fgSet = 1};
            for (std::size_t i = 0UZ; i < 8UZ; ++i) {
                double coverage = static_cast<double>(i + 1) / 8.0;
                double x        = static_cast<double>(partialBarOffset + i);
                auto   style    = interpolateStyles(std::vector{yellow, red}, coverage);
                canvas.hBar({x, 0.0}, x + coverage, {style});
            }
            for (std::size_t i = 0UZ; i < 8UZ; ++i) {
                double coverage = static_cast<double>(i + 1UZ) / 8.0;
                auto   style    = interpolateStyles(std::vector{yellow, red}, coverage);
                double x        = static_cast<double>(partialBarOffset + i + 10UZ);
                canvas.vBar({x, 0.0}, 0.0 + coverage, {style});
            }
            return canvas;
        });

        render("colour bars", [] {
            ImCanvas<40UZ, 2UZ> canvas;
            canvas.hBar<false>({2.0, 0.0}, 20.0, {Style{.fg = {255, 0, 0}, .fgSet = 1}, Style{.fg = {0, 0, 255}, .fgSet = 1}});
            canvas.hBar<true>({2.0, 1.0}, 20.0, {Style{.fg = {255, 0, 0}, .fgSet = 1}, Style{.fg = {0, 0, 255}, .fgSet = 1}});
            return canvas;
        });

        render("box with lines", [] {
            ImCanvas<40UZ, 6UZ> canvas;
            Style               box{.fg = {100, 200, 255}, .fgSet = 1};
            canvas.hLine({0, 0}, {10, 0}, box);
            canvas.hLine({0, 5}, {10, 5}, box);
            canvas.vLine({0, 0}, {0, 5}, box);
            canvas.vLine({10, 0}, {10, 5}, box);

            canvas.hLine({10, 3}, {15, 3}, box);
            canvas.vLine({15, 1}, {15, 4}, box);
            canvas.hLine({15, 1}, {17, 1}, box);
            canvas.hLine({15, 4}, {13, 4}, box);
            return canvas;
        });

        render("sine wave", [] {
            ImCanvas<40UZ, 5UZ> canvas;
            for (std::size_t i = 0UZ; i < 100UZ; ++i) {
                const double x = 2.0 + static_cast<double>(i) * 0.35;
                const double y = 2.5 + 2.0 * std::sin(x);
                canvas.put(Point{x, y}, Style{.fg = {0, 255, 0}, .fgSet = 1});
            }
            return canvas;
        });

        render("star pattern", [] {
            ImCanvas<20UZ, 10UZ> canvas;
            for (int angle = 0; angle < 360; angle += 30) {
                constexpr Point<double> centre{10.0, 5.0};
                constexpr double        radius = 8.0;
                double                  rad    = angle * std::numbers::pi_v<double> / 180.0;
                Point<double>           end{centre.x + radius * std::cos(rad), centre.y + radius * std::sin(rad) * 0.5};

                Style                   centreStyle{.fg = {255, 255, 255}, .fgSet = 1};
                color::ColourHSV<float> hsv{static_cast<float>(angle), 1.0f, 1.0f};
                Style                   edgeStyle{.fg = color::hsvToRGB(hsv), .fgSet = 1};

                canvas.put(std::vector{centre, end}, std::vector{centreStyle, edgeStyle}, true);
            }
            return canvas;
        });

        render("sun pattern", [] {
            ImCanvas<20UZ, 10UZ> canvas;
            for (double deg = 0; deg < 360.0; deg += 2.5) {
                constexpr Point<double> centre{10.0, 5.0};
                constexpr double        radius = 6.5;
                const double            th     = deg * std::numbers::pi_v<double> / 180.0;
                for (double rho = 0.0; rho <= radius; rho += 0.5) {
                    const double t = rho / radius; // 0..1
                    Style        s{.fg = color::hsvToRGB(color::ColourHSV<float>{60.0f * float(1.0 - t), 1.0f, 1.0f}), .fgSet = 1};
                    canvas.putArea(Point{centre.x + rho * std::cos(th), centre.y + 0.5 * rho * std::sin(th)}, s); // 0.5 to counter the cell aspect ratio
                }
            }
            return canvas;
        });
    };
};

const suite<"Edge cases"> edgeSuite = [] {
    "UTF-8 decode"_test = [] {
        expect(decodeLength("") == 0UZ);
        expect(decodeLength("a") == 1UZ);
        expect(decodeLength("€") == 3UZ);
        expect(decodeLength("𝄞") == 4UZ);
        expect(decodeLength("\x80") == 0UZ); // Invalid
    };

    "Box symbols mapping"_test = [] {
        using namespace std::string_view_literals;
        using gr::utf8::detail::mapBoxSymbols;

        "basic directions"_test = [] {
            expect(eq(mapBoxSymbols(std::to_underlying(Direction::North), true), "│"sv));
            expect(eq(mapBoxSymbols(std::to_underlying(Direction::South), true), "│"sv));
            expect(eq(mapBoxSymbols(std::to_underlying(Direction::North | Direction::South), true), "│"sv));
            expect(eq(mapBoxSymbols(std::to_underlying(Direction::East), true), "─"sv));
            expect(eq(mapBoxSymbols(std::to_underlying(Direction::West), true), "─"sv));
            expect(eq(mapBoxSymbols(std::to_underlying(Direction::West | Direction::East), true), "─"sv));
        };

        "corners"_test = [] {
            expect(mapBoxSymbols(std::to_underlying(Direction::North | Direction::East), true) == "╰"sv);
            expect(mapBoxSymbols(std::to_underlying(Direction::North | Direction::West), true) == "╯"sv);
            expect(mapBoxSymbols(std::to_underlying(Direction::South | Direction::East), true) == "╭"sv);
            expect(mapBoxSymbols(std::to_underlying(Direction::South | Direction::West), true) == "╮"sv);
        };

        "crossings"_test = [] {
            expect(mapBoxSymbols(std::to_underlying(Direction::North | Direction::South | Direction::East | Direction::West), true) == "┼"sv);  // Connected
            expect(mapBoxSymbols(std::to_underlying(Direction::North | Direction::South | Direction::East | Direction::West), false) == "⌒"sv); // Not connected
        };

        // Diagonals
        expect(mapBoxSymbols(std::to_underlying(Direction::SouthWest | Direction::NorthEast), true) == "╱"sv);
        expect(mapBoxSymbols(std::to_underlying(Direction::SouthEast | Direction::NorthWest), true) == "╲"sv);
    };

    "Canvas bounds checking"_test = [] {
        ImCanvas<10UZ, 10UZ> cv;

        // out-of-bounds operations (should silently fail)
        cv.put({100, 100}, "X", {});
        cv.put(Point{-1.0, 5.0}, {});
        cv.putArea({100.0, 100.0}, {});

        // check nothing was written
        bool allEmpty = std::ranges::all_of(cv._grapheme, [](const auto& g) { return g == " "; });
        expect(allEmpty);
    };
};

// technical demo with additional examples
#include <cmath>
#include <print>

int main() { // only needed for technical demonstration, UT unit-tests do not need this.
    using namespace gr::utf8;

    // Create canvas
    ImCanvas<> canvas(80UZ, 32UZ);

    constexpr auto centreText = [](auto& cv, std::size_t x, std::size_t y, std::string_view text, Style style) {
        const std::size_t width = text.length();
        cv.put({x - width / 2UZ, y}, text, style);
    };

    // title bar spanning full width
    constexpr Style titleStyle{.fg = {255, 255, 255}, .fgSet = 1, .bold = 1};
    constexpr Style labelStyle{.fg = {200, 200, 200}, .fgSet = 1};
    for (std::size_t x = 0UZ; x < canvas.width(); ++x) {
        canvas.put({x, 0}, "═", titleStyle);
        canvas.put({x, canvas.height() - 1UZ}, "═", titleStyle);
    }
    canvas.put({0, 0}, "╔", titleStyle);
    canvas.put({canvas.width() - 1UZ, 0}, "╗", titleStyle);
    canvas.put({0, canvas.height() - 1UZ}, "╚", titleStyle);
    canvas.put({canvas.width() - 1UZ, canvas.height() - 1UZ}, "╝", titleStyle);
    centreText(canvas, canvas.width() / 2UZ, 0, " ImCanvas Technical Demonstration ", titleStyle);

    // default palette with dark/bright pairs
    canvas.put({2, 1}, "Default Palette:", labelStyle);
    using Pal = color::palette::Default;
    for (std::size_t i = 0UZ; i < Pal::all.size() / 2; ++i) {
        std::size_t x = 2UZ + i * 3UZ;
        canvas.put({x, 2UZ}, "█", {.fg = Pal::all[i], .fgSet = 1});
        canvas.put({x + 1UZ, 2UZ}, "█", {.fg = Pal::all[i + 8], .fgSet = 1});
    }

    // RGB vs HSV interpolation bars (blue to red)
    constexpr Style red{.fg = {255, 0, 0}, .fgSet = 1};
    constexpr Style blue{.fg = {0, 0, 255}, .fgSet = 1};
    constexpr Style green{.fg = {0, 255, 0}, .fgSet = 1};
    constexpr Style yellow{.fg = {255, 210, 0}, .fgSet = 1};

    canvas.put({2, 4}, "RGB:", labelStyle);
    canvas.hBar<false>({8.0, 4.0}, 35.0, {blue, red});
    canvas.put({2, 5}, "HSV:", labelStyle);
    canvas.hBar<true>({8.0, 5.0}, 35.0, {blue, red});
    canvas.put({2, 7}, "R:", labelStyle);
    canvas.put({4, 7}, "H:", labelStyle);
    canvas.vBar<false>({2.0, 20.0}, 8.0, {blue, green, red});
    canvas.vBar<false>({3.0, 20.0}, 8.0, {blue, green, red});
    canvas.vBar<true>({4.0, 20.0}, 8.0, {blue, green, red});
    canvas.vBar<true>({5.0, 20.0}, 8.0, {blue, green, red});

    // partial bars
    std::size_t partialBarOffset = 40UZ;
    canvas.put({partialBarOffset, 1UZ}, "Partial bars:", labelStyle);
    for (std::size_t i = 0UZ; i < 8UZ; ++i) {
        double coverage = static_cast<double>(i + 1) / 8.0;
        double x        = static_cast<double>(partialBarOffset + i);
        auto   style    = interpolateStyles(std::vector{yellow, red}, coverage);
        canvas.hBar({x, 2.0}, x + coverage, {style});
    }
    for (std::size_t i = 0UZ; i < 8UZ; ++i) {
        double coverage = static_cast<double>(i + 1UZ) / 8.0;
        auto   style    = interpolateStyles(std::vector{yellow, red}, coverage);
        double x        = static_cast<double>(partialBarOffset + i + 10UZ);
        canvas.vBar({x, 2.0}, 2.0 + coverage, {style});
    }

    // crossing lines
    constexpr auto e1 = pack_edge(1, 100);
    constexpr auto e2 = pack_edge(2, 200);

    for (std::size_t x = 10; x < 50; ++x) { // horizontal axis
        canvas.addMask({x, 15}, Direction::East | Direction::West, e1, green);
    }
    for (std::size_t y = 8UZ; y < 28UZ; ++y) { // vertical axis crossing
        canvas.addMask({23, y}, Direction::North | Direction::South, e1, green);
    }
    for (std::size_t y = 8UZ; y < 28UZ; ++y) { // vertical axis non-crossing
        canvas.addMask({25, y}, Direction::North | Direction::South, e2, yellow);
    }

    // diagonal crossings
    for (std::size_t k = 0UZ; k < 12UZ; ++k) {
        canvas.addMask({30UZ + k, 11UZ + k}, Direction::NorthWest | Direction::SouthEast, e1, green);
        canvas.addMask({35UZ + k, 20UZ - k}, Direction::SouthWest | Direction::NorthEast, e1, green);
        canvas.addMask({37UZ + k, 20UZ - k}, Direction::SouthWest | Direction::NorthEast, e2, red);
    }

    // filled circle area
    for (double deg = 0; deg < 360.0; deg += 2.5) {
        constexpr Point<double> c{73.5, 6.5};
        constexpr double        r  = 6.5;
        const double            th = deg * std::numbers::pi_v<double> / 180.0;
        for (double rho = 0.0; rho <= r; rho += 0.5) {
            const double t = rho / r; // 0..1
            Style        s{.fg = color::hsvToRGB(color::ColourHSV<float>{60.0f * float(1.0 - t), 1.0f, 1.0f}), .fgSet = 1};
            canvas.putArea(Point{c.x + rho * std::cos(th), c.y + 0.5 * rho * std::sin(th)}, s); // 0.5 to counter the cell aspect ratio
        }
    }

    // sine waves and lines
    std::vector<Point<double>> sine1(200UZ);
    std::vector<Point<double>> sine2(sine1.size());
    std::vector<Point<double>> sine3(sine1.size());
    std::vector<Style>         style2(sine1.size());
    for (std::size_t i = 0UZ; i < sine1.size(); ++i) {
        double x  = 10.0 + double(i) * 0.25;
        double y1 = 13.0 + 3.0 * std::sin(double(i) * 0.05);
        double y2 = 11.0 + 3.0 * std::sin(double(i) * 0.05 + 0.25);
        sine1[i]  = {x, y1};
        sine2[i]  = {x, y2};
        sine3[i]  = {x, y2 + 0.05};
        float hue = static_cast<float>(i) * 1.4f; // rainbow style
        style2[i] = {.fg = color::hsvToRGB(color::ColourHSV{hue, 1.0f, 1.0f}), .fgSet = 1};
    }
    canvas.put(sine1, green);
    canvas.put({{27.0, 24.0}, {46.0, 27.2}}, green);
    canvas.put({{27.0, 25.05}, {46.0, 28.25}});
    canvas.put(sine2, style2);
    canvas.put(sine3, style2);

    // gradient landscape with vertical bars (lower right quadrant)
    for (std::size_t x = 50; x < 80; ++x) {
        double height = 3.0 + 2.0 * std::sin((static_cast<double>(x) - 50.) * 0.22);
        double xd     = static_cast<double>(x);
        double yd     = static_cast<double>(canvas.height() - 2UZ);
        canvas.vBar({xd, yd}, yd - height, {{.fg = {34, 139, 34}, .fgSet = 1}, {.fg = Pal::BrightCyan, .fgSet = 1}});
    }

    // star pattern with gradient lines (lower right, no background)
    for (int angle = 0; angle < 360; angle += 30) {
        constexpr Point<double> centre{60.0, 20.0};
        constexpr double        radius = 6.0;
        double                  rad    = angle * std::numbers::pi_v<double> / 180.0;
        Point<double>           end{centre.x + radius * std::cos(rad), centre.y + radius * std::sin(rad) * 0.5};

        Style                   centreStyle{.fg = {255, 255, 255}, .fgSet = 1};
        color::ColourHSV<float> hsv{static_cast<float>(angle), 1.0f, 1.0f};
        Style                   edgeStyle{.fg = color::hsvToRGB(hsv), .fgSet = 1};

        canvas.put(std::vector{centre, end}, std::vector{centreStyle, edgeStyle}, true);
    }

    // stars
    Style starStyle{.fg = {255, 255, 200}, .fgSet = 1};
    canvas.put({50.3, 18.8}, starStyle);
    canvas.put({53.7, 19.2}, starStyle);
    canvas.put({65.1, 18.5}, starStyle);
    canvas.put({70.6, 20.3}, starStyle);
    canvas.put({75.4, 21.9}, starStyle);
    canvas.put({77.8, 19.3}, starStyle);

    // Box drawing with label on frame
    Style boxStyle{.fg = {100, 150, 255}, .fgSet = 1};
    canvas.hLine({4, 21}, {18, 21}, boxStyle);
    canvas.hLine({4, 25}, {18, 25}, boxStyle);
    canvas.vLine({4, 21}, {4, 25}, boxStyle);
    canvas.vLine({18, 21}, {18, 25}, boxStyle);
    canvas.hLine({18, 23}, {21, 23}, boxStyle); // line leaving the box on the right
    canvas.put({7, 21}, "Box Label", boxStyle);

    // output with ANSI colors
    std::print("{}", canvas.toString({}, true));

    // optional: non-colored version
    std::print("\n--- Without Colors ---\n{}", canvas.toString({}, false));

    return 0;
}
