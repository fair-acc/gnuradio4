#include <gnuradio-4.0/algorithm/Colour.hpp>

#include <boost/ut.hpp>

#include <array>
#include <bit>
#include <cstdio>
#include <limits>
#include <random>
#include <span>

#include <format>

#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/algorithm/dataset/DataSetUtils.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>

const boost::ut::suite<"ImChart"> windowTests = [] {
    using namespace boost::ut;
    using namespace boost::ut::reflection;

    "omitting the source footer leaves both direction names visible beside a centred vertical axis"_test = [] {
        gr::graphs::ImChart<100UZ, 14UZ> chart({{-180.0, 180.0}, {0.0, 1.0}});
        constexpr std::array             kAngles{-150.0, 0.0, 150.0};
        constexpr std::array             kPower{0.25, 0.5, 0.05};
        chart.draw(kAngles, kPower, "azimuth");
        chart.draw(kAngles, kPower, "elevation");
        chart.draw(std::nullopt);
        std::string legend;
        for (const auto& cell : chart._screen.back()) {
            legend += cell;
        }
        expect(legend.contains("azimuth"));
        expect(legend.contains("elevation"));
        expect(!legend.contains("qa_ImChart.cpp"));
        const auto unannotated = chart._screen.back();
        expect(neq(unannotated[91UZ], std::string(" ")));
        chart.draw();
        expect(eq(chart._screen.back(), unannotated));
        chart.draw(std::nullopt);
        expect(eq(chart._screen.back(), unannotated));
    };

    "heatmap normalisation preserves signed values, clips limits and handles constant and extreme ranges"_test = [] {
        using gr::graphs::normaliseHeatmapValue;
        expect(eq(normaliseHeatmapValue(-2.0, -2.0, 2.0), 0.0));
        expect(eq(normaliseHeatmapValue(0.0, -2.0, 2.0), 0.5));
        expect(eq(normaliseHeatmapValue(2.0, -2.0, 2.0), 1.0));
        expect(eq(normaliseHeatmapValue(-3.0, -2.0, 2.0), 0.0));
        expect(eq(normaliseHeatmapValue(3.0, -2.0, 2.0), 1.0));
        expect(eq(normaliseHeatmapValue(7.0, 7.0, 7.0), 0.5));
        constexpr double kLargest = std::numeric_limits<double>::max();
        expect(eq(normaliseHeatmapValue(0.0, -kLargest, kLargest), 0.5));
        expect(eq(normaliseHeatmapValue(3.0, 2.0, 6.0), 0.25));
    };

    "heatmap sampling preserves endpoints and defines nearest-neighbour ties and singletons"_test = [] {
        constexpr std::array kExpected{0UZ, 1UZ, 1UZ, 2UZ, 2UZ};
        for (std::size_t i = 0UZ; i < kExpected.size(); ++i) {
            expect(eq(gr::graphs::heatmapSampleIndex(i, 5UZ, 3UZ), kExpected[i]));
            expect(eq(gr::graphs::heatmapSampleIndex(i, 5UZ, 1UZ), 0UZ));
        }
        expect(eq(gr::graphs::heatmapSampleIndex(0UZ, 1UZ, 5UZ), 0UZ));
        expect(eq(gr::graphs::heatmapSampleIndex(1UZ, 3UZ, 5UZ), 2UZ));
        expect(eq(gr::graphs::heatmapSampleIndex(2UZ, 3UZ, 5UZ), 4UZ));
        expect(gr::graphs::heatmapColour(0.0) == gr::utf8::color::Colour{0U, 32U, 128U});
        expect(gr::graphs::heatmapColour(0.5) == gr::utf8::color::Colour{240U, 240U, 240U});
        expect(gr::graphs::heatmapColour(1.0) == gr::utf8::color::Colour{160U, 0U, 0U});
        expect(gr::graphs::heatmapColour(0.25) == gr::utf8::color::Colour{120U, 136U, 184U});
    };

    "normalised heatmap indices round ties upwards and map signed-angle marker positions to nearest cells"_test = [] {
        expect(eq(gr::graphs::heatmapSampleIndex(0.0, 47UZ), 0UZ));
        expect(eq(gr::graphs::heatmapSampleIndex(0.5, 47UZ), 23UZ));
        expect(eq(gr::graphs::heatmapSampleIndex(1.0, 47UZ), 46UZ));
        expect(eq(gr::graphs::heatmapSampleIndex(0.25, 3UZ), 1UZ));
        expect(eq(gr::graphs::heatmapSampleIndex(0.75, 3UZ), 2UZ));
        expect(eq(gr::graphs::heatmapSampleIndex(7.0 / 12.0, 47UZ), 27UZ));
        expect(eq(gr::graphs::heatmapSampleIndex(5.0 / 12.0, 47UZ), 19UZ));
        expect(eq(gr::graphs::heatmapSampleIndex(0.75, 1UZ), 0UZ));
        std::array<bool, 37UZ> covered{};
        for (std::size_t row = 0UZ; row < 47UZ; ++row) {
            covered[gr::graphs::heatmapSampleIndex(row, 47UZ, 37UZ)] = true;
        }
        expect(std::ranges::all_of(covered, std::identity{}));
    };

    "heatmap markers at plus and minus fifteen degrees align with sampled peaks"_test = [] {
        gr::graphs::ImChart<100UZ, 51UZ> chart({{-90.0, 90.0}, {-90.0, 90.0}});
        std::vector<float>               values(37UZ * 37UZ, 0.f);
        values[21UZ * 37UZ + 18UZ] = 1.f;
        values[15UZ * 37UZ + 18UZ] = 1.f;
        expect(chart.drawHeatmap(std::span<const float>(values), 37UZ, 37UZ, {0.0, 1.0}, {}, gr::graphs::HeatmapStyle::monochrome).has_value());
        expect(eq(chart._screen[20UZ][52UZ], std::string("⣿")));
        expect(eq(chart._screen[28UZ][52UZ], std::string("⣿")));
        constexpr std::array kMarkers{std::array{0.0, 15.0}, std::array{0.0, -15.0}};
        expect(chart.drawHeatmap(std::span<const float>(values), 37UZ, 37UZ, {0.0, 1.0}, kMarkers, gr::graphs::HeatmapStyle::monochrome).has_value());
        expect(eq(chart._screen[20UZ][52UZ], std::string("1")));
        expect(eq(chart._screen[28UZ][52UZ], std::string("2")));
    };

    "axis normalisation places linear midpoints and logarithmic geometric means at one half"_test = [] {
        expect(eq(gr::graphs::LinearAxisTransform::normalise(0.0, -2.0, 2.0), 0.5));
        expect(eq(gr::graphs::LogAxisTransform::normalise(1.0, 1.0, 16.0), 0.0));
        expect(eq(gr::graphs::LogAxisTransform::normalise(4.0, 1.0, 16.0), 0.5));
        expect(eq(gr::graphs::LogAxisTransform::normalise(16.0, 1.0, 16.0), 1.0));
    };

    "logarithmic heatmap ticks use readable decades with separated labels and inclusive endpoints"_test = [] {
        const auto           ticks = gr::graphs::detail::heatmapAxisTicks<gr::graphs::LogAxisTransform>(100.0, 10000.0, 21UZ);
        constexpr std::array kPositions{0UZ, 3UZ, 7UZ, 10UZ, 13UZ, 17UZ, 20UZ};
        constexpr std::array kValues{100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0};
        expect(fatal(eq(ticks.size(), kValues.size())));
        for (std::size_t i = 0UZ; i < kValues.size(); ++i) {
            expect(eq(ticks[i].first, kPositions[i]));
            expect(eq(ticks[i].second, kValues[i]));
        }
        const auto narrow = gr::graphs::detail::heatmapAxisTicks<gr::graphs::LogAxisTransform>(100.0, 1000.0, 2UZ);
        expect(eq(narrow.size(), 2UZ));
        expect(eq(narrow.front().first, 0UZ));
        expect(eq(narrow.back().first, 1UZ));
    };

    "logarithmic x heatmaps place geometric midpoints and markers in the central cell"_test = [] {
        using namespace gr::graphs;
        ImChart<64UZ, 9UZ, LogAxisTransform> chart({{1.0, 16.0}, {-1.0, 1.0}});
        constexpr std::array                 kValues{0.0, 1.0, 0.0};
        expect(chart.drawHeatmap(std::span(kValues), 1UZ, 3UZ, {0.0, 1.0}, {}, HeatmapStyle::monochrome).has_value());
        expect(eq(chart._screen[3UZ][34UZ], std::string("⣿")));
        constexpr std::array<std::array<double, 2UZ>, 1UZ> kMarker{std::array{4.0, 0.0}};
        expect(chart.drawHeatmap(std::span(kValues), 1UZ, 3UZ, {0.0, 1.0}, kMarker, HeatmapStyle::monochrome).has_value());
        expect(eq(chart._screen[3UZ][34UZ], std::string("1")));
        chart.axis_min_x = 0.0;
        expect(!chart.drawHeatmap(std::span(kValues), 1UZ, 3UZ, {0.0, 1.0}));
        chart.axis_min_x = 1e100;
        chart.axis_max_x = std::nextafter(1e100, std::numeric_limits<double>::infinity());
        expect(!chart.drawHeatmap(std::span(kValues), 1UZ, 3UZ, {0.0, 1.0}));
    };

    "logarithmic y intervals preserve nonuniform physical boundaries"_test = [] {
        using namespace gr::graphs;
        ImChart<64UZ, 9UZ, LinearAxisTransform, LogAxisTransform> chart({{-1.0, 1.0}, {100.0, 1600.0}});
        constexpr std::array                                      kValues{0.0, 0.5, 1.0};
        constexpr std::array                                      kEdges{100.0, 200.0, 800.0, 1600.0};
        expect(chart.drawHeatmap(std::span(kValues), 3UZ, 1UZ, {0.0, 1.0}, {}, HeatmapStyle::monochrome, stdout, kEdges).has_value());
        expect(eq(chart._screen[5UZ][62UZ], std::string("⠀")));
        expect(eq(chart._screen[4UZ][62UZ], std::string("⣉")));
        expect(eq(chart._screen[3UZ][62UZ], std::string("⣉")));
        expect(eq(chart._screen[2UZ][62UZ], std::string("⣿")));
        expect(eq(chart._screen[1UZ][62UZ], std::string("⣿")));
        chart.axis_min_y = -1.0;
        expect(!chart.drawHeatmap(std::span(kValues), 3UZ, 1UZ, {0.0, 1.0}));
    };

    "two logarithmic axes share geometric marker positions and invalid row edges fail before drawing"_test = [] {
        using namespace gr::graphs;
        ImChart<64UZ, 9UZ, LogAxisTransform, LogAxisTransform> chart({{1.0, 16.0}, {100.0, 1600.0}});
        constexpr std::array                                   kValues{1.0};
        constexpr std::array<std::array<double, 2UZ>, 1UZ>     kMarkers{std::array{4.0, 400.0}};
        expect(chart.drawHeatmap(std::span(kValues), 1UZ, 1UZ, {0.0, 1.0}, kMarkers, HeatmapStyle::monochrome).has_value());
        expect(eq(chart._screen[3UZ][37UZ], std::string("1")));
        constexpr std::array kInvalidEdges{std::array{100.0, 1600.0, 1600.0}, std::array{100.0, 1700.0, 1600.0}, std::array{101.0, 400.0, 1600.0}, std::array{100.0, 400.0, 1599.0}, std::array{100.0, std::numeric_limits<double>::quiet_NaN(), 1600.0}};
        constexpr std::array kTwoRows{0.0, 1.0};
        for (const auto& edges : kInvalidEdges) {
            expect(!chart.drawHeatmap(std::span(kTwoRows), 2UZ, 1UZ, {0.0, 1.0}, {}, HeatmapStyle::monochrome, stdout, edges));
        }
        expect(!chart.drawHeatmap(std::span(kValues), 1UZ, 1UZ, {0.0, 1.0}, {}, HeatmapStyle::monochrome, stdout, kInvalidEdges[0UZ]));
    };

    "heatmap braille density rounds to nested zero through eight dot masks"_test = [] {
        for (unsigned int dots = 0U; dots <= 8U; ++dots) {
            const auto mask = gr::graphs::heatmapBrailleMask(static_cast<double>(dots) / 8.0);
            expect(eq(std::popcount(mask), static_cast<int>(dots)));
            if (dots > 0U) {
                const auto previous = gr::graphs::heatmapBrailleMask(static_cast<double>(dots - 1U) / 8.0);
                expect(eq(mask & previous, static_cast<int>(previous)));
            }
        }
        expect(eq(gr::graphs::heatmapBrailleMask(0.0), std::uint8_t{0x00U}));
        expect(eq(gr::graphs::heatmapBrailleMask(0.5), std::uint8_t{0xc9U}));
        expect(eq(gr::graphs::heatmapBrailleMask(1.0), std::uint8_t{0xffU}));
        expect(eq(std::popcount(gr::graphs::heatmapBrailleMask(0.0624)), 0));
        expect(eq(std::popcount(gr::graphs::heatmapBrailleMask(0.0625)), 1));
        expect(eq(std::popcount(gr::graphs::heatmapBrailleMask(0.9374)), 7));
        expect(eq(std::popcount(gr::graphs::heatmapBrailleMask(0.9375)), 8));
    };

    "printing a screen preserves UTF-8 colours empty cells row breaks and the final reset"_test = [] {
        gr::graphs::ImChart<3UZ, 2UZ> chart;
        chart._screen           = {{"A", "⣉", "\x1b[31m⣿\x1b[0m"}, {"", "B", " "}};
        std::FILE* const output = std::tmpfile();
        expect(fatal(output != nullptr));
        const gr::on_scope_exit closeOutput([output] { std::fclose(output); });
        chart.printScreen(output);
        std::rewind(output);
        std::array<char, 128UZ> bytes{};
        const std::size_t       count = std::fread(bytes.data(), 1UZ, bytes.size(), output);
        expect(eq(std::string(bytes.data(), count), std::string("A⣉\x1b[31m⣿\x1b[0m\nB \n\x1b[39m")));
        expect(std::feof(output) != 0);
    };

    "heatmaps default to coloured braille for every cell and can write the frame and scale to a supplied stream"_test = [] {
        using gr::graphs::HeatmapStyle;
        gr::graphs::ImChart<16UZ, 8UZ> implicit({{-1.0, 1.0}, {-1.0, 1.0}});
        gr::graphs::ImChart<16UZ, 8UZ> explicitStyle({{-1.0, 1.0}, {-1.0, 1.0}});
        constexpr std::array           kValues{-2.0, 0.0, 1.0, 2.0};
        const std::array               markers{std::array{-1.0, -1.0}, std::array{1.0, 1.0}};
        expect(implicit.drawHeatmap(std::span(kValues), 2UZ, 2UZ, {-2.0, 2.0}, markers).has_value());
        std::FILE* const output = std::tmpfile();
        expect(fatal(output != nullptr));
        const gr::on_scope_exit closeOutput([output] { std::fclose(output); });
        expect(explicitStyle.drawHeatmap(std::span(kValues), 2UZ, 2UZ, {-2.0, 2.0}, markers, HeatmapStyle::braille, output).has_value());
        expect(std::ranges::equal(implicit._screen, explicitStyle._screen));
        std::rewind(output);
        std::array<char, 4096UZ> bytes{};
        const std::size_t        count = std::fread(bytes.data(), 1UZ, bytes.size(), output);
        const std::string        text(bytes.data(), count);
        expect(text.contains("linear dot-density scale: -2 "));
        expect(text.contains("⣉"));
        expect(text.contains("⣿"));
        expect(!text.contains("█"));
        expect(std::feof(output) != 0);
        expect(eq(std::ranges::count(text, '\n'), 9));
    };

    "braille heatmaps encode the same scores in colour and monochrome with unchanged axes"_test = [] {
        using gr::graphs::HeatmapStyle;
        gr::graphs::ImChart<16UZ, 8UZ> coloured({{-1.0, 1.0}, {-1.0, 1.0}});
        gr::graphs::ImChart<16UZ, 8UZ> monochrome({{-1.0, 1.0}, {-1.0, 1.0}});
        constexpr std::array           kValues{-2.0, 0.0, 2.0, std::numeric_limits<double>::quiet_NaN()};
        expect(coloured.drawHeatmap(std::span(kValues), 2UZ, 2UZ, {-2.0, 2.0}, {}, HeatmapStyle::braille).has_value());
        expect(monochrome.drawHeatmap(std::span(kValues), 2UZ, 2UZ, {-2.0, 2.0}, {}, HeatmapStyle::monochrome).has_value());
        expect(eq(coloured._screen[1UZ][5UZ], std::string("\x1b[38;2;160;0;0m⣿\x1b[0m")));
        expect(eq(coloured._screen[4UZ][5UZ], std::string("\x1b[38;2;0;32;128m⠀\x1b[0m")));
        expect(eq(coloured._screen[4UZ][14UZ], std::string("\x1b[38;2;240;240;240m⣉\x1b[0m")));
        expect(eq(monochrome._screen[1UZ][5UZ], std::string("⣿")));
        expect(eq(monochrome._screen[4UZ][5UZ], std::string("⠀")));
        expect(eq(monochrome._screen[4UZ][14UZ], std::string("⣉")));
        expect(eq(monochrome._screen[1UZ][14UZ], std::string(" ")));
        expect(eq(monochrome._screen[6UZ], coloured._screen[6UZ]));
        const auto hasNoEscape = [](const auto& row) { return std::ranges::none_of(row, [](const auto& cell) { return cell.contains('\x1b'); }); };
        expect(std::ranges::all_of(monochrome._screen, hasNoEscape));
        const std::array markers{std::array{-1.0, -1.0}, std::array{1.0, 1.0}};
        expect(monochrome.drawHeatmap(std::span(kValues), 2UZ, 2UZ, {-2.0, 2.0}, markers, HeatmapStyle::monochrome).has_value());
        expect(eq(monochrome._screen[4UZ][5UZ], std::string("1")));
        expect(eq(monochrome._screen[1UZ][14UZ], std::string("2")));
        constexpr std::array kSingle{7.0};
        expect(monochrome.drawHeatmap(std::span(kSingle), 1UZ, 1UZ, {7.0, 7.0}, {}, HeatmapStyle::monochrome).has_value());
        expect(eq(monochrome._screen[1UZ][5UZ], std::string("⣉")));
    };

    "heatmap rows rise from minimum y and coloured cells restore terminal attributes"_test = [] {
        gr::graphs::ImChart<16UZ, 8UZ> chart({{-1.0, 1.0}, {-1.0, 1.0}});
        constexpr std::array           kValues{-2.0, 0.0, 1.0, 2.0};
        expect(chart.drawHeatmap(std::span(kValues), 2UZ, 2UZ, {-2.0, 2.0}).has_value());
        expect(eq(chart._screen[1UZ][5UZ], std::string("\x1b[38;2;200;120;120m⣭\x1b[0m")));
        expect(eq(chart._screen[1UZ][14UZ], std::string("\x1b[38;2;160;0;0m⣿\x1b[0m")));
        expect(eq(chart._screen[4UZ][5UZ], std::string("\x1b[38;2;0;32;128m⠀\x1b[0m")));
        expect(eq(chart._screen[4UZ][14UZ], std::string("\x1b[38;2;240;240;240m⣉\x1b[0m")));
    };

    "heatmap margins fit interior tick labels without overwriting axes or samples"_test = [] {
        gr::graphs::ImChart<16UZ, 14UZ> chart({{-1.0, 1.0}, {-1.0, 1.0}});
        constexpr std::array            kValues{0.0};
        expect(chart.drawHeatmap(std::span(kValues), 1UZ, 1UZ, {-1.0, 1.0}).has_value());
        for (const auto& [row, label] : std::array{std::pair{4UZ, "+0.333"}, std::pair{7UZ, "-0.333"}}) {
            for (std::size_t column = 0UZ; column < 6UZ; ++column) {
                expect(eq(chart._screen[row][column + 1UZ], std::string(1UZ, label[column])));
            }
            expect(eq(chart._screen[row][7UZ], std::string(" ")));
            expect(eq(chart._screen[row][8UZ], std::string("┤")));
        }
        for (std::size_t row = 1UZ; row <= 10UZ; ++row) {
            expect(eq(chart._screen[row][9UZ], std::string("\x1b[38;2;240;240;240m⣉\x1b[0m")));
        }
        gr::graphs::ImChart<11UZ, 14UZ> narrow({{-1.0, 1.0}, {-1.0, 1.0}});
        expect(!narrow.drawHeatmap(std::span(kValues), 1UZ, 1UZ, {-1.0, 1.0}));
        expect(std::ranges::all_of(narrow._screen, [](const auto& row) { return std::ranges::all_of(row, [](const auto& cell) { return cell == " "; }); }));
    };

    "axis labels are centred and clamped to the available columns"_test = [] {
        using gr::graphs::detail::axisLabelStartColumn;
        expect(eq(axisLabelStartColumn(9UZ, 7UZ, 16UZ), 6UZ));
        expect(eq(axisLabelStartColumn(14UZ, 7UZ, 16UZ), 9UZ));
        expect(eq(axisLabelStartColumn(1UZ, 6UZ, 16UZ), 0UZ));
        expect(eq(axisLabelStartColumn(8UZ, 16UZ, 16UZ), 0UZ));
        expect(eq(axisLabelStartColumn(8UZ, 0UZ, 16UZ), 8UZ));
    };

    "heatmap x ticks lose crowded interior labels and reject overlapping endpoints before drawing"_test = [] {
        constexpr std::array           kValues{0.0};
        gr::graphs::ImChart<16UZ, 8UZ> chart({{-1.0, 1.0}, {-1.0, 1.0}});
        expect(chart.drawHeatmap(std::span(kValues), 1UZ, 1UZ, {-1.0, 1.0}).has_value());
        const std::string labels = chart._screen[6UZ] | std::views::join | std::ranges::to<std::string>();
        expect(eq(labels, std::string("    -1       +1 ")));
        gr::graphs::ImChart<16UZ, 14UZ> scientific({{-1e100, 1e100}, {-1.0, 1.0}});
        expect(!scientific.drawHeatmap(std::span(kValues), 1UZ, 1UZ, {-1.0, 1.0}));
        expect(std::ranges::all_of(scientific._screen, [](const auto& row) { return std::ranges::all_of(row, [](const auto& cell) { return cell == " "; }); }));
        gr::graphs::ImChart<29UZ, 14UZ> wide({{-1e100, 1e100}, {-1.0, 1.0}});
        expect(wide.drawHeatmap(std::span(kValues), 1UZ, 1UZ, {-1.0, 1.0}).has_value());
        const std::string wideLabels = wide._screen[12UZ] | std::views::join | std::ranges::to<std::string>();
        expect(eq(wideLabels, std::string("      -1e+100    +0   +1e+100")));
        expect(eq(wide._screen[11UZ][9UZ], std::string("┌")));
        expect(eq(wide._screen[11UZ][18UZ], std::string("┬")));
        expect(eq(wide._screen[11UZ][27UZ], std::string("┬")));
        expect(eq(wide._screen[11UZ][15UZ], std::string("─")));
        expect(eq(wide._screen[11UZ][21UZ], std::string("─")));
    };

    "heatmap endpoint markers are inclusive and invalid markers or non-finite cells are omitted"_test = [] {
        gr::graphs::ImChart<16UZ, 8UZ> chart({{-1.0, 1.0}, {-1.0, 1.0}});
        const std::array               values{-2.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(), 2.0};
        expect(chart.drawHeatmap(std::span(values), 2UZ, 2UZ, {-2.0, 2.0}).has_value());
        expect(eq(chart._screen[1UZ][5UZ], std::string(" ")));
        expect(eq(chart._screen[4UZ][14UZ], std::string(" ")));
        const std::array markers{std::array{-1.0, -1.0}, std::array{1.0, 1.0}, std::array{-2.0, 0.0}, std::array{std::numeric_limits<double>::quiet_NaN(), 0.0}};
        expect(chart.drawHeatmap(std::span(values), 2UZ, 2UZ, {-2.0, 2.0}, markers).has_value());
        expect(eq(chart._screen[4UZ][5UZ], std::string("\x1b[1;30;107m1\x1b[0m")));
        expect(eq(chart._screen[1UZ][14UZ], std::string("\x1b[1;30;107m2\x1b[0m")));
    };

    "heatmap singletons and constant fields render while invalid shapes and ranges fail before drawing"_test = [] {
        gr::graphs::ImChart<16UZ, 8UZ> chart({{10.0, 20.0}, {5.0, 6.0}});
        constexpr std::array           kSingle{7.f};
        expect(chart.drawHeatmap(std::span(kSingle), 1UZ, 1UZ, {7.0, 7.0}).has_value());
        expect(eq(chart._screen[1UZ][5UZ], std::string("\x1b[38;2;240;240;240m⣉\x1b[0m")));
        expect(!chart.drawHeatmap(std::span(kSingle), 2UZ, 1UZ, {0.0, 1.0}));
        expect(!chart.drawHeatmap(std::span<const float>{}, 0UZ, 0UZ, {0.0, 1.0}));
        expect(!chart.drawHeatmap(std::span(kSingle), std::numeric_limits<std::size_t>::max(), 2UZ, {0.0, 1.0}));
        expect(!chart.drawHeatmap(std::span(kSingle), 1UZ, 1UZ, {1.0, 0.0}));
        expect(!chart.drawHeatmap(std::span(kSingle), 1UZ, 1UZ, {0.0, std::numeric_limits<double>::infinity()}));
        chart.axis_min_y = chart.axis_max_y;
        expect(!chart.drawHeatmap(std::span(kSingle), 1UZ, 1UZ, {0.0, 1.0}));
        gr::graphs::ImChart<8UZ, 4UZ> small({{0.0, 1.0}, {0.0, 1.0}});
        expect(!small.drawHeatmap(std::span(kSingle), 1UZ, 1UZ, {0.0, 1.0}));
        gr::graphs::ImChart<9UZ, 6UZ> scientific({{-1.23456e100, -1e100}, {0.0, 1.0}});
        expect(!scientific.drawHeatmap(std::span(kSingle), 1UZ, 1UZ, {0.0, 1.0}));
    };

    "axes remain inside positive negative and zero-crossing coordinate ranges"_test = [] {
        gr::graphs::ImChart<40UZ, 12UZ> positive({{10.0, 20.0}, {5.0, 10.0}});
        expect(eq(positive.getVerticalAxisPositionX(), 5UZ));
        expect(eq(positive.getHorizontalAxisPositionY(), 9UZ));
        positive.drawAxes();
        gr::graphs::ImChart<40UZ, 12UZ> negative({{-20.0, -10.0}, {-10.0, -5.0}});
        expect(eq(negative.getVerticalAxisPositionX(), 39UZ));
        expect(eq(negative.getHorizontalAxisPositionY(), 0UZ));
        negative.drawAxes();
        gr::graphs::ImChart<40UZ, 12UZ> crossing({{-1.0, 1.0}, {-1.0, 1.0}});
        expect(eq(crossing.getVerticalAxisPositionX(), 19UZ));
        expect(eq(crossing.getHorizontalAxisPositionY(), 6UZ));
        crossing.drawAxes();
        expect(eq(crossing._screen[6UZ][19UZ], std::string("┼")));
    };

    "linear axis transform"_test = [](const double valueCoordinate) {
        using namespace gr::graphs;
        constexpr std::size_t axisOffset = 5UZ;
        constexpr std::size_t axisWidth  = 65UZ;
        constexpr double      xMin       = 10.;
        constexpr double      xMax       = 100.;
        //
        auto identity = [](double value) {
            const auto screenCoordinate = LinearAxisTransform::toScreen(value, xMin, xMax, axisOffset, axisWidth);
            return LinearAxisTransform::fromScreen(screenCoordinate, xMin, xMax, axisOffset, axisWidth);
        };
        expect(approx(valueCoordinate, identity(valueCoordinate), 2.2)); // binning limited

        expect(eq(axisOffset, LinearAxisTransform::toScreen(xMin, xMin, xMax, axisOffset, axisWidth))) << "xMin does not correspond to min axis index";
        expect(eq(axisWidth - 1UZ, LinearAxisTransform::toScreen(xMax, xMin, xMax, axisOffset, axisWidth))) << "xMax does not correspond to max axis index";
    } | std::vector{10., 20., 42., 50., 100.};

    "log axis transform"_test = [](const double valueCoordinate) {
        using namespace gr::graphs;
        constexpr std::size_t axisOffset = 5UZ;
        constexpr std::size_t axisWidth  = 65UZ;
        constexpr double      xMin       = 10.;
        constexpr double      xMax       = 100.;
        //
        auto identity = [](double value) {
            const auto screenCoordinate = LogAxisTransform::toScreen(value, xMin, xMax, axisOffset, axisWidth);
            return LogAxisTransform::fromScreen(screenCoordinate, xMin, xMax, axisOffset, axisWidth);
        };
        expect(approx(valueCoordinate, identity(valueCoordinate), 2.2)); // binning limited

        expect(eq(axisOffset, LogAxisTransform::toScreen(xMin, xMin, xMax, axisOffset, axisWidth))) << "xMin does not correspond to min axis index";
        expect(eq(axisWidth - 1UZ, LogAxisTransform::toScreen(xMax, xMin, xMax, axisOffset, axisWidth))) << "xMax does not correspond to max axis index";

        expect(throws([] { std::ignore = LogAxisTransform::toScreen(0.0, 10., 100., 5UZ, 65UZ); }));
        expect(throws([] { std::ignore = LogAxisTransform::toScreen(1.0, 0., 100., 5UZ, 65UZ); }));
        expect(throws([] { std::ignore = LogAxisTransform::toScreen(1.0, 10., 0., 5UZ, 65UZ); }));
        expect(throws([] { std::ignore = LogAxisTransform::fromScreen(40, 0., 100., 5UZ, 65UZ); }));
        expect(throws([] { std::ignore = LogAxisTransform::fromScreen(40, 10., 0., 5UZ, 65UZ); }));
    } | std::vector{10., 20., 42., 50., 100.};

    "optimal tick position"_test = [](std::size_t axisWidth) {
        for (std::size_t minGapSize : {1UZ, 2UZ, 3UZ}) {
            auto tickPositions = gr::graphs::detail::optimalTickScreenPositions(axisWidth, minGapSize);

            // first and last tick index fulfill [0, axisWidth - 1].
            expect(!tickPositions.empty());
            expect(ge(tickPositions.size(), 2UZ));
            expect(eq(tickPositions.front(), 0UZ));
            expect(eq(tickPositions.back(), axisWidth - 1UZ));

            // validate even spacing and minimum gap, if there are more than two ticks.
            if (tickPositions.size() == 2UZ) {
                continue;
            }
            auto expectedGap = tickPositions[1UZ] - tickPositions[0UZ];
            for (size_t i = 1; i < tickPositions.size() - 1; ++i) {
                auto actualGap = tickPositions[i + 1] - tickPositions[i];
                expect(eq(actualGap, expectedGap)) << std::format("uneven spacing {} exp. {} found at axis width {}", actualGap, expectedGap, axisWidth);
                expect(ge(actualGap, minGapSize)) << std::format("gap size {} less than minimum required {} at axis width {}", actualGap, minGapSize, axisWidth);
            }
        }
    } | std::vector{10UZ, 20UZ, 42UZ, 50UZ, 100UZ};

    [[maybe_unused]] constexpr std::size_t sizeX = 120;
    [[maybe_unused]] constexpr std::size_t sizeY = 16;
    [[maybe_unused]] constexpr double      xMin  = 0.0;
    [[maybe_unused]] constexpr double      xMax  = 100.0;
    [[maybe_unused]] constexpr double      yMin  = -5.0;
    [[maybe_unused]] constexpr double      yMax  = +5.0;

    "basic chart - lines"_test = [&](bool defaultConstructor) {
        using namespace gr::graphs;

        // drawing sine, and cosine curves
        std::vector<double> xValues(2000);
        std::vector<double> sineYValues(xValues.size());
        std::vector<double> cosineYValues(xValues.size());

        for (std::size_t i = 0UZ; i < xValues.size(); ++i) {
            xValues[i]       = xMin + (xMax - xMin) * static_cast<double>(i) / static_cast<double>(xValues.size());
            sineYValues[i]   = 3.0 * std::sin(xValues[i] * 0.2);
            cosineYValues[i] = 3.0 * std::cos(xValues[i] * 0.2);
        }

        auto chart = defaultConstructor ? gr::graphs::ImChart<sizeX, sizeY>() : gr::graphs::ImChart<sizeX, sizeY>({{xMin, xMax}, {yMin, yMax}});

        expect(nothrow([&] { chart.draw(xValues, sineYValues, "sine-like"); }));
        expect(nothrow([&] { chart.draw(xValues, cosineYValues, "cosine-like"); }));

        expect(nothrow([&] { chart.draw(); }));
    } | std::vector{true, false};

    "basic chart - no data"_test = [&]() {
        using namespace gr::graphs;
        auto chart = gr::graphs::ImChart<sizeX, sizeY>();
        expect(nothrow([&] { chart.draw(std::vector<double>(), std::vector<double>(), "sine-like"); }));
    };

    "basic chart - bars"_test = [&]() {
        using namespace gr::graphs;

        // drawing gauss curves
        std::vector<double> xValues(2000);
        std::vector<double> gauss1(xValues.size());
        std::vector<double> gauss2(xValues.size());

        for (std::size_t i = 0UZ; i < xValues.size(); ++i) {
            xValues[i]              = xMin + (xMax - xMin) * static_cast<double>(i) / static_cast<double>(xValues.size());
            constexpr double mu1    = 55.0;
            constexpr double sigma1 = 5.0;
            gauss1[i]               = 3.0 * std::exp(-std::pow(xValues[i] - mu1, 2.) / (2.0 * std::pow(sigma1, 2.))); // Gaussian function
            constexpr double mu2    = 70.0;
            constexpr double sigma2 = 10.0;
            gauss2[i]               = 2.0 * std::exp(-std::pow(xValues[i] - mu2, 2.) / (2.0 * std::pow(sigma2, 2.)));
        }

        auto chart        = gr::graphs::ImChart<sizeX, sizeY>({{xMin, xMax}, {0.0, 5.0}});
        chart.draw_border = true;
        expect(nothrow([&] { chart.draw<Style::Bars>(xValues, gauss1, "gauss-like1"); }));
        expect(nothrow([&] { chart.draw<Style::Bars>(xValues, gauss2, "gauss-like2"); }));

        expect(nothrow([&] { chart.draw(); }));
    };

    "basic chart - drawing markers"_test = [&]() {
        using namespace gr::graphs;

        // drawing sine curve
        std::vector<double> xValues(2000);
        std::vector<double> sineYValues(xValues.size());

        for (std::size_t i = 0UZ; i < xValues.size(); ++i) {
            xValues[i]     = xMin + (xMax - xMin) * static_cast<double>(i) / static_cast<double>(xValues.size());
            sineYValues[i] = 3.0 * std::sin(xValues[i] * 0.2);
        }

        auto chart = gr::graphs::ImChart<sizeX, sizeY>({{xMin, xMax}, {yMin, yMax}});
        expect(nothrow([&] { chart.draw<Style::Marker>(xValues, sineYValues, "sine-like"); }));

        expect(nothrow([&] { chart.draw(); }));
    };

    "basic chart - axis top"_test = [&]() {
        using namespace gr::graphs;

        // drawing sine curve
        std::vector<double> xValues(2000);
        std::vector<double> sineYValues(xValues.size());

        for (std::size_t i = 0UZ; i < xValues.size(); ++i) {
            xValues[i]     = xMin + (xMax - xMin) * static_cast<double>(i) / static_cast<double>(xValues.size());
            sineYValues[i] = -5.0 + 3.0 * std::sin(xValues[i] * 0.2);
        }

        auto chart = gr::graphs::ImChart<sizeX, sizeY>({{xMin, xMax}, {-10, 0}});
        expect(nothrow([&] { chart.draw(xValues, sineYValues, "sine-like"); }));

        expect(nothrow([&] { chart.draw(); }));
    };

    "basic chart - axis top/centre"_test = [&]() {
        using namespace gr::graphs;

        // drawing sine curve
        std::vector<double> xValues(2000);
        std::vector<double> sineYValues(xValues.size());

        for (std::size_t i = 0UZ; i < xValues.size(); ++i) {
            xValues[i]     = -50. + 100. * static_cast<double>(i) / static_cast<double>(xValues.size());
            sineYValues[i] = -5.0 + 3.0 * std::sin(xValues[i] * 0.2);
        }

        auto chart = gr::graphs::ImChart<sizeX, sizeY>({{-50, +50}, {-10, 0}});
        expect(nothrow([&] { chart.draw(xValues, sineYValues, "sine-like"); }));

        expect(nothrow([&] { chart.draw(); }));
    };

    "basic chart - log axis"_test = []() {
        using namespace gr::graphs;
        constexpr std::size_t width = 120;

        auto sequence_range = [](auto start, auto end, auto step) {
            using namespace std::views; // for iota and transform
            return iota(1, (end - start) / step + 2) | transform([=](auto i) { return start + step * (i - 1); });
        };

        std::vector<double> xValues;
        for (auto subrange : {sequence_range(0.1, 0.9, 0.1), sequence_range(1.0, 9.0, 1.0), sequence_range(10.0, 90.0, 10.0), sequence_range(100.0, 1000.0, 100.0)}) {
            std::ranges::move(subrange, std::back_inserter(xValues));
        }
        std::vector<double> response1(xValues.size());
        std::vector<double> response2(xValues.size());

        for (std::size_t i = 0UZ; i < xValues.size(); ++i) {
            response1[i] = -5. + 20. * std::log10(1.0 / (1.0 + std::pow(2.0, xValues[i] / 10)));
            response2[i] = -10. + 20. * std::log10(1.0 / (1.0 + std::pow(4.0, xValues[i] / 200.)));
            if (xValues[i] > 5000) {
                response1[i] = -10.;
                response2[i] = -10.;
            }
        }

        auto chart = gr::graphs::ImChart<width, 16, LogAxisTransform>({{0.1, 1'000.0}, {-100, 0}});
        expect(nothrow([&] { chart.draw(xValues, response1, "low-pass1"); }));
        expect(nothrow([&] { chart.draw(xValues, response2, "low-pass2"); }));

        expect(nothrow([&] { chart.draw(); }));
    };

    "DataSet chart"_test = []() {
        using namespace gr::dataset;
        constexpr std::size_t kLength       = 1024UZ;
        constexpr float       kSamplingRate = 1000.f;
        constexpr float       kFrequency    = 5.f;
        constexpr float       kAmplitude    = 1.f;
        constexpr float       kOffset       = 0.2f;

        auto sinDataSet    = generate::waveform<float>(generate::WaveType::Sine, kLength, kSamplingRate, kFrequency, kAmplitude, +kOffset);
        auto cosDataSet    = generate::waveform<float>(generate::WaveType::Cosine, kLength, kSamplingRate, kFrequency, kAmplitude, -kOffset);
        auto mergedDataSet = merge(sinDataSet, cosDataSet);

        expect(eq(sinDataSet.signal_values.size(), kLength));
        expect(eq(cosDataSet.signal_values.size(), kLength));
        expect(eq(mergedDataSet.signal_values.size(), 2UZ * kLength));

        draw(sinDataSet);
        draw(cosDataSet);
        draw(mergedDataSet);
    };
};

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
