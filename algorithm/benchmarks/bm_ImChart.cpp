#include "benchmark.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include <gnuradio-4.0/algorithm/ImChart.hpp>

const boost::ut::suite<"ImChart"> chartBenchmarks = [] {
    using namespace benchmark;
    using namespace boost::ut;
#ifdef _WIN32
    constexpr const char* kNullDevice = "NUL";
#else
    constexpr const char* kNullDevice = "/dev/null";
#endif
    std::FILE* const output = std::fopen(kNullDevice, "w");
    expect(fatal(output != nullptr));
    const gr::on_scope_exit closeOutput([output] { std::fclose(output); });
    constexpr std::size_t   kRows    = 19UZ;
    constexpr std::size_t   kColumns = 73UZ;
    std::vector<float>      values(kRows * kColumns);
    for (std::size_t row = 0UZ; row < kRows; ++row) {
        for (std::size_t column = 0UZ; column < kColumns; ++column) {
            values[row * kColumns + column] = static_cast<float>(std::sin(static_cast<double>(row)) * std::cos(static_cast<double>(column)));
        }
    }
    constexpr std::array kMarkers{std::array{-90.0, 30.0}, std::array{30.0, 40.0}, std::array{135.0, 60.0}};
    for (const auto style : {gr::graphs::HeatmapStyle::braille, gr::graphs::HeatmapStyle::monochrome}) {
        gr::graphs::ImChart<100UZ, 14UZ> chart({{-180.0, 180.0}, {0.0, 90.0}});
        chart.axis_name_x = "azimuth [deg]";
        chart.axis_name_y = "elevation [deg]";
        expect(chart.drawHeatmap(std::span<const float>(values), kRows, kColumns, {-1.0, 1.0}, kMarkers, style, output).has_value());
        const auto name                                                                      = gr::meta::enumName(style).value();
        ::benchmark::benchmark<1'000>(std::format("{} reused 100x14 heatmap to null", name)) = [&] {
            const auto result = chart.drawHeatmap(std::span<const float>(values), kRows, kColumns, {-1.0, 1.0}, kMarkers, style, output);
            fake_read(result);
        };
        ::benchmark::benchmark<1'000>(std::format("{} fresh 100x14 heatmap to null", name)) = [&] {
            gr::graphs::ImChart<100UZ, 14UZ> fresh({{-180.0, 180.0}, {0.0, 90.0}});
            fresh.axis_name_x = "azimuth [deg]";
            fresh.axis_name_y = "elevation [deg]";
            const auto result = fresh.drawHeatmap(std::span<const float>(values), kRows, kColumns, {-1.0, 1.0}, kMarkers, style, output);
            fake_read(result);
        };
        ::benchmark::benchmark<1'000>(std::format("{} prepared 100x14 screen to null", name)) = [&] { chart.printScreen(output); };
    }
};

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
