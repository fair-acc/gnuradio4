#ifndef GNURADIO_TESTING_TEST_PEAKSPECTRUMCHART_HPP
#define GNURADIO_TESTING_TEST_PEAKSPECTRUMCHART_HPP

// Shared ImChart visual-debugging helper for the generator qa suites in this directory.
// Linear axes only: ImChart's LogAxisTransform throws on non-positive bounds. Degenerate spectra
// are guarded so that drawing a chart can never affect pass/fail.

#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/algorithm/ImChart.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gr::testing::qa {

[[nodiscard]] inline bool verboseCharts() { return std::getenv("GR_QA_VERBOSE") != nullptr; }

// idx_pmt_map's value type does not depend on T, so one float instantiation covers every generator
using IdxPmtMap = gr::DataSet<float>::idx_pmt_map;

struct PeakMarkerSet {
    std::string_view        label;
    std::vector<double>     centres;
    gr::graphs::Color::Type color;
};

[[nodiscard]] inline PeakMarkerSet designedPeakMarkers(const std::vector<IdxPmtMap>& events, std::string_view label = "designed (ground truth)", gr::graphs::Color::Type color = gr::graphs::Color::Type::LightGreen) {
    PeakMarkerSet markers{.label = label, .centres = {}, .color = color};
    markers.centres.reserve(events.size());
    for (const auto& [idx, props] : events) {
        markers.centres.push_back(static_cast<double>(props.value_or<float>("centre", 0.f)));
    }
    return markers;
}

template<typename T>
inline void printSpectrumChart(std::string_view title, std::span<const T> spectrum, std::span<const PeakMarkerSet> markerSets = {}) {
    const std::size_t n = spectrum.size();
    if (n == 0) {
        return;
    }

    auto sampleAt = [&](double centre) -> double {
        const auto bin = static_cast<std::size_t>(std::clamp(std::lround(centre), 0L, static_cast<long>(n - 1)));
        return static_cast<double>(spectrum[bin]);
    };

    std::vector<double> xAxis(n), ySpectrum(n);
    for (std::size_t i = 0; i < n; ++i) {
        xAxis[i]     = static_cast<double>(i);
        ySpectrum[i] = static_cast<double>(spectrum[i]);
    }

    std::println("\n=== {} ===", title);
    gr::graphs::ImChart<112, 32> chart;
    chart.axis_name_x = "frequency bin []";
    chart.axis_name_y = "magnitude [a.u.]";

    chart._lastColor = gr::graphs::Color::Type::Blue;
    chart.draw<gr::graphs::Style::Braille>(xAxis, ySpectrum, "spectrum");

    for (const auto& markers : markerSets) {
        if (markers.centres.empty()) {
            continue;
        }
        std::vector<double> y;
        y.reserve(markers.centres.size());
        for (double centre : markers.centres) {
            y.push_back(sampleAt(centre));
        }
        chart._lastColor = markers.color;
        chart.draw<gr::graphs::Style::Marker>(markers.centres, y, markers.label);
    }

    chart.draw();
}

} // namespace gr::testing::qa

#endif // GNURADIO_TESTING_TEST_PEAKSPECTRUMCHART_HPP
