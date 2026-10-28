#include <boost/ut.hpp>

#include <array>
#include <cmath>
#include <numbers>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/math/TensorPeakFinder.hpp>
#include <gnuradio-4.0/test/DeviceTestHelper.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TensorSource.hpp>

namespace {
using namespace boost::ut;
using gr::blocks::math::TensorPeakFinder;
using gr::testing::TensorCollector;
using gr::testing::TensorSource;

template<typename T>
using Frame = typename TensorSource<T>::Frame;

struct Found {
    std::vector<std::vector<double>>                      rows;
    std::vector<std::vector<std::size_t>>                 shapes;
    std::vector<std::pair<std::size_t, gr::property_map>> tags;
    bool                                                  ran = false;
};

template<typename T>
[[nodiscard]] Found findPeaksOn(auto& fixture, std::vector<Frame<T>> frames, gr::property_map settings, gr::property_map firstTag = {}, std::string_view domain = {}) {
    if (!domain.empty()) {
        settings.insert_or_assign("gr:compute_domain", std::string(domain) + "!");
    }
    auto& source     = fixture.template emplace<TensorSource<T>>();
    source._frames   = std::move(frames);
    source._firstTag = std::move(firstTag);
    auto& finder     = fixture.template emplace<TensorPeakFinder<T>>(std::move(settings));
    auto& sink       = fixture.template emplace<TensorCollector<T>>();
    expect(fixture.template connect<"out", "in">(source, finder).has_value());
    expect(fixture.template connect<"out", "in">(finder, sink).has_value());
    Found found{.rows = {}, .shapes = {}, .tags = {}, .ran = fixture.run().has_value()};
    for (const auto& elements : sink._elements) {
        found.rows.emplace_back(elements.begin(), elements.end());
    }
    found.shapes = sink._shapes;
    found.tags   = sink._tags;
    return found;
}

[[nodiscard]] Found findPeaks(std::vector<std::size_t> extents, std::vector<double> scores, gr::property_map settings, gr::property_map firstTag = {}) {
    gr::testing::GraphFixture<> fixture;
    return findPeaksOn<double>(fixture, {{.extents = std::move(extents), .elements = std::move(scores)}}, std::move(settings), std::move(firstTag));
}

void expectRows(const Found& found, std::size_t frame, const std::vector<std::vector<double>>& expected, std::string_view context, double tolerance = 1e-12) {
    expect(fatal(gt(found.rows.size(), frame))) << context << "frame" << frame << "arrives";
    const std::size_t width = found.shapes[frame][1];
    expect(eq(found.shapes[frame][0], expected.size())) << context << "frame" << frame << "peaks";
    for (std::size_t k = 0UZ; k < std::min(expected.size(), found.shapes[frame][0]); ++k) {
        expect(eq(width, expected[k].size())) << context;
        for (std::size_t c = 0UZ; c < std::min(width, expected[k].size()); ++c) {
            expect(le(std::abs(found.rows[frame][k * width + c] - expected[k][c]), tolerance)) << context << "frame" << frame << "row" << k << "column" << c;
        }
    }
}

[[nodiscard]] gr::property_map axesOf(std::vector<double> origin, std::vector<double> step, std::vector<bool> periodic = {}) {
    const std::string originKey{gr::tag::kAxisOrigin.key()};
    const std::string stepKey{gr::tag::kAxisStep.key()};
    gr::property_map  axes{{originKey, gr::Tensor<double>(gr::data_from, origin)}, {stepKey, gr::Tensor<double>(gr::data_from, step)}};
    if (!periodic.empty()) {
        gr::Tensor<bool> flags(gr::extents_from, {periodic.size()});
        for (std::size_t d = 0UZ; d < periodic.size(); ++d) {
            flags[d] = periodic[d];
        }
        axes.insert_or_assign(std::string(gr::tag::kAxisPeriodic.key()), flags);
    }
    return axes;
}

// [El,Az] = [3,8], azimuth periodic: the peak 9 at (1,0) beside the seam; parabolas (1, 9, 2) along El and (7, 9, 5) along Az
const std::vector<double> kElAz{1, 1, 1, 1, 1, 1, 1, 1, 9, 5, 0, 0, 0, 0, 0, 7, 2, 2, 2, 2, 2, 2, 2, 2};
const std::vector<double> kElAzPeak{-0.5 + (1.0 + 1.0 / 30.0) * 0.5, std::numbers::pi - std::numbers::pi / 24.0, 9.0 + 1.0 / 120.0 + 1.0 / 12.0, 1.0};

[[nodiscard]] gr::property_map elAzAxes() { return axesOf({-0.5, -std::numbers::pi}, {0.5, std::numbers::pi / 4.0}, {false, true}); }

const auto peaks = [] {
    "two peaks of a line, by score, each placed by its parabola"_test = []<typename T>() {
        // at 3: (2, 5, 4) -> offset 0.25, score 5.125, the reference; at 7: (1, 3, 1) -> 3
        gr::testing::GraphFixture<> fixture;
        const Found                 found = findPeaksOn<T>(fixture, {{.extents = {10UZ}, .elements = {0, 0, 2, 5, 4, 0, 1, 3, 1, 0}}}, {{"max_peaks", gr::Size_t(2)}, {"threshold", 0.5f}});
        expect(found.ran);
        expectRows(found, 0UZ, {{3.25, 5.125, 1.0}, {7.0, 3.0, 3.0 / 5.125}}, "line", std::same_as<T, float> ? 1e-6 : 1e-12);
    } | std::tuple<float, double>{};

    "threshold, max_peaks and min_separation each drop the lower peak"_test = [] {
        for (const gr::property_map& settings : {gr::property_map{{"max_peaks", gr::Size_t(2)}, {"threshold", 0.7f}}, gr::property_map{{"max_peaks", gr::Size_t(1)}, {"threshold", 0.f}}, //
                 gr::property_map{{"max_peaks", gr::Size_t(2)}, {"threshold", 0.f}, {"min_separation", 4.f}}}) {
            const Found found = findPeaks({10UZ}, {0, 0, 2, 5, 4, 0, 1, 3, 1, 0}, settings);
            expect(found.ran);
            expectRows(found, 0UZ, {{3.25, 5.125, 1.0}}, "only the higher");
        }
    };

    "a plateau is no peak, a shoulder plateau gives way to the peak beside it"_test = [] {
        const Found flatTop = findPeaks({6UZ}, {0, 1, 4, 4, 1, 0}, {{"max_peaks", gr::Size_t(3)}, {"threshold", 0.f}});
        expect(flatTop.ran);
        expectRows(flatTop, 0UZ, {}, "flat top");
        // (4, 5, 0) at 3: offset -1/3, rise 1/3
        const Found shoulder = findPeaks({5UZ}, {0, 4, 4, 5, 0}, {{"max_peaks", gr::Size_t(3)}, {"threshold", 0.f}});
        expect(shoulder.ran);
        expectRows(shoulder, 0UZ, {{3.0 - 1.0 / 3.0, 5.0 + 1.0 / 3.0, 1.0}}, "shoulder");
    };

    "peaks are ranked by their refined score, not by their cell"_test = [] {
        // cell 1: (0, 5, 0) -> 5; cell 5: (4, 4.9, 0) -> offset 4 / (2 (4 - 9.8)), score 4.9 - offset, higher
        const double offset = 4.0 / (2.0 * (4.0 - 9.8));
        const Found  found  = findPeaks({7UZ}, {0, 5, 0, 0, 4, 4.9, 0}, {{"max_peaks", gr::Size_t(1)}, {"threshold", 0.f}});
        expect(found.ran);
        expectRows(found, 0UZ, {{5.0 + offset, 4.9 - offset, 1.0}}, "the refined 5.24 beats the cell 5");
    };

    "a periodic line wraps its peak across the seam; open, its ends are no peaks"_test = [] {
        const std::vector<double> scores{5, 3, 0, 0, 0, 0, 0, 4};
        const gr::property_map    settings{{"max_peaks", gr::Size_t(2)}, {"threshold", 0.f}};
        const Found               open = findPeaks({8UZ}, scores, settings);
        expect(open.ran);
        expectRows(open, 0UZ, {}, "open: ends");
        const Found ring = findPeaks({8UZ}, scores, settings, axesOf({-4.0}, {1.0}, {true}));
        expect(ring.ran);
        expectRows(ring, 0UZ, {{-4.0 - 1.0 / 6.0 + 8.0, 5.0 + 1.0 / 24.0, 1.0}}, "ring");
    };

    "an elevation-azimuth map's peak beside the azimuth seam, in radians"_test = [] {
        const Found found = findPeaks({3UZ, 8UZ}, kElAz, {{"max_peaks", gr::Size_t(4)}, {"threshold", 0.f}}, elAzAxes());
        expect(found.ran);
        expectRows(found, 0UZ, {kElAzPeak}, "[El,Az]");
    };

    "a map's peak in physical units, other keys forwarded with it, axis keys not"_test = [] {
        // [3,4]: the peak 9 at (1,2); along rows (0, 9, 3): offset 0.1, rise 0.075; along columns (1, 9, 5): offset 1/6, rise 1/6
        gr::property_map tag = axesOf({-1.0, 10.0}, {0.5, 2.0});
        tag.insert_or_assign("note", std::string("kept"));
        const Found  found = findPeaks({3UZ, 4UZ}, {0, 8, 0, 0, 0, 1, 9, 5, 0, 0, 3, 0}, {{"max_peaks", gr::Size_t(4)}, {"threshold", 0.f}}, tag);
        const double score = 9.0 + 0.075 + 1.0 / 6.0;
        expect(found.ran);
        expectRows(found, 0UZ, {{-1.0 + 1.1 * 0.5, 10.0 + (2.0 + 1.0 / 6.0) * 2.0, score, 1.0}}, "physical units");
        expect(fatal(eq(found.tags.size(), 1UZ)));
        expect(eq(found.tags[0].first, 0UZ)) << "with frame 0's peaks";
        expect(found.tags[0].second.contains("note")) << "other keys go with the peaks";
        const bool hasGridAxes = found.tags[0].second.contains(gr::tag::kAxisOrigin.key()) || found.tags[0].second.contains(gr::tag::kAxisStep.key());
        expect(!hasGridAxes) << "a list of peaks has no grid axes";
    };

    "a cube's centre beats its 26 neighbours"_test = [] {
        std::vector<double> scores(27UZ, 0.0);
        scores[13]        = 2.0;
        scores[1]         = 1.0;
        const Found found = findPeaks({3UZ, 3UZ, 3UZ}, scores, {{"max_peaks", gr::Size_t(3)}, {"threshold", 0.f}});
        expect(found.ran);
        expectRows(found, 0UZ, {{1.0, 1.0, 1.0, 2.0, 1.0}}, "the centre");
    };

    "flat, zero and negative maps give no peaks; a single cell is its own"_test = [] {
        const gr::property_map settings{{"max_peaks", gr::Size_t(2)}, {"threshold", 0.f}};
        expectRows(findPeaks({4UZ}, {1, 1, 1, 1}, settings), 0UZ, {}, "flat");
        expectRows(findPeaks({4UZ}, {0, 0, 0, 0}, settings), 0UZ, {}, "zero");
        expectRows(findPeaks({5UZ}, {-3, -2, -1, -2, -3}, settings), 0UZ, {}, "negative: no positive reference");
        expectRows(findPeaks({1UZ}, {2}, settings), 0UZ, {{0.0, 2.0, 1.0}}, "a single cell");
    };

    "axes hold for later frames of their rank; a frame of another rank is in cells"_test = [] {
        // [5]: the peak at 2, (2, 5, 1): offset -1/14, rise 1/56; at origin 10, step 2 -> 10 + (2 - 1/14) 2
        gr::testing::GraphFixture<> fixture;
        const Frame<double>         line{.extents = {5UZ}, .elements = {0, 2, 5, 1, 0}};
        const Frame<double>         square{.extents = {3UZ, 3UZ}, .elements = {0, 0, 0, 0, 1, 0, 0, 0, 0}};
        const Found                 found = findPeaksOn<double>(fixture, {line, line, square}, {{"threshold", 0.f}}, axesOf({10.0}, {2.0}));
        const std::vector<double>   inUnits{10.0 + (2.0 - 1.0 / 14.0) * 2.0, 5.0 + 1.0 / 56.0, 1.0};
        expect(found.ran);
        expectRows(found, 0UZ, {inUnits}, "frame 0, with the axes");
        expectRows(found, 1UZ, {inUnits}, "frame 1, axes kept");
        expectRows(found, 2UZ, {{1.0, 1.0, 1.0, 1.0}}, "frame 2, rank 2: cells");
    };

    "a zero step, an axis of the wrong length, and rank 4 stop the search"_test = [] {
        const std::vector<double> line{0, 2, 5, 1, 0};
        expect(!findPeaks({5UZ}, line, {}, axesOf({0.0}, {0.0})).ran) << "step 0";
        expect(!findPeaks({5UZ}, line, {}, axesOf({0.0, 0.0}, {1.0, 1.0})).ran) << "two entries for rank 1";
        expect(!findPeaks({2UZ, 2UZ, 2UZ, 2UZ}, std::vector<double>(16UZ, 1.0), {}).ran) << "rank 4";
    };
};

const auto domains = [] {
    using gr::testing::operator""_domain_test;
    "an elevation-azimuth map's peak on each served domain"_domain_test = [](auto& fixture, std::string_view domain) {
        const Found found = findPeaksOn<double>(fixture, {{.extents = {3UZ, 8UZ}, .elements = kElAz}}, {{"max_peaks", gr::Size_t(4)}, {"threshold", 0.f}}, elAzAxes(), domain);
        expect(found.ran) << domain;
        expectRows(found, 0UZ, {kElAzPeak}, domain);
    } | gr::testing::overGraphs(gr::testing::kAllDomains);
};
} // namespace

int main() {
    "TensorPeakFinder"_test                 = peaks;
    "TensorPeakFinder on every domain"_test = domains;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
