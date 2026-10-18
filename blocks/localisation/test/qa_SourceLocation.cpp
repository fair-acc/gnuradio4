#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/localisation/PairCrossSpectra.hpp>
#include <gnuradio-4.0/localisation/SourceLocation.hpp>
#include <gnuradio-4.0/math/TensorPeakFinder.hpp>
#include <gnuradio-4.0/test/DeviceTestHelper.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TensorSource.hpp>

namespace {
using namespace boost::ut;
using gr::algorithm::Field;
using gr::localisation::locate;
using gr::localisation::scanGridOf;
using gr::localisation::SourceFromDelays;
using gr::localisation::SteeredPowerMap;
using gr::testing::TensorCollector;
using gr::testing::TensorSource;

constexpr double kPi = std::numbers::pi;

// the pinned scene: three elements, waves at 1000 m/s sampled at 1 kHz (1 m = 1 sample); a far source at azimuth 0 reaches element 1,
// 3 m towards it, 3 samples early: pair lags (0,1) +3, (0,2) 0, (1,2) -3
[[nodiscard]] gr::Tensor<double> scenePositions() { return gr::Tensor<double>(std::array<std::size_t, 2>{3UZ, 3UZ}, std::vector<double>{0, 0, 0, 3, 0, 0, 0, 5, 0}); }
constexpr std::array<int, 3>     kSceneLags{3, 0, -3};

[[nodiscard]] gr::property_map freeGeometry(const gr::Tensor<double>& positions) { return {{"element_positions", positions}}; }

[[nodiscard]] gr::property_map sceneGeometry() {
    gr::property_map geometry = freeGeometry(scenePositions());
    geometry.insert_or_assign("propagation_speed", 1000.0);
    geometry.insert_or_assign("field", std::string("far"));
    return geometry;
}

[[nodiscard]] std::vector<float> triangles(std::span<const int> lags, std::size_t L) {
    std::vector<float> curves(lags.size() * L, 0.f);
    for (std::size_t p = 0UZ; p < lags.size(); ++p) {
        const int at = static_cast<int>(L / 2UZ) + lags[p];
        for (std::size_t l = 0UZ; l < L; ++l) {
            curves[p * L + l] = std::max(0.f, 1.f - static_cast<float>(std::abs(static_cast<int>(l) - at)));
        }
    }
    return curves;
}

const auto grids = [] {
    "a full azimuth turn excludes its end and wraps; elevation includes both ends"_test = [] {
        const auto grid = scanGridOf(Field::far, gr::Tensor<double>(gr::data_from, {-kPi / 2, -kPi}), gr::Tensor<double>(gr::data_from, {kPi / 2, kPi}), gr::Tensor<gr::Size_t>(gr::data_from, {gr::Size_t(7), gr::Size_t(8)}));
        expect(fatal(grid.has_value()));
        expect(le(std::abs(grid->grid.step[0] - kPi / 6.0), 1e-12) && !grid->grid.periodic[0]) << "elevation: 7 points over pi";
        expect(le(std::abs(grid->grid.step[1] - kPi / 4.0), 1e-12) && grid->grid.periodic[1]) << "azimuth: 8 points over a full turn";
        // cell 34 = (4, 2): elevation -pi/2 + 4 pi/6 = pi/6, azimuth -pi + 2 pi/4 = -pi/2
        const auto                  u = grid->pointOf(34UZ);
        const std::array<double, 3> expected{0.0, -std::sqrt(3.0) / 2.0, 0.5};
        for (std::size_t c = 0UZ; c < 3UZ; ++c) {
            expect(le(std::abs(u[c] - expected[c]), 1e-12)) << "component" << c;
        }
    };

    "a near grid [Z,Y,X] gives positions x, y, z"_test = [] {
        const auto grid = scanGridOf(Field::near, gr::Tensor<double>(gr::data_from, {0.0, -1.0, -2.0}), gr::Tensor<double>(gr::data_from, {1.0, 1.0, 2.0}), gr::Tensor<gr::Size_t>(gr::data_from, {gr::Size_t(2), gr::Size_t(3), gr::Size_t(5)}));
        expect(fatal(grid.has_value()));
        expect(grid->pointOf(29UZ) == std::array{2.0, 1.0, 1.0}) << "cell 29 = (1, 2, 4): x -2 + 4, y -1 + 2, z 0 + 1";
    };

    "a far grid of rank 3, a near one of rank 1, or no points is refused"_test = [] {
        const gr::Tensor<double>     three(gr::data_from, {0.0, 0.0, 0.0});
        const gr::Tensor<gr::Size_t> threeSteps(gr::data_from, {gr::Size_t(2), gr::Size_t(2), gr::Size_t(2)});
        expect(!scanGridOf(Field::far, three, three, threeSteps).has_value());
        expect(!scanGridOf(Field::near, gr::Tensor<double>(gr::data_from, {0.0}), gr::Tensor<double>(gr::data_from, {1.0}), gr::Tensor<gr::Size_t>(gr::data_from, {gr::Size_t(4)})).has_value());
        expect(!scanGridOf(Field::far, gr::Tensor<double>(gr::data_from, {0.0}), gr::Tensor<double>(gr::data_from, {1.0}), gr::Tensor<gr::Size_t>(gr::data_from, {gr::Size_t(0)})).has_value());
    };
};

void sceneMapOn(auto& fixture, std::string_view domain = {}) {
    auto& source              = fixture.template emplace<TensorSource<float>>();
    source._frames            = {{.extents = {3UZ, 17UZ}, .elements = triangles(kSceneLags, 17UZ)}};
    gr::property_map settings = sceneGeometry();
    settings.insert_or_assign("sample_rate", 1000.f);
    settings.insert_or_assign("grid_steps", gr::Tensor<gr::Size_t>(gr::data_from, {gr::Size_t(360)}));
    if (!domain.empty()) {
        settings.insert_or_assign("gr:compute_domain", std::string(domain) + "!");
    }
    auto& srp  = fixture.template emplace<SteeredPowerMap<float>>(settings);
    auto& sink = fixture.template emplace<TensorCollector<float>>();
    expect(fixture.template connect<"out", "in">(source, srp).has_value());
    expect(fixture.template connect<"out", "in">(srp, sink).has_value());
    const auto ran = fixture.run();
    expect(ran.has_value()) << domain << (ran.has_value() ? std::string{} : ran.error().message);
    expect(fatal(eq(sink._elements.size(), 1UZ))) << domain;
    const auto&       map  = sink._elements[0];
    const std::size_t peak = static_cast<std::size_t>(std::ranges::max_element(map) - map.begin());
    expect(sink._shapes[0] == std::vector<std::size_t>{360UZ}) << domain;
    expect(eq(peak, 180UZ)) << domain << "cell 180: azimuth -pi + 180 degrees = 0";
    expect(le(std::abs(map[peak] - 3.f), 1e-5f)) << domain << "three pairs, each 1 at its lag";
}

const auto maps = [] {
    "three pairs' curves sum to 3 at the source's azimuth, and the map names its axes"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       source = fixture.emplace<TensorSource<float>>();
        source._frames                     = {{.extents = {3UZ, 17UZ}, .elements = triangles(kSceneLags, 17UZ)}, {.extents = {3UZ, 17UZ}, .elements = triangles(kSceneLags, 17UZ)}};
        gr::property_map settings          = sceneGeometry();
        settings.insert_or_assign("sample_rate", 1000.f);
        settings.insert_or_assign("grid_steps", gr::Tensor<gr::Size_t>(gr::data_from, {gr::Size_t(360)}));
        auto& srp  = fixture.emplace<SteeredPowerMap<float>>(settings);
        auto& sink = fixture.emplace<TensorCollector<float>>();
        expect(fixture.connect<"out", "in">(source, srp).has_value());
        expect(fixture.connect<"out", "in">(srp, sink).has_value());
        expect(fixture.run().has_value());
        expect(fatal(eq(sink._elements.size(), 2UZ)));
        const auto axesTags = std::ranges::count_if(sink._tags, [](const auto& tag) { return tag.second.contains(gr::tag::AXIS_STEP.key()); });
        expect(eq(axesTags, 1)) << "the axes once, with the first map";
        const auto axes = std::ranges::find_if(sink._tags, [](const auto& tag) { return tag.second.contains(gr::tag::AXIS_STEP.key()); });
        expect(fatal(axes != sink._tags.end()));
        expect(eq(axes->first, 0UZ));
        const auto& map  = axes->second;
        const auto  step = map.value_or<gr::Tensor<double>>(gr::tag::AXIS_STEP.key(), gr::Tensor<double>{});
        const auto  wrap = map.value_or<gr::Tensor<bool>>(gr::tag::AXIS_PERIODIC.key(), gr::Tensor<bool>{});
        const auto  name = map.value_or<gr::Tensor<gr::pmt::Value>>(gr::tag::AXIS_NAME.key(), gr::Tensor<gr::pmt::Value>{});
        expect(step.size() == 1UZ && std::abs(step[0] - 2.0 * kPi / 360.0) < 1e-12);
        expect(wrap.size() == 1UZ && wrap[0]);
        expect(name.size() == 1UZ && name[0].value_or(std::string{}) == "azimuth");
    };

    "the scene's map on the host"_test = [] {
        gr::testing::GraphFixture<> fixture;
        sceneMapOn(fixture);
    };

    "a near map [Y,X] peaks at the point as far from all three elements"_test = [] {
        // elements at (0,0), (6,0), (0,8): (3,4) is 5 m from each, every pair lag 0; on a 0.5 m grid it is cell (8, 6) of [17,13]
        gr::testing::GraphFixture<> fixture;
        auto&                       source = fixture.emplace<TensorSource<float>>();
        source._frames                     = {{.extents = {3UZ, 17UZ}, .elements = triangles(std::array{0, 0, 0}, 17UZ)}};
        gr::property_map settings          = freeGeometry(gr::Tensor<double>(std::array<std::size_t, 2>{3UZ, 3UZ}, std::vector<double>{0, 0, 0, 6, 0, 0, 0, 8, 0}));
        settings.merge(gr::property_map{{"propagation_speed", 1000.0}, {"field", std::string("near")}, {"sample_rate", 1000.f}, {"grid_min", gr::Tensor<double>(gr::data_from, {0.0, 0.0})}, //
            {"grid_max", gr::Tensor<double>(gr::data_from, {8.0, 6.0})}, {"grid_steps", gr::Tensor<gr::Size_t>(gr::data_from, {gr::Size_t(17), gr::Size_t(13)})}});
        auto& srp  = fixture.emplace<SteeredPowerMap<float>>(settings);
        auto& sink = fixture.emplace<TensorCollector<float>>();
        expect(fixture.connect<"out", "in">(source, srp).has_value());
        expect(fixture.connect<"out", "in">(srp, sink).has_value());
        expect(fixture.run().has_value());
        expect(fatal(eq(sink._elements.size(), 1UZ)));
        expect(sink._shapes[0] == std::vector<std::size_t>{17UZ, 13UZ});
        const auto& map = sink._elements[0];
        expect(eq(static_cast<std::size_t>(std::ranges::max_element(map) - map.begin()), 8UZ * 13UZ + 6UZ));
        expect(le(std::abs(map[8UZ * 13UZ + 6UZ] - 3.f), 1e-5f));
    };

    "curves of a pair count other than the elements' stop the map"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       source = fixture.emplace<TensorSource<float>>();
        source._frames                     = {{.extents = {2UZ, 17UZ}, .elements = std::vector<float>(34UZ, 0.f)}};
        auto& srp                          = fixture.emplace<SteeredPowerMap<float>>(sceneGeometry());
        auto& sink                         = fixture.emplace<TensorCollector<float>>();
        expect(fixture.connect<"out", "in">(source, srp).has_value());
        expect(fixture.connect<"out", "in">(srp, sink).has_value());
        expect(!fixture.run().has_value());
        expect(sink._elements.empty());
    };

    "curves of even length or of the wrong rank, no elements, or an empty grid stop the map with a message"_test = [] {
        struct Misconfiguration {
            std::string              name;
            std::vector<std::size_t> extents;
            gr::property_map         settings;
            std::string              expectedMessage;
        };
        gr::property_map noElements = sceneGeometry();
        noElements.erase("element_positions");
        gr::property_map emptyGrid = sceneGeometry();
        emptyGrid.insert_or_assign("grid_steps", gr::Tensor<gr::Size_t>(gr::data_from, {gr::Size_t(0)}));
        const std::vector<Misconfiguration> cases{
            {"even length", {3UZ, 16UZ}, sceneGeometry(), "with L odd, at least 3"},
            {"wrong rank", {48UZ}, sceneGeometry(), "with L odd, at least 3"},
            {"no elements", {3UZ, 17UZ}, noElements, "element_positions have shape"},
            {"empty grid", {3UZ, 17UZ}, emptyGrid, "at least one point each"},
        };
        for (const auto& [name, extents, settings, expectedMessage] : cases) {
            gr::testing::GraphFixture<> fixture;
            auto&                       source = fixture.emplace<TensorSource<float>>();
            source._frames                     = {{.extents = extents, .elements = std::vector<float>(3UZ * 17UZ, 0.f)}};
            auto& srp                          = fixture.emplace<SteeredPowerMap<float>>(settings);
            auto& sink                         = fixture.emplace<TensorCollector<float>>();
            expect(fixture.connect<"out", "in">(source, srp).has_value());
            expect(fixture.connect<"out", "in">(srp, sink).has_value());
            gr::MsgPortIn fromScheduler;
            expect(fatal(fixture.scheduler().msgOut.connect(fromScheduler).has_value()));
            std::ignore = fixture.run();
            std::string       errors;
            const std::size_t available = fromScheduler.streamReader().available();
            if (available != 0UZ) {
                auto span = fromScheduler.streamReader().get(available);
                for (const gr::Message& message : span) {
                    if (!message.data.has_value()) {
                        errors += message.data.error().message + "\n";
                    }
                }
                std::ignore = span.consume(available);
            }
            expect(errors.contains(expectedMessage)) << name << ": " << errors;
            expect(sink._elements.empty()) << name << ": no map of a misconfigured block";
        }
    };
};

/** the pinned scene end to end: noise delayed by 0, -3 and 0 samples at 1 kHz -> GCC-PHAT -> SRP-PHAT -> TensorPeakFinder, and -> PairDelay -> SourceFromDelays */
const auto scene = [] {
    "GCC-PHAT, SRP-PHAT with TensorPeakFinder, and PairDelay with SourceFromDelays all place the source at azimuth 0"_test = [] {
        constexpr std::size_t kN    = 64UZ;
        const auto            noise = gr::testing::whiteNoise(kN, 9U);
        std::vector<float>    frame(3UZ * kN);
        constexpr std::array  kDelays{0, -3, 0};
        for (std::size_t m = 0UZ; m < 3UZ; ++m) {
            for (std::size_t n = 0UZ; n < kN; ++n) {
                frame[m * kN + n] = static_cast<float>(noise[(n + kN - static_cast<std::size_t>((kDelays[m] + static_cast<int>(kN)) % static_cast<int>(kN))) % kN]);
            }
        }
        gr::testing::GraphFixture<> fixture;
        auto&                       source = fixture.emplace<TensorSource<float>>();
        source._frames                     = {{.extents = {3UZ, kN}, .elements = frame}};
        source._sampleRate                 = 1000.f;
        auto&            gcc               = fixture.emplace<gr::localisation::PairCrossCorrelation<float>>({{"max_lag", gr::Size_t(8)}, {"eps", 0.f}});
        gr::property_map map               = sceneGeometry();
        map.insert_or_assign("grid_steps", gr::Tensor<gr::Size_t>(gr::data_from, {gr::Size_t(360)}));
        auto& srp       = fixture.emplace<SteeredPowerMap<float>>(map);
        auto& peaks     = fixture.emplace<gr::blocks::math::TensorPeakFinder<float>>({{"threshold", 0.5f}});
        auto& peakSink  = fixture.emplace<TensorCollector<float>>();
        auto& lagPeak   = fixture.emplace<gr::localisation::PairDelay<float>>();
        auto& localiser = fixture.emplace<gr::localisation::SourceFromDelays<float>>(sceneGeometry());
        auto& tdoaSink  = fixture.emplace<TensorCollector<float>>();
        expect(fixture.connect<"out", "in">(source, gcc).has_value());
        expect(fixture.connect<"out", "in">(gcc, srp).has_value());
        expect(fixture.connect<"out", "in">(srp, peaks).has_value());
        expect(fixture.connect<"out", "in">(peaks, peakSink).has_value());
        expect(fixture.connect<"out", "in">(gcc, lagPeak).has_value());
        expect(fixture.connect<"out", "in">(lagPeak, localiser).has_value());
        expect(fixture.connect<"out", "in">(localiser, tdoaSink).has_value());
        const auto ran = fixture.run();
        expect(ran.has_value()) << (ran.has_value() ? std::string{} : ran.error().message);

        expect(fatal(eq(peakSink._elements.size(), 1UZ)));
        expect(peakSink._shapes[0] == std::vector<std::size_t>{1UZ, 3UZ}) << "one peak: azimuth, score, confidence";
        expect(le(std::abs(peakSink._elements[0][0]), 1e-4f)) << "SRP-PHAT: azimuth 0, the map symmetric about it";
        expect(le(std::abs(peakSink._elements[0][1] - 3.f), 1e-3f)) << "three PHAT peaks of 1";

        expect(fatal(eq(tdoaSink._elements.size(), 1UZ)));
        const auto& row = tdoaSink._elements[0];
        expect(tdoaSink._shapes[0] == std::vector<std::size_t>{1UZ, 5UZ}) << "elevation, azimuth, residual, iterations, rank";
        expect(le(std::abs(row[0]), 1e-3f)) << "TDoA: elevation 0, the planar array's normal side at its edge";
        expect(le(std::abs(row[1]), 1e-4f)) << "azimuth 0";
        expect(le(row[2], 1e-6f)) << "integer delays: no residual";
        expect(eq(row[4], 2.f)) << "a planar array: rank 2";
    };
};

const auto domains = [] {
    using gr::testing::operator""_domain_test;
    "the scene's map on each served domain"_domain_test = [](auto& fixture, std::string_view domain) { sceneMapOn(fixture, domain); } | gr::testing::overGraphs(gr::testing::kAllDomains);
};

[[nodiscard]] gr::Tensor<double> positionsOf(std::vector<double> xyz) { return gr::Tensor<double>(std::array<std::size_t, 2>{xyz.size() / 3UZ, 3UZ}, xyz); }

/** the pairs' delays tau_i - tau_j, pairs (0,1), (0,2), ..., written out: far tau_m = -p_m . u / c, near tau_m = |q - p_m| / c */
[[nodiscard]] std::vector<double> pairDelays(const gr::Tensor<double>& positions, std::array<double, 3> point, Field field, double c) {
    const std::size_t   M = positions.extent(0UZ);
    std::vector<double> tau(M);
    for (std::size_t m = 0UZ; m < M; ++m) {
        const double dx = point[0] - positions[m, 0UZ];
        const double dy = point[1] - positions[m, 1UZ];
        const double dz = point[2] - positions[m, 2UZ];
        tau[m]          = field == Field::far ? -(positions[m, 0UZ] * point[0] + positions[m, 1UZ] * point[1] + positions[m, 2UZ] * point[2]) / c : std::sqrt(dx * dx + dy * dy + dz * dz) / c;
    }
    std::vector<double> pairs;
    for (std::size_t i = 0UZ; i < M; ++i) {
        for (std::size_t j = i + 1UZ; j < M; ++j) {
            pairs.push_back(tau[i] - tau[j]);
        }
    }
    return pairs;
}

[[nodiscard]] std::array<double, 3> unitOf(double azimuth, double elevation) { return {std::cos(elevation) * std::cos(azimuth), std::cos(elevation) * std::sin(azimuth), std::sin(elevation)}; }

void expectPoint(const std::array<double, 3>& found, const std::array<double, 3>& expected, double tolerance, std::string_view context) {
    for (std::size_t c = 0UZ; c < 3UZ; ++c) {
        expect(le(std::abs(found[c] - expected[c]), tolerance)) << context << "component" << c;
    }
}

const auto solutions = [] {
    "a 3-D array finds a far direction exactly, rank 3"_test = [] {
        const auto positions = positionsOf({0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1});
        const auto u         = unitOf(kPi / 6.0, kPi / 9.0);
        const auto located   = locate(positions, pairDelays(positions, u, Field::far, 343.0), 343.0, Field::far, 20UZ, 1e-9);
        expect(fatal(located.has_value()));
        expectPoint(located->point, u, 1e-9, "tetrahedron");
        expect(eq(located->rank, 3UZ));
        expect(le(located->rmsResidual, 1e-12));
    };

    "a planar array takes the side of +z: rank 2"_test = [] {
        const auto positions = positionsOf({0, 0, 0, 3, 0, 0, 0, 5, 0});
        const auto u         = unitOf(2.0 * kPi / 9.0, kPi / 9.0);
        const auto located   = locate(positions, pairDelays(positions, u, Field::far, 1000.0), 1000.0, Field::far, 20UZ, 1e-9);
        expect(fatal(located.has_value()));
        expectPoint(located->point, u, 1e-9, "above the plane");
        expect(eq(located->rank, 2UZ));
    };

    "a linear array knows the cone angle and takes the cone's +y side: rank 1"_test = [] {
        const auto positions = positionsOf({0, 0, 0, 1, 0, 0, 2.5, 0, 0});
        const auto u         = unitOf(kPi / 3.0, 0.0);
        const auto located   = locate(positions, pairDelays(positions, u, Field::far, 343.0), 343.0, Field::far, 20UZ, 1e-9);
        expect(fatal(located.has_value()));
        expectPoint(located->point, u, 1e-9, "60 degrees from the axis, in the x-y plane");
        expect(eq(located->rank, 1UZ));
    };

    "a vertical line knows the elevation and takes the cone's +y side, never NaN"_test = [] {
        const auto positions = positionsOf({0, 0, 0, 0, 0, 1, 0, 0, 2.5});
        const auto u         = unitOf(kPi / 6.0, 2.0 * kPi / 9.0);
        const auto located   = locate(positions, pairDelays(positions, u, Field::far, 343.0), 343.0, Field::far, 20UZ, 1e-9);
        expect(fatal(located.has_value()));
        expectPoint(located->point, {0.0, std::cos(2.0 * kPi / 9.0), std::sin(2.0 * kPi / 9.0)}, 1e-9, "elevation 40 degrees, towards +y");
        expect(eq(located->rank, 1UZ));
    };

    "Gauss-Newton from beside a planar array finds a near source above it"_test = [] {
        const auto                  positions = positionsOf({0, 0, 0, 3, 0, 0, 0, 5, 0, 3, 5, 0});
        const std::array<double, 3> source{1.0, 2.0, 1.5};
        const auto                  located = locate(positions, pairDelays(positions, source, Field::near, 343.0), 343.0, Field::near, 50UZ, 1e-9);
        expect(fatal(located.has_value()));
        expectPoint(located->point, source, 1e-6, "above the plane");
        expect(le(located->rmsResidual, 1e-12));
    };

    "Gauss-Newton from the centroid finds a near source"_test = [] {
        const auto                  positions = positionsOf({0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 2, 2, 2});
        const std::array<double, 3> source{1.5, 0.7, 0.4};
        const auto                  located = locate(positions, pairDelays(positions, source, Field::near, 343.0), 343.0, Field::near, 20UZ, 1e-9);
        expect(fatal(located.has_value()));
        expectPoint(located->point, source, 1e-6, "near source");
        expect(le(located->rmsResidual, 1e-12));
        expect(le(located->iterations, 20UZ));
        expect(eq(located->rank, 3UZ));
    };

    "a delay or a position that is NaN is refused with a message, not located"_test = [] {
        const double nan       = std::numeric_limits<double>::quiet_NaN();
        const auto   positions = positionsOf({0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1});
        auto         tau       = pairDelays(positions, {0.3, 0.2, 0.1}, Field::near, 343.0);
        tau[0]                 = nan;
        const auto delay       = locate(positions, tau, 343.0, Field::near, 20UZ, 1e-9);
        expect(!delay.has_value() && delay.error().contains("diverged after 1 steps")) << (delay ? "located" : delay.error());
        for (const Field field : {Field::far, Field::near}) {
            const auto broken = locate(positionsOf({0, 0, 0, nan, 0, 0, 0, 1, 0, 0, 0, 1}), tau, 343.0, field, 20UZ, 1e-9);
            expect(!broken.has_value()) << gr::meta::enumName(field).value_or("");
        }
    };
};

const auto block = [] {
    "a frame of pair delays gives one row: elevation, azimuth, residual, iterations, rank"_test = [] {
        const auto          positions = positionsOf({0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1});
        const auto          tau       = pairDelays(positions, unitOf(-kPi / 4.0, kPi / 12.0), Field::far, 343.0);
        std::vector<double> frame;
        for (const double delay : tau) {
            frame.insert(frame.end(), {delay, 1.0, 10.0});
        }
        gr::testing::GraphFixture<> fixture;
        auto&                       source = fixture.emplace<TensorSource<double>>();
        source._frames                     = {{.extents = {tau.size(), 3UZ}, .elements = frame}};
        gr::property_map settings          = freeGeometry(positions);
        settings.insert_or_assign("propagation_speed", 343.0);
        auto& localiser = fixture.emplace<SourceFromDelays<double>>(settings);
        auto& sink      = fixture.emplace<TensorCollector<double>>();
        expect(fixture.connect<"out", "in">(source, localiser).has_value());
        expect(fixture.connect<"out", "in">(localiser, sink).has_value());
        expect(fixture.run().has_value());
        expect(fatal(eq(sink._elements.size(), 1UZ)));
        expect(sink._shapes[0] == std::vector<std::size_t>{1UZ, 5UZ});
        const auto& row = sink._elements[0];
        expect(le(std::abs(row[0] - kPi / 12.0), 1e-9)) << "elevation";
        expect(le(std::abs(row[1] + kPi / 4.0), 1e-9)) << "azimuth";
        expect(le(row[2], 1e-12)) << "residual";
        expect(eq(row[3], 0.0)) << "closed form: no iterations";
        expect(eq(row[4], 3.0)) << "rank";
    };

    "a near row gives z, y, x in map order"_test = [] {
        const auto                  positions = positionsOf({0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 2, 2, 2});
        const std::array<double, 3> source{1.5, 0.7, 0.4};
        const auto                  tau = pairDelays(positions, source, Field::near, 343.0);
        gr::testing::GraphFixture<> fixture;
        auto&                       delays = fixture.emplace<TensorSource<double>>();
        delays._frames                     = {{.extents = {tau.size(), 1UZ}, .elements = tau}};
        gr::property_map settings          = freeGeometry(positions);
        settings.merge(gr::property_map{{"propagation_speed", 343.0}, {"field", std::string("near")}});
        auto& localiser = fixture.emplace<SourceFromDelays<double>>(settings);
        auto& sink      = fixture.emplace<TensorCollector<double>>();
        expect(fixture.connect<"out", "in">(delays, localiser).has_value());
        expect(fixture.connect<"out", "in">(localiser, sink).has_value());
        expect(fixture.run().has_value());
        expect(fatal(eq(sink._elements.size(), 1UZ)));
        expect(sink._shapes[0] == std::vector<std::size_t>{1UZ, 6UZ});
        expectPoint({sink._elements[0][2], sink._elements[0][1], sink._elements[0][0]}, source, 1e-6, "z, y, x");
    };

    "delays for another number of pairs, or no propagation speed, stop the localiser"_test = [] {
        for (const double speed : {343.0, 0.0}) {
            gr::testing::GraphFixture<> fixture;
            auto&                       source = fixture.emplace<TensorSource<double>>();
            source._frames                     = {{.extents = {speed > 0.0 ? 2UZ : 6UZ, 1UZ}, .elements = std::vector<double>(speed > 0.0 ? 2UZ : 6UZ, 0.0)}};
            gr::property_map settings          = freeGeometry(positionsOf({0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1}));
            settings.insert_or_assign("propagation_speed", speed);
            auto& localiser = fixture.emplace<SourceFromDelays<double>>(settings);
            auto& sink      = fixture.emplace<TensorCollector<double>>();
            expect(fixture.connect<"out", "in">(source, localiser).has_value());
            expect(fixture.connect<"out", "in">(localiser, sink).has_value());
            expect(!fixture.run().has_value()) << "speed" << speed;
            expect(sink._elements.empty());
        }
    };
};
/** SourceFromDelays placed by `settings`, given the pair delays of a far source in direction `u` at 343 m/s: its row {elevation, azimuth, residual, iterations, rank} */
[[nodiscard]] std::vector<double> locatedBy(gr::property_map settings, const std::array<double, 3>& u) {
    constexpr double kSpeed = 343.0;
    settings.insert_or_assign("propagation_speed", kSpeed);
    gr::testing::GraphFixture<> fixture;
    auto&                       delays    = fixture.emplace<TensorSource<double>>();
    auto&                       localiser = fixture.emplace<SourceFromDelays<double>>(settings);
    auto&                       sink      = fixture.emplace<TensorCollector<double>>();
    const auto                  tau       = pairDelays(localiser.element_positions.value, u, Field::far, kSpeed);
    delays._frames                        = {{.extents = {tau.size(), 1UZ}, .elements = tau}};
    expect(fixture.connect<"out", "in">(delays, localiser).has_value());
    expect(fixture.connect<"out", "in">(localiser, sink).has_value());
    expect(fixture.run().has_value());
    expect(fatal(eq(sink._elements.size(), 1UZ)));
    return sink._elements[0];
}

void expectDirection(const std::vector<double>& row, double elevation, double azimuth, double rank, std::string_view layout) {
    expect(le(std::abs(row[0] - elevation), 1e-9)) << layout << "elevation" << row[0];
    expect(le(std::abs(row[1] - azimuth), 1e-9)) << layout << "azimuth" << row[1];
    expect(eq(row[4], rank)) << layout << "rank";
}

const auto layouts = [] {
    "a line of four along x knows the cone angle about x: a source at azimuth 60 degrees in the x-y plane, rank 1"_test = [] {
        const auto row = locatedBy(freeGeometry(gr::algorithm::geometry::line(4UZ, 1.0)), unitOf(kPi / 3.0, 0.0));
        expectDirection(row, 0.0, kPi / 3.0, 1.0, "line");
    };

    "the 4x4 grid of 0.042 m finds a source in front of it and mirrors one behind it to the front: rank 2"_test = [] {
        const gr::property_map grid = freeGeometry(gr::algorithm::geometry::grid(4UZ, 4UZ));
        expectDirection(locatedBy(grid, unitOf(kPi / 6.0, 5.0 * kPi / 18.0)), 5.0 * kPi / 18.0, kPi / 6.0, 2.0, "grid, in front");
        expectDirection(locatedBy(grid, unitOf(kPi / 6.0, -5.0 * kPi / 18.0)), 5.0 * kPi / 18.0, kPi / 6.0, 2.0, "grid, behind");
    };

    "the tetrahedron of 0.3 m resolves a source below the horizon: rank 3"_test = [] {
        const auto row = locatedBy(freeGeometry(gr::algorithm::geometry::tetrahedron()), unitOf(3.0 * kPi / 4.0, -2.0 * kPi / 9.0));
        expectDirection(row, -2.0 * kPi / 9.0, 3.0 * kPi / 4.0, 3.0, "tetrahedron");
    };

    "the 4x4 grid's [El,Az] map over its front hemisphere peaks at the source, elevation 60 degrees, azimuth 0"_test = [] {
        /** 0.042 m at 343 m/s sampled at 2 c / d: towards the source pair (i, j) lags by column(j) - column(i) samples */
        constexpr std::size_t kColumns = 4UZ;
        constexpr std::size_t kM       = 16UZ;
        std::vector<int>      lags;
        for (std::size_t i = 0UZ; i < kM; ++i) {
            for (std::size_t j = i + 1UZ; j < kM; ++j) {
                lags.push_back(static_cast<int>(j % kColumns) - static_cast<int>(i % kColumns));
            }
        }
        gr::testing::GraphFixture<> fixture;
        auto&                       source = fixture.emplace<TensorSource<float>>();
        source._frames                     = {{.extents = {lags.size(), 17UZ}, .elements = triangles(lags, 17UZ)}};
        auto& srp                          = fixture.emplace<SteeredPowerMap<float>>({{"element_positions", gr::algorithm::geometry::grid(4UZ, 4UZ)}, {"propagation_speed", 343.0}, {"sample_rate", static_cast<float>(2.0 * 343.0 / 0.042)}, //
                                     {"grid_min", gr::Tensor<double>(gr::data_from, {0.0, -kPi})}, {"grid_max", gr::Tensor<double>(gr::data_from, {kPi / 2.0, kPi})}, {"grid_steps", gr::Tensor<gr::Size_t>(gr::data_from, {gr::Size_t(7), gr::Size_t(24)})}});
        auto& sink                         = fixture.emplace<TensorCollector<float>>();
        expect(fixture.connect<"out", "in">(source, srp).has_value());
        expect(fixture.connect<"out", "in">(srp, sink).has_value());
        expect(fixture.run().has_value());
        expect(fatal(eq(sink._elements.size(), 1UZ)));
        expect(sink._shapes[0] == std::vector<std::size_t>{7UZ, 24UZ});
        const auto& map = sink._elements[0];
        expect(eq(static_cast<std::size_t>(std::ranges::max_element(map) - map.begin()), 4UZ * 24UZ + 12UZ)) << "elevation 4 x 15 degrees, azimuth -180 + 12 x 15 degrees";
        expect(le(std::abs(map[4UZ * 24UZ + 12UZ] - 120.f), 1e-3f)) << "all 120 pairs, each 1 at its lag";
    };

    "a geometry change to another element count than the frames carried is refused, the previous geometry kept"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       source = fixture.emplace<TensorSource<float>>();
        source._frames                     = {{.extents = {3UZ, 17UZ}, .elements = triangles(kSceneLags, 17UZ)}};
        gr::property_map settings          = sceneGeometry();
        settings.insert_or_assign("sample_rate", 1000.f);
        auto& srp  = fixture.emplace<SteeredPowerMap<float>>(settings);
        auto& sink = fixture.emplace<TensorCollector<float>>();
        expect(fixture.connect<"out", "in">(source, srp).has_value());
        expect(fixture.connect<"out", "in">(srp, sink).has_value());
        expect(fixture.run().has_value());
        expect(fatal(eq(sink._elements.size(), 1UZ)));
        std::ignore = srp.settings().setStaged({{"element_positions", gr::algorithm::geometry::grid(2UZ, 3UZ)}});
        std::ignore = srp.settings().applyStagedParameters();
        expect(std::ranges::equal(srp.element_positions.value, scenePositions())) << "six elements for frames of three: refused, the scene kept";
        std::ignore = srp.settings().setStaged({{"element_positions", gr::algorithm::geometry::line(3UZ, 2.0)}});
        std::ignore = srp.settings().applyStagedParameters();
        expect(std::ranges::equal(srp.element_positions.value, gr::algorithm::geometry::line(3UZ, 2.0))) << "three elements: accepted";
        std::ignore = srp.settings().setStaged({{"element_positions", gr::Tensor<double>({3UZ, 2UZ})}});
        std::ignore = srp.settings().applyStagedParameters();
        expect(std::ranges::equal(srp.element_positions.value, gr::algorithm::geometry::line(3UZ, 2.0))) << "[3,2] is no geometry: refused, the line kept";

        SourceFromDelays<double> localiser(sceneGeometry());
        localiser.init(localiser.progress);
        localiser._inputElements = 3UZ;
        std::ignore              = localiser.settings().setStaged({{"element_positions", gr::algorithm::geometry::tetrahedron()}});
        std::ignore              = localiser.settings().applyStagedParameters();
        expect(std::ranges::equal(localiser.element_positions.value, scenePositions())) << "a tetrahedron of four for three: refused";
    };
};
} // namespace

int main() {
    "scan grids"_test                      = grids;
    "SteeredPowerMap"_test                 = maps;
    "the pinned scene"_test                = scene;
    "SteeredPowerMap on every domain"_test = domains;
    "locate"_test                          = solutions;
    "SourceFromDelays"_test                = block;
    "array layouts"_test                   = layouts;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
