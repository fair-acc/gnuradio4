#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <numbers>
#include <optional>
#include <print>
#include <ranges>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/algorithm/ArrayGeometry.hpp>
#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/basic/StreamToTensor.hpp>
#include <gnuradio-4.0/localisation/FrontProjection.hpp>
#include <gnuradio-4.0/localisation/PairCrossSpectra.hpp>
#include <gnuradio-4.0/localisation/SourceLocation.hpp>
#include <gnuradio-4.0/math/TensorPeakFinder.hpp>
#include <gnuradio-4.0/testing/ArrayScene.hpp>
#include <gnuradio-4.0/testing/ExampleRun.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace {

constexpr double      kPi           = std::numbers::pi;
constexpr double      kSampleRate   = gr::testing::kExampleSampleRateHz;
constexpr double      kSpeedOfSound = gr::testing::kSpeedOfSoundInAir;
constexpr std::size_t kChartColumns = 100UZ;
constexpr std::size_t kChartRows    = 14UZ;
constexpr auto        kRedrawPeriod = std::chrono::seconds(1);

struct KnownSource {
    double elevationDegrees;
    double azimuthDegrees;
    double gain;
    double distanceMetres = std::numeric_limits<double>::infinity();
};

[[nodiscard]] std::vector<float> sweepingNoise(double x, std::size_t nSamples, double sampleRate) {
    constexpr std::size_t kMargin = gr::testing::kMaxArrivalDelaySamples;
    const auto            noise   = gr::testing::whiteNoise(nSamples + 2UZ * kMargin, 21U);
    std::vector<float>    heard(nSamples);
    for (std::size_t n = 0UZ; n < nSamples; ++n) {
        const double azimuth  = kPi / 2.0 - gr::testing::sweepBearingRadians(n, nSamples);
        const double position = static_cast<double>(n + kMargin) + x * std::cos(azimuth) / kSpeedOfSound * sampleRate;
        heard[n]              = static_cast<float>(gr::testing::interpolateAt(noise, position));
    }
    return heard;
}

struct SteeredPowerProjections : gr::Block<SteeredPowerProjections> {
    using Description = gr::Doc<"Plots the horizontal and vertical projections of a steered-power map and lists its peaks against the known sources.">;

    gr::TensorPortIn<float> map;
    gr::TensorPortIn<float> peaks;
    gr::TensorPortIn<float> located;
    gr::SampleRate          sample_rate = 0.f;
    gr::FrameRate           frame_rate  = 0.f;
    GR_MAKE_REFLECTABLE(SteeredPowerProjections, map, peaks, located, sample_rate, frame_rate);

    std::string                           _inputDescription;
    std::vector<double>                   _azimuthDegrees;
    std::vector<double>                   _elevationDegrees;
    std::vector<KnownSource>              _truth;
    std::vector<float>                    _latestMap;
    std::vector<std::vector<float>>       _latestPeaks;
    std::vector<float>                    _latestLocated;
    std::size_t                           _framesSeen            = 0UZ;
    std::size_t                           _sweepSceneSamples     = 0UZ;
    double                                _requestedSampleRateHz = 0.0;
    std::chrono::steady_clock::time_point _printed{};

    void start() { gr::graphs::resetView(); }

    gr::work::Status processBulk(gr::InputSpanLike auto& maps, gr::InputSpanLike auto& found, gr::InputSpanLike auto& directions) {
        const std::size_t nMaps = std::min(maps.size(), found.size());
        if (nMaps > 0UZ) {
            const auto& latest = maps[nMaps - 1UZ];
            _latestMap.assign(latest.data(), latest.data() + latest.size());
            const auto& rows = found[nMaps - 1UZ];
            _latestPeaks.clear();
            for (std::size_t k = 0UZ; rows.rank() == 2UZ && k < rows.extent(0UZ); ++k) {
                _latestPeaks.emplace_back(rows.data() + k * rows.extent(1UZ), rows.data() + (k + 1UZ) * rows.extent(1UZ));
            }
        }
        if (directions.size() > 0UZ) {
            const auto& row = directions[directions.size() - 1UZ];
            _latestLocated.assign(row.data(), row.data() + row.size());
        }
        _framesSeen += nMaps;
        std::ignore              = maps.consume(nMaps);
        std::ignore              = found.consume(nMaps);
        std::ignore              = directions.consume(directions.size());
        const auto redrawStarted = std::chrono::steady_clock::now();
        if (redrawStarted - _printed >= kRedrawPeriod && !_latestMap.empty()) {
            _printed = redrawStarted;
            print();
        }
        return gr::work::Status::OK;
    }

    void print() {
        const bool        camera        = !_elevationDegrees.empty();
        const std::size_t nAzimuth      = _azimuthDegrees.size();
        const std::size_t nElevation    = camera ? _elevationDegrees.size() : 1UZ;
        const auto        displayAngles = [camera](double azimuthRadians, double elevationRadians) {
            const auto direction = camera ? gr::algorithm::directionOf(azimuthRadians, elevationRadians) : std::array{std::cos(azimuthRadians), 0.0, std::sin(azimuthRadians)};
            return gr::localisation::frontAnglesDegrees(direction);
        };
        const auto frontMap = [&] {
            if (camera) {
                return gr::localisation::projectFrontPower(_latestMap, nAzimuth, nElevation);
            }
            gr::localisation::FrontPowerMap result;
            result.power         = _latestMap | std::views::reverse | std::ranges::to<std::vector>();
            result.anglesDegrees = _azimuthDegrees | std::views::reverse | std::views::transform([](double azimuth) { return gr::localisation::kFrontAngleLimitDegrees - azimuth; }) | std::ranges::to<std::vector>();
            return result;
        }();
        const std::size_t frontColumns = frontMap.anglesDegrees.size();
        const std::size_t frontRows    = camera ? frontColumns : 1UZ;

        std::vector<double> horizontalPower(frontColumns, -std::numeric_limits<double>::infinity());
        std::vector<double> verticalPower(frontRows, -std::numeric_limits<double>::infinity());
        for (std::size_t row = 0UZ; row < frontRows; ++row) {
            for (std::size_t column = 0UZ; column < frontColumns; ++column) {
                const double power = static_cast<double>(frontMap.power[row * frontColumns + column]);
                if (std::isfinite(power)) {
                    horizontalPower[column] = std::max(horizontalPower[column], power);
                    verticalPower[row]      = std::max(verticalPower[row], power);
                }
            }
        }
        const double                                   bottom = static_cast<double>(*std::ranges::min_element(_latestMap));
        const double                                   top    = static_cast<double>(std::max(*std::ranges::max_element(_latestMap), std::numeric_limits<float>::min()));
        gr::graphs::ImChart<kChartColumns, kChartRows> chart({{-gr::localisation::kFrontAngleLimitDegrees, gr::localisation::kFrontAngleLimitDegrees}, {std::min(0.0, bottom), 1.05 * top}});
        chart.axis_name_x         = "angle [deg]";
        chart.axis_name_y         = "steered power";
        const auto drawProjection = [&](const std::vector<double>& power, const char* label) {
            std::vector<double> angles;
            std::vector<double> scores;
            for (std::size_t i = 0UZ; i < power.size(); ++i) {
                if (std::isfinite(power[i])) {
                    angles.push_back(frontMap.anglesDegrees[i]);
                    scores.push_back(power[i]);
                }
            }
            chart.draw(angles, scores, label);
        };
        drawProjection(horizontalPower, "horizontal");
        if (camera) {
            drawProjection(verticalPower, "vertical");
        }
        chart.reset();
        chart.draw(std::nullopt);

        std::println("input {:<60.60} {:6.0f} Hz received, {:.0f} Hz requested\033[K", _inputDescription, static_cast<double>(sample_rate), _requestedSampleRateHz);
        const auto truth = [&] {
            if (_sweepSceneSamples == 0UZ) {
                return _truth;
            }
            const double latestFrameCentreSample = static_cast<double>(_framesSeen) * static_cast<double>(sample_rate) / static_cast<double>(frame_rate);
            const double bearing                 = gr::testing::sweepBearingRadians(static_cast<std::size_t>(latestFrameCentreSample), _sweepSceneSamples);
            return std::vector<KnownSource>{{0.0, gr::algorithm::degreesOf(kPi / 2.0 - bearing), 1.0}};
        }();
        const auto errorColumn = [&](double azimuthRadians, double elevationRadians) {
            if (truth.empty()) {
                return std::string{};
            }
            const auto estimate = gr::algorithm::directionOf(azimuthRadians, elevationRadians);
            double     nearest  = std::numeric_limits<double>::infinity();
            for (const auto& source : truth) {
                nearest = std::min(nearest, gr::localisation::angularSeparationDegrees(estimate, gr::algorithm::directionOf(gr::algorithm::radiansOf(source.azimuthDegrees), gr::algorithm::radiansOf(source.elevationDegrees))));
            }
            return std::format(" | err {:5.1f} deg", nearest);
        };
        for (std::size_t k = 0UZ; k < _latestPeaks.size(); ++k) {
            const auto&       peak          = _latestPeaks[k];
            const double      peakAzimuth   = static_cast<double>(peak[camera ? 1UZ : 0UZ]);
            const double      peakElevation = camera ? static_cast<double>(peak[0UZ]) : 0.0;
            const auto        angles        = displayAngles(peakAzimuth, peakElevation);
            const std::string error         = errorColumn(peakAzimuth, peakElevation);
            std::println("peak {:1} H {:+6.1f} V {:+6.1f} deg | power {:7.2f}{}\033[K", k, angles[0UZ], angles[1UZ], static_cast<double>(peak[camera ? 2UZ : 1UZ]), error);
        }
        if (_latestLocated.size() >= 5UZ) {
            const double elevation = static_cast<double>(_latestLocated[0UZ]);
            const double azimuth   = camera ? static_cast<double>(_latestLocated[1UZ]) : std::abs(static_cast<double>(_latestLocated[1UZ]));
            const auto   angles    = displayAngles(azimuth, elevation);
            std::println("delay   H {:+6.1f} V {:+6.1f} deg | residual {:6.3f} ms | rank {:1}{}\033[K", angles[0UZ], angles[1UZ], 1000.0 * static_cast<double>(_latestLocated[2UZ]), static_cast<int>(_latestLocated[4UZ]), errorColumn(azimuth, elevation));
        }
        for (std::size_t k = 0UZ; k < truth.size(); ++k) {
            const auto angles = displayAngles(gr::algorithm::radiansOf(truth[k].azimuthDegrees), gr::algorithm::radiansOf(truth[k].elevationDegrees));
            std::println("truth {:1} H {:+6.1f} V {:+6.1f} deg | r {:6.2f} m | gain {:4.2f}\033[K", k, angles[0UZ], angles[1UZ], truth[k].distanceMetres, truth[k].gain);
        }
        std::print("\033[J");
    }
};

} // namespace

int main(int argc, char** argv) {
    std::string                   device;
    bool                          synthetic  = false;
    bool                          noGpu      = false;
    bool                          arrayGiven = false;
    double                        seconds    = 0.0;
    double                        spacing    = 0.0;
    std::size_t                   detectors  = 8UZ;
    gr::Size_t                    rows       = 4U;
    gr::Size_t                    columns    = 4U;
    std::string                   layout     = "grid";
    std::string                   positionsText;
    gr::testing::ExampleArguments args(argc, argv);
    while (args.next()) {
        const bool arrayOption = args.number("--spacing", spacing) || args.number("--detectors", detectors) || args.number("--rows", rows) || args.number("--columns", columns) || args.text("--layout", layout) || args.text("--positions", positionsText);
        const bool known       = arrayOption || args.flag("--synthetic", synthetic) || args.flag("--nogpu", noGpu) || args.number("--seconds", seconds) || args.text("--device", device);
        arrayGiven             = arrayGiven || arrayOption;
        if (!known) {
            args.refuse();
        }
    }
    constexpr std::size_t kMaxDetectors      = gr::testing::kMaxExampleElements;
    const bool            durationValid      = seconds >= 0.0;
    const bool            detectorCountValid = detectors >= 2UZ && detectors <= kMaxDetectors;
    const bool            gridCountsValid    = rows > 0U && columns > 0U && rows <= kMaxDetectors && columns <= kMaxDetectors;
    const bool            gridFits           = gridCountsValid && std::size_t{rows} * columns <= kMaxDetectors;
    if (!args.valid || !durationValid || !detectorCountValid || !gridFits || spacing < 0.0) {
        std::println(stderr, "invalid arguments: supply finite values, 2..32 detectors, non-negative seconds (0: unlimited) and a supported geometry");
        return 1;
    }
    if (const auto deviceArray = synthetic || arrayGiven ? std::nullopt : gr::testing::microphoneArray(device)) {
        layout    = deviceArray->layout;
        detectors = deviceArray->elements;
        rows      = static_cast<gr::Size_t>(deviceArray->rows);
        columns   = static_cast<gr::Size_t>(deviceArray->columns);
        spacing   = deviceArray->spacing;
    }

    const bool camera    = layout != "line";
    const auto positions = positionsText.empty() ? gr::algorithm::geometry::layoutOf(layout, detectors, rows, columns, spacing) : gr::algorithm::geometry::parsePositions(positionsText);
    const auto elements  = positions.and_then(gr::algorithm::placedElements);
    if (!elements) {
        std::println(stderr, "array refused: {}", elements.error());
        return 1;
    }
    const std::size_t M              = *elements;
    const double      stepDegrees    = camera ? 5.0 : 1.0;
    const double      azimuthStart   = camera ? -180.0 : 0.0;
    const auto        azimuthSteps   = static_cast<std::size_t>((camera ? 360.0 : 180.0) / stepDegrees) + (camera ? 0UZ : 1UZ);
    const auto        elevationSteps = static_cast<std::size_t>(90.0 / stepDegrees) + 1UZ;
    const auto        axisOf         = [stepDegrees](double start, std::size_t n) { return std::views::iota(0UZ, n) | std::views::transform([=](std::size_t i) { return start + stepDegrees * static_cast<double>(i); }) | std::ranges::to<std::vector>(); };
    gr::property_map  geometry{{"element_positions", *positions}, {"propagation_speed", kSpeedOfSound}, {"field", std::string("far")}};
    gr::property_map  scanned = geometry;
    const auto        gridOf  = []<typename T>(std::vector<T> values) { return gr::Tensor<T>(std::array<std::size_t, 1>{values.size()}, values); };
    scanned.insert_or_assign("grid_min", camera ? gridOf(std::vector{0.0, gr::algorithm::radiansOf(azimuthStart)}) : gridOf(std::vector{0.0}));
    scanned.insert_or_assign("grid_max", camera ? gridOf(std::vector{kPi / 2.0, kPi}) : gridOf(std::vector{kPi}));
    scanned.insert_or_assign("grid_steps", camera ? gridOf(std::vector{static_cast<gr::Size_t>(elevationSteps), static_cast<gr::Size_t>(azimuthSteps)}) : gridOf(std::vector{static_cast<gr::Size_t>(azimuthSteps)}));

    const auto frame = gr::testing::correlationFrameFor(gr::algorithm::apertureOf(*positions), kSpeedOfSound, kSampleRate);
    if (!frame) {
        std::println(stderr, "{}", frame.error());
        return 1;
    }
    const gr::Size_t frameSamples = frame->samples;
    const gr::Size_t maxLag       = frame->maxLag;
    const gr::Size_t hopSamples   = frameSamples / 2U;
    if (!gr::testing::kLiveAudio) {
        synthetic = true;
    }
    const bool sweep = !camera && M == 2UZ;
    if (synthetic && sweep) {
        std::println("input: synthesised audio; repeating {:.0f} s scene, one source sweeping from {:+.0f} to {:+.0f} deg bearing", gr::testing::kExampleSceneSeconds, -gr::testing::kSweepBearingLimitDegrees, gr::testing::kSweepBearingLimitDegrees);
    } else if (synthetic) {
        std::println("input: synthesised audio; repeating {:.0f} s scene with interpolated delays", gr::testing::kExampleSceneSeconds);
    }

    const auto buildGraph = [&](std::string_view domain) -> std::expected<gr::Graph, std::string> {
        gr::Graph graph;
        auto&     frames              = graph.emplaceBlock<gr::basic::StreamsToTensor<float>>({{"n_ports", gr::Size_t(M)}, {"size", frameSamples}, {"stride", hopSamples}, {"compute_domain", std::string(domain)}});
        auto&     report              = graph.emplaceBlock<SteeredPowerProjections>({{"frame_rate", static_cast<float>(kSampleRate / hopSamples)}});
        report._requestedSampleRateHz = kSampleRate;
        report._azimuthDegrees        = axisOf(azimuthStart, azimuthSteps);
        report._elevationDegrees      = camera ? axisOf(0.0, elevationSteps) : std::vector<double>{};
        if (!synthetic) {
            if (const auto connected = gr::testing::connectMicrophones(graph, frames, device, M, kSampleRate); !connected) {
                return std::unexpected(connected.error());
            }
        }
        if (synthetic) {
            const auto nSamples = static_cast<std::size_t>(gr::testing::kExampleSceneSeconds * kSampleRate);
            if (sweep) {
                report._sweepSceneSamples = nSamples;
            } else {
                report._truth = camera ? std::vector<KnownSource>{{40.0, 30.0, 1.0}, {25.0, -70.0, 0.7}} : std::vector<KnownSource>{{0.0, 40.0, 1.0}, {0.0, 120.0, 0.7}};
            }
            for (std::size_t m = 0UZ; m < M; ++m) {
                const std::array<double, 3>       position{(*positions)[m, 0UZ], (*positions)[m, 1UZ], (*positions)[m, 2UZ]};
                std::vector<gr::testing::Arrival> arrivals;
                for (std::size_t k = 0UZ; k < report._truth.size(); ++k) {
                    const KnownSource&          source        = report._truth[k];
                    const auto                  u             = gr::algorithm::directionOf(gr::algorithm::radiansOf(source.azimuthDegrees), gr::algorithm::radiansOf(source.elevationDegrees));
                    const bool                  near          = std::isfinite(source.distanceMetres);
                    const std::array<double, 3> point         = near ? std::array{u[0] * source.distanceMetres, u[1] * source.distanceMetres, u[2] * source.distanceMetres} : u;
                    const double                originDelay   = near ? source.distanceMetres / kSpeedOfSound : 0.0;
                    const double                relativeDelay = gr::algorithm::delayOf<double>(position.data(), point, near ? gr::algorithm::Field::near : gr::algorithm::Field::far, kSpeedOfSound) - originDelay;
                    arrivals.push_back({.delaySamples = relativeDelay * kSampleRate, .gain = source.gain, .seed = 21U * (1U + static_cast<std::uint32_t>(k))});
                }
                std::vector<float> noisySamples = sweep ? sweepingNoise(position[0UZ], nSamples, kSampleRate) : gr::testing::delayedNoises(arrivals, nSamples);
                gr::testing::addMicrophoneNoise(noisySamples, m);
                gr::testing::addElementSource(graph, frames, m, std::move(noisySamples), kSampleRate);
            }
        }

        report._inputDescription         = synthetic ? gr::testing::syntheticInputDescription() : gr::testing::liveInputDescription(device);
        auto& lagCurve                   = graph.emplaceBlock<gr::localisation::PairCrossCorrelation<float>>({{"max_lag", maxLag}, {"f_min", 0.f}, {"f_max", static_cast<float>(kSampleRate / 2.0)}, {"compute_domain", std::string(domain)}});
        std::ignore                      = graph.connect<"out", "in">(frames, lagCurve);
        gr::property_map scannedOnDomain = scanned;
        scannedOnDomain.insert_or_assign("compute_domain", std::string(domain));
        auto& srp    = graph.emplaceBlock<gr::localisation::SteeredPowerMap<float>>(scannedOnDomain);
        auto& finder = graph.emplaceBlock<gr::blocks::math::TensorPeakFinder<float>>({{"max_peaks", gr::Size_t(2)}, {"threshold", 0.3f}, {"min_separation", camera ? 4.f : 20.f}, {"compute_domain", std::string(domain)}});
        auto& delay  = graph.emplaceBlock<gr::localisation::PairDelay<float>>({{"max_lag", maxLag}, {"compute_domain", std::string(domain)}});
        auto& locate = graph.emplaceBlock<gr::localisation::SourceFromDelays<float>>(geometry);
        std::ignore  = graph.connect<"out", "in">(lagCurve, srp);
        std::ignore  = graph.connect<"out", "map">(srp, report);
        std::ignore  = graph.connect<"out", "in">(srp, finder);
        std::ignore  = graph.connect<"out", "peaks">(finder, report);
        std::ignore  = graph.connect<"out", "in">(lagCurve, delay);
        std::ignore  = graph.connect<"out", "in">(delay, locate);
        std::ignore  = graph.connect<"out", "located">(locate, report);
        return graph;
    };

    gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::multiThreaded> scheduler;
    return gr::testing::runExample(buildGraph, gr::testing::chooseExampleDomain(noGpu), seconds, scheduler);
}
