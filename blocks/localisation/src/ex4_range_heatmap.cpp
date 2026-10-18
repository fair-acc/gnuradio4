#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
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
#include <gnuradio-4.0/localisation/SteeredPowerRange.hpp>
#include <gnuradio-4.0/testing/ArrayScene.hpp>
#include <gnuradio-4.0/testing/ExampleRun.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace {

constexpr double      kSampleRate   = gr::testing::kExampleSampleRateHz;
constexpr double      kSpeedOfSound = gr::testing::kSpeedOfSoundInAir;
constexpr std::size_t kChartColumns = 100UZ;
constexpr std::size_t kChartRows    = 14UZ;
constexpr std::size_t kHeatmapRows  = 51UZ;
constexpr auto        kRedrawPeriod = std::chrono::milliseconds(500);

struct KnownSource {
    double elevationDegrees;
    double azimuthDegrees;
    double gain;
    double distanceMetres = std::numeric_limits<double>::infinity();
};

struct RangeHeatmap : gr::Block<RangeHeatmap> {
    using Description = gr::Doc<"Scans the lag curves of a planar array over direction and distance and plots the angular heatmap, its projections and the range profile of each peak.">;

    gr::TensorPortIn<float> lagCurves;
    gr::SampleRate          sample_rate = 0.f;
    GR_MAKE_REFLECTABLE(RangeHeatmap, lagCurves, sample_rate);

    std::string                           _inputDescription;
    gr::graphs::HeatmapStyle              _heatmapStyle = gr::graphs::HeatmapStyle::braille;
    double                                _colourLimit  = 1.0;
    gr::localisation::SteeredPowerRange   _rangeScan;
    std::vector<float>                    _latestLagCurves;
    std::size_t                           _lagLength = 0UZ;
    std::vector<KnownSource>              _truth;
    double                                _requestedSampleRateHz = 0.0;
    std::chrono::steady_clock::time_point _printed{};

    void start() { gr::graphs::resetView(); }

    gr::work::Status processBulk(gr::InputSpanLike auto& correlations) {
        if (correlations.size() > 0UZ) {
            const auto& curves = correlations[correlations.size() - 1UZ];
            _lagLength         = curves.rank() == 2UZ ? curves.extent(1UZ) : 0UZ;
            _latestLagCurves.assign(curves.data(), curves.data() + curves.size());
        }
        std::ignore              = correlations.consume(correlations.size());
        const auto redrawStarted = std::chrono::steady_clock::now();
        if (redrawStarted - _printed >= kRedrawPeriod && !_latestLagCurves.empty()) {
            _printed = redrawStarted;
            print();
        }
        return gr::work::Status::OK;
    }

    void print() {
        _rangeScan.sampleRateHz = static_cast<double>(sample_rate.value);
        auto rangeResult        = gr::localisation::scanSteeredPowerRange(_rangeScan, _latestLagCurves, _lagLength);
        if (!rangeResult) {
            this->emitErrorMessage("range scan", rangeResult.error());
            this->requestStop();
            return;
        }
        const std::vector<float>        angularPower = rangeResult->angularPower;
        std::vector<std::vector<float>> peaks;
        for (const auto& profile : rangeResult->profiles) {
            peaks.push_back({static_cast<float>(profile.elevationRadians), static_cast<float>(profile.azimuthRadians), profile.score});
        }
        const auto        displayAngles = [](double azimuthRadians, double elevationRadians) { return gr::localisation::frontAnglesDegrees(gr::algorithm::directionOf(azimuthRadians, elevationRadians)); };
        const auto        frontMap      = gr::localisation::projectFrontPower(angularPower, gr::localisation::SteeredPowerRange::kAzimuthSteps, gr::localisation::SteeredPowerRange::kElevationSteps);
        const std::size_t frontColumns  = frontMap.anglesDegrees.size();

        gr::graphs::ImChart<kChartColumns, std::dynamic_extent> heatmap({{-gr::localisation::kFrontAngleLimitDegrees, gr::localisation::kFrontAngleLimitDegrees}, {-gr::localisation::kFrontAngleLimitDegrees, gr::localisation::kFrontAngleLimitDegrees}}, kChartColumns, kHeatmapRows);
        heatmap.axis_name_x = "horizontal bearing [deg]";
        heatmap.axis_name_y = "vertical elevation [deg]";
        std::vector<std::array<double, 2>> markers;
        for (const auto& peak : peaks) {
            const auto angles = displayAngles(static_cast<double>(peak[1UZ]), static_cast<double>(peak[0UZ]));
            markers.push_back({angles[0UZ], angles[1UZ]});
        }
        heatmap.reset();
        const auto bounds = std::ranges::minmax(angularPower);
        if (const auto rendered = heatmap.drawHeatmap(std::span<const float>(frontMap.power), frontColumns, frontColumns, std::pair<double, double>{bounds.min, bounds.max}, markers, _heatmapStyle); !rendered) {
            this->emitErrorMessage("heatmap", rendered.error());
            this->requestStop();
            return;
        }
        std::println("map   {:<10} | H/V +/-90 deg | bins 5 deg | relative maxima\033[K", gr::meta::enumName(_heatmapStyle).value());

        std::vector<double> horizontalPower(frontColumns, -std::numeric_limits<double>::infinity());
        std::vector<double> verticalPower(frontColumns, -std::numeric_limits<double>::infinity());
        for (std::size_t row = 0UZ; row < frontColumns; ++row) {
            for (std::size_t column = 0UZ; column < frontColumns; ++column) {
                const double power = static_cast<double>(frontMap.power[row * frontColumns + column]);
                if (std::isfinite(power)) {
                    horizontalPower[column] = std::max(horizontalPower[column], power);
                    verticalPower[row]      = std::max(verticalPower[row], power);
                }
            }
        }
        const double                                   bottom = static_cast<double>(*std::ranges::min_element(angularPower));
        const double                                   top    = static_cast<double>(std::max(*std::ranges::max_element(angularPower), std::numeric_limits<float>::min()));
        gr::graphs::ImChart<kChartColumns, kChartRows> projections({{-gr::localisation::kFrontAngleLimitDegrees, gr::localisation::kFrontAngleLimitDegrees}, {std::min(0.0, bottom), 1.05 * top}});
        projections.axis_name_x   = "angle [deg]";
        projections.axis_name_y   = "steered power";
        const auto drawProjection = [&](const std::vector<double>& power, const char* label) {
            std::vector<double> angles;
            std::vector<double> scores;
            for (std::size_t i = 0UZ; i < power.size(); ++i) {
                if (std::isfinite(power[i])) {
                    angles.push_back(frontMap.anglesDegrees[i]);
                    scores.push_back(power[i]);
                }
            }
            projections.draw(angles, scores, label);
        };
        drawProjection(horizontalPower, "horizontal");
        drawProjection(verticalPower, "vertical");
        projections.draw(std::nullopt);

        std::println("scan  {:5.2f}..{:5.2f} m + far | H/V: max over distance and other angle\033[K", _rangeScan.minDistanceMetres, _rangeScan.maxDistanceMetres);
        gr::graphs::ImChart<kChartColumns, kChartRows, gr::graphs::LogAxisTransform> rangeChart({{_rangeScan.minDistanceMetres, _rangeScan.maxDistanceMetres}, {-_colourLimit, _colourLimit}});
        rangeChart.axis_name_x  = "distance from array centre [m]";
        rangeChart.axis_name_y  = "steered power";
        const auto distanceAxis = rangeResult->distancesMetres | std::views::transform([](double radius) { return static_cast<float>(radius); }) | std::ranges::to<std::vector>();
        for (std::size_t k = 0UZ; k < rangeResult->profiles.size(); ++k) {
            rangeChart.draw(distanceAxis, rangeResult->profiles[k].power, std::format("peak {}", k + 1UZ));
        }
        rangeChart.draw(std::nullopt);
        std::println("range fixed direction | MAX: finite maximum | EDGE: boundary | FAR: far wins; flat = unresolved\033[K");

        std::println("input {:<60.60} {:6.0f} Hz received, {:.0f} Hz requested\033[K", _inputDescription, static_cast<double>(sample_rate), _requestedSampleRateHz);
        const auto errorColumn = [&](double azimuthRadians, double elevationRadians) {
            if (_truth.empty()) {
                return std::string{};
            }
            const auto estimate = gr::algorithm::directionOf(azimuthRadians, elevationRadians);
            double     nearest  = std::numeric_limits<double>::infinity();
            for (const auto& source : _truth) {
                nearest = std::min(nearest, gr::localisation::angularSeparationDegrees(estimate, gr::algorithm::directionOf(gr::algorithm::radiansOf(source.azimuthDegrees), gr::algorithm::radiansOf(source.elevationDegrees))));
            }
            return std::format(" | err {:5.1f} deg", nearest);
        };
        for (std::size_t k = 0UZ; k < peaks.size(); ++k) {
            const auto&       peak          = peaks[k];
            const double      peakAzimuth   = static_cast<double>(peak[1UZ]);
            const double      peakElevation = static_cast<double>(peak[0UZ]);
            const auto        angles        = displayAngles(peakAzimuth, peakElevation);
            const std::string error         = errorColumn(peakAzimuth, peakElevation);
            const auto&       profile       = rangeResult->profiles[k];
            const auto        best          = std::ranges::max_element(profile.power);
            const std::size_t index         = static_cast<std::size_t>(best - profile.power.begin());
            const bool        boundary      = index == 0UZ || index + 1UZ == profile.power.size();
            const char*       rangeKind     = [&] {
                if (profile.farPower >= *best) {
                    return "FAR";
                }
                return boundary ? "EDGE" : "MAX";
            }();
            std::println("peak {:1} H {:+6.1f} V {:+6.1f} deg | rMax {:5.2f} m {:4} | Pmax {:7.2f} far {:7.2f}{}\033[K", k + 1UZ, angles[0UZ], angles[1UZ], rangeResult->distancesMetres[index], rangeKind, *best, profile.farPower, error);
        }
        for (std::size_t k = 0UZ; k < _truth.size(); ++k) {
            const auto angles = displayAngles(gr::algorithm::radiansOf(_truth[k].azimuthDegrees), gr::algorithm::radiansOf(_truth[k].elevationDegrees));
            std::println("truth {:1} H {:+6.1f} V {:+6.1f} deg | r {:6.2f} m | gain {:4.2f}\033[K", k, angles[0UZ], angles[1UZ], _truth[k].distanceMetres, _truth[k].gain);
        }
        std::print("\033[J");
    }
};

} // namespace

int main(int argc, char** argv) {
    std::string                         device;
    bool                                synthetic  = false;
    bool                                noGpu      = false;
    bool                                arrayGiven = false;
    double                              seconds    = 0.0;
    double                              spacing    = 0.0;
    std::size_t                         detectors  = 8UZ;
    gr::Size_t                          rows       = 4U;
    gr::Size_t                          columns    = 4U;
    std::string                         layout     = "grid";
    std::string                         positionsText;
    gr::localisation::SteeredPowerRange rangeScan;
    std::string                         mapStyle = "braille";
    gr::testing::ExampleArguments       args(argc, argv);
    while (args.next()) {
        const bool arrayOption = args.number("--spacing", spacing) || args.number("--detectors", detectors) || args.number("--rows", rows) || args.number("--columns", columns) || args.text("--layout", layout) || args.text("--positions", positionsText);
        const bool known       = arrayOption || args.flag("--synthetic", synthetic) || args.flag("--nogpu", noGpu) || args.number("--seconds", seconds) || args.text("--device", device) || args.text("--map-style", mapStyle) //
                           || args.number("--range-min", rangeScan.minDistanceMetres) || args.number("--range-max", rangeScan.maxDistanceMetres) || args.number("--peaks", rangeScan.maxPeaks);
        arrayGiven = arrayGiven || arrayOption;
        if (!known) {
            args.refuse();
        }
    }
    const auto heatmapStyle = gr::meta::parseEnum<gr::graphs::HeatmapStyle>(mapStyle);
    if (!heatmapStyle) {
        std::println(stderr, "--map-style: use braille or monochrome");
        return 1;
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

    const auto deviceArray = synthetic || arrayGiven ? std::nullopt : gr::testing::microphoneArray(device);
    if (deviceArray) {
        layout    = deviceArray->layout;
        detectors = deviceArray->elements;
        rows      = static_cast<gr::Size_t>(deviceArray->rows);
        columns   = static_cast<gr::Size_t>(deviceArray->columns);
        spacing   = deviceArray->spacing;
    }
    if (layout == "line") {
        if (!deviceArray) {
            std::println(stderr, "the angular heatmap requires a planar array");
        } else {
            std::println(stderr, "the angular heatmap needs a planar array, the device's {} input channels place a line: use another --device, or --synthetic", deviceArray->elements);
        }
        return 1;
    }
    const auto positions = positionsText.empty() ? gr::algorithm::geometry::layoutOf(layout, detectors, rows, columns, spacing) : gr::algorithm::geometry::parsePositions(positionsText);
    const auto elements  = positions.and_then(gr::algorithm::placedElements);
    if (!elements) {
        std::println(stderr, "array refused: {}", elements.error());
        return 1;
    }
    const std::size_t M                       = *elements;
    rangeScan.propagationSpeedMetresPerSecond = kSpeedOfSound;
    for (std::size_t m = 0UZ; m < M; ++m) {
        rangeScan.positionsMetres.push_back({(*positions)[m, 0UZ], (*positions)[m, 1UZ], (*positions)[m, 2UZ]});
    }
    if (const auto valid = rangeScan.validate(); !valid) {
        std::println(stderr, "{}", valid.error());
        return 1;
    }
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
    if (synthetic) {
        std::println("input: synthesised audio; repeating {:.0f} s scene with interpolated delays", gr::testing::kExampleSceneSeconds);
    }

    const auto buildGraph = [&](std::string_view domain) -> std::expected<gr::Graph, std::string> {
        gr::Graph graph;
        auto&     frames              = graph.emplaceBlock<gr::basic::StreamsToTensor<float>>({{"n_ports", gr::Size_t(M)}, {"size", frameSamples}, {"stride", hopSamples}, {"compute_domain", std::string(domain)}});
        auto&     report              = graph.emplaceBlock<RangeHeatmap>();
        report._requestedSampleRateHz = kSampleRate;
        report._heatmapStyle          = *heatmapStyle;
        report._rangeScan             = rangeScan;
        report._colourLimit           = static_cast<double>(M * (M - 1UZ) / 2UZ);
        if (!synthetic) {
            if (const auto connected = gr::testing::connectMicrophones(graph, frames, device, M, kSampleRate); !connected) {
                return std::unexpected(connected.error());
            }
        }
        if (synthetic) {
            const auto nSamples = static_cast<std::size_t>(gr::testing::kExampleSceneSeconds * kSampleRate);
            report._truth       = std::vector<KnownSource>{{40.0, 30.0, 1.0, 0.3}, {25.0, -70.0, 0.7}};
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
                std::vector<float> noisySamples = gr::testing::delayedNoises(arrivals, nSamples);
                gr::testing::addMicrophoneNoise(noisySamples, m);
                gr::testing::addElementSource(graph, frames, m, std::move(noisySamples), kSampleRate);
            }
        }

        report._inputDescription = synthetic ? gr::testing::syntheticInputDescription() : gr::testing::liveInputDescription(device);
        auto& lagCurve           = graph.emplaceBlock<gr::localisation::PairCrossCorrelation<float>>({{"max_lag", maxLag}, {"f_min", 0.f}, {"f_max", static_cast<float>(kSampleRate / 2.0)}, {"compute_domain", std::string(domain)}});
        std::ignore              = graph.connect<"out", "in">(frames, lagCurve);
        std::ignore              = graph.connect<"out", "lagCurves">(lagCurve, report);
        return graph;
    };

    gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::multiThreaded> scheduler;
    return gr::testing::runExample(buildGraph, gr::testing::chooseExampleDomain(noGpu), seconds, scheduler);
}
