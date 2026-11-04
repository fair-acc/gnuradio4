#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <numbers>
#include <numeric>
#include <optional>
#include <print>
#include <ranges>
#include <span>
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
#include <gnuradio-4.0/fourier/STFT.hpp>
#include <gnuradio-4.0/localisation/FrontProjection.hpp>
#include <gnuradio-4.0/localisation/PairCrossSpectra.hpp>
#include <gnuradio-4.0/localisation/SourceLocation.hpp>
#include <gnuradio-4.0/localisation/SteeredPowerHistory.hpp>
#include <gnuradio-4.0/math/TensorPeakFinder.hpp>
#include <gnuradio-4.0/testing/ArrayScene.hpp>
#include <gnuradio-4.0/testing/ExampleRun.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace {

constexpr double      kPi                 = std::numbers::pi;
constexpr double      kSampleRate         = 96000.0;
constexpr std::size_t kDetectors          = 2UZ;
constexpr double      kSpeedOfSound       = gr::testing::kSpeedOfSoundInAir;
constexpr std::size_t kChartColumns       = 100UZ;
constexpr std::size_t kChartRows          = 14UZ;
constexpr auto        kRedrawPeriod       = std::chrono::milliseconds(500);
constexpr double      kSilentMeanBinPower = 1e-12;

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

struct LineWaterfall : gr::Block<LineWaterfall> {
    using Description = gr::Doc<"Plots the pair lag curves and a scrolling bearing waterfall of a line array's steered power, with the peaks against the known sources.">;

    gr::TensorPortIn<float>               map;
    gr::TensorPortIn<float>               peaks;
    gr::TensorPortIn<float>               lagCurves;
    gr::TensorPortIn<float>               located;
    gr::TensorPortIn<std::complex<float>> spectra;
    gr::SampleRate                        sample_rate = 0.f;
    gr::FrameRate                         frame_rate  = 0.f;
    GR_MAKE_REFLECTABLE(LineWaterfall, map, peaks, lagCurves, located, spectra, sample_rate, frame_rate);

    std::string                           _inputDescription;
    gr::graphs::HeatmapStyle              _heatmapStyle = gr::graphs::HeatmapStyle::braille;
    gr::localisation::SteeredPowerHistory _history;
    double                                _colourLimit = 1.0;
    std::vector<float>                    _latestLagCurves;
    std::size_t                           _lagLength = 0UZ;
    std::vector<double>                   _azimuthDegrees;
    std::vector<KnownSource>              _truth;
    std::vector<float>                    _latestMap;
    std::vector<std::vector<float>>       _latestPeaks;
    std::vector<float>                    _latestLocated;
    std::size_t                           _elements              = 0UZ;
    std::size_t                           _framesSeen            = 0UZ;
    std::size_t                           _sweepSceneSamples     = 0UZ;
    gr::Size_t                            _maxLagSamples         = 0U;
    double                                _requestedSampleRateHz = 0.0;
    std::vector<double>                   _loudestSinceRedraw;
    std::vector<bool>                     _silentReported;
    std::chrono::steady_clock::time_point _printed{};

    void start() { gr::graphs::resetView(); }

    gr::work::Status processBulk(gr::InputSpanLike auto& maps, gr::InputSpanLike auto& found, gr::InputSpanLike auto& correlations, gr::InputSpanLike auto& directions, gr::InputSpanLike auto& frameSpectra) {
        const std::size_t nMaps = std::min(maps.size(), found.size());
        for (std::size_t frame = 0UZ; frame < nMaps; ++frame) {
            const auto& frameMap = maps[frame];
            if (const auto appended = _history.append(std::span<const float>(frameMap.data(), frameMap.size()), frame_rate); !appended) {
                this->emitErrorMessage("waterfall", appended.error());
                this->requestStop();
                return gr::work::Status::ERROR;
            }
        }
        if (nMaps > 0UZ) {
            const auto& latest = maps[nMaps - 1UZ];
            _latestMap.assign(latest.data(), latest.data() + latest.size());
            const auto& rows = found[nMaps - 1UZ];
            _latestPeaks.clear();
            for (std::size_t k = 0UZ; rows.rank() == 2UZ && k < rows.extent(0UZ); ++k) {
                _latestPeaks.emplace_back(rows.data() + k * rows.extent(1UZ), rows.data() + (k + 1UZ) * rows.extent(1UZ));
            }
        }
        if (correlations.size() > 0UZ) {
            const auto& curves = correlations[correlations.size() - 1UZ];
            _lagLength         = curves.rank() == 2UZ ? curves.extent(1UZ) : 0UZ;
            _latestLagCurves.assign(curves.data(), curves.data() + curves.size());
        }
        if (directions.size() > 0UZ) {
            const auto& row = directions[directions.size() - 1UZ];
            _latestLocated.assign(row.data(), row.data() + row.size());
        }
        for (std::size_t frame = 0UZ; frame < frameSpectra.size(); ++frame) {
            const auto&       rows    = frameSpectra[frame];
            const std::size_t nInputs = rows.extent(0UZ);
            const std::size_t nBins   = rows.extent(1UZ);
            _loudestSinceRedraw.resize(nInputs, 0.0);
            for (std::size_t m = 0UZ; m < nInputs; ++m) {
                const auto*  row          = rows.data() + m * nBins;
                const double meanBinPower = std::accumulate(row, row + nBins, 0.0, [](double sum, std::complex<float> X) { return sum + static_cast<double>(std::norm(X)); }) / static_cast<double>(nBins);
                _loudestSinceRedraw[m]    = std::max(_loudestSinceRedraw[m], meanBinPower);
            }
        }
        std::ignore = frameSpectra.consume(frameSpectra.size());
        _framesSeen += nMaps;
        std::ignore              = maps.consume(nMaps);
        std::ignore              = found.consume(nMaps);
        std::ignore              = directions.consume(directions.size());
        std::ignore              = correlations.consume(correlations.size());
        const auto redrawStarted = std::chrono::steady_clock::now();
        if (redrawStarted - _printed >= kRedrawPeriod && !_latestMap.empty()) {
            _printed = redrawStarted;
            print();
            _silentReported.resize(_loudestSinceRedraw.size(), false);
            for (std::size_t m = 0UZ; m < _loudestSinceRedraw.size(); ++m) {
                if (_loudestSinceRedraw[m] < kSilentMeanBinPower && !_silentReported[m]) {
                    _silentReported[m] = true;
                    std::println(stderr, "input {} is silent: check the device's channel mapping or the cable", m);
                }
            }
            _loudestSinceRedraw.clear();
        }
        return gr::work::Status::OK;
    }

    void print() {
        const auto        displayAngles = [](double azimuthRadians, double /*elevationRadians*/) { return gr::localisation::frontAnglesDegrees({std::cos(azimuthRadians), 0.0, std::sin(azimuthRadians)}); };
        const std::size_t frontColumns  = _azimuthDegrees.size();
        if (_lagLength > 0UZ) {
            const double                                   maxLagMs = 1000.0 * static_cast<double>(_maxLagSamples) / static_cast<double>(sample_rate);
            gr::graphs::ImChart<kChartColumns, kChartRows> lagChart({{-maxLagMs, maxLagMs}, {-0.25, 1.1}});
            lagChart.axis_name_x            = "lag [ms], +: input 0 hears later";
            lagChart.axis_name_y            = "GCC-PHAT";
            const std::vector<double> lagMs = gr::testing::lagAxisMilliseconds(_lagLength, static_cast<double>(sample_rate));
            std::size_t               pair  = 0UZ;
            for (std::size_t i = 0UZ; i < _elements; ++i) {
                for (std::size_t j = i + 1UZ; j < _elements; ++j, ++pair) {
                    const auto                curveBegin = _latestLagCurves.begin() + static_cast<std::ptrdiff_t>(pair * _lagLength);
                    const std::vector<double> curve(curveBegin, curveBegin + static_cast<std::ptrdiff_t>(_lagLength));
                    lagChart.draw<gr::graphs::Style::Bars>(lagMs, curve, std::format("pair {}-{}", i, j));
                }
            }
            lagChart.reset();
            lagChart.draw(std::nullopt);
        }
        const double                                            yMin = std::min(_history.ageSeconds(0UZ), -1.0 / static_cast<double>(frame_rate));
        gr::graphs::ImChart<kChartColumns, std::dynamic_extent> chart({{-gr::localisation::kFrontAngleLimitDegrees, gr::localisation::kFrontAngleLimitDegrees}, {yMin, 0.0}}, kChartColumns, kChartRows);
        chart.axis_name_x = "horizontal bearing [deg]";
        chart.axis_name_y = "age [s]";
        std::vector<float> values;
        values.reserve(_history.rows.size() * frontColumns);
        for (const auto& row : _history.rows) {
            values.insert(values.end(), row.rbegin(), row.rend());
        }
        std::vector<std::array<double, 2>> markers;
        for (const auto& peak : _latestPeaks) {
            const auto angles = displayAngles(static_cast<double>(peak[0UZ]), 0.0);
            markers.push_back({angles[0UZ], 0.0});
        }
        if (const auto rendered = chart.drawHeatmap(std::span<const float>(values), _history.rows.size(), frontColumns, std::pair{-_colourLimit, _colourLimit}, markers, _heatmapStyle); !rendered) {
            this->emitErrorMessage("heatmap", rendered.error());
            this->requestStop();
            return;
        }
        std::println("waterfall: {} frames at {:.3f} Hz; broadside 0 deg, right positive; elevation is not measured", _history.rows.size(), static_cast<float>(frame_rate));

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
            const auto&       peak        = _latestPeaks[k];
            const double      peakAzimuth = static_cast<double>(peak[0UZ]);
            const auto        angles      = displayAngles(peakAzimuth, 0.0);
            const std::string error       = errorColumn(peakAzimuth, 0.0);
            std::println("peak {:1} H {:+6.1f} V {:+6.1f} deg | power {:7.2f}{}\033[K", k + 1UZ, angles[0UZ], angles[1UZ], static_cast<double>(peak[1UZ]), error);
        }
        if (_latestLocated.size() >= 5UZ) {
            const double elevation = static_cast<double>(_latestLocated[0UZ]);
            const double azimuth   = std::abs(static_cast<double>(_latestLocated[1UZ]));
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
    bool                          synthetic = false;
    bool                          noGpu     = false;
    double                        seconds   = 0.0;
    double                        spacing   = 0.0;
    std::size_t                   detectors = kDetectors;
    gr::Size_t                    rows      = 4U;
    gr::Size_t                    columns   = 4U;
    std::string                   layout    = "line";
    std::string                   positionsText;
    std::string                   mapStyle = "braille";
    std::string                   bandText;
    gr::testing::ExampleArguments args(argc, argv);
    while (args.next()) {
        const bool arrayOption = args.number("--spacing", spacing) || args.number("--detectors", detectors) || args.number("--rows", rows) || args.number("--columns", columns) || args.text("--layout", layout) || args.text("--positions", positionsText);
        const bool known       = arrayOption || args.flag("--synthetic", synthetic) || args.flag("--nogpu", noGpu) || args.number("--seconds", seconds) || args.text("--device", device) || args.text("--map-style", mapStyle) || args.text("--band", bandText);
        if (!known) {
            args.refuse();
        }
    }
    const auto heatmapStyle = gr::meta::parseEnum<gr::graphs::HeatmapStyle>(mapStyle);
    if (!heatmapStyle) {
        std::println(stderr, "--map-style: use braille or monochrome");
        return 1;
    }
    std::array<double, 2> bandHz{0.0, kSampleRate / 2.0};
    if (!bandText.empty()) {
        const auto parseHz = [](std::string_view text, double& hz) {
            const auto parsed = std::from_chars(text.data(), text.data() + text.size(), hz);
            return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() && std::isfinite(hz);
        };
        const std::string_view argument = bandText;
        const std::size_t      colon    = argument.find(':');
        const bool             parsed   = colon != std::string_view::npos && parseHz(argument.substr(0UZ, colon), bandHz[0UZ]) && parseHz(argument.substr(colon + 1UZ), bandHz[1UZ]);
        if (!parsed || bandHz[0UZ] < 0.0 || bandHz[0UZ] >= bandHz[1UZ] || bandHz[1UZ] > kSampleRate / 2.0) {
            std::println(stderr, "--band: use f_min:f_max in Hz with 0 <= f_min < f_max <= {:.0f}", kSampleRate / 2.0);
            return 1;
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
    if (layout != "line") {
        std::println(stderr, "the waterfall requires a line array");
        return 1;
    }
    const auto positions = positionsText.empty() ? gr::algorithm::geometry::layoutOf(layout, detectors, rows, columns, spacing) : gr::algorithm::geometry::parsePositions(positionsText);
    const auto elements  = positions.and_then(gr::algorithm::placedElements);
    if (!elements) {
        std::println(stderr, "array refused: {}", elements.error());
        return 1;
    }
    const std::size_t M            = *elements;
    const double      stepDegrees  = 1.0;
    const auto        azimuthSteps = static_cast<std::size_t>(180.0 / stepDegrees) + 1UZ;
    const auto        axisOf       = [stepDegrees](double start, std::size_t n) { return std::views::iota(0UZ, n) | std::views::transform([=](std::size_t i) { return start + stepDegrees * static_cast<double>(i); }) | std::ranges::to<std::vector>(); };
    gr::property_map  geometry{{"element_positions", *positions}, {"propagation_speed", kSpeedOfSound}, {"field", std::string("far")}};
    gr::property_map  scanned = geometry;
    const auto        gridOf  = []<typename T>(std::vector<T> values) { return gr::Tensor<T>(std::array<std::size_t, 1>{values.size()}, values); };
    scanned.insert_or_assign("grid_min", gridOf(std::vector{0.0}));
    scanned.insert_or_assign("grid_max", gridOf(std::vector{kPi}));
    scanned.insert_or_assign("grid_steps", gridOf(std::vector{static_cast<gr::Size_t>(azimuthSteps)}));

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
    const bool sweep = M == 2UZ;
    if (synthetic && sweep) {
        std::println("input: synthesised audio; repeating {:.0f} s scene, one source sweeping from {:+.0f} to {:+.0f} deg bearing", gr::testing::kExampleSceneSeconds, -gr::testing::kSweepBearingLimitDegrees, gr::testing::kSweepBearingLimitDegrees);
    } else if (synthetic) {
        std::println("input: synthesised audio; repeating {:.0f} s scene with interpolated delays", gr::testing::kExampleSceneSeconds);
    }

    const auto buildGraph = [&](std::string_view domain) -> std::expected<gr::Graph, std::string> {
        gr::Graph graph;
        auto&     frames              = graph.emplaceBlock<gr::basic::StreamsToTensor<float>>({{"n_ports", gr::Size_t(M)}, {"size", frameSamples}, {"stride", hopSamples}, {"compute_domain", std::string(domain)}});
        auto&     report              = graph.emplaceBlock<LineWaterfall>({{"frame_rate", static_cast<float>(kSampleRate / hopSamples)}});
        report._elements              = M;
        report._maxLagSamples         = maxLag;
        report._requestedSampleRateHz = kSampleRate;
        report._heatmapStyle          = *heatmapStyle;
        report._colourLimit           = static_cast<double>(M * (M - 1UZ) / 2UZ);
        report._azimuthDegrees        = axisOf(0.0, azimuthSteps);
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
                report._truth = std::vector<KnownSource>{{0.0, 40.0, 1.0}, {0.0, 120.0, 0.7}};
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
        auto& stft                       = graph.emplaceBlock<gr::blocks::fft::STFT<float>>({{"compute_domain", std::string(domain)}});
        std::ignore                      = graph.connect<"out", "in">(frames, stft);
        std::ignore                      = graph.connect<"out", "spectra">(stft, report);
        auto& lagCurve                   = graph.emplaceBlock<gr::localisation::PairCrossCorrelation<float>>({{"max_lag", maxLag}, {"f_min", static_cast<float>(bandHz[0UZ])}, {"f_max", static_cast<float>(bandHz[1UZ])}, {"compute_domain", std::string(domain)}});
        std::ignore                      = graph.connect<"out", "in">(frames, lagCurve);
        gr::property_map scannedOnDomain = scanned;
        scannedOnDomain.insert_or_assign("compute_domain", std::string(domain));
        auto& srp    = graph.emplaceBlock<gr::localisation::SteeredPowerMap<float>>(scannedOnDomain);
        auto& finder = graph.emplaceBlock<gr::blocks::math::TensorPeakFinder<float>>({{"max_peaks", gr::Size_t(2)}, {"threshold", 0.3f}, {"min_separation", 20.f}, {"compute_domain", std::string(domain)}});
        auto& delay  = graph.emplaceBlock<gr::localisation::PairDelay<float>>({{"max_lag", maxLag}, {"compute_domain", std::string(domain)}});
        auto& locate = graph.emplaceBlock<gr::localisation::SourceFromDelays<float>>(geometry);
        std::ignore  = graph.connect<"out", "in">(lagCurve, srp);
        std::ignore  = graph.connect<"out", "map">(srp, report);
        std::ignore  = graph.connect<"out", "in">(srp, finder);
        std::ignore  = graph.connect<"out", "peaks">(finder, report);
        std::ignore  = graph.connect<"out", "in">(lagCurve, delay);
        std::ignore  = graph.connect<"out", "in">(delay, locate);
        std::ignore  = graph.connect<"out", "located">(locate, report);
        std::ignore  = graph.connect<"out", "lagCurves">(lagCurve, report);
        return graph;
    };

    gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::multiThreaded> scheduler;
    return gr::testing::runExample(buildGraph, gr::testing::chooseExampleDomain(noGpu), seconds, scheduler);
}
