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
#include <gnuradio-4.0/localisation/BandSteeredPower.hpp>
#include <gnuradio-4.0/localisation/FrontProjection.hpp>
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
constexpr double      kBandAverageSeconds = 0.1;
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

struct LineBands : gr::Block<LineBands> {
    using Description = gr::Doc<"Plots the steered power of a line array over bearing and half-octave frequency bands, with its broadband projection and verdict.">;

    gr::TensorPortIn<std::complex<float>> spectra;
    gr::SampleRate                        sample_rate = 0.f;
    gr::FrameRate                         frame_rate  = 0.f;
    GR_MAKE_REFLECTABLE(LineBands, spectra, sample_rate, frame_rate);

    std::string                           _inputDescription;
    gr::graphs::HeatmapStyle              _heatmapStyle = gr::graphs::HeatmapStyle::braille;
    std::vector<KnownSource>              _truth;
    std::size_t                           _elements              = 0UZ;
    std::size_t                           _framesSeen            = 0UZ;
    std::size_t                           _sweepSceneSamples     = 0UZ;
    double                                _requestedSampleRateHz = 0.0;
    std::chrono::steady_clock::time_point _printed{};

    std::vector<double>              _bearingsDegrees;
    std::vector<double>              _bearingPairDelays;
    std::array<double, 2>            _bandHz{};
    std::vector<std::complex<float>> _recentSpectra;
    std::size_t                      _spectrumBins = 0UZ;
    std::vector<double>              _loudestSinceRedraw;
    std::vector<bool>                _silentReported;

    void start() { gr::graphs::resetView(); }

    gr::work::Status processBulk(gr::InputSpanLike auto& frameSpectra) {
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
            if (nBins != _spectrumBins) {
                _recentSpectra.clear();
                _spectrumBins = nBins;
            }
            _recentSpectra.insert(_recentSpectra.end(), rows.data(), rows.data() + rows.size());
            ++_framesSeen;
        }
        if (!_recentSpectra.empty()) {
            const auto        averagedFrames = std::max(1UZ, static_cast<std::size_t>(std::ceil(kBandAverageSeconds * static_cast<double>(frame_rate))));
            const std::size_t keptValues     = averagedFrames * _elements * _spectrumBins;
            if (_recentSpectra.size() > keptValues) {
                _recentSpectra.erase(_recentSpectra.begin(), _recentSpectra.end() - static_cast<std::ptrdiff_t>(keptValues));
            }
        }
        std::ignore              = frameSpectra.consume(frameSpectra.size());
        const auto redrawStarted = std::chrono::steady_clock::now();
        if (redrawStarted - _printed >= kRedrawPeriod && !_recentSpectra.empty()) {
            _printed = redrawStarted;
            print();
            _silentReported.resize(_loudestSinceRedraw.size(), false);
            for (std::size_t m = 0UZ; m < _loudestSinceRedraw.size(); ++m) {
                const double silenceLimit = kSilentMeanBinPower * static_cast<double>(gr::algorithm::fft::samplesOf<float>(_spectrumBins)) / 1024.0;
                if (_loudestSinceRedraw[m] < silenceLimit && !_silentReported[m]) {
                    _silentReported[m] = true;
                    std::println(stderr, "input {} is silent: check the device's channel mapping or the cable", m);
                }
            }
            _loudestSinceRedraw.clear();
        }
        return gr::work::Status::OK;
    }

    void print() {
        const double receivedNyquistHz = static_cast<double>(sample_rate) / 2.0;
        const double fMinHz            = _bandHz[0UZ];
        const double fMaxHz            = std::min(_bandHz[1UZ], receivedNyquistHz);
        if (fMinHz >= fMaxHz) {
            this->emitErrorMessage("bands", std::format("--band {:.0f}..{:.0f} Hz lies above the received Nyquist frequency of {:.0f} Hz", _bandHz[0UZ], _bandHz[1UZ], receivedNyquistHz));
            this->requestStop();
            return;
        }
        const std::size_t fftSamples      = gr::algorithm::fft::samplesOf<float>(_spectrumBins);
        const auto        requiredSamples = gr::localisation::bandFrameSamples(static_cast<double>(sample_rate), fMinHz, fMaxHz);
        if (!requiredSamples || *requiredSamples > fftSamples) {
            const std::string error = requiredSamples ? std::format("received {:.0f} Hz needs {} FFT samples for the lowest band; this run has {}", static_cast<double>(sample_rate), *requiredSamples, fftSamples) : std::string(requiredSamples.error());
            this->emitErrorMessage("bands", error);
            this->requestStop();
            return;
        }
        const std::size_t nFrames   = _recentSpectra.size() / (_elements * _spectrumBins);
        const auto        edges     = gr::localisation::halfOctaveEdges(fMinHz, fMaxHz);
        const std::size_t nBands    = edges.size() - 1UZ;
        const double      binHz     = static_cast<double>(sample_rate) / static_cast<double>(fftSamples);
        const auto        bandPower = gr::localisation::bandSteeredPower(_recentSpectra, _elements, _spectrumBins, binHz, _bearingPairDelays, edges);
        const std::array  broadbandEdges{fMinHz, fMaxHz};
        const auto        broadband = gr::localisation::bandSteeredPower(_recentSpectra, _elements, _spectrumBins, binHz, _bearingPairDelays, broadbandEdges);

        constexpr std::size_t kHeatmapAxisRows = 4UZ;
        const double          bandTop          = std::max(static_cast<double>(std::ranges::max(bandPower)), gr::localisation::kMinVerdictPeak);

        const std::pair   bearingBounds{-gr::localisation::kFrontAngleLimitDegrees, gr::localisation::kFrontAngleLimitDegrees};
        const std::tuple  bandBounds{bearingBounds, std::pair{fMinHz, fMaxHz}};
        const std::size_t plotRows = std::max(2UZ, nBands + (nBands + 1UZ) / 2UZ);

        auto bandChart        = gr::graphs::ImChart<kChartColumns, std::dynamic_extent, gr::graphs::LinearAxisTransform, gr::graphs::LogAxisTransform>(bandBounds, kChartColumns, plotRows + kHeatmapAxisRows);
        bandChart.axis_name_x = "bearing [deg], right positive";
        bandChart.axis_name_y = "frequency [Hz], log";
        bandChart.reset();
        if (const auto rendered = bandChart.drawHeatmap(std::span<const float>(bandPower), nBands, _bearingsDegrees.size(), {0.0, bandTop}, {}, _heatmapStyle, stdout, edges); !rendered) {
            this->emitErrorMessage("heatmap", rendered.error());
            this->requestStop();
            return;
        }
        const double observedSeconds = static_cast<double>(fftSamples) / static_cast<double>(sample_rate) + static_cast<double>(nFrames - 1UZ) / static_cast<double>(frame_rate);
        std::println("bands {:.0f}..{:.0f} Hz, {} half-octaves (last clipped) | SRP-PHAT, 1: coherent | {} pairs, {} frames spanning {:.3f} s\033[K", fMinHz, fMaxHz, nBands, gr::algorithm::pairsOf(_elements), nFrames, observedSeconds);
        std::println("FFT {} samples, {:.3f} Hz/bin, {:.3f} s/window | input 0 left, input 1 right\033[K", fftSamples, binHz, static_cast<double>(fftSamples) / static_cast<double>(sample_rate));

        const std::vector<double> broadbandPower(broadband.begin(), broadband.end());
        const auto [lowest, peak]                                  = std::ranges::minmax_element(broadbandPower);
        const auto                                     peakBearing = _bearingsDegrees[static_cast<std::size_t>(peak - broadbandPower.begin())];
        const double                                   bottom      = std::min(0.0, *lowest);
        const double                                   top         = 1.05 * std::max(*peak, gr::localisation::kMinVerdictPeak);
        gr::graphs::ImChart<kChartColumns, kChartRows> projection({{-gr::localisation::kFrontAngleLimitDegrees, gr::localisation::kFrontAngleLimitDegrees}, {bottom, top}});
        projection.axis_name_x = "bearing [deg]";
        projection.axis_name_y = "broadband SRP-PHAT";
        projection.draw(_bearingsDegrees, broadbandPower, "mean over frequency bins");
        projection.draw<gr::graphs::Style::Marker>(std::vector{peakBearing}, std::vector{*peak}, "peak");
        projection.draw(std::nullopt);
        std::println("{}\033[K", gr::localisation::bearingVerdict(broadband));

        std::println("input {:<60.60} {:6.0f} Hz received, {:.0f} Hz requested\033[K", _inputDescription, static_cast<double>(sample_rate), _requestedSampleRateHz);
        const auto truth = [&] {
            if (_sweepSceneSamples == 0UZ) {
                return _truth;
            }
            const double latestFrameCentreSample = static_cast<double>(_framesSeen) * static_cast<double>(sample_rate) / static_cast<double>(frame_rate);
            const double bearing                 = gr::testing::sweepBearingRadians(static_cast<std::size_t>(latestFrameCentreSample), _sweepSceneSamples);
            return std::vector<KnownSource>{{0.0, gr::algorithm::degreesOf(kPi / 2.0 - bearing), 1.0}};
        }();
        for (std::size_t k = 0UZ; k < truth.size(); ++k) {
            const auto angles = gr::localisation::frontAnglesDegrees({std::cos(gr::algorithm::radiansOf(truth[k].azimuthDegrees)), 0.0, std::sin(gr::algorithm::radiansOf(truth[k].azimuthDegrees))});
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
    std::array<double, 2> bandHz{gr::localisation::kMinBandHz, kSampleRate / 2.0};
    if (!bandText.empty()) {
        const auto parseHz = [](std::string_view text, double& hz) {
            const auto parsed = std::from_chars(text.data(), text.data() + text.size(), hz);
            return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() && std::isfinite(hz);
        };
        const std::string_view argument = bandText;
        const std::size_t      colon    = argument.find(':');
        const bool             parsed   = colon != std::string_view::npos && parseHz(argument.substr(0UZ, colon), bandHz[0UZ]) && parseHz(argument.substr(colon + 1UZ), bandHz[1UZ]);
        if (!parsed || bandHz[0UZ] <= 0.0 || bandHz[0UZ] >= bandHz[1UZ] || bandHz[1UZ] > kSampleRate / 2.0) {
            std::println(stderr, "--band: use f_min:f_max in Hz with 0 < f_min < f_max <= {:.0f}", kSampleRate / 2.0);
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
        std::println(stderr, "the bands require a line array");
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

    const auto frame = gr::testing::correlationFrameFor(gr::algorithm::apertureOf(*positions), kSpeedOfSound, kSampleRate);
    if (!frame) {
        std::println(stderr, "{}", frame.error());
        return 1;
    }
    const auto bandSamples = gr::localisation::bandFrameSamples(kSampleRate, bandHz[0UZ], bandHz[1UZ]);
    if (!bandSamples) {
        std::println(stderr, "{}", bandSamples.error());
        return 1;
    }
    const gr::Size_t frameSamples = static_cast<gr::Size_t>(std::max(static_cast<std::size_t>(frame->samples), *bandSamples));
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
        auto&     report              = graph.emplaceBlock<LineBands>({{"frame_rate", static_cast<float>(kSampleRate / hopSamples)}});
        report._elements              = M;
        report._requestedSampleRateHz = kSampleRate;
        report._heatmapStyle          = *heatmapStyle;
        report._bandHz                = bandHz;
        report._bearingsDegrees       = axisOf(-gr::localisation::kFrontAngleLimitDegrees, azimuthSteps);
        report._bearingPairDelays     = gr::localisation::bearingPairDelays(std::span<const double>(positions->data(), positions->size()), report._bearingsDegrees, kSpeedOfSound);
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

        report._inputDescription = synthetic ? gr::testing::syntheticInputDescription() : gr::testing::liveInputDescription(device);
        auto& stft               = graph.emplaceBlock<gr::blocks::fft::STFT<float>>({{"compute_domain", std::string(domain)}});
        std::ignore              = graph.connect<"out", "in">(frames, stft);
        std::ignore              = graph.connect<"out", "spectra">(stft, report);
        return graph;
    };

    gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::multiThreaded> scheduler;
    return gr::testing::runExample(buildGraph, gr::testing::chooseExampleDomain(noGpu), seconds, scheduler);
}
