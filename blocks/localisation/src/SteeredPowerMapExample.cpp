#include "SteeredPowerMapExample.hpp"
#include "SteeredPowerRange.hpp"

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
#include <gnuradio-4.0/localisation/PairCrossSpectra.hpp>
#include <gnuradio-4.0/localisation/SourceLocation.hpp>
#include <gnuradio-4.0/math/TensorPeakFinder.hpp>
#include <gnuradio-4.0/testing/ArrayScene.hpp>
#include <gnuradio-4.0/testing/ExampleRun.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace {

constexpr double      kPi                      = std::numbers::pi;
constexpr double      kLineSampleRateHz        = 96000.0;
constexpr std::size_t kLineDetectors           = 2UZ;
constexpr double      kSpeedOfSound            = gr::testing::kSpeedOfSoundInAir;
constexpr std::size_t kChartRows               = 14UZ;
constexpr auto        kProjectionsRedrawPeriod = std::chrono::seconds(1);
constexpr auto        kHeatmapRedrawPeriod     = std::chrono::milliseconds(500);
constexpr double      kBandAverageSeconds      = 0.1;
/** mean |X_k|^2 of an input's spectrum: a 1024-sample Hann frame of noise at a 24-bit converter's least significant bit (2^-24 rms) gives 1.4e-12 */
constexpr double kSilentMeanBinPower = 1e-12;

struct KnownSource {
    double elevationDegrees;
    double azimuthDegrees;
    double gain;
    double distanceMetres = std::numeric_limits<double>::infinity();
};

/** the noise heard by an element at `x` m on the line while the source's bearing sweeps from -kSweepBearingLimitDegrees to +kSweepBearingLimitDegrees; the right (positive x) hears it early */
[[nodiscard]] std::vector<float> sweepingNoise(double x, std::size_t nSamples, double sampleRate) {
    constexpr std::size_t kMargin = gr::testing::kMaxArrivalDelaySamples;
    const auto            noise   = gr::testing::whiteNoise(nSamples + 2UZ * kMargin, 21U);
    std::vector<float>    heard(nSamples);
    for (std::size_t n = 0UZ; n < nSamples; ++n) {
        const double azimuth  = kPi / 2.0 - gr::examples::sweepBearingRadians(n, nSamples);
        const double position = static_cast<double>(n + kMargin) + x * std::cos(azimuth) / kSpeedOfSound * sampleRate;
        heard[n]              = static_cast<float>(gr::testing::interpolateAt(noise, position));
    }
    return heard;
}

struct MapReport : gr::Block<MapReport> {
    using Description = gr::Doc<"Plots steered-power maps as projections, angular heatmaps, a time waterfall or bearing x frequency bands.">;

    gr::TensorPortIn<float, gr::Optional>               map;
    gr::TensorPortIn<float, gr::Optional>               peaks;
    gr::TensorPortIn<float, gr::Optional>               lagCurves;
    gr::TensorPortIn<float, gr::Optional>               located;
    gr::TensorPortIn<std::complex<float>, gr::Optional> spectra;
    gr::SampleRate                                      sample_rate = 0.f;
    gr::FrameRate                                       frame_rate  = 0.f;
    GR_MAKE_REFLECTABLE(MapReport, map, peaks, lagCurves, located, spectra, sample_rate, frame_rate);

    std::string                       _inputDescription;
    gr::examples::MapDisplay          _display      = gr::examples::MapDisplay::projections;
    gr::graphs::HeatmapStyle          _heatmapStyle = gr::graphs::HeatmapStyle::braille;
    gr::examples::SteeredPowerHistory _history;
    double                            _colourLimit = 1.0;

    gr::examples::SteeredPowerRange       _rangeScan;
    std::vector<float>                    _latestLagCurves;
    std::size_t                           _lagLength = 0UZ;
    std::vector<double>                   _azimuthDegrees;
    std::vector<double>                   _elevationDegrees;
    std::vector<KnownSource>              _truth;
    std::vector<float>                    _latestMap;
    std::vector<std::vector<float>>       _latestPeaks;
    std::vector<float>                    _latestLocated;
    std::size_t                           _elements              = 0UZ;
    std::size_t                           _framesSeen            = 0UZ;
    std::size_t                           _sweepSceneSamples     = 0UZ;
    gr::Size_t                            _maxLagSamples         = 0U;
    double                                _requestedSampleRateHz = 0.0;
    std::chrono::steady_clock::time_point _printed{};

    std::vector<double>              _bearingsDegrees;
    std::vector<double>              _bearingPairDelays; /**< [bearings, P] in s */
    std::array<double, 2>            _bandHz{};
    std::vector<std::complex<float>> _recentSpectra; /**< [frames, M, _spectrumBins] */
    std::size_t                      _spectrumBins = 0UZ;
    std::vector<double>              _loudestSinceRedraw; /**< per input, the largest mean bin power of a frame */
    std::vector<bool>                _silentReported;

    void start() { gr::graphs::resetView(); }

    gr::work::Status processBulk(gr::InputSpanLike auto& maps, gr::InputSpanLike auto& found, gr::InputSpanLike auto& correlations, gr::InputSpanLike auto& directions, gr::InputSpanLike auto& frameSpectra) {
        const bool        rangeView = _display == gr::examples::MapDisplay::heatmap;
        const bool        bandsView = _display == gr::examples::MapDisplay::bands;
        const std::size_t nMaps     = std::min(maps.size(), found.size());
        if (_display == gr::examples::MapDisplay::waterfall) {
            for (std::size_t frame = 0UZ; frame < nMaps; ++frame) {
                const auto& frameMap = maps[frame];
                if (const auto appended = _history.append(std::span<const float>(frameMap.data(), frameMap.size()), frame_rate); !appended) {
                    this->emitErrorMessage("waterfall", appended.error());
                    this->requestStop();
                    return gr::work::Status::ERROR;
                }
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
            if (bandsView) {
                if (nBins != _spectrumBins) {
                    _recentSpectra.clear();
                    _spectrumBins = nBins;
                }
                _recentSpectra.insert(_recentSpectra.end(), rows.data(), rows.data() + rows.size());
                ++_framesSeen;
            }
        }
        if (bandsView && !_recentSpectra.empty()) {
            const auto        averagedFrames = std::max(1UZ, static_cast<std::size_t>(std::ceil(kBandAverageSeconds * static_cast<double>(frame_rate))));
            const std::size_t keptValues     = averagedFrames * _elements * _spectrumBins;
            if (_recentSpectra.size() > keptValues) {
                _recentSpectra.erase(_recentSpectra.begin(), _recentSpectra.end() - static_cast<std::ptrdiff_t>(keptValues));
            }
        }
        std::ignore = frameSpectra.consume(frameSpectra.size());
        _framesSeen += rangeView ? correlations.size() : nMaps;
        if (rangeView) {
            std::ignore = correlations.consume(correlations.size());
        } else {
            std::ignore = maps.consume(nMaps);
            std::ignore = found.consume(nMaps);
            std::ignore = directions.consume(directions.size());
            if (_display == gr::examples::MapDisplay::waterfall) {
                std::ignore = correlations.consume(correlations.size());
            }
        }
        const auto redrawPeriod  = _display == gr::examples::MapDisplay::projections ? kProjectionsRedrawPeriod : kHeatmapRedrawPeriod;
        const auto redrawStarted = std::chrono::steady_clock::now();
        if (redrawStarted - _printed >= redrawPeriod && hasFrameToPrint()) {
            _printed = redrawStarted;
            print();
            _silentReported.resize(_loudestSinceRedraw.size(), false);
            for (std::size_t m = 0UZ; m < _loudestSinceRedraw.size(); ++m) {
                const double silenceLimit = kSilentMeanBinPower * (bandsView ? static_cast<double>(gr::algorithm::fft::samplesOf<float>(_spectrumBins)) / 1024.0 : 1.0);
                if (_loudestSinceRedraw[m] < silenceLimit && !_silentReported[m]) {
                    _silentReported[m] = true;
                    std::println(stderr, "input {} is silent: check the device's channel mapping or the cable", m);
                }
            }
            _loudestSinceRedraw.clear();
        }
        return gr::work::Status::OK;
    }

    [[nodiscard]] bool hasFrameToPrint() const noexcept {
        switch (_display) {
        case gr::examples::MapDisplay::projections:
        case gr::examples::MapDisplay::waterfall: return !_latestMap.empty();
        case gr::examples::MapDisplay::heatmap: return !_latestLagCurves.empty();
        case gr::examples::MapDisplay::bands: return !_recentSpectra.empty();
        }
        std::unreachable();
    }

    void print() {
        if (!hasFrameToPrint()) {
            return;
        }
        if (_display == gr::examples::MapDisplay::bands) {
            const double receivedNyquistHz = static_cast<double>(sample_rate) / 2.0;
            const double fMinHz            = _bandHz[0UZ];
            const double fMaxHz            = std::min(_bandHz[1UZ], receivedNyquistHz);
            if (fMinHz >= fMaxHz) {
                this->emitErrorMessage("bands", std::format("--band {:.0f}..{:.0f} Hz lies above the received Nyquist frequency of {:.0f} Hz", _bandHz[0UZ], _bandHz[1UZ], receivedNyquistHz));
                this->requestStop();
                return;
            }
            const std::size_t fftSamples      = gr::algorithm::fft::samplesOf<float>(_spectrumBins);
            const auto        requiredSamples = gr::examples::bandFrameSamples(static_cast<double>(sample_rate), fMinHz, fMaxHz);
            if (!requiredSamples || *requiredSamples > fftSamples) {
                const std::string error = requiredSamples ? std::format("received {:.0f} Hz needs {} FFT samples for the lowest band; this run has {}", static_cast<double>(sample_rate), *requiredSamples, fftSamples) : std::string(requiredSamples.error());
                this->emitErrorMessage("bands", error);
                this->requestStop();
                return;
            }
            const std::size_t nFrames   = _recentSpectra.size() / (_elements * _spectrumBins);
            const auto        edges     = gr::examples::halfOctaveEdges(fMinHz, fMaxHz);
            const std::size_t nBands    = edges.size() - 1UZ;
            const double      binHz     = static_cast<double>(sample_rate) / static_cast<double>(gr::algorithm::fft::samplesOf<float>(_spectrumBins));
            const auto        bandPower = gr::examples::bandSteeredPower(_recentSpectra, _elements, _spectrumBins, binHz, _bearingPairDelays, edges);
            const std::array  broadbandEdges{fMinHz, fMaxHz};
            const auto        broadband = gr::examples::bandSteeredPower(_recentSpectra, _elements, _spectrumBins, binHz, _bearingPairDelays, broadbandEdges);

            constexpr std::size_t kHeatmapAxisRows = 4UZ;
            const double          bandTop          = std::max(static_cast<double>(std::ranges::max(bandPower)), gr::examples::kMinVerdictPeak);

            const std::pair   bearingBounds{-gr::examples::kFrontAngleLimitDegrees, gr::examples::kFrontAngleLimitDegrees};
            const std::tuple  bandBounds{bearingBounds, std::pair{fMinHz, fMaxHz}};
            const std::size_t plotRows = std::max(2UZ, nBands + (nBands + 1UZ) / 2UZ);

            auto bandChart        = gr::graphs::ImChart<gr::examples::kMapChartColumns, std::dynamic_extent, gr::graphs::LinearAxisTransform, gr::graphs::LogAxisTransform>(bandBounds, gr::examples::kMapChartColumns, plotRows + kHeatmapAxisRows);
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
            const auto [lowest, peak]                                                   = std::ranges::minmax_element(broadbandPower);
            const auto                                                      peakBearing = _bearingsDegrees[static_cast<std::size_t>(peak - broadbandPower.begin())];
            const double                                                    bottom      = std::min(0.0, *lowest);
            const double                                                    top         = 1.05 * std::max(*peak, gr::examples::kMinVerdictPeak);
            gr::graphs::ImChart<gr::examples::kMapChartColumns, kChartRows> projection({{-gr::examples::kFrontAngleLimitDegrees, gr::examples::kFrontAngleLimitDegrees}, {bottom, top}});
            projection.axis_name_x = "bearing [deg]";
            projection.axis_name_y = "broadband SRP-PHAT";
            projection.draw(_bearingsDegrees, broadbandPower, "mean over frequency bins");
            projection.draw<gr::graphs::Style::Marker>(std::vector{peakBearing}, std::vector{*peak}, "peak");
            projection.draw(std::nullopt);
            std::println("{}\033[K", gr::examples::bearingVerdict(broadband));
        }
        gr::examples::SteeredPowerRangeScan rangeResult;
        if (_display == gr::examples::MapDisplay::heatmap) {
            _rangeScan.sampleRateHz = static_cast<double>(sample_rate.value);
            auto scanned            = gr::examples::scanSteeredPowerRange(_rangeScan, _latestLagCurves, _lagLength);
            if (!scanned) {
                this->emitErrorMessage("range scan", scanned.error());
                this->requestStop();
                return;
            }
            rangeResult = std::move(*scanned);
            _latestMap  = rangeResult.angularPower;
            _latestPeaks.clear();
            for (const auto& profile : rangeResult.profiles) {
                _latestPeaks.push_back({static_cast<float>(profile.elevationRadians), static_cast<float>(profile.azimuthRadians), profile.score});
            }
        }
        const bool        camera        = !_elevationDegrees.empty();
        const std::size_t nAzimuth      = _azimuthDegrees.size();
        const std::size_t nElevation    = camera ? _elevationDegrees.size() : 1UZ;
        const auto        displayAngles = [camera](double azimuthRadians, double elevationRadians) {
            const auto direction = camera ? gr::algorithm::directionOf(azimuthRadians, elevationRadians) : std::array{std::cos(azimuthRadians), 0.0, std::sin(azimuthRadians)};
            return gr::examples::frontAnglesDegrees(direction);
        };
        const auto frontMap = [&] {
            if (camera) {
                return gr::examples::projectFrontPower(_latestMap, nAzimuth, nElevation);
            }
            gr::examples::FrontPowerMap result;
            result.power         = _latestMap | std::views::reverse | std::ranges::to<std::vector>();
            result.anglesDegrees = _azimuthDegrees | std::views::reverse | std::views::transform([](double azimuth) { return gr::examples::kFrontAngleLimitDegrees - azimuth; }) | std::ranges::to<std::vector>();
            return result;
        }();
        const std::size_t frontColumns = frontMap.anglesDegrees.size();
        const std::size_t frontRows    = camera ? frontColumns : 1UZ;
        const bool        waterfall    = _display == gr::examples::MapDisplay::waterfall;
        if (waterfall && _lagLength > 0UZ) {
            const double                                                    maxLagMs = 1000.0 * static_cast<double>(_maxLagSamples) / static_cast<double>(sample_rate);
            gr::graphs::ImChart<gr::examples::kMapChartColumns, kChartRows> lagChart({{-maxLagMs, maxLagMs}, {-0.25, 1.1}});
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
        if (_display == gr::examples::MapDisplay::heatmap || waterfall) {
            const double                                                             yMin     = waterfall ? std::min(_history.ageSeconds(0UZ), -1.0 / static_cast<double>(frame_rate)) : -gr::examples::kFrontAngleLimitDegrees;
            const double                                                             yMax     = waterfall ? 0.0 : gr::examples::kFrontAngleLimitDegrees;
            const std::size_t                                                        plotRows = waterfall ? _history.rows.size() : frontRows;
            gr::graphs::ImChart<gr::examples::kMapChartColumns, std::dynamic_extent> chart({{-gr::examples::kFrontAngleLimitDegrees, gr::examples::kFrontAngleLimitDegrees}, {yMin, yMax}}, gr::examples::kMapChartColumns, waterfall ? kChartRows : gr::examples::kAngularHeatmapRows);
            chart.axis_name_x = "horizontal bearing [deg]";
            chart.axis_name_y = waterfall ? "age [s]" : "vertical elevation [deg]";
            std::vector<float> values;
            if (waterfall) {
                values.reserve(plotRows * frontColumns);
                for (const auto& row : _history.rows) {
                    values.insert(values.end(), row.rbegin(), row.rend());
                }
            }
            const std::span<const float>       field(waterfall ? values : frontMap.power);
            std::vector<std::array<double, 2>> markers;
            for (const auto& peak : _latestPeaks) {
                const auto angles = displayAngles(static_cast<double>(peak[camera ? 1UZ : 0UZ]), camera ? static_cast<double>(peak[0UZ]) : 0.0);
                markers.push_back({angles[0UZ], waterfall ? 0.0 : angles[1UZ]});
            }
            if (!waterfall) {
                chart.reset();
            }
            const auto displayRange = [&] {
                if (waterfall) {
                    return std::pair{-_colourLimit, _colourLimit};
                }
                const auto bounds = std::ranges::minmax(_latestMap);
                return std::pair<double, double>{bounds.min, bounds.max};
            }();
            if (const auto rendered = chart.drawHeatmap(field, plotRows, frontColumns, displayRange, markers, _heatmapStyle); !rendered) {
                this->emitErrorMessage("heatmap", rendered.error());
                this->requestStop();
                return;
            }
            if (waterfall) {
                std::println("waterfall: {} frames at {:.3f} Hz; broadside 0 deg, right positive; elevation is not measured", _history.rows.size(), static_cast<float>(frame_rate));
            } else {
                std::println("map   {:<10} | H/V +/-90 deg | bins 5 deg | relative maxima\033[K", gr::meta::enumName(_heatmapStyle).value());
            }
        }
        if (_display == gr::examples::MapDisplay::projections || _display == gr::examples::MapDisplay::heatmap) {
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
            const double                                                    bottom = static_cast<double>(*std::ranges::min_element(_latestMap));
            const double                                                    top    = static_cast<double>(std::max(*std::ranges::max_element(_latestMap), std::numeric_limits<float>::min()));
            gr::graphs::ImChart<gr::examples::kMapChartColumns, kChartRows> chart({{-gr::examples::kFrontAngleLimitDegrees, gr::examples::kFrontAngleLimitDegrees}, {std::min(0.0, bottom), 1.05 * top}});
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
            if (_display == gr::examples::MapDisplay::projections) {
                chart.reset();
            }
            chart.draw(std::nullopt);
        }
        if (_display == gr::examples::MapDisplay::heatmap) {
            std::println("scan  {:5.2f}..{:5.2f} m + far | H/V: max over distance and other angle\033[K", _rangeScan.minDistanceMetres, _rangeScan.maxDistanceMetres);
            gr::graphs::ImChart<gr::examples::kMapChartColumns, kChartRows, gr::graphs::LogAxisTransform> chart({{_rangeScan.minDistanceMetres, _rangeScan.maxDistanceMetres}, {-_colourLimit, _colourLimit}});
            chart.axis_name_x       = "distance from array centre [m]";
            chart.axis_name_y       = "steered power";
            const auto distanceAxis = rangeResult.distancesMetres | std::views::transform([](double radius) { return static_cast<float>(radius); }) | std::ranges::to<std::vector>();
            for (std::size_t k = 0UZ; k < rangeResult.profiles.size(); ++k) {
                chart.draw(distanceAxis, rangeResult.profiles[k].power, std::format("peak {}", k + 1UZ));
            }
            chart.draw(std::nullopt);
            std::println("range fixed direction | MAX: finite maximum | EDGE: boundary | FAR: far wins; flat = unresolved\033[K");
        }
        std::println("input {:<60.60} {:6.0f} Hz received, {:.0f} Hz requested\033[K", _inputDescription, static_cast<double>(sample_rate), _requestedSampleRateHz);
        const auto truth = [&] {
            if (_sweepSceneSamples == 0UZ) {
                return _truth;
            }
            const double latestFrameCentreSample = static_cast<double>(_framesSeen) * static_cast<double>(sample_rate) / static_cast<double>(frame_rate);
            const double bearing                 = gr::examples::sweepBearingRadians(static_cast<std::size_t>(latestFrameCentreSample), _sweepSceneSamples);
            return std::vector<KnownSource>{{0.0, gr::algorithm::degreesOf(kPi / 2.0 - bearing), 1.0}};
        }();
        const auto errorColumn = [&](double azimuthRadians, double elevationRadians) {
            if (truth.empty()) {
                return std::string{};
            }
            const auto estimate = gr::algorithm::directionOf(azimuthRadians, elevationRadians);
            double     nearest  = std::numeric_limits<double>::infinity();
            for (const auto& source : truth) {
                nearest = std::min(nearest, gr::examples::angularSeparationDegrees(estimate, gr::algorithm::directionOf(gr::algorithm::radiansOf(source.azimuthDegrees), gr::algorithm::radiansOf(source.elevationDegrees))));
            }
            return std::format(" | err {:5.1f} deg", nearest);
        };
        for (std::size_t k = 0UZ; k < _latestPeaks.size(); ++k) {
            const auto&       peak          = _latestPeaks[k];
            const double      peakAzimuth   = static_cast<double>(peak[camera ? 1UZ : 0UZ]);
            const double      peakElevation = camera ? static_cast<double>(peak[0UZ]) : 0.0;
            const auto        angles        = displayAngles(peakAzimuth, peakElevation);
            const std::string error         = errorColumn(peakAzimuth, peakElevation);
            const std::size_t peakNumber    = _display == gr::examples::MapDisplay::projections ? k : k + 1UZ;
            if (_display == gr::examples::MapDisplay::heatmap) {
                const auto&       profile   = rangeResult.profiles[k];
                const auto        best      = std::ranges::max_element(profile.power);
                const std::size_t index     = static_cast<std::size_t>(best - profile.power.begin());
                const bool        boundary  = index == 0UZ || index + 1UZ == profile.power.size();
                const char*       rangeKind = [&] {
                    if (profile.farPower >= *best) {
                        return "FAR";
                    }
                    return boundary ? "EDGE" : "MAX";
                }();
                std::println("peak {:1} H {:+6.1f} V {:+6.1f} deg | rMax {:5.2f} m {:4} | Pmax {:7.2f} far {:7.2f}{}\033[K", peakNumber, angles[0UZ], angles[1UZ], rangeResult.distancesMetres[index], rangeKind, *best, profile.farPower, error);
            } else {
                std::println("peak {:1} H {:+6.1f} V {:+6.1f} deg | power {:7.2f}{}\033[K", peakNumber, angles[0UZ], angles[1UZ], static_cast<double>(peak[camera ? 2UZ : 1UZ]), error);
            }
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

int gr::examples::runSteeredPowerMapExample(int argc, char** argv, MapDisplay display) {
    std::string device;
    bool        synthetic     = false;
    bool        noGpu         = false;
    bool        arrayGiven    = false;
    double      seconds       = 0.0;
    double      spacing       = 0.0;
    const bool  lineExample   = display == MapDisplay::waterfall;
    std::size_t detectors     = lineExample ? kLineDetectors : 8UZ;
    gr::Size_t  rows          = 4U;
    gr::Size_t  columns       = 4U;
    const auto  defaultLayout = [display] {
        switch (display) {
        case MapDisplay::projections: return "grid";
        case MapDisplay::heatmap: return "grid";
        case MapDisplay::waterfall:
        case MapDisplay::bands: return "line";
        }
        std::unreachable();
    };
    std::string                   layout = defaultLayout();
    std::string                   positionsText;
    SteeredPowerRange             rangeScan;
    std::string                   mapStyle    = "braille";
    std::string                   lineDisplay = "bands";
    std::string                   bandText;
    gr::testing::ExampleArguments args(argc, argv);
    while (args.next()) {
        const bool heatmap     = display == MapDisplay::heatmap;
        const bool arrayOption = args.number("--spacing", spacing) || args.number("--detectors", detectors) || args.number("--rows", rows) || args.number("--columns", columns) || args.text("--layout", layout) || args.text("--positions", positionsText);
        const bool known       = arrayOption || args.flag("--synthetic", synthetic) || args.flag("--nogpu", noGpu) || args.number("--seconds", seconds) || args.text("--device", device)                      //
                           || (display != MapDisplay::projections && args.text("--map-style", mapStyle))                                                                                                      //
                           || (heatmap && (args.number("--range-min", rangeScan.minDistanceMetres) || args.number("--range-max", rangeScan.maxDistanceMetres) || args.number("--peaks", rangeScan.maxPeaks))) //
                           || (lineExample && (args.text("--display", lineDisplay) || args.text("--band", bandText)));
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
    if (lineExample) {
        const auto chosen = gr::meta::parseEnum<MapDisplay>(lineDisplay);
        if (!chosen || (*chosen != MapDisplay::bands && *chosen != MapDisplay::waterfall)) {
            std::println(stderr, "--display: use bands or waterfall");
            return 1;
        }
        display = *chosen;
    }
    const double          sampleRate = lineExample ? kLineSampleRateHz : gr::testing::kExampleSampleRateHz;
    std::array<double, 2> bandHz{display == MapDisplay::bands ? kMinBandHz : 0.0, sampleRate / 2.0};
    if (!bandText.empty()) {
        const auto parseHz = [](std::string_view text, double& hz) {
            const auto parsed = std::from_chars(text.data(), text.data() + text.size(), hz);
            return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() && std::isfinite(hz);
        };
        const std::string_view argument = bandText;
        const std::size_t      colon    = argument.find(':');
        const bool             parsed   = colon != std::string_view::npos && parseHz(argument.substr(0UZ, colon), bandHz[0UZ]) && parseHz(argument.substr(colon + 1UZ), bandHz[1UZ]);
        if (!parsed || bandHz[0UZ] < 0.0 || (display == MapDisplay::bands && bandHz[0UZ] == 0.0) || bandHz[0UZ] >= bandHz[1UZ] || bandHz[1UZ] > sampleRate / 2.0) {
            std::println(stderr, "--band: use f_min:f_max in Hz with 0 {} f_min < f_max <= {:.0f}", display == MapDisplay::bands ? "<" : "<=", sampleRate / 2.0);
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

    const auto deviceArray = synthetic || arrayGiven || lineExample ? std::nullopt : gr::testing::microphoneArray(device);
    if (deviceArray) {
        layout    = deviceArray->layout;
        detectors = deviceArray->elements;
        rows      = static_cast<gr::Size_t>(deviceArray->rows);
        columns   = static_cast<gr::Size_t>(deviceArray->columns);
        spacing   = deviceArray->spacing;
    }

    const bool camera = layout != "line";
    if ((display == MapDisplay::heatmap && !camera) || (lineExample && camera)) {
        if (!deviceArray) {
            std::println(stderr, "the angular heatmap requires a planar array; the bands and the waterfall require a line array");
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

    if (display == MapDisplay::heatmap) {
        rangeScan.propagationSpeedMetresPerSecond = kSpeedOfSound;
        for (std::size_t m = 0UZ; m < M; ++m) {
            rangeScan.positionsMetres.push_back({(*positions)[m, 0UZ], (*positions)[m, 1UZ], (*positions)[m, 2UZ]});
        }
        if (const auto valid = rangeScan.validate(); !valid) {
            std::println(stderr, "{}", valid.error());
            return 1;
        }
    }
    const auto frame = gr::testing::correlationFrameFor(gr::algorithm::apertureOf(*positions), kSpeedOfSound, sampleRate);
    if (!frame) {
        std::println(stderr, "{}", frame.error());
        return 1;
    }
    const auto bandSamples = display == MapDisplay::bands ? bandFrameSamples(sampleRate, bandHz[0UZ], bandHz[1UZ]) : std::expected<std::size_t, std::string_view>(0UZ);
    if (!bandSamples) {
        std::println(stderr, "{}", bandSamples.error());
        return 1;
    }
    const gr::Size_t frameSamples = static_cast<gr::Size_t>(std::max(static_cast<std::size_t>(frame->samples), *bandSamples));
    const gr::Size_t maxLag       = frame->maxLag;
    const gr::Size_t hopSamples   = frameSamples / 2U;
    if (!gr::testing::kLiveAudio) {
        synthetic = true;
    }
    const bool sweep = !camera && M == 2UZ;
    if (synthetic && sweep) {
        std::println("input: synthesised audio; repeating {:.0f} s scene, one source sweeping from {:+.0f} to {:+.0f} deg bearing", gr::testing::kExampleSceneSeconds, -gr::examples::kSweepBearingLimitDegrees, gr::examples::kSweepBearingLimitDegrees);
    } else if (synthetic) {
        std::println("input: synthesised audio; repeating {:.0f} s scene with interpolated delays", gr::testing::kExampleSceneSeconds);
    }

    const auto buildGraph = [&](std::string_view domain) -> std::expected<gr::Graph, std::string> {
        gr::Graph graph;
        auto&     frames              = graph.emplaceBlock<gr::basic::StreamsToTensor<float>>({{"n_ports", gr::Size_t(M)}, {"size", frameSamples}, {"stride", hopSamples}, {"compute_domain", std::string(domain)}});
        auto&     report              = graph.emplaceBlock<MapReport>({{"frame_rate", static_cast<float>(sampleRate / hopSamples)}});
        report._display               = display;
        report._elements              = M;
        report._maxLagSamples         = maxLag;
        report._requestedSampleRateHz = sampleRate;
        report._heatmapStyle          = *heatmapStyle;
        report._rangeScan             = rangeScan;
        report._colourLimit           = static_cast<double>(M * (M - 1UZ) / 2UZ);
        report._azimuthDegrees        = axisOf(azimuthStart, azimuthSteps);
        report._elevationDegrees      = camera ? axisOf(0.0, elevationSteps) : std::vector<double>{};
        report._bandHz                = bandHz;
        if (display == MapDisplay::bands) {
            report._bearingsDegrees   = axisOf(-kFrontAngleLimitDegrees, azimuthSteps);
            report._bearingPairDelays = gr::examples::bearingPairDelays(std::span<const double>(positions->data(), positions->size()), report._bearingsDegrees, kSpeedOfSound);
        }
        if (!synthetic) {
            if (const auto connected = gr::testing::connectMicrophones(graph, frames, device, M, sampleRate); !connected) {
                return std::unexpected(connected.error());
            }
        }
        if (synthetic) {
            const auto nSamples = static_cast<std::size_t>(gr::testing::kExampleSceneSeconds * sampleRate);
            if (sweep) {
                report._sweepSceneSamples = nSamples;
            } else {
                report._truth = camera ? std::vector<KnownSource>{{40.0, 30.0, 1.0}, {25.0, -70.0, 0.7}} : std::vector<KnownSource>{{0.0, 40.0, 1.0}, {0.0, 120.0, 0.7}};
                if (display == MapDisplay::heatmap) {
                    report._truth.front().distanceMetres = 0.3;
                }
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
                    arrivals.push_back({.delaySamples = relativeDelay * sampleRate, .gain = source.gain, .seed = 21U * (1U + static_cast<std::uint32_t>(k))});
                }
                std::vector<float> noisySamples = sweep ? sweepingNoise(position[0UZ], nSamples, sampleRate) : gr::testing::delayedNoises(arrivals, nSamples);
                gr::testing::addMicrophoneNoise(noisySamples, m);
                gr::testing::addElementSource(graph, frames, m, std::move(noisySamples), sampleRate);
            }
        }

        report._inputDescription = synthetic ? gr::testing::syntheticInputDescription() : gr::testing::liveInputDescription(device);
        if (lineExample) {
            auto& stft  = graph.emplaceBlock<gr::blocks::fft::STFT<float>>({{"compute_domain", std::string(domain)}});
            std::ignore = graph.connect<"out", "in">(frames, stft);
            std::ignore = graph.connect<"out", "spectra">(stft, report);
        }
        if (display == MapDisplay::bands) {
            return graph;
        }
        auto& lagCurve = graph.emplaceBlock<gr::localisation::PairCrossCorrelation<float>>({{"max_lag", maxLag}, {"f_min", static_cast<float>(bandHz[0UZ])}, {"f_max", static_cast<float>(bandHz[1UZ])}, {"compute_domain", std::string(domain)}});
        std::ignore    = graph.connect<"out", "in">(frames, lagCurve);
        if (display == MapDisplay::heatmap) {
            std::ignore = graph.connect<"out", "lagCurves">(lagCurve, report);
        } else {
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
            if (display == MapDisplay::waterfall) {
                std::ignore = graph.connect<"out", "lagCurves">(lagCurve, report);
            }
        }
        return graph;
    };

    gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::multiThreaded> scheduler;
    return gr::testing::runExample(buildGraph, gr::testing::chooseExampleDomain(noGpu), seconds, scheduler);
}
