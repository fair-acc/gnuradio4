/**
 * Delays between the detectors of an array: GCC lag curve of every pair -> peak delay and confidence, redrawn each second; the array's aperture sets the lag range.
 * Live: the sound card's input channels placed by the array options or, without them, by its channel count (2 a stereo line, 4 a tetrahedron, k² a k × k grid, otherwise a line);
 * with `--synthetic`, or without an audio backend: a line of 4 hearing noise from a known azimuth, the measured delays printed against the analytic tau_i - tau_j (positive: detector i hears later; azimuth from +x towards +y).
 * Usage: ex1_pair_delays [--synthetic] [--nogpu] [--device selector] [--layout line|grid|tetrahedron|circle|circle-centre] [--detectors 4] [--rows 4] [--columns 4] [--spacing 0] [--positions "x,y,z;..."] [--weighting none|phat] [--azimuth 60] [--seconds N]
 *        spacing 0 is the layout default
 */

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <expected>
#include <format>
#include <numbers>
#include <optional>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/algorithm/ArrayGeometry.hpp>
#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/basic/StreamToTensor.hpp>
#include <gnuradio-4.0/localisation/PairCrossSpectra.hpp>
#include <gnuradio-4.0/testing/ArrayScene.hpp>
#include <gnuradio-4.0/testing/ExampleRun.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace {

constexpr double      kSampleRate   = gr::testing::kExampleSampleRateHz;
constexpr double      kSpeedOfSound = gr::testing::kSpeedOfSoundInAir;
constexpr std::size_t kPlottedPairs = 6UZ;
constexpr std::size_t kChartColumns = 100UZ;
constexpr std::size_t kChartRows    = 14UZ;
constexpr auto        kRedrawPeriod = std::chrono::seconds(1);

/** keeps the latest frame's lag curves [P,L] and pair delays [P,3] = {delay in s, peak, confidence} */
struct DelayReport : gr::Block<DelayReport> {
    using Description = gr::Doc<"Plots pair correlation curves and reports measured and geometric delays.">;

    gr::TensorPortIn<float> lags;
    gr::TensorPortIn<float> delays;
    gr::SampleRate          sample_rate = static_cast<float>(kSampleRate);
    GR_MAKE_REFLECTABLE(DelayReport, lags, delays, sample_rate);

    std::string _inputDescription;

    std::size_t                           _detectors = 0UZ;
    std::vector<double>                   _analyticSeconds;
    std::vector<float>                    _latestLags;
    std::vector<float>                    _latestDelays;
    std::vector<double>                   _delaySum;
    std::size_t                           _frames    = 0UZ;
    std::size_t                           _lagLength = 0UZ;
    std::chrono::steady_clock::time_point _printed{};

    void start() { gr::graphs::resetView(); }

    void stop() { print(); }

    gr::work::Status processBulk(gr::InputSpanLike auto& curves, gr::InputSpanLike auto& peaks) {
        const std::size_t nFrames = std::min(curves.size(), peaks.size());
        for (std::size_t k = 0UZ; k < nFrames; ++k) {
            const auto& peak = peaks[k];
            _delaySum.resize(peak.size(), 0.0);
            std::ranges::transform(_delaySum, std::span(peak.data(), peak.size()), _delaySum.begin(), [](double sum, float value) { return sum + static_cast<double>(value); });
            ++_frames;
        }
        if (nFrames > 0UZ) {
            const auto& curve = curves[nFrames - 1UZ];
            const auto& peak  = peaks[nFrames - 1UZ];
            _latestLags.assign(curve.data(), curve.data() + curve.size());
            _lagLength = curve.extent(1UZ);
            _latestDelays.assign(peak.data(), peak.data() + peak.size());
        }
        std::ignore = curves.consume(nFrames);
        std::ignore = peaks.consume(nFrames);
        if (std::chrono::steady_clock::now() - _printed >= kRedrawPeriod && !_latestDelays.empty()) {
            print();
            _printed = std::chrono::steady_clock::now();
        }
        return gr::work::Status::OK;
    }

    void print() const {
        if (_lagLength == 0UZ) {
            return;
        }
        const std::vector<double> lagMs                       = gr::testing::lagAxisMilliseconds(_lagLength, static_cast<double>(sample_rate));
        const auto [lowest, highest]                          = std::ranges::minmax_element(_latestLags);
        const double                                   margin = 0.05 * static_cast<double>(*highest - *lowest);
        gr::graphs::ImChart<kChartColumns, kChartRows> chart({{lagMs.front(), lagMs.back()}, {static_cast<double>(*lowest) - margin, static_cast<double>(*highest) + margin}});
        chart.axis_name_x = "lag [ms], positive: first detector of the pair hears later";
        chart.axis_name_y = "cross-correlation";
        for (std::size_t p = 0UZ; p < std::min(kPlottedPairs, _latestLags.size() / _lagLength); ++p) {
            const auto curve = std::span(_latestLags).subspan(p * _lagLength, _lagLength);
            chart.draw(lagMs, std::vector<double>(curve.begin(), curve.end()), std::format("pair {}", p));
        }
        chart.reset();
        chart.draw();
        std::println("input: {}", _inputDescription);
        std::size_t p = 0UZ;
        for (std::size_t i = 0UZ; i < _detectors; ++i) {
            for (std::size_t j = i + 1UZ; j < _detectors && p < _latestDelays.size() / 3UZ; ++j, ++p) {
                const std::string analytic    = p < _analyticSeconds.size() ? std::format(", analytic {:+8.4f} ms", 1000.0 * _analyticSeconds[p]) : std::string{};
                const double      meanSeconds = _delaySum[3UZ * p] / static_cast<double>(_frames);
                std::println("pair {}-{}: delay {:+8.4f} ms (latest {:+8.4f}){}, confidence {:6.2f}", i, j, 1000.0 * meanSeconds, 1000.0 * static_cast<double>(_latestDelays[3UZ * p]), analytic, static_cast<double>(_latestDelays[3UZ * p + 2UZ]));
            }
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
    double                        azimuth    = 60.0;
    std::size_t                   detectors  = 4UZ;
    gr::Size_t                    rows       = 4U;
    gr::Size_t                    columns    = 4U;
    std::string                   layout     = "line";
    std::string                   positionsText;
    std::string                   weighting = "phat";
    gr::testing::ExampleArguments args(argc, argv);
    while (args.next()) {
        const bool arrayOption = args.number("--spacing", spacing) || args.number("--detectors", detectors) || args.number("--rows", rows) || args.number("--columns", columns) || args.text("--layout", layout) || args.text("--positions", positionsText);
        const bool known       = arrayOption || args.flag("--synthetic", synthetic) || args.flag("--nogpu", noGpu) || args.number("--seconds", seconds) || args.number("--azimuth", azimuth) || args.text("--weighting", weighting) || args.text("--device", device);
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
    const bool            weightingValid     = weighting == "none" || weighting == "phat";
    if (!args.valid || !durationValid || !detectorCountValid || !gridFits || spacing < 0.0 || !weightingValid) {
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
    const double azimuthRadians = gr::algorithm::radiansOf(azimuth);
    const auto   positions      = positionsText.empty() ? gr::algorithm::geometry::layoutOf(layout, detectors, rows, columns, spacing) : gr::algorithm::geometry::parsePositions(positionsText);
    const auto   elements       = positions.and_then(gr::algorithm::placedElements);
    if (!elements) {
        std::println(stderr, "array refused: {}", elements.error());
        return 1;
    }
    const std::size_t M     = *elements;
    const auto        frame = gr::testing::correlationFrameFor(gr::algorithm::apertureOf(*positions), kSpeedOfSound, kSampleRate);
    if (!frame) {
        std::println(stderr, "{}", frame.error());
        return 1;
    }
    if (!gr::testing::kLiveAudio) {
        synthetic = true;
    }
    if (synthetic) {
        std::println("input: synthesised audio; repeating {:.0f} s scene with interpolated delays", gr::testing::kExampleSceneSeconds);
        std::println("synthetic scene: noise from azimuth {:.1f} deg on {} detectors, {:.3g} m aperture", azimuth, M, gr::algorithm::apertureOf(*positions));
    }

    const auto buildGraph = [&](std::string_view domain) -> std::expected<gr::Graph, std::string> {
        gr::Graph graph;
        auto&     frames  = graph.emplaceBlock<gr::basic::StreamsToTensor<float>>({{"n_ports", gr::Size_t(M)}, {"size", frame->samples}, {"stride", frame->samples / 2U}, {"compute_domain", std::string(domain)}});
        auto&     report  = graph.emplaceBlock<DelayReport>();
        report._detectors = M;
        if (!synthetic) {
            if (const auto connected = gr::testing::connectMicrophones(graph, frames, device, M, kSampleRate); !connected) {
                return std::unexpected(connected.error());
            }
        }
        if (synthetic) {
            const auto                  nSamples  = static_cast<std::size_t>(gr::testing::kExampleSceneSeconds * kSampleRate);
            const std::array<double, 3> direction = gr::algorithm::directionOf(azimuthRadians, 0.0);
            for (std::size_t m = 0UZ; m < M; ++m) {
                const double               delaySeconds = gr::algorithm::delayOf<double>(positions->data() + 3UZ * m, direction, gr::algorithm::Field::far, kSpeedOfSound);
                const gr::testing::Arrival arrival{.delaySamples = delaySeconds * kSampleRate, .gain = 1.0, .seed = 21U};
                std::vector<float>         noisySamples = gr::testing::delayedNoises(std::span(&arrival, 1UZ), nSamples);
                gr::testing::addMicrophoneNoise(noisySamples, m);
                gr::testing::addElementSource(graph, frames, m, std::move(noisySamples), kSampleRate);
            }
            for (std::size_t p = 0UZ; p < gr::algorithm::pairsOf(M); ++p) {
                report._analyticSeconds.push_back(gr::algorithm::pairDelayOf(positions->data(), gr::algorithm::pairOf(p, M), direction, gr::algorithm::Field::far, kSpeedOfSound));
            }
        }

        report._inputDescription = synthetic ? gr::testing::syntheticInputDescription() : gr::testing::liveInputDescription(device);
        auto& lagCurve           = graph.emplaceBlock<gr::localisation::PairCrossCorrelation<float>>({{"max_lag", frame->maxLag}, {"weighting", weighting}, {"compute_domain", std::string(domain)}});
        auto& delay              = graph.emplaceBlock<gr::localisation::PairDelay<float>>({{"max_lag", frame->maxLag}, {"compute_domain", std::string(domain)}});
        std::ignore              = graph.connect<"out", "in">(frames, lagCurve);
        std::ignore              = graph.connect<"out", "in">(lagCurve, delay);
        std::ignore              = graph.connect<"out", "lags">(lagCurve, report);
        std::ignore              = graph.connect<"out", "delays">(delay, report);
        return graph;
    };

    gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::multiThreaded> scheduler;
    return gr::testing::runExample(buildGraph, gr::testing::chooseExampleDomain(noGpu), seconds, scheduler);
}
