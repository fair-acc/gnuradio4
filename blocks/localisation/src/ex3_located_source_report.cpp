/**
 * The source that explains the pair delays: GCC-PHAT pair delays -> least-squares direction (`--field far`) or Gauss-Newton position (`near`).
 * Live: the sound card's channels placed by `--layout` line (along x), grid (x-y plane, facing +z) or tetrahedron (4 channels),
 * or without array options by the device's channel count (2 a stereo line left and right, 4 a tetrahedron facing the source, 16 the UMA-16, other k² a k × k grid, otherwise a line);
 * with `--synthetic`, or without an audio backend: one noise source at a known direction or position, charted over time against the result with the RMS delay residual and the iterations.
 * A line cannot tell the sides of its axis apart, a grid those of its plane (+z is taken).
 * Usage: ex3_located_source_report [--synthetic] [--nogpu] [--field far|near] [--layout line|grid|tetrahedron|circle|circle-centre|uma16] [--detectors 4] [--rows 4] [--columns 4] [--spacing 0] [--positions "x,y,z;..."] [--seconds N]
 */

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <deque>
#include <expected>
#include <format>
#include <functional>
#include <limits>
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
#include <gnuradio-4.0/localisation/FrontProjection.hpp>
#include <gnuradio-4.0/localisation/PairCrossSpectra.hpp>
#include <gnuradio-4.0/localisation/SourceLocation.hpp>
#include <gnuradio-4.0/testing/ArrayScene.hpp>
#include <gnuradio-4.0/testing/ExampleRun.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace {

constexpr double      kSampleRate    = gr::testing::kExampleSampleRateHz;
constexpr double      kSpeedOfSound  = gr::testing::kSpeedOfSoundInAir;
constexpr std::size_t kChartColumns  = 100UZ;
constexpr std::size_t kChartRows     = 14UZ;
constexpr std::size_t kHistoryFrames = 240UZ;
constexpr auto        kRedrawPeriod  = std::chrono::milliseconds(500);

/** the fitted source per frame: its display angles (far) or coordinates (near), with the latest fit, its residual and the truth as rows */
struct LocatedReport : gr::Block<LocatedReport> {
    using Description = gr::Doc<"Plots fitted source candidates over time and reports their delay residuals.">;

    gr::TensorPortIn<float> located;
    gr::FrameRate           frame_rate = 0.f;
    GR_MAKE_REFLECTABLE(LocatedReport, located, frame_rate);

    std::string _inputDescription;

    gr::algorithm::Field                  _field     = gr::algorithm::Field::far;
    bool                                  _lineArray = false;
    std::vector<double>                   _truth;
    std::deque<std::vector<double>>       _history;
    std::size_t                           _frames = 0UZ;
    std::chrono::steady_clock::time_point _printed{};

    void start() { gr::graphs::resetView(); }

    void stop() { print(); }

    gr::work::Status processBulk(gr::InputSpanLike auto& rows) {
        for (std::size_t k = 0UZ; k < rows.size(); ++k) {
            const std::size_t columns = rows[k].extent(1UZ);
            _history.emplace_back(rows[k].data(), rows[k].data() + columns);
            if (_history.size() > kHistoryFrames) {
                _history.pop_front();
            }
            ++_frames;
        }
        std::ignore = rows.consume(rows.size());
        if (std::chrono::steady_clock::now() - _printed >= kRedrawPeriod && _frames > 0UZ) {
            print();
            _printed = std::chrono::steady_clock::now();
        }
        return gr::work::Status::OK;
    }

    /** far: the display angles (bearing, elevation) of a row's (elevation, azimuth), a line's cone angle as the bearing with broadside zero; near: its (x, y, z) from the row's (z, y, x) */
    [[nodiscard]] std::array<double, 3> shownOf(std::span<const double> coordinates) const {
        if (_field == gr::algorithm::Field::far) {
            const auto direction = _lineArray ? std::array{std::cos(coordinates[1]), 0.0, std::sin(coordinates[1])} : gr::algorithm::directionOf(coordinates[1], coordinates[0]);
            const auto angles    = gr::localisation::frontAnglesDegrees(direction);
            return {angles[0], angles[1], 0.0};
        }
        return {coordinates[2], coordinates[1], coordinates[0]};
    }

    void print() const {
        if (_history.empty()) {
            return;
        }
        const bool                            far   = _field == gr::algorithm::Field::far;
        const std::size_t                     D     = far ? 2UZ : 3UZ;
        const std::size_t                     n     = _history.size();
        const double                          span  = static_cast<double>(n - 1UZ) / static_cast<double>(frame_rate);
        const std::array<std::string_view, 3> names = far ? std::array<std::string_view, 3>{"H [deg]", "V [deg]", ""} : std::array<std::string_view, 3>{"x [m]", "y [m]", "z [m]"};
        std::vector<double>                   ageSeconds(n);
        std::array<std::vector<double>, 3>    shown;
        for (std::size_t i = 0UZ; i < n; ++i) {
            ageSeconds[i]  = -span + static_cast<double>(i) / static_cast<double>(frame_rate);
            const auto xyz = shownOf(_history[i]);
            for (std::size_t c = 0UZ; c < 3UZ; ++c) {
                shown[c].push_back(xyz[c]);
            }
        }
        const std::array<double, 3> truthShown = _truth.empty() ? std::array<double, 3>{} : shownOf(_truth);
        const auto                  yRange     = [&]() -> std::pair<double, double> {
            if (far) {
                return {-gr::localisation::kFrontAngleLimitDegrees, gr::localisation::kFrontAngleLimitDegrees};
            }
            double low  = _truth.empty() ? std::numeric_limits<double>::infinity() : std::ranges::min(truthShown);
            double high = _truth.empty() ? -std::numeric_limits<double>::infinity() : std::ranges::max(truthShown);
            for (const auto& series : shown) {
                low  = std::min(low, std::ranges::min(series));
                high = std::max(high, std::ranges::max(series));
            }
            const double margin = std::max(0.1, 0.1 * (high - low));
            return {low - margin, high + margin};
        }();
        gr::graphs::ImChart<kChartColumns, kChartRows> chart({{std::min(-span, -1.0 / static_cast<double>(frame_rate)), 0.0}, yRange});
        chart.axis_name_x = "age [s]";
        chart.axis_name_y = far ? "front-centred angle [deg]" : "position [m]";
        for (std::size_t c = 0UZ; c < (far ? 2UZ : 3UZ); ++c) {
            chart.draw(ageSeconds, shown[c], names[c]);
            if (!_truth.empty()) {
                chart.draw(ageSeconds, std::vector<double>(n, truthShown[c]), std::format("truth {}", names[c].substr(0UZ, 1UZ)));
            }
        }
        chart.reset();
        chart.draw(std::nullopt);
        const auto& latest = _history.back();
        const auto  fitted = shownOf(latest);
        std::println("input {:<78.78} {:7.0f} Hz[K", _inputDescription, kSampleRate);
        const std::string error = [&] {
            if (_truth.empty()) {
                return std::string{};
            }
            if (far) {
                return std::format(" | err {:5.1f} deg", gr::localisation::angularSeparationDegrees(gr::algorithm::directionOf(latest[1], latest[0]), gr::algorithm::directionOf(_truth[1], _truth[0])));
            }
            return std::format(" | err {:5.2f} m", std::hypot(fitted[0] - truthShown[0], fitted[1] - truthShown[1], fitted[2] - truthShown[2]));
        }();
        if (far) {
            std::println("delay   H {:+6.1f} V {:+6.1f} deg | residual {:6.3f} ms | iterations {:2} | rank {:1}{}[K", fitted[0], fitted[1], 1e3 * latest[D], static_cast<int>(latest[D + 1UZ]), static_cast<int>(latest[D + 2UZ]), error);
        } else {
            std::println("point   x {:+6.2f} y {:+6.2f} z {:+6.2f} m | residual {:6.3f} ms | iterations {:2} | rank {:1}{}[K", fitted[0], fitted[1], fitted[2], 1e3 * latest[D], static_cast<int>(latest[D + 1UZ]), static_cast<int>(latest[D + 2UZ]), error);
        }
        if (!_truth.empty()) {
            if (far) {
                std::println("truth   H {:+6.1f} V {:+6.1f} deg[K", truthShown[0], truthShown[1]);
            } else {
                std::println("truth   x {:+6.2f} y {:+6.2f} z {:+6.2f} m[K", truthShown[0], truthShown[1], truthShown[2]);
            }
        }
        std::println("frames {} of the last {} shown[K", _frames, n);
        std::print("[J");
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
    std::size_t                   detectors  = 4UZ;
    gr::Size_t                    rows       = 4U;
    gr::Size_t                    columns    = 4U;
    std::string                   layout     = "tetrahedron";
    std::string                   positionsText;
    std::string                   field = "far";
    gr::testing::ExampleArguments args(argc, argv);
    while (args.next()) {
        const bool arrayOption = args.number("--spacing", spacing) || args.number("--detectors", detectors) || args.number("--rows", rows) || args.number("--columns", columns) || args.text("--layout", layout) || args.text("--positions", positionsText);
        const bool known       = arrayOption || args.flag("--synthetic", synthetic) || args.flag("--nogpu", noGpu) || args.number("--seconds", seconds) || args.text("--field", field) || args.text("--device", device);
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

    const bool line = layout == "line";
    if (!(field == "far" || field == "near")) {
        std::println(stderr, "--field {}: use far or near", field);
        return 1;
    }
    const bool far       = field == "far";
    const auto positions = positionsText.empty() ? gr::algorithm::geometry::layoutOf(layout, detectors, rows, columns, spacing) : gr::algorithm::geometry::parsePositions(positionsText);
    const auto elements  = positions.and_then(gr::algorithm::placedElements);
    if (!elements) {
        std::println(stderr, "array refused: {}", elements.error());
        return 1;
    }
    const std::size_t M = *elements;

    const auto frame = gr::testing::correlationFrameFor(gr::algorithm::apertureOf(*positions), kSpeedOfSound, kSampleRate);
    if (!frame) {
        std::println(stderr, "{}", frame.error());
        return 1;
    }
    if (!gr::testing::kLiveAudio) {
        synthetic = true;
    }
    if (synthetic) {
        std::println("input: synthesised audio; repeating {:.0f} s scene with interpolated delays", gr::testing::kExampleSceneSeconds);
        if (!far) {
            std::println("near-field candidates can be ambiguous: four detectors can admit two positions with identical pair delays");
        }
    }

    const auto buildGraph = [&](std::string_view domain) -> std::expected<gr::Graph, std::string> {
        gr::Graph              graph;
        const gr::property_map geometry{{"element_positions", *positions}, {"propagation_speed", kSpeedOfSound}, {"field", field}};
        auto&                  locate     = graph.emplaceBlock<gr::localisation::SourceFromDelays<float>>(geometry);
        const gr::Size_t       hopSamples = frame->samples / 2U;
        auto&                  frames     = graph.emplaceBlock<gr::basic::StreamsToTensor<float>>({{"n_ports", gr::Size_t(M)}, {"size", frame->samples}, {"stride", hopSamples}, {"compute_domain", std::string(domain)}});
        auto&                  report     = graph.emplaceBlock<LocatedReport>({{"frame_rate", static_cast<float>(kSampleRate / hopSamples)}});
        report._field                     = far ? gr::algorithm::Field::far : gr::algorithm::Field::near;
        report._lineArray                 = line;
        if (!synthetic) {
            if (const auto connected = gr::testing::connectMicrophones(graph, frames, device, M, kSampleRate); !connected) {
                return std::unexpected(connected.error());
            }
        }
        if (synthetic) {
            const double                elevation = line ? 0.0 : gr::algorithm::radiansOf(30.0);
            const double                azimuth   = gr::algorithm::radiansOf(50.0);
            const std::array<double, 3> position{0.3, 0.5, line ? 0.0 : 0.4};
            const std::array<double, 3> point = far ? gr::algorithm::directionOf(azimuth, elevation) : position;
            report._truth                     = far ? std::vector{elevation, azimuth} : std::vector{position[2], position[1], position[0]};
            const auto nSamples               = static_cast<std::size_t>(gr::testing::kExampleSceneSeconds * kSampleRate);
            for (std::size_t m = 0UZ; m < M; ++m) {
                const gr::testing::Arrival arrival{.delaySamples = gr::algorithm::delayOf<double>(positions->data() + 3UZ * m, point, locate.field.value, kSpeedOfSound) * kSampleRate, .gain = 1.0, .seed = 21U};
                std::vector<float>         noisySamples = gr::testing::delayedNoises(std::span(&arrival, 1UZ), nSamples);
                gr::testing::addMicrophoneNoise(noisySamples, m);
                gr::testing::addElementSource(graph, frames, m, std::move(noisySamples), kSampleRate);
            }
        }

        report._inputDescription = synthetic ? gr::testing::syntheticInputDescription() : gr::testing::liveInputDescription(device);
        auto& lagCurve           = graph.emplaceBlock<gr::localisation::PairCrossCorrelation<float>>({{"max_lag", frame->maxLag}, {"compute_domain", std::string(domain)}});
        auto& delay              = graph.emplaceBlock<gr::localisation::PairDelay<float>>({{"max_lag", frame->maxLag}, {"compute_domain", std::string(domain)}});
        std::ignore              = graph.connect<"out", "in">(frames, lagCurve);
        std::ignore              = graph.connect<"out", "in">(lagCurve, delay);
        std::ignore              = graph.connect<"out", "in">(delay, locate);
        std::ignore              = graph.connect<"out", "located">(locate, report);
        return graph;
    };

    gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::multiThreaded> scheduler;
    return gr::testing::runExample(buildGraph, gr::testing::chooseExampleDomain(noGpu), seconds, scheduler);
}
