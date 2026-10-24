// Locates sound sources with an array and listens to each: GCC-PHAT -> SRP-PHAT -> TensorPeakFinder gives their directions, SteeringVectors -> MVDR one beam per direction.
// Live: the sound card's M input channels placed by `--layout`: `line` (default) along x, front and back alike, the azimuth from 0 to pi;
// `grid` with `--rows` x `--columns` = M, an acoustic camera over the grid's front hemisphere; `tetrahedron` (M = 4), an [El,Az] map over the full sphere; `--spacing` in m, 0 for the layout's default.
// With fewer than two channels, without an audio backend, or with `--synthetic`: a 2x2 grid of 4 m pitch and two noise sources at azimuth 0 and pi/2, in units where 1 m is one sample at 1 kHz.
// The per-frame peaks feed a Histogram2D that forgets over seconds; its peaks (TensorPeakFinder) are the persistent items: the sources and their reflections.
// `--nogpu` keeps the array blocks on the host instead of gpu:sycl.

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
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/basic/StreamToTensor.hpp>
#include <gnuradio-4.0/beamforming/Beamformer.hpp>
#include <gnuradio-4.0/beamforming/CrossSpectralMatrix.hpp>
#include <gnuradio-4.0/fourier/STFT.hpp>
#include <gnuradio-4.0/localisation/PairCrossSpectra.hpp>
#include <gnuradio-4.0/localisation/SourceLocation.hpp>
#include <gnuradio-4.0/math/Histogram.hpp>
#include <gnuradio-4.0/math/TensorPeakFinder.hpp>
#include <gnuradio-4.0/testing/ArrayScene.hpp>
#include <gnuradio-4.0/testing/ExampleRun.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/testing/TensorSource.hpp>

namespace {

constexpr gr::Size_t kSyntheticFrameSamples = 256U;

struct Options {
    std::string device;
    bool        synthetic = false;
    bool        noGpu     = false;
    std::size_t channels  = 2UZ;
    std::string layout    = "line";
    std::string positionsText;
    gr::Size_t  rows    = 4U;
    gr::Size_t  columns = 4U;
    double      spacing = 0.0;
    double      seconds = 0.0;
};

/** the scan of the map, slowest dimension first, in rad: [Az] for a line or the synthetic scene, [El,Az] for a grid or a tetrahedron */
struct Scan {
    std::vector<double>     minima;
    std::vector<double>     maxima;
    std::vector<gr::Size_t> steps;
};

/**
 * plots, each second and at the end: the latest map's azimuth (and, for an [El,Az] map, elevation) power curve and the first pairs' lag curves,
 * then the latest sources with their beam levels and the persistent items
 */
struct Report : gr::Block<Report> {
    gr::TensorPortIn<float>               peaks;
    gr::TensorPortIn<std::complex<float>> beams;
    gr::TensorPortIn<float>               map;
    gr::TensorPortIn<float, gr::Async>    items;
    gr::TensorPortIn<float>               lags;
    GR_MAKE_REFLECTABLE(Report, peaks, beams, map, items, lags);

    std::string _inputDescription;

    static constexpr std::size_t kChartColumns = 100UZ;
    static constexpr std::size_t kChartRows    = 14UZ;
    static constexpr std::size_t kPlottedPairs = 6UZ;

    Scan                                  _scan;
    double                                _sampleRate = 1.0;
    std::vector<std::vector<float>>       _found;
    std::vector<double>                   _levels;
    std::vector<std::vector<float>>       _items;
    std::vector<float>                    _latestMap;
    std::vector<std::size_t>              _latestShape;
    std::vector<float>                    _latestLags;
    std::size_t                           _lagLength = 0UZ;
    std::chrono::steady_clock::time_point _printed{};

    void start() { gr::graphs::resetView(); }

    void stop() {
        print();
        if (_levels.empty()) {
            std::println("no beam levels available; capture more data after a source is found");
        }
    }

    gr::work::Status processBulk(gr::InputSpanLike auto& found, gr::InputSpanLike auto& beamSpectra, gr::InputSpanLike auto& maps, gr::InputSpanLike auto& persistent, gr::InputSpanLike auto& lagCurves) {
        const auto rowsOf = [](const auto& tensor) {
            std::vector<std::vector<float>> rows;
            for (std::size_t k = 0UZ; tensor.rank() == 2UZ && k < tensor.extent(0UZ); ++k) {
                rows.emplace_back(tensor.data() + k * tensor.extent(1UZ), tensor.data() + (k + 1UZ) * tensor.extent(1UZ));
            }
            return rows;
        };
        if (found.size() > 0UZ) {
            _found = rowsOf(found[found.size() - 1UZ]);
        }
        if (persistent.size() > 0UZ) {
            _items = rowsOf(persistent[persistent.size() - 1UZ]);
        }
        std::ignore = persistent.consume(persistent.size());
        if (maps.size() > 0UZ) {
            const auto& latest = maps[maps.size() - 1UZ];
            _latestMap.assign(latest.data(), latest.data() + latest.size());
            _latestShape.assign(latest.extents().begin(), latest.extents().end());
        }
        if (lagCurves.size() > 0UZ) {
            const auto& latest = lagCurves[lagCurves.size() - 1UZ];
            _latestLags.assign(latest.data(), latest.data() + latest.size());
            _lagLength = latest.extent(1UZ);
        }
        if (beamSpectra.size() > 0UZ) {
            const auto&       latest = beamSpectra[beamSpectra.size() - 1UZ];
            const std::size_t K      = latest.extent(0UZ);
            const std::size_t F      = latest.extent(1UZ);
            _levels.assign(K, 0.0);
            for (std::size_t k = 0UZ; k < K; ++k) {
                double power = 0.0;
                for (std::size_t bin = 0UZ; bin < F; ++bin) {
                    power += static_cast<double>(std::norm(latest.data()[k * F + bin]));
                }
                _levels[k] = 10.0 * std::log10(power / static_cast<double>(F) + 1e-30);
            }
        }
        if (std::chrono::steady_clock::now() - _printed >= std::chrono::seconds(1) && !_levels.empty()) {
            print();
            _printed = std::chrono::steady_clock::now();
        }
        return gr::work::Status::OK;
    }

    void print() const {
        constexpr float kDeg        = 180.f / std::numbers::pi_v<float>;
        const bool      camera      = _latestShape.size() == 2UZ;
        const auto      toDoubles   = [](std::span<const float> values) { return std::vector<double>(values.begin(), values.end()); };
        const auto      axisDegrees = [](double low, double high, std::size_t nPoints) {
            const bool          fullTurn = high - low >= 2.0 * std::numbers::pi - 1e-9;
            const double        step     = (high - low) / static_cast<double>(fullTurn ? nPoints : nPoints - 1UZ);
            std::vector<double> axis(nPoints);
            for (std::size_t i = 0UZ; i < nPoints; ++i) {
                axis[i] = (low + step * static_cast<double>(i)) * 180.0 / std::numbers::pi;
            }
            return axis;
        };
        if (!_latestMap.empty()) {
            const std::size_t   nAzimuth   = _latestShape.back();
            const std::size_t   nElevation = camera ? _latestShape[0] : 1UZ;
            std::vector<double> azimuthPower(nAzimuth, 0.0);
            std::vector<double> elevationPower(nElevation, 0.0);
            for (std::size_t row = 0UZ; row < nElevation; ++row) {
                for (std::size_t column = 0UZ; column < nAzimuth; ++column) {
                    const double power   = static_cast<double>(_latestMap[row * nAzimuth + column]);
                    azimuthPower[column] = std::max(azimuthPower[column], power);
                    elevationPower[row]  = std::max(elevationPower[row], power);
                }
            }
            const auto                                     azimuthDegrees   = axisDegrees(_scan.minima.back(), _scan.maxima.back(), nAzimuth);
            const auto                                     elevationDegrees = camera ? axisDegrees(_scan.minima[0], _scan.maxima[0], nElevation) : std::vector<double>{};
            const double                                   xMin             = std::ranges::fold_left(elevationDegrees, std::ranges::fold_left(azimuthDegrees, std::numeric_limits<double>::infinity(), std::ranges::min), std::ranges::min);
            const double                                   xMax             = std::ranges::fold_left(elevationDegrees, std::ranges::fold_left(azimuthDegrees, -std::numeric_limits<double>::infinity(), std::ranges::max), std::ranges::max);
            const double                                   peak             = static_cast<double>(std::max(*std::ranges::max_element(_latestMap), std::numeric_limits<float>::min()));
            gr::graphs::ImChart<kChartColumns, kChartRows> mapChart({{xMin, xMax}, {0.0, 1.05 * peak}});
            mapChart.axis_name_x = "angle [deg]";
            mapChart.axis_name_y = "steered power";
            mapChart.draw(azimuthDegrees, azimuthPower, camera ? "azimuth (max over elevation)" : "azimuth");
            if (camera) {
                mapChart.draw(elevationDegrees, elevationPower, "elevation (max over azimuth)");
            }
            mapChart.reset();
            mapChart.draw();
        }
        if (_lagLength > 0UZ) {
            const std::vector<double>                      lagMs = gr::testing::lagAxisMilliseconds(_lagLength, _sampleRate);
            gr::graphs::ImChart<kChartColumns, kChartRows> lagChart({{lagMs.front(), lagMs.back()}, {-0.25, 1.1}});
            lagChart.axis_name_x = "lag [ms]";
            lagChart.axis_name_y = "GCC-PHAT";
            for (std::size_t p = 0UZ; p < std::min(kPlottedPairs, _latestLags.size() / _lagLength); ++p) {
                lagChart.draw(lagMs, toDoubles(std::span(_latestLags).subspan(p * _lagLength, _lagLength)), std::format("pair {}", p));
            }
            lagChart.draw();
        }
        std::println("input: {}", _inputDescription);
        const auto direction = [](float elevation, float azimuth) { return std::format("elevation {:6.1f} deg, azimuth {:7.1f} deg", elevation * kDeg, azimuth * kDeg); };
        for (std::size_t k = 0UZ; k < _found.size() && k < _levels.size(); ++k) {
            std::println("source {}: {}, beam level {:6.1f} dB", k, direction(camera ? _found[k][0] : 0.f, camera ? _found[k][1] : _found[k][0]), _levels[k]);
        }
        for (std::size_t k = 0UZ; k < _items.size(); ++k) {
            std::println("item {}: {}, histogram score {:.2f}", k, direction(camera ? _items[k][1] : 0.f, camera ? _items[k][0] : _items[k][1]), _items[k][2]);
        }
        std::print("\033[J");
    }
};

[[nodiscard]] std::expected<Options, std::string> parse(int argc, char** argv) {
    Options                       options;
    gr::testing::ExampleArguments args(argc, argv);
    while (args.next()) {
        const bool known = args.flag("--synthetic", options.synthetic) || args.flag("--nogpu", options.noGpu) || args.text("--device", options.device) || args.number("--channels", options.channels) || args.text("--layout", options.layout) //
                           || args.text("--positions", options.positionsText) || args.number("--rows", options.rows) || args.number("--columns", options.columns) || args.number("--spacing", options.spacing)                                 //
                           || args.number("--seconds", options.seconds);
        if (!known) {
            args.refuse();
        }
    }
    if (!args.valid || options.seconds < 0.0) {
        return std::unexpected("invalid arguments: supply finite values and non-negative seconds (0: unlimited)");
    }
    return options;
}

/**
 * the chain from frames [M,N] on, its lags from the array's aperture at `sampleRate`, the largest lag returned;
 * the per-frame peaks go into a histogram [Az,El] (2-D) or [score,Az] (1-D), padded by one bin per side so no item sits at its edge
 */
gr::Size_t addChain(gr::Graph& graph, auto& frames, const gr::property_map& geometry, const Scan& scan, double sampleRate, std::string_view inputDescription, std::string_view domain) {
    const std::string domainText(domain);
    gr::property_map  scanned = geometry;
    scanned.insert_or_assign("compute_domain", domainText);
    scanned.insert_or_assign("grid_min", gr::Tensor<double>(std::array<std::size_t, 1>{scan.minima.size()}, scan.minima));
    scanned.insert_or_assign("grid_max", gr::Tensor<double>(std::array<std::size_t, 1>{scan.maxima.size()}, scan.maxima));
    scanned.insert_or_assign("grid_steps", gr::Tensor<gr::Size_t>(std::array<std::size_t, 1>{scan.steps.size()}, scan.steps));
    auto&             srp         = graph.emplaceBlock<gr::localisation::SteeredPowerMap<float>>(scanned);
    const std::size_t M           = gr::algorithm::placedElements(srp.element_positions.value).value_or(0UZ);
    const auto        maxLag      = static_cast<gr::Size_t>(std::ceil(gr::algorithm::apertureOf(srp.element_positions.value) / srp.propagation_speed.value * sampleRate)) + 2U;
    const gr::Size_t  coordinates = static_cast<gr::Size_t>(scan.steps.size());
    const bool        camera      = coordinates == 2U;
    const double      pairs       = static_cast<double>(M * (M - 1UZ) / 2UZ);
    constexpr double  kPi         = std::numbers::pi;
    const auto        padded      = [](double low, double high, gr::Size_t bins) {
        const double step = (high - low) / static_cast<double>(bins);
        return std::array{low - step, high + step, static_cast<double>(bins + 2U)};
    };
    const auto       x          = camera ? padded(scan.minima[0], scan.maxima[0], 18U) : padded(scan.minima[0], scan.maxima[0], 72U);
    const auto       y          = camera ? padded(-kPi, kPi, 72U) : padded(0.0, pairs, 4U);
    const auto       names      = camera ? std::array{"azimuth", "elevation"} : std::array{"score", "azimuth"};
    const auto       units      = camera ? std::array{"rad", "rad"} : std::array{"a.u.", "rad"};
    gr::property_map binning    = {{"n_bins_x", static_cast<gr::Size_t>(x[2])}, {"n_bins_y", static_cast<gr::Size_t>(y[2])}, {"x_min", x[0]}, {"x_max", x[1]}, {"y_min", y[0]}, {"y_max", y[1]}, //
           {"weight_column", camera ? std::int32_t{2} : std::int32_t{-1}}, {"time_constant", 3.0}, {"n_samples", gr::Size_t(64)}, {"timeout", 1.0}, {"reset_on_publish", false},                 //
           {"axis_name", gr::Tensor<gr::pmt::Value>(gr::data_from, {gr::pmt::Value(names[0]), gr::pmt::Value(names[1])})}, {"axis_unit", gr::Tensor<gr::pmt::Value>(gr::data_from, {gr::pmt::Value(units[0]), gr::pmt::Value(units[1])})}};
    auto&            histogram  = graph.emplaceBlock<gr::blocks::math::Histogram2D<float>>(binning);
    auto&            itemFinder = graph.emplaceBlock<gr::blocks::math::TensorPeakFinder<float>>({{"max_peaks", gr::Size_t(4)}, {"threshold", 0.2f}, {"min_separation", 2.f}, {"compute_domain", domainText}});
    gr::property_map steered    = geometry;
    steered.insert_or_assign("coordinates", coordinates);
    steered.insert_or_assign("compute_domain", domainText);
    auto& gcc                = graph.emplaceBlock<gr::localisation::PairCrossCorrelation<float>>({{"max_lag", maxLag}, {"compute_domain", domainText}});
    auto& peaks              = graph.emplaceBlock<gr::blocks::math::TensorPeakFinder<float>>({{"max_peaks", gr::Size_t(2)}, {"threshold", 0.3f}, {"min_separation", camera ? 4.f : 20.f}, {"compute_domain", domainText}});
    auto& stft               = graph.emplaceBlock<gr::blocks::fft::STFT<float>>({{"compute_domain", domainText}});
    auto& covariance         = graph.emplaceBlock<gr::beamforming::CrossSpectralMatrix<float>>({{"alpha", 0.05f}, {"compute_domain", domainText}});
    auto& steering           = graph.emplaceBlock<gr::beamforming::SteeringVectors<float>>(steered);
    auto& mvdr               = graph.emplaceBlock<gr::beamforming::MinimumVarianceBeamformer<float>>({{"diagonal_loading", 0.05f}, {"compute_domain", domainText}});
    auto& report             = graph.emplaceBlock<Report>();
    report._scan             = scan;
    report._sampleRate       = sampleRate;
    report._inputDescription = inputDescription;
    std::ignore              = graph.connect<"out", "lags">(gcc, report);
    std::ignore              = graph.connect<"out", "map">(srp, report);
    std::ignore              = graph.connect<"out", "in">(frames, gcc);
    std::ignore              = graph.connect<"out", "in">(gcc, srp);
    std::ignore              = graph.connect<"out", "in">(srp, peaks);
    std::ignore              = graph.connect<"out", "peaks">(peaks, steering);
    std::ignore              = graph.connect<"out", "peaks">(peaks, report);
    std::ignore              = graph.connect<"out", "in">(frames, stft);
    std::ignore              = graph.connect<"out", "in">(stft, covariance);
    std::ignore              = graph.connect<"out", "in">(stft, steering);
    std::ignore              = graph.connect<"out", "in">(stft, mvdr);
    std::ignore              = graph.connect<"out", "steering">(steering, mvdr);
    std::ignore              = graph.connect<"out", "covariance">(covariance, mvdr);
    std::ignore              = graph.connect<"out", "beams">(mvdr, report);
    std::ignore              = graph.connect<"out", "points">(peaks, histogram);
    std::ignore              = graph.connect<"out", "in">(histogram, itemFinder);
    std::ignore              = graph.connect<"out", "items">(itemFinder, report);
    return maxLag;
}

/** noise from azimuth 0 and, 6 dB lower, from azimuth pi/2, heard by the four elements of gr::testing::kSquareArray */
void addSyntheticScene(gr::Graph& graph, std::string_view domain) {
    const std::size_t samples = static_cast<std::size_t>(gr::testing::kExampleSceneSeconds * 1000.0);
    auto&             frames  = graph.emplaceBlock<gr::basic::StreamsToTensor<float, gr::testing::kSquareArrayElements>>({{"size", kSyntheticFrameSamples}, {"stride", gr::Size_t(128)}, {"compute_domain", std::string(domain)}});
    for (std::size_t m = 0UZ; m < gr::testing::kSquareArrayElements; ++m) {
        std::vector<float> noisySamples = gr::testing::squareArraySamples(m, samples);
        gr::testing::addMicrophoneNoise(noisySamples, m);
        gr::testing::addElementSource(graph, frames, m, std::move(noisySamples), 1000.0);
    }
    std::ignore = addChain(graph, frames, gr::testing::squareArrayGeometry(), Scan{.minima = {-std::numbers::pi}, .maxima = {std::numbers::pi}, .steps = {360U}}, 1000.0, gr::testing::syntheticInputDescription(), domain);
}

} // namespace

int main(int argc, char** argv) {
    const auto parsed = parse(argc, argv);
    if (!parsed) {
        std::println(stderr, "{}", parsed.error());
        return 1;
    }
    const Options options  = *parsed;
    const auto    runGraph = [&options]<gr::scheduler::ExecutionPolicy execution>(const auto& buildGraph) -> int {
        constexpr std::size_t            maxWorkItems = execution == gr::scheduler::ExecutionPolicy::singleThreaded ? kSyntheticFrameSamples : std::numeric_limits<std::size_t>::max();
        gr::scheduler::Simple<execution> scheduler({{"max_work_items", static_cast<std::uint64_t>(maxWorkItems)}});
        return gr::testing::runExample(buildGraph, gr::testing::chooseExampleDomain(options.noGpu), options.seconds, scheduler);
    };
    if (gr::testing::kLiveAudio && !options.synthetic && options.channels >= 2UZ) {
        constexpr double  kSpeedOfSound = gr::testing::kSpeedOfSoundInAir;
        constexpr double  kRate         = gr::testing::kExampleSampleRateHz;
        constexpr double  kPi           = std::numbers::pi;
        const std::size_t M             = options.channels;
        const bool        line          = options.layout == "line";
        const bool        grid          = options.layout == "grid";
        const auto        positions     = options.positionsText.empty() ? gr::algorithm::geometry::layoutOf(options.layout, M, options.rows, options.columns, options.spacing) : gr::algorithm::geometry::parsePositions(options.positionsText);
        const auto        elements      = positions.and_then(gr::algorithm::placedElements);
        if (!elements || *elements != M) {
            std::println(stderr, "array refused for {} channels: {}", M, elements ? std::format("it places {} elements", *elements) : elements.error());
            return 1;
        }
        std::println("{} channels as a {}: {}", M, options.layout, line ? "azimuth 0..180 deg along +x, front and back alike" : (grid ? "an [El,Az] camera over the front hemisphere (+z)" : "an [El,Az] map over the full sphere"));
        return runGraph.template operator()<gr::scheduler::ExecutionPolicy::multiThreaded>([&](std::string_view domain) -> std::expected<gr::Graph, std::string> {
            gr::Graph graph;
            auto&     frames = graph.emplaceBlock<gr::basic::StreamsToTensor<float>>({{"n_ports", gr::Size_t(M)}, {"size", gr::Size_t(1024)}, {"stride", gr::Size_t(512)}, {"compute_domain", std::string(domain)}});
            if (const auto connected = gr::testing::connectMicrophones(graph, frames, options.device, M, kRate); !connected) {
                return std::unexpected(connected.error());
            }
            const gr::property_map geometry{{"element_positions", *positions}, {"propagation_speed", kSpeedOfSound}, {"field", std::string("far")}};
            const Scan             lineScan{.minima = {0.0}, .maxima = {kPi}, .steps = {181U}};
            const Scan             frontHemisphere{.minima = {0.0, -kPi}, .maxima = {kPi / 2.0, kPi}, .steps = {19U, 72U}};
            const Scan             fullSphere{.minima = {-kPi / 2.0, -kPi}, .maxima = {kPi / 2.0, kPi}, .steps = {37U, 72U}};
            const auto             maxLag       = addChain(graph, frames, geometry, line ? lineScan : (grid ? frontHemisphere : fullSphere), kRate, gr::testing::liveInputDescription(options.device), domain);
            const auto             frameSamples = static_cast<gr::Size_t>(gr::testing::correlationFrameSamples(maxLag));
            if (frameSamples > gr::testing::kMaxCorrelationFrameSamples) {
                return std::unexpected(std::format("array aperture exceeds the largest correlation frame of {} samples; reduce --spacing or the channel count", gr::testing::kMaxCorrelationFrameSamples));
            }
            if (const auto refused = frames.settings().set({{"size", frameSamples}, {"stride", frameSamples / 2U}}); !refused.empty()) {
                return std::unexpected(std::format("frame size {} refused", frameSamples));
            }
            return graph;
        });
    }
    std::println("input: synthesised audio; repeating {:.0f} s scene", gr::testing::kExampleSceneSeconds);
    std::println("synthetic scene: sources at azimuth 0 and 90 deg, the second 6 dB lower");
    return runGraph.template operator()<gr::scheduler::ExecutionPolicy::singleThreaded>([](std::string_view domain) -> std::expected<gr::Graph, std::string> {
        gr::Graph graph;
        addSyntheticScene(graph, domain);
        return graph;
    });
}
