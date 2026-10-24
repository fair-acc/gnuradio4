/**
 * Beams towards given directions, no localisation: StreamsToTensor -> STFT -> SteeringVectors (fed the directions) -> the chosen beamformer, the beam level [dB] per direction redrawn each second.
 * `--beamformer conventional|mvdr|lcmv` (LCMV nulls the other directions), `--directions 0,90` azimuths in degrees.
 * Live: the sound card's `--channels` placed by `--layout line|grid|tetrahedron` (`--rows`, `--columns`, `--spacing` m, 0 for the default);
 * with `--synthetic`, or without audio: the 2x2 array of gr::testing::ArrayScene.hpp with noise at azimuth 0 and 90 deg, the second 6 dB lower, and a summary of each beam against the truth.
 * Usage: ex0_beam_directions [--synthetic] [--nogpu] [--beamformer mvdr] [--directions 0,90] [--seconds N]
 */

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <expected>
#include <format>
#include <numbers>
#include <optional>
#include <print>
#include <ranges>
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
#include <gnuradio-4.0/testing/ArrayScene.hpp>
#include <gnuradio-4.0/testing/ExampleRun.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/testing/TensorSource.hpp>

namespace {

constexpr double      kPi                  = std::numbers::pi;
constexpr double      kSyntheticRate       = 1000.0;
constexpr gr::Size_t  kSyntheticFrame      = 256U;
constexpr gr::Size_t  kSyntheticHop        = 128U;
constexpr double      kUniformNoiseVar     = 1.0 / 3.0;                                                         // gr::testing::whiteNoise is uniform in [-1, 1)
const double          kUnitSourceDb        = 10.0 * std::log10(kUniformNoiseVar * 3.0 / 8.0 * kSyntheticFrame); // Hann window: sum of w^2 = 3N/8, STFT is unscaled
constexpr double      kSourceAzimuthsDeg[] = {0.0, 90.0};
constexpr double      kSourceAmplitudes[]  = {1.0, 0.5};
constexpr std::size_t kWarmUpFrames        = 10UZ;
constexpr std::size_t kChartColumns        = 100UZ;
constexpr std::size_t kChartRows           = 14UZ;
constexpr std::size_t kHistoryFrames       = 240UZ;

struct Options {
    std::string         device;
    bool                synthetic  = false;
    bool                noGpu      = false;
    std::string         beamformer = "mvdr";
    std::vector<double> directionsDeg{0.0, 90.0};
    std::size_t         channels = 2UZ;
    std::string         layout   = "line";
    std::string         positionsText;
    gr::Size_t          rows    = 4U;
    gr::Size_t          columns = 4U;
    double              spacing = 0.0;
    double              seconds = 0.0;
};

struct BeamDisplay : gr::Block<BeamDisplay> {
    using Description = gr::Doc<"Plots beam levels and compares synthetic source levels after covariance warm-up.">;

    gr::TensorPortIn<std::complex<float>> beams;
    gr::SampleRate                        sample_rate = static_cast<float>(kSyntheticRate);
    GR_MAKE_REFLECTABLE(BeamDisplay, beams, sample_rate);

    std::string _inputDescription;

    std::vector<double>                   _directionsDeg;
    double                                _hopSamples = 1.0;
    bool                                  _synthetic  = false;
    std::vector<std::vector<double>>      _levelsDb;
    std::vector<double>                   _meanPower;
    std::size_t                           _frames = 0UZ;
    std::chrono::steady_clock::time_point _drawn{};

    void start() { gr::graphs::resetView(); }

    void stop() {
        draw();
        for (std::size_t k = 0UZ; k < _directionsDeg.size() && _frames > kWarmUpFrames; ++k) {
            const double meanDb = 10.0 * std::log10(_meanPower[k] / static_cast<double>(_frames - kWarmUpFrames) + 1e-30);
            const auto   truth  = std::ranges::find_if(kSourceAzimuthsDeg, [&](double azimuth) { return std::abs(azimuth - _directionsDeg[k]) < 1.0; });
            if (_synthetic && truth != std::end(kSourceAzimuthsDeg)) {
                const double truthDb = kUnitSourceDb + 20.0 * std::log10(kSourceAmplitudes[truth - std::begin(kSourceAzimuthsDeg)]);
                std::println("beam {:6.1f} deg: level {:6.2f} dB, truth {:6.2f} dB, difference from ideal source level {:+6.2f} dB", _directionsDeg[k], meanDb, truthDb, meanDb - truthDb);
            } else {
                std::println("beam {:6.1f} deg: level {:6.2f} dB", _directionsDeg[k], meanDb);
            }
        }
    }

    gr::work::Status processBulk(gr::InputSpanLike auto& frames) {
        const std::size_t K = _directionsDeg.size();
        _levelsDb.resize(K);
        _meanPower.resize(K, 0.0);
        for (std::size_t n = 0UZ; n < frames.size(); ++n) {
            const auto&       beamSpectra       = frames[n];
            const std::size_t firstBin          = _synthetic ? 1UZ : 0UZ;
            const std::size_t endBin            = _synthetic ? 32UZ : beamSpectra.extent(1UZ);
            const double      F                 = static_cast<double>(endBin - firstBin);
            const bool        directionsPending = beamSpectra.extent(0UZ) < K;
            if (directionsPending) {
                continue;
            }
            for (std::size_t k = 0UZ; k < K; ++k) {
                double power = 0.0;
                for (std::size_t bin = firstBin; bin < endBin; ++bin) {
                    power += static_cast<double>(std::norm(beamSpectra.data()[k * beamSpectra.extent(1UZ) + bin]));
                }
                _levelsDb[k].push_back(10.0 * std::log10(power / F + 1e-30));
                if (_levelsDb[k].size() > kHistoryFrames) {
                    _levelsDb[k].erase(_levelsDb[k].begin());
                }
                _meanPower[k] += _frames >= kWarmUpFrames ? power / F : 0.0;
            }
            ++_frames;
        }
        std::ignore = frames.consume(frames.size());
        if (std::chrono::steady_clock::now() - _drawn >= std::chrono::seconds(1) && _frames > 0UZ) {
            draw();
            _drawn = std::chrono::steady_clock::now();
        }
        return gr::work::Status::OK;
    }

    void draw() const {
        if (_levelsDb.empty() || _levelsDb[0].empty()) {
            return;
        }
        std::vector<double> time(_levelsDb[0].size());
        std::ranges::generate(time, [&, n = _frames - time.size()]() mutable { return _hopSamples / static_cast<double>(sample_rate) * static_cast<double>(n++); });
        const auto                                     low  = std::ranges::min(_levelsDb | std::views::join);
        const auto                                     high = std::ranges::max(_levelsDb | std::views::join);
        gr::graphs::ImChart<kChartColumns, kChartRows> chart({{time.front(), time.back() + 1e-9}, {low - 1.0, high + 1.0}});
        chart.axis_name_x = "time [s]";
        chart.axis_name_y = "beam level [dB]";
        for (std::size_t k = 0UZ; k < _levelsDb.size(); ++k) {
            chart.draw(time, _levelsDb[k], std::format("{:.0f} deg", _directionsDeg[k]));
        }
        chart.reset();
        chart.draw();
        std::println("input: {}", _inputDescription);
        std::print("\033[J");
    }
};

template<typename TBeamformer>
void addBeamformer(gr::Graph& graph, auto& stft, auto& steering, auto& display, std::string_view domain, gr::property_map settings) {
    settings.insert_or_assign("compute_domain", std::string(domain));
    auto& beamformer = graph.emplaceBlock<TBeamformer>(settings);
    std::ignore      = graph.connect<"out", "in">(stft, beamformer);
    std::ignore      = graph.connect<"out", "steering">(steering, beamformer);
    if constexpr (TBeamformer::kUsesCovariance) {
        auto& covariance = graph.emplaceBlock<gr::beamforming::CrossSpectralMatrix<float>>({{"alpha", 0.05f}, {"compute_domain", std::string(domain)}});
        std::ignore      = graph.connect<"out", "in">(stft, covariance);
        std::ignore      = graph.connect<"out", "covariance">(covariance, beamformer);
    }
    std::ignore = graph.connect<"out", "beams">(beamformer, display);
}

void addChain(gr::Graph& graph, auto& frames, const gr::property_map& geometry, const Options& options, double sampleRate, double hop, std::string_view domain) {
    auto&              stft       = graph.emplaceBlock<gr::blocks::fft::STFT<float>>({{"compute_domain", std::string(domain)}});
    auto&              directions = graph.emplaceBlock<gr::testing::TensorSource<float>>();
    std::vector<float> azimuths;
    std::ranges::transform(options.directionsDeg, std::back_inserter(azimuths), [](double degrees) { return static_cast<float>(std::remainder(degrees, 360.0) * (kPi / 180.0)); });
    directions._frames       = {{.extents = {azimuths.size(), 1UZ}, .elements = azimuths}};
    gr::property_map steered = geometry;
    steered.insert_or_assign("coordinates", gr::Size_t(1));
    steered.insert_or_assign("compute_domain", std::string(domain));
    auto& steering                = graph.emplaceBlock<gr::beamforming::SteeringVectors<float>>(steered);
    auto& display                 = graph.emplaceBlock<BeamDisplay>();
    display._directionsDeg        = options.directionsDeg;
    display._hopSamples           = hop;
    display.sample_rate           = static_cast<float>(sampleRate);
    display._synthetic            = options.synthetic;
    const std::string inputDevice = options.device.empty() ? "default" : options.device;
    display._inputDescription     = options.synthetic ? gr::testing::syntheticInputDescription() : std::format("true audio, device {}", inputDevice);
    std::ignore                   = graph.connect<"out", "in">(frames, stft);
    std::ignore                   = graph.connect<"out", "in">(stft, steering);
    std::ignore                   = graph.connect<"out", "peaks">(directions, steering);
    if (options.beamformer == "conventional") {
        addBeamformer<gr::beamforming::ConventionalBeamformer<float>>(graph, stft, steering, display, domain, {});
    } else {
        const gr::property_map loading{{"diagonal_loading", 0.05f}};
        if (options.beamformer == "lcmv") {
            addBeamformer<gr::beamforming::NullingBeamformer<float>>(graph, stft, steering, display, domain, loading);
        } else {
            addBeamformer<gr::beamforming::MinimumVarianceBeamformer<float>>(graph, stft, steering, display, domain, loading);
        }
    }
}

[[nodiscard]] std::expected<Options, std::string> parse(int argc, char** argv) {
    Options                       options;
    std::optional<std::string>    directionsText;
    gr::testing::ExampleArguments args(argc, argv);
    while (args.next()) {
        const bool known = args.flag("--synthetic", options.synthetic) || args.flag("--nogpu", options.noGpu) || args.text("--beamformer", options.beamformer) || args.text("--directions", directionsText) || args.text("--device", options.device) //
                           || args.number("--channels", options.channels) || args.text("--layout", options.layout) || args.text("--positions", options.positionsText) || args.number("--rows", options.rows)                                         //
                           || args.number("--columns", options.columns) || args.number("--spacing", options.spacing) || args.number("--seconds", options.seconds);
        if (!known) {
            args.refuse();
        }
    }
    bool validArguments = args.valid;
    if (directionsText) {
        options.directionsDeg.clear();
        for (const auto part : std::views::split(std::string_view(*directionsText), ',')) {
            double     degrees = 0.0;
            const auto parsed  = std::from_chars(part.data(), part.data() + part.size(), degrees);
            validArguments     = validArguments && parsed.ec == std::errc{} && parsed.ptr == part.data() + part.size() && std::isfinite(degrees);
            options.directionsDeg.push_back(degrees);
        }
    }
    constexpr std::size_t kMaxChannels      = gr::testing::kMaxExampleElements;
    const bool            beamformerKnown   = options.beamformer == "conventional" || options.beamformer == "mvdr" || options.beamformer == "lcmv";
    const bool            channelCountValid = options.channels >= 2UZ && options.channels <= kMaxChannels;
    const bool            gridCountsValid   = options.rows > 0U && options.columns > 0U && options.rows <= kMaxChannels && options.columns <= kMaxChannels;
    const bool            durationValid     = options.seconds >= 0.0;
    const bool            directionsValid   = !options.directionsDeg.empty() && options.directionsDeg.size() <= kMaxChannels;
    const bool            geometryValid     = channelCountValid && gridCountsValid && options.spacing >= 0.0;
    if (!validArguments || !beamformerKnown || !geometryValid || !durationValid || !directionsValid) {
        return std::unexpected("invalid arguments: use conventional|mvdr|lcmv, 2..32 channels, finite directions and non-negative seconds (0: unlimited)");
    }
    return options;
}

} // namespace

int main(int argc, char** argv) {
    const auto parsed = parse(argc, argv);
    if (!parsed) {
        std::println(stderr, "{}", parsed.error());
        return 1;
    }
    Options options = *parsed;
    if (options.beamformer == "lcmv") {
        std::println("LCMV: bins with numerically dependent directional constraints, including DC, are omitted (zero output)");
    }
    const auto runGraph = [&options](const auto& buildGraph) {
        gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::multiThreaded> scheduler;
        return gr::testing::runExample(buildGraph, gr::testing::chooseExampleDomain(options.noGpu), options.seconds, scheduler);
    };
    if (gr::testing::kLiveAudio && !options.synthetic && options.channels >= 2UZ) {
        constexpr double  kRate     = gr::testing::kExampleSampleRateHz;
        const std::size_t M         = options.channels;
        const auto        positions = options.positionsText.empty() ? gr::algorithm::geometry::layoutOf(options.layout, M, options.rows, options.columns, options.spacing) : gr::algorithm::geometry::parsePositions(options.positionsText);
        const auto        elements  = positions.and_then(gr::algorithm::placedElements);
        if (!elements || *elements != M) {
            std::println(stderr, "array refused for {} channels: {}", M, elements ? std::format("it places {} elements", *elements) : elements.error());
            return 1;
        }
        std::println("{} channels as a {}: beams towards {} deg ({})", M, options.layout, options.directionsDeg, options.beamformer);
        return runGraph([&](std::string_view domain) -> std::expected<gr::Graph, std::string> {
            gr::Graph graph;
            auto&     frames = graph.emplaceBlock<gr::basic::StreamsToTensor<float>>({{"n_ports", gr::Size_t(M)}, {"size", gr::Size_t(1024)}, {"stride", gr::Size_t(512)}, {"compute_domain", std::string(domain)}});
            if (const auto connected = gr::testing::connectMicrophones(graph, frames, options.device, M, kRate); !connected) {
                return std::unexpected(connected.error());
            }
            const gr::property_map geometry{{"element_positions", *positions}, {"propagation_speed", gr::testing::kSpeedOfSoundInAir}, {"field", std::string("far")}};
            addChain(graph, frames, geometry, options, kRate, 512.0, domain);
            return graph;
        });
    }
    options.synthetic = true;
    std::println("input: synthesised audio; repeating {:.0f} s scene", gr::testing::kExampleSceneSeconds);
    std::println("synthetic beam levels use bins 1..31 (3.90625..121.09375 Hz), below the spatial-alias limit of 125 Hz");
    std::println("synthetic scene: noise at azimuth 0 and 90 deg, the second 6 dB lower; beams towards {} deg ({})", options.directionsDeg, options.beamformer);
    return runGraph([&options](std::string_view domain) -> std::expected<gr::Graph, std::string> {
        const std::size_t samples = static_cast<std::size_t>(gr::testing::kExampleSceneSeconds * kSyntheticRate);
        gr::Graph         graph;
        auto&             frames = graph.emplaceBlock<gr::basic::StreamsToTensor<float, gr::testing::kSquareArrayElements>>({{"size", kSyntheticFrame}, {"stride", kSyntheticHop}, {"compute_domain", std::string(domain)}});
        for (std::size_t m = 0UZ; m < gr::testing::kSquareArrayElements; ++m) {
            std::vector<float> noisySamples = gr::testing::squareArraySamples(m, samples);
            gr::testing::addMicrophoneNoise(noisySamples, m);
            gr::testing::addElementSource(graph, frames, m, std::move(noisySamples), kSyntheticRate);
        }
        addChain(graph, frames, gr::testing::squareArrayGeometry(), options, kSyntheticRate, kSyntheticHop, domain);
        return graph;
    });
}
