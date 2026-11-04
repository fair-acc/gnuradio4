#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <format>
#include <numbers>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/algorithm/fourier/window.hpp>
#include <gnuradio-4.0/basic/StreamToTensor.hpp>
#include <gnuradio-4.0/fourier/STFT.hpp>
#include <gnuradio-4.0/test/DeviceTestHelper.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/testing/TensorSource.hpp>

namespace {
using namespace boost::ut;
using gr::blocks::fft::InverseSTFT;
using gr::blocks::fft::STFT;
using gr::testing::TensorCollector;
using gr::testing::TensorSource;

template<typename T>
[[nodiscard]] T sampleAt(std::size_t frame, std::size_t m, std::size_t n) {
    using Real       = decltype(std::real(T{}));
    const double phi = 2.0 * std::numbers::pi * static_cast<double>(n);
    const double x   = std::cos(phi * (3.0 + static_cast<double>(m)) / 64.0) + 0.5 * std::sin(phi * 11.0 / 64.0 + static_cast<double>(frame));
    if constexpr (gr::meta::complex_like<T>) {
        return T(static_cast<Real>(x), static_cast<Real>(0.25 * std::cos(phi * 7.0 / 64.0)));
    } else {
        return static_cast<T>(x);
    }
}

/** `nFrames` frames of `rows` rows; frame f has `columnSequence[f % size]` columns */
template<typename T>
[[nodiscard]] std::vector<typename gr::testing::TensorSource<T>::Frame> framesOf(const std::vector<std::size_t>& columnSequence = {64UZ}, std::size_t nFrames = 4UZ, std::size_t rows = 3UZ) {
    std::vector<typename gr::testing::TensorSource<T>::Frame> frames(nFrames);
    for (std::size_t f = 0UZ; f < nFrames; ++f) {
        const std::size_t N = columnSequence[f % columnSequence.size()];
        frames[f].extents   = {rows, N};
        for (std::size_t m = 0UZ; m < rows; ++m) {
            for (std::size_t j = 0UZ; j < N; ++j) {
                frames[f].elements.push_back(sampleAt<T>(f, m, j));
            }
        }
    }
    return frames;
}

/** w[n] = 0.5 - 0.5 cos(2 pi n / N), the periodic Hann window: sin^2(pi n / N), so w[n] + w[n + N/2] = 1 */
[[nodiscard]] std::vector<double> periodicHann(std::size_t N) {
    std::vector<double> w(N);
    for (std::size_t n = 0UZ; n < N; ++n) {
        w[n] = 0.5 - 0.5 * std::cos(2.0 * std::numbers::pi * static_cast<double>(n) / static_cast<double>(N));
    }
    return w;
}

/** X[k] = sum_n w[n] x[n] exp(-j 2 pi k n / N), written out */
template<typename T>
[[nodiscard]] std::vector<std::complex<double>> directDft(std::span<const T> row, std::span<const double> window, std::size_t bins) {
    const std::size_t                 N = row.size();
    std::vector<std::complex<double>> spectrum(bins);
    for (std::size_t k = 0UZ; k < bins; ++k) {
        for (std::size_t n = 0UZ; n < N; ++n) {
            const double               angle = -2.0 * std::numbers::pi * static_cast<double>(k * n) / static_cast<double>(N);
            const std::complex<double> x_n(static_cast<double>(std::real(row[n])), static_cast<double>(std::imag(row[n])));
            spectrum[k] += x_n * window[n] * std::polar(1.0, angle);
        }
    }
    return spectrum;
}

template<typename T>
void expectDirectDft(const gr::testing::TensorSource<T>& source, const auto& sink, gr::algorithm::window::Type window, std::string_view context) {
    constexpr std::size_t kRows = 3UZ;
    expect(eq(sink._elements.size(), source._frames.size())) << context;
    for (std::size_t f = 0UZ; f < sink._elements.size(); ++f) {
        const std::size_t         N    = source._frames[f].extents[1];
        const std::size_t         bins = gr::meta::complex_like<T> ? N : N / 2UZ + 1UZ;
        const std::vector<double> w    = window == gr::algorithm::window::Type::Hann ? periodicHann(N) : std::vector<double>(N, 1.0);
        expect(sink._shapes[f] == std::vector<std::size_t>{kRows, bins}) << context << "frame" << f;
        for (std::size_t m = 0UZ; m < kRows; ++m) {
            std::vector<T> row(N);
            for (std::size_t n = 0UZ; n < N; ++n) {
                row[n] = sampleAt<T>(f, m, n);
            }
            const auto expected = directDft<T>(row, w, bins);
            double     maxError = 0.0;
            for (std::size_t k = 0UZ; k < bins; ++k) {
                maxError = std::max(maxError, std::abs(std::complex<double>(sink._elements[f][m * bins + k]) - expected[k]) / static_cast<double>(N));
            }
            const double tolerance = std::same_as<gr::meta::fundamental_base_value_type_t<T>, float> ? 1e-5 : 1e-12;
            expect(le(maxError, tolerance)) << context << "frame" << f << "row" << m;
        }
    }
}

template<typename T>
using SpectrumCollector = TensorCollector<std::complex<gr::meta::fundamental_base_value_type_t<T>>>;

const auto spectra = [] {
    "each row's spectrum is a direct DFT of the windowed row: one-sided for real samples, full for complex ones"_test = []<typename T>() {
        for (const auto window : {gr::algorithm::window::Type::None, gr::algorithm::window::Type::Hann}) {
            gr::testing::GraphFixture<> fixture;
            auto&                       source = fixture.emplace<TensorSource<T>>();
            source._frames                     = framesOf<T>();
            auto& stft                         = fixture.emplace<STFT<T>>({{"window", std::string(gr::meta::enumName(window).value_or(""))}});
            auto& sink                         = fixture.emplace<SpectrumCollector<T>>();
            expect(fixture.connect<"out", "in">(source, stft).has_value());
            expect(fixture.connect<"out", "in">(stft, sink).has_value());
            expect(fixture.run().has_value());
            expectDirectDft(source, sink, window, gr::meta::enumName(window).value_or(""));
        }
    } | std::tuple<float, double, std::complex<float>, std::complex<double>>{};

    "frames whose N changes from frame to frame each get their own spectrum"_test = []<typename T>() {
        gr::testing::GraphFixture<> fixture;
        auto&                       source = fixture.emplace<TensorSource<T>>();
        source._frames                     = framesOf<T>({64UZ, 16UZ, 128UZ, 2UZ});
        auto& stft                         = fixture.emplace<STFT<T>>();
        auto& sink                         = fixture.emplace<SpectrumCollector<T>>();
        expect(fixture.connect<"out", "in">(source, stft).has_value());
        expect(fixture.connect<"out", "in">(stft, sink).has_value());
        expect(fixture.run().has_value());
        expectDirectDft(source, sink, gr::algorithm::window::Type::Hann, "N 64, 16, 128, 2");
    } | std::tuple<float, std::complex<double>>{};

    "a frame of one column, or of an odd number of real samples, stops the STFT before it"_test = []<typename T>() {
        for (const gr::Size_t columns : {gr::Size_t(1), gr::Size_t(63)}) {
            const bool                  rejected = columns == 1U || !gr::meta::complex_like<T>;
            gr::testing::GraphFixture<> fixture;
            auto&                       source = fixture.emplace<TensorSource<T>>();
            source._frames                     = framesOf<T>({64UZ, columns});
            auto& stft                         = fixture.emplace<STFT<T>>();
            auto& sink                         = fixture.emplace<SpectrumCollector<T>>();
            expect(fixture.connect<"out", "in">(source, stft).has_value());
            expect(fixture.connect<"out", "in">(stft, sink).has_value());
            expect(eq(fixture.run().has_value(), !rejected)) << "N" << columns;
            expect(eq(sink._elements.size(), rejected ? 1UZ : 4UZ)) << "N" << columns << ": the frame before it, of N 64, went out";
        }
    } | std::tuple<float, std::complex<float>>{};
};

template<typename T>
using Source = gr::testing::TagSource<T, gr::testing::ProcessFunction::USE_PROCESS_BULK>;
template<typename T>
using Sink = gr::testing::TagSink<T, gr::testing::ProcessFunction::USE_PROCESS_BULK>;

constexpr std::size_t kFrameSize = 64UZ, kSamples = 1024UZ;

template<typename T>
struct RoundTrip {
    std::vector<Sink<T>*>       sinks;
    std::vector<std::vector<T>> inputs;
};

template<typename T>
[[nodiscard]] RoundTrip<T> roundTripGraph(auto& fixture, gr::Size_t stride, gr::property_map inverse, float sampleRate = 0.f, std::string_view domain = {}) {
    using gr::basic::StreamsToTensor;
    using gr::basic::TensorToStreams;
    const gr::property_map placed = domain.empty() ? gr::property_map{} : gr::property_map{{"gr:compute_domain", std::string(domain) + "!"}};
    for (const auto& [key, value] : placed) {
        inverse.insert_or_assign(key, value);
    }
    auto&        toFrames  = fixture.template emplace<StreamsToTensor<T, 2UZ>>({{"size", gr::Size_t(kFrameSize)}, {"stride", stride}});
    auto&        stft      = fixture.template emplace<STFT<T>>(placed);
    auto&        istft     = fixture.template emplace<InverseSTFT<T>>(std::move(inverse));
    auto&        toStreams = fixture.template emplace<TensorToStreams<T, 2UZ>>();
    RoundTrip<T> graph{.sinks = {}, .inputs = std::vector<std::vector<T>>(2UZ)};
    for (std::size_t port = 0UZ; port < 2UZ; ++port) {
        for (std::size_t i = 0UZ; i < kSamples; ++i) {
            graph.inputs[port].push_back(sampleAt<T>(port, port, i));
        }
        auto& source = fixture.template emplace<Source<T>>({{"values", graph.inputs[port]}, {"n_samples_max", gr::Size_t(kSamples)}, {"mark_tag", false}});
        if (sampleRate > 0.f) {
            source._tags = {{0UZ, {{std::string(gr::tag::kSampleRate.key()), sampleRate}}}};
        }
        graph.sinks.push_back(&fixture.template emplace<Sink<T>>({{"log_samples", true}}));
        expect(fixture.graph.connect(source, "out", toFrames, std::format("in#{}", port)).has_value());
        expect(fixture.graph.connect(toStreams, std::format("out#{}", port), *graph.sinks.back(), "in").has_value());
    }
    expect(fixture.template connect<"out", "in">(toFrames, stft).has_value());
    expect(fixture.template connect<"out", "in">(stft, istft).has_value());
    expect(fixture.template connect<"out", "in">(istft, toStreams).has_value());
    return graph;
}

template<typename T>
void expectReconstructed(const RoundTrip<T>& graph, std::size_t stride, std::string_view context) {
    constexpr bool    kFloat  = std::same_as<gr::meta::fundamental_base_value_type_t<T>, float>;
    const std::size_t nFrames = (kSamples - kFrameSize) / stride + 1UZ;
    for (std::size_t port = 0UZ; port < 2UZ; ++port) {
        const auto& out = graph.sinks[port]->_samples;
        expect(eq(out.size(), nFrames * stride)) << context << "stride" << stride << "output" << port;
        double maxError = 0.0;
        for (std::size_t i = kFloat ? kFrameSize - stride : 1UZ; i < std::min(out.size(), kSamples); ++i) {
            maxError = std::max(maxError, static_cast<double>(std::abs(out[i] - graph.inputs[port][i])));
        }
        expect(le(maxError, kFloat ? 1e-5 : 1e-12)) << context << "stride" << stride << "output" << port;
    }
}

/** row m of the test signal: a tone of 2 + m periods per 16 samples, phase 0.3 rad; complex samples are exp(j phase) */
template<typename T>
[[nodiscard]] T toneAt(std::size_t m, std::size_t n) {
    using Real         = gr::meta::fundamental_base_value_type_t<T>;
    const double phase = 2.0 * std::numbers::pi * (2.0 + static_cast<double>(m)) * static_cast<double>(n) / 16.0 + 0.3;
    if constexpr (gr::meta::complex_like<T>) {
        return T(static_cast<Real>(std::cos(phase)), static_cast<Real>(std::sin(phase)));
    } else {
        return static_cast<T>(std::cos(phase));
    }
}

struct InverseCase {
    std::size_t                 fftSize;
    std::size_t                 hop;
    bool                        hopFromRates; // else N/2
    gr::algorithm::window::Type window;
};

/** the one-sided (real T) or full (complex T) spectra of `nFrames` windowed tone segments */
template<typename T>
[[nodiscard]] auto toneSpectra(const InverseCase& c, std::size_t nFrames, std::size_t rows) {
    using Complex                                             = std::complex<gr::meta::fundamental_base_value_type_t<T>>;
    const std::size_t                                  bins   = gr::meta::complex_like<T> ? c.fftSize : c.fftSize / 2UZ + 1UZ;
    const auto                                         window = c.window == gr::algorithm::window::Type::Hann ? periodicHann(c.fftSize) : std::vector<double>(c.fftSize, 1.0);
    std::vector<typename TensorSource<Complex>::Frame> frames(nFrames);
    for (std::size_t f = 0UZ; f < nFrames; ++f) {
        frames[f].extents = {rows, bins};
        for (std::size_t m = 0UZ; m < rows; ++m) {
            std::vector<T> segment(c.fftSize);
            for (std::size_t n = 0UZ; n < c.fftSize; ++n) {
                segment[n] = toneAt<T>(m, f * c.hop + n);
            }
            for (const std::complex<double>& bin : directDft<T>(segment, window, bins)) {
                frames[f].elements.push_back(Complex(bin));
            }
        }
    }
    return frames;
}

template<typename T>
void expectToneReconstructed(const InverseCase& c, std::string_view context) {
    using Complex                       = std::complex<gr::meta::fundamental_base_value_type_t<T>>;
    constexpr bool              kFloat  = std::same_as<gr::meta::fundamental_base_value_type_t<T>, float>;
    constexpr std::size_t       kRows   = 2UZ;
    constexpr std::size_t       kFrames = 6UZ;
    gr::testing::GraphFixture<> fixture;
    auto&                       source = fixture.emplace<TensorSource<Complex>>();
    source._frames                     = toneSpectra<T>(c, kFrames, kRows);
    if (c.hopFromRates) {
        source._sampleRate = 48'000.f;
        source._firstTag   = gr::property_map{{std::string(gr::tag::kFrameRate.key()), 48'000.f / static_cast<float>(c.hop)}};
    }
    auto& istft = fixture.emplace<InverseSTFT<T>>({{"window", std::string(gr::meta::enumName(c.window).value_or(""))}});
    auto& sink  = fixture.emplace<TensorCollector<T>>();
    expect(fixture.connect<"out", "in">(source, istft).has_value()) << context;
    expect(fixture.connect<"out", "in">(istft, sink).has_value()) << context;
    expect(fixture.run().has_value()) << context;
    expect(eq(sink._elements.size(), kFrames)) << context;

    const double tolerance = kFloat ? 1e-4 : 1e-12; // float: the start-up division by a small w[j] amplifies the rounding of the inverse FFT
    for (std::size_t f = 0UZ; f < std::min(sink._elements.size(), kFrames); ++f) {
        expect(sink._shapes[f] == std::vector<std::size_t>{kRows, c.hop}) << context << "frame" << f;
        expect(eq(sink._elements[f].size(), kRows * c.hop)) << context << "frame" << f;
        double maxError = 0.0;
        for (std::size_t i = 0UZ; i < std::min(sink._elements[f].size(), kRows * c.hop); ++i) {
            const std::size_t m         = i / c.hop;
            const std::size_t j         = i % c.hop;
            const bool        uncovered = f == 0UZ && j == 0UZ && c.window == gr::algorithm::window::Type::Hann; // w[0] = 0: no window has covered it yet
            const T           expected  = uncovered ? T{} : toneAt<T>(m, f * c.hop + j);
            maxError                    = std::max(maxError, static_cast<double>(std::abs(sink._elements[f][i] - expected)));
        }
        expect(le(maxError, tolerance)) << context << "frame" << f;
    }
}

const auto roundTrip = [] {
    "streams -> frames -> STFT -> ISTFT -> streams reproduces the input wherever a window covers it"_test = []<typename T>() {
        for (const gr::Size_t stride : {gr::Size_t(32), gr::Size_t(16), gr::Size_t(24)}) { // 24 does not divide N = 64
            gr::testing::GraphFixture<> fixture;
            const auto                  graph = roundTripGraph<T>(fixture, stride, {}, 48'000.f);
            expect(fixture.run().has_value());
            expectReconstructed(graph, stride, "stride from the rates");
        }
    } | std::tuple<float, double, std::complex<float>, std::complex<double>>{};

    "without a rate the inverse steps by N/2"_test = [] {
        gr::testing::GraphFixture<> fixture;
        const auto                  graph = roundTripGraph<double>(fixture, gr::Size_t(kFrameSize / 2UZ), {});
        expect(fixture.run().has_value());
        expectReconstructed(graph, kFrameSize / 2UZ, "default step");
    };

    "gr:sample_rate and gr:frame_rate leave the inverse as they came"_test = [] {
        constexpr float             kRate = 48'000.f;
        gr::testing::GraphFixture<> fixture;
        const auto                  graph = roundTripGraph<double>(fixture, gr::Size_t(24), {}, kRate);
        expect(fixture.run().has_value());
        expectReconstructed(graph, 24UZ, "stride from the rates");
        for (const Sink<double>* sink : graph.sinks) {
            const std::string_view sampleRateKey{gr::tag::kSampleRate.key()};
            const auto             rateTag = std::ranges::find_if(sink->_tags, [&](const auto& tag) { return tag.map.contains(sampleRateKey); });
            expect(rateTag != sink->_tags.end() && rateTag->map.template value_or<float>(sampleRateKey, 0.f) == kRate) << "the signal's rate on every stream";
            const bool framed = std::ranges::any_of(sink->_tags, [](const auto& tag) { return tag.map.contains(gr::tag::kFrameRate.key()); });
            expect(!framed) << "streams again: no frame rate";
        }
    };

    "a window whose squares leave a gap at the stride stops the inverse"_test = [] {
        gr::testing::GraphFixture<> fixture;
        const auto                  graph = roundTripGraph<float>(fixture, gr::Size_t(kFrameSize), {}, 48'000.f);
        expect(!fixture.run().has_value());
        expect(graph.sinks[0]->_samples.empty());
    };

    "a spectrum of one bin stops the inverse"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       oneBin = fixture.emplace<TensorSource<std::complex<float>>>();
        oneBin._frames                     = framesOf<std::complex<float>>({1UZ});
        auto& istft                        = fixture.emplace<InverseSTFT<float>>();
        auto& samples                      = fixture.emplace<TensorCollector<float>>();
        expect(fixture.connect<"out", "in">(oneBin, istft).has_value());
        expect(fixture.connect<"out", "in">(istft, samples).has_value());
        expect(!fixture.run().has_value());
        expect(samples._elements.empty());
    };

    // each frame's inverse FFT gives w x; times w and divided by the squares of the windows overlapping so far, that is x again
    "Hann-windowed spectra of a tone, cut every N/2, N/4 and 6 of N = 16 samples, overlap-add back to the tone; the rectangular ones too"_test = []<typename T>() {
        using gr::algorithm::window::Type;
        const std::vector<InverseCase> cases{{16UZ, 8UZ, false, Type::Hann}, {16UZ, 4UZ, true, Type::Hann}, {16UZ, 6UZ, true, Type::Hann}, //
            {16UZ, 8UZ, false, Type::None}, {16UZ, 16UZ, true, Type::None}, {2UZ, 1UZ, false, Type::Hann}};
        for (const InverseCase& c : cases) {
            expectToneReconstructed<T>(c, std::format("N {} hop {} {} {}", c.fftSize, c.hop, gr::meta::enumName(c.window).value_or(""), c.hopFromRates ? "from the rates" : "default"));
        }
    } | std::tuple<float, double, std::complex<float>, std::complex<double>>{};

    // x = 1 has X[0] = N and the inverse FFT gives 1 per sample: frame 0 yields w / w^2 = 1/w, later ones (w[j] + w[j + 4]) / (w[j]^2 + w[j + 4]^2) with w[j] + w[j + 4] = 1
    "a spectrum of the constant 1 with N = 8 comes out as 1/w[j] at start-up (0 where w = 0), then as sum(w) / sum(w^2) = 1, 4/3, 2, 4/3 with two frames overlapping"_test = []<typename T>() {
        using Real                                = gr::meta::fundamental_base_value_type_t<T>;
        using Complex                             = std::complex<Real>;
        constexpr std::size_t          N          = 8UZ;
        constexpr std::size_t          kHop       = N / 2UZ;
        constexpr std::size_t          kFrames    = 3UZ;
        const std::size_t              bins       = gr::meta::complex_like<T> ? N : N / 2UZ + 1UZ;
        const double                   sqrt2      = std::numbers::sqrt2; // Hann: w[1] = (1 - 1/sqrt2) / 2, w[2] = 1/2, w[3] = (1 + 1/sqrt2) / 2
        const std::array<double, kHop> startUp    = {0.0, 4.0 + 2.0 * sqrt2, 2.0, 4.0 - 2.0 * sqrt2};
        const std::array<double, kHop> overlapped = {1.0, 4.0 / 3.0, 2.0, 4.0 / 3.0};

        gr::testing::GraphFixture<> fixture;
        auto&                       source = fixture.emplace<TensorSource<Complex>>();
        source._frames.resize(kFrames);
        for (auto& frame : source._frames) {
            frame.extents     = {1UZ, bins};
            frame.elements    = std::vector<Complex>(bins);
            frame.elements[0] = Complex(static_cast<Real>(N));
        }
        auto& istft = fixture.emplace<InverseSTFT<T>>();
        auto& sink  = fixture.emplace<TensorCollector<T>>();
        expect(fixture.connect<"out", "in">(source, istft).has_value());
        expect(fixture.connect<"out", "in">(istft, sink).has_value());
        expect(fixture.run().has_value());
        expect(eq(sink._elements.size(), kFrames));
        const double tolerance = std::same_as<Real, float> ? 1e-5 : 1e-12;
        for (std::size_t f = 0UZ; f < std::min(sink._elements.size(), kFrames); ++f) {
            expect(sink._shapes[f] == std::vector<std::size_t>{1UZ, kHop}) << "frame" << f;
            expect(eq(sink._elements[f].size(), kHop)) << "frame" << f;
            for (std::size_t j = 0UZ; j < std::min(sink._elements[f].size(), kHop); ++j) {
                const T expected = T(static_cast<Real>(f == 0UZ ? startUp[j] : overlapped[j]));
                expect(le(static_cast<double>(std::abs(sink._elements[f][j] - expected)), tolerance)) << "frame" << f << "sample" << j;
            }
        }
    } | std::tuple<float, double, std::complex<float>, std::complex<double>>{};

    "a frame_rate of a third of the sample_rate of the first frame sets the hop to 3 of N = 8 for every frame, and a tag moves with its frame"_test = [] {
        const std::string           triggerKey{gr::tag::kTriggerName.key()};
        constexpr std::size_t       kHop    = 3UZ;
        constexpr std::size_t       kFrames = 4UZ;
        gr::testing::GraphFixture<> fixture;
        auto&                       source = fixture.emplace<TensorSource<std::complex<float>>>();
        source._frames.resize(kFrames);
        for (auto& frame : source._frames) {
            frame.extents     = {1UZ, 8UZ};
            frame.elements    = std::vector<std::complex<float>>(8UZ);
            frame.elements[0] = 8.f;
        }
        source._sampleRate = 48'000.f;
        source._firstTag   = gr::property_map{{std::string(gr::tag::kFrameRate.key()), 16'000.f}};
        source._frameTags  = {{2UZ, gr::property_map{{triggerKey, std::string("here")}}}};
        auto& istft        = fixture.emplace<InverseSTFT<std::complex<float>>>();
        auto& sink         = fixture.emplace<TensorCollector<std::complex<float>>>();
        expect(fixture.connect<"out", "in">(source, istft).has_value());
        expect(fixture.connect<"out", "in">(istft, sink).has_value());
        expect(fixture.run().has_value());
        expect(eq(sink._elements.size(), kFrames));
        for (std::size_t f = 0UZ; f < sink._shapes.size(); ++f) {
            expect(sink._shapes[f] == std::vector<std::size_t>{1UZ, kHop}) << "frame" << f << ": 48000 / 16000 = 3";
        }
        expect(!sink._elements.empty() && sink._elements[0][0] == std::complex<float>{}) << "sample 0 of the first frame: w[0] = 0, nothing covers it";
        std::vector<std::size_t> triggered;
        for (const auto& [index, map] : sink._tags) {
            if (map.contains(triggerKey)) {
                triggered.push_back(index);
            }
        }
        expect(triggered == std::vector<std::size_t>{2UZ}) << "the tag of frame 2 comes out with frame 2";
    };

    "a hop longer than N, or a frame of one dimension, stops the inverse before any output"_test = [] {
        for (const bool tooLongHop : {true, false}) {
            gr::testing::GraphFixture<> fixture;
            auto&                       source = fixture.emplace<TensorSource<std::complex<float>>>();
            source._frames.resize(2UZ);
            for (auto& frame : source._frames) {
                frame.extents  = tooLongHop ? std::vector<std::size_t>{1UZ, 8UZ} : std::vector<std::size_t>{8UZ};
                frame.elements = std::vector<std::complex<float>>(8UZ);
            }
            if (tooLongHop) {
                source._sampleRate = 48'000.f;
                source._firstTag   = gr::property_map{{std::string(gr::tag::kFrameRate.key()), 3'000.f}}; // hop 16 > N = 8
            }
            auto& istft = fixture.emplace<InverseSTFT<std::complex<float>>>();
            auto& sink  = fixture.emplace<TensorCollector<std::complex<float>>>();
            expect(fixture.connect<"out", "in">(source, istft).has_value());
            expect(fixture.connect<"out", "in">(istft, sink).has_value());
            expect(!fixture.run().has_value()) << (tooLongHop ? "hop 16 of N 8" : "rank 1");
            expect(sink._elements.empty()) << (tooLongHop ? "hop 16 of N 8" : "rank 1");
        }
    };

    "a changed window setting drops the window coefficients to be rebuilt, any other setting leaves them"_test = []<typename T>() {
        const auto expectDropsOnWindowOnly = [](auto& block) {
            block._windowCoefficients.assign(8UZ, T{1});
            block.settingsChanged({}, gr::property_map{{"unrelated", 1.f}});
            expect(eq(block._windowCoefficients.size(), 8UZ));
            block.settingsChanged({}, gr::property_map{{"window", std::string("None")}});
            expect(block._windowCoefficients.empty());
        };
        STFT<T>        forward;
        InverseSTFT<T> inverse;
        expectDropsOnWindowOnly(forward);
        expectDropsOnWindowOnly(inverse);
    } | std::tuple<float, double>{};

    "start of the inverse forgets the hop of an earlier run and its window coefficients"_test = [] {
        InverseSTFT<float> block;
        block._hopFromRates = 6UZ;
        block._windowCoefficients.assign(8UZ, 1.f);
        block.start();
        expect(eq(block._hopFromRates, 0UZ));
        expect(block._windowCoefficients.empty());
    };
};

template<typename T>
void spectraOnDomain(auto& fixture, std::string_view domain) {
    auto& source   = fixture.template emplace<TensorSource<T>>();
    source._frames = framesOf<T>();
    auto& stft     = fixture.template emplace<STFT<T>>({{"gr:compute_domain", std::string(domain) + "!"}});
    auto& sink     = fixture.template emplace<SpectrumCollector<T>>();
    expect(fixture.template connect<"out", "in">(source, stft).has_value());
    expect(fixture.template connect<"out", "in">(stft, sink).has_value());
    const auto ran = fixture.run();
    expect(ran.has_value()) << domain << (ran.has_value() ? std::string{} : ran.error().message);
    expectDirectDft(source, sink, gr::algorithm::window::Type::Hann, domain);
}

template<typename T>
void reconstructsOnDomain(auto& fixture, std::string_view domain) {
    const auto graph = roundTripGraph<T>(fixture, gr::Size_t(16), {}, 48'000.f, domain);
    const auto ran   = fixture.run();
    expect(ran.has_value()) << domain << (ran.has_value() ? std::string{} : ran.error().message);
    expectReconstructed(graph, 16UZ, domain);
}

template<typename TIn, typename TBlock, typename TSink>
void takesNonPowerOfTwoOnHostOnly(auto& fixture, std::string_view domain, std::size_t columns) {
    const bool onDevice = domain != gr::testing::kHostDomain;
    auto&      source   = fixture.template emplace<TensorSource<TIn>>();
    source._frames      = framesOf<TIn>({columns});
    auto& block         = fixture.template emplace<TBlock>({{"gr:compute_domain", std::string(domain) + "!"}});
    auto& sink          = fixture.template emplace<TSink>();
    expect(fixture.template connect<"out", "in">(source, block).has_value());
    expect(fixture.template connect<"out", "in">(block, sink).has_value());
    expect(eq(fixture.run().has_value(), !onDevice)) << domain;
    expect(eq(sink._elements.empty(), onDevice)) << domain << ": a device FFT refuses the frame before any output";
}

const auto domains = [] {
    using gr::testing::operator""_domain_test;
    using gr::testing::overGraphs;
    using gr::testing::kAllDomains;
    "real frames' spectra on each served domain match a direct DFT"_domain_test                                     = [](auto& fixture, std::string_view domain) { spectraOnDomain<float>(fixture, domain); } | overGraphs(kAllDomains);
    "complex frames' spectra on each served domain match a direct DFT"_domain_test                                  = [](auto& fixture, std::string_view domain) { spectraOnDomain<std::complex<float>>(fixture, domain); } | overGraphs(kAllDomains);
    "real streams survive STFT and ISTFT on each served domain"_domain_test                                         = [](auto& fixture, std::string_view domain) { reconstructsOnDomain<float>(fixture, domain); } | overGraphs(kAllDomains);
    "complex streams survive STFT and ISTFT on each served domain"_domain_test                                      = [](auto& fixture, std::string_view domain) { reconstructsOnDomain<std::complex<float>>(fixture, domain); } | overGraphs(kAllDomains);
    "12 samples are no power of two: the STFT refuses the frame on a device, not on the host"_domain_test           = [](auto& fixture, std::string_view domain) { takesNonPowerOfTwoOnHostOnly<float, STFT<float>, SpectrumCollector<float>>(fixture, domain, 12UZ); } | overGraphs(kAllDomains);
    "a one-sided spectrum of 7 bins is 12 samples: the inverse refuses it on a device, not on the host"_domain_test = [](auto& fixture, std::string_view domain) { takesNonPowerOfTwoOnHostOnly<std::complex<float>, InverseSTFT<float>, TensorCollector<float>>(fixture, domain, 7UZ); } | overGraphs(kAllDomains);
};
} // namespace

int main() {
    "STFT"_test                        = spectra;
    "STFT and InverseSTFT"_test        = roundTrip;
    "STFT on every served domain"_test = domains;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
