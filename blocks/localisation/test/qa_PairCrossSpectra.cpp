#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <format>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/localisation/PairCrossSpectra.hpp>
#include <gnuradio-4.0/test/DeviceTestHelper.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TensorSource.hpp>

namespace {
using namespace boost::ut;
using gr::localisation::PairCrossCorrelation;
using gr::localisation::PairCrossSpectra;
using gr::localisation::PairDelay;
using gr::localisation::Weighting;
using gr::testing::TensorCollector;
using gr::testing::TensorSource;

constexpr std::array kDelays{0, 3, -5};
constexpr std::array kLags{-3, 5, 8};

template<typename T>
[[nodiscard]] std::vector<T> delayedRows(std::span<const int> delays, std::size_t N) {
    using Real        = gr::meta::fundamental_base_value_type_t<T>;
    const auto     re = gr::testing::whiteNoise(N, 7U);
    const auto     im = gr::testing::whiteNoise(N, 11U);
    std::vector<T> frame(delays.size() * N);
    for (std::size_t m = 0UZ; m < delays.size(); ++m) {
        const std::size_t delay = static_cast<std::size_t>((delays[m] + static_cast<int>(N)) % static_cast<int>(N));
        for (std::size_t n = 0UZ; n < N; ++n) {
            const std::size_t source = (n + N - delay) % N;
            if constexpr (gr::meta::complex_like<T>) {
                frame[m * N + n] = T(static_cast<Real>(re[source]), static_cast<Real>(im[source]));
            } else {
                frame[m * N + n] = static_cast<T>(re[source]);
            }
        }
    }
    return frame;
}

template<typename T>
[[nodiscard]] constexpr double toleranceOf() {
    return std::same_as<gr::meta::fundamental_base_value_type_t<T>, float> ? 1e-4 : 1e-10;
}

const auto pairsAndBands = [] {
    "pairs of four elements in lexicographic order"_test = [] {
        constexpr std::array<std::array<std::size_t, 2>, 6> kExpected{{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}};
        expect(eq(gr::localisation::detail::pairsOf(4UZ), 6UZ));
        for (std::size_t p = 0UZ; p < kExpected.size(); ++p) {
            const auto pair = gr::localisation::detail::pairOf(p, 4UZ);
            expect(eq(pair.i, kExpected[p][0]) && eq(pair.j, kExpected[p][1])) << "pair" << p;
        }
    };

    "a band holds |f| for real samples and the signed offset for complex ones, N even or odd"_test = [] {
        // N = 8 at 8 Hz: bins 0..7 lie at 0, 1, 2, 3, -4, -3, -2, -1 Hz; N = 5 at 5 Hz: bins 0..4 at 0, 1, 2, -2, -1 Hz
        const gr::localisation::detail::Band<double> real8{.binHz = 1.0, .fftSize = 8UZ, .realSamples = true, .fMin = 1.0, .fMax = 3.0};
        const gr::localisation::detail::Band<double> complex8{.binHz = 1.0, .fftSize = 8UZ, .realSamples = false, .fMin = -2.0, .fMax = 1.0};
        const gr::localisation::detail::Band<double> real5{.binHz = 1.0, .fftSize = 5UZ, .realSamples = true, .fMin = 2.0, .fMax = 2.0};
        const gr::localisation::detail::Band<double> complex5{.binHz = 1.0, .fftSize = 5UZ, .realSamples = false, .fMin = 0.0, .fMax = 2.0};
        constexpr std::array<bool, 8>                kReal8{false, true, true, true, false, true, true, true};
        constexpr std::array<bool, 8>                kComplex8{true, true, false, false, false, false, true, true};
        constexpr std::array<bool, 5>                kReal5{false, false, true, true, false};
        constexpr std::array<bool, 5>                kComplex5{true, true, true, false, false};
        for (std::size_t k = 0UZ; k < 8UZ; ++k) {
            expect(eq(real8.holds(k), kReal8[k])) << "real, N 8, bin" << k;
            expect(eq(complex8.holds(k), kComplex8[k])) << "complex, N 8, bin" << k;
        }
        for (std::size_t k = 0UZ; k < 5UZ; ++k) {
            expect(eq(real5.holds(k), kReal5[k])) << "real, N 5, bin" << k;
            expect(eq(complex5.holds(k), kComplex5[k])) << "complex, N 5, bin" << k;
        }
    };
};

void expectSpectra(const TensorCollector<std::complex<double>>& sink, const std::vector<std::vector<std::complex<double>>>& expected, std::string_view context) {
    expect(eq(sink._elements.size(), expected.size())) << context;
    for (std::size_t f = 0UZ; f < std::min(sink._elements.size(), expected.size()); ++f) {
        for (std::size_t k = 0UZ; k < expected[f].size(); ++k) {
            expect(le(std::abs(sink._elements[f][k] - expected[f][k]), 1e-12)) << context << "frame" << f << "bin" << k;
        }
    }
}

const auto crossSpectra = [] {
    using C = std::complex<double>;

    "the cross-spectrum of each pair is X_i conj(X_j)"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       source = fixture.emplace<TensorSource<C>>();
        source._frames                     = {{.extents = {3UZ, 2UZ}, .elements = {C(1, 0), C(0, 1), C(2, 1), C(1, -1), C(0, -3), C(-1, 2)}}};
        auto& cross                        = fixture.emplace<PairCrossSpectra<double>>();
        auto& sink                         = fixture.emplace<TensorCollector<C>>();
        expect(fixture.connect<"out", "in">(source, cross).has_value());
        expect(fixture.connect<"out", "in">(cross, sink).has_value());
        expect(fixture.run().has_value());
        // (0,1): 1*(2-i) = 2-i, i*(1+i) = -1+i; (0,2): 1*(3i) = 3i, i*(-1-2i) = 2-i; (1,2): (2+i)(3i) = -3+6i, (1-i)(-1-2i) = -3-i
        expect(sink._shapes.size() == 1UZ && sink._shapes[0] == std::vector<std::size_t>{3UZ, 2UZ});
        expectSpectra(sink, {{C(2, -1), C(-1, 1), C(0, 3), C(2, -1), C(-3, 6), C(-3, -1)}}, "three elements");
    };

    "spectra of a single element stop the cross-spectrum"_test = [] {
        using Cf = std::complex<float>;
        gr::testing::GraphFixture<> fixture;
        auto&                       source = fixture.emplace<TensorSource<Cf>>();
        source._frames                     = {{.extents = {1UZ, 2UZ}, .elements = {Cf(1, 0), Cf(0, 1)}}};
        auto& cross                        = fixture.emplace<PairCrossSpectra<Cf>>();
        auto& sink                         = fixture.emplace<TensorCollector<Cf>>();
        expect(fixture.connect<"out", "in">(source, cross).has_value());
        expect(fixture.connect<"out", "in">(cross, sink).has_value());
        expect(!fixture.run().has_value());
        expect(sink._elements.empty());
    };

    "PHAT leaves unit magnitudes, none passes, the band zeroes the rest"_test = [] {
        using Dc = gr::complex<double>;
        // one-sided F = 5, N = 8 at 8 Hz: bins at 0, 1, 2, 3, 4 Hz; the band keeps 1..3
        const std::array<Dc, 5>                      G{Dc(4, 0), Dc(0, 2), Dc(-1, 1), Dc(3, 4), Dc(0, -0.5)};
        const gr::localisation::detail::Band<double> band = gr::localisation::detail::bandOf<double, float>(8.f, 8UZ, 1.f, 3.f);
        const std::array<C, 5>                       phat{C(0, 0), C(0, 1), C(-1, 1) / std::sqrt(2.0), C(0.6, 0.8), C(0, 0)};
        const std::array<C, 5>                       none{C(0, 0), C(0, 2), C(-1, 1), C(3, 4), C(0, 0)};
        for (const Weighting weighting : {Weighting::phat, Weighting::none}) {
            std::array<Dc, 5> out{};
            gr::localisation::detail::weightPair(G.data(), out.data(), G.size(), weighting, 0.0, band);
            const auto& expected = weighting == Weighting::phat ? phat : none;
            for (std::size_t k = 0UZ; k < out.size(); ++k) {
                expect(le(std::abs(C(out[k].real(), out[k].imag()) - expected[k]), 1e-12)) << gr::meta::enumName(weighting).value_or("") << "bin" << k;
            }
        }
    };

    "SCOT divides by sqrt(G_ii G_jj), Roth by G_ii of the pair's first element"_test = [] {
        using Dc = gr::complex<double>;
        const std::array<Dc, 2>                      G{Dc(2, 0), Dc(0, 3)};
        const std::array<double, 2>                  powerI{4.0, 9.0};
        const std::array<double, 2>                  powerJ{1.0, 1.0};
        const gr::localisation::detail::Band<double> band = gr::localisation::detail::bandOf<double, std::complex<float>>(1.f, 2UZ, std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());
        std::array<Dc, 2>                            out{};
        gr::localisation::detail::weightPair(G.data(), out.data(), G.size(), Weighting::scot, 0.0, band, powerI.data(), powerJ.data());
        expect(le(std::abs(C(out[0].real(), out[0].imag()) - C(1, 0)), 1e-12)) << "2 / sqrt(4 * 1)";
        expect(le(std::abs(C(out[1].real(), out[1].imag()) - C(0, 1)), 1e-12)) << "3j / sqrt(9 * 1)";
        gr::localisation::detail::weightPair(G.data(), out.data(), G.size(), Weighting::roth, 0.0, band, powerI.data(), powerJ.data());
        expect(le(std::abs(C(out[0].real(), out[0].imag()) - C(0.5, 0)), 1e-12)) << "2 / 4";
        expect(le(std::abs(C(out[1].real(), out[1].imag()) - C(0, 1. / 3.)), 1e-12)) << "3j / 9";
    };

    "eps adds a share of the band's largest magnitude to each divisor"_test = [] {
        using Dc = gr::complex<double>;
        const std::array<Dc, 2>                      G{Dc(3, 4), Dc(1, 0)};
        const gr::localisation::detail::Band<double> band = gr::localisation::detail::bandOf<double, std::complex<float>>(1.f, 2UZ, std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());
        std::array<Dc, 2>                            out{};
        gr::localisation::detail::weightPair(G.data(), out.data(), G.size(), Weighting::phat, 0.5, band);
        // |G| 5 and 1, largest 5: divisors 5 + 2.5 and 1 + 2.5
        expect(le(std::abs(C(out[0].real(), out[0].imag()) - C(3, 4) / 7.5), 1e-12));
        expect(le(std::abs(C(out[1].real(), out[1].imag()) - C(1, 0) / 3.5), 1e-12));
    };
};

template<typename T>
void expectKnownDelays(const TensorCollector<gr::meta::fundamental_base_value_type_t<T>>& sink, std::size_t maxLag, double height, std::string_view context) {
    const std::size_t width = 2UZ * maxLag + 1UZ;
    expect(eq(sink._elements.size(), 1UZ)) << context;
    if (sink._elements.size() != 1UZ) {
        return;
    }
    expect(sink._shapes[0] == std::vector<std::size_t>{3UZ, width}) << context;
    for (std::size_t p = 0UZ; p < 3UZ; ++p) {
        const auto        curve = std::span(sink._elements[0]).subspan(p * width, width);
        const std::size_t peak  = static_cast<std::size_t>(std::ranges::max_element(curve) - curve.begin());
        expect(eq(static_cast<int>(peak) - static_cast<int>(maxLag), kLags[p])) << context << "pair" << p;
        expect(le(std::abs(static_cast<double>(curve[peak]) - height), toleranceOf<T>())) << context << "pair" << p << "peak height";
    }
}

template<typename T>
void gccPhatOf(auto& fixture, std::string_view domain = {}) {
    using Real     = gr::meta::fundamental_base_value_type_t<T>;
    auto& source   = fixture.template emplace<TensorSource<T>>();
    source._frames = {{.extents = {3UZ, 64UZ}, .elements = delayedRows<T>(kDelays, 64UZ)}};
    gr::property_map settings{{"max_lag", gr::Size_t(12)}, {"eps", 0.f}};
    if (!domain.empty()) {
        settings.insert_or_assign("gr:compute_domain", std::string(domain) + "!");
    }
    auto& gcc  = fixture.template emplace<PairCrossCorrelation<T>>(std::move(settings));
    auto& sink = fixture.template emplace<TensorCollector<Real>>();
    expect(fixture.template connect<"out", "in">(source, gcc).has_value());
    expect(fixture.template connect<"out", "in">(gcc, sink).has_value());
    const auto ran = fixture.run();
    expect(ran.has_value()) << domain << (ran.has_value() ? std::string{} : ran.error().message);
    expectKnownDelays<T>(sink, 12UZ, 1.0, domain);
}

[[nodiscard]] std::vector<std::vector<double>> correlationsOf(std::size_t nFrames, gr::property_map settings) {
    gr::testing::GraphFixture<> fixture;
    auto&                       source = fixture.emplace<TensorSource<double>>();
    for (std::size_t f = 0UZ; f < nFrames; ++f) {
        source._frames.push_back({.extents = {3UZ, 64UZ}, .elements = delayedRows<double>(kDelays, 64UZ)});
    }
    settings.insert_or_assign("max_lag", gr::Size_t(12));
    settings.insert_or_assign("eps", 0.f);
    auto& gcc  = fixture.emplace<PairCrossCorrelation<double>>(std::move(settings));
    auto& sink = fixture.emplace<TensorCollector<double>>();
    expect(fixture.connect<"out", "in">(source, gcc).has_value());
    expect(fixture.connect<"out", "in">(gcc, sink).has_value());
    expect(fixture.run().has_value());
    return sink._elements;
}

const auto axes = [] {
    "the lag curves carry their axes: pairs, then lags in s from -max_lag / sample_rate by 1 / sample_rate; PairDelay its own pair and column axes"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       source = fixture.emplace<TensorSource<float>>();
        source._frames                     = {{.extents = {3UZ, 64UZ}, .elements = delayedRows<float>(kDelays, 64UZ)}, {.extents = {3UZ, 64UZ}, .elements = delayedRows<float>(kDelays, 64UZ)}};
        source._sampleRate                 = 1000.f;
        auto& gcc                          = fixture.emplace<PairCrossCorrelation<float>>({{"max_lag", gr::Size_t(12)}});
        auto& curves                       = fixture.emplace<TensorCollector<float>>();
        auto& peak                         = fixture.emplace<PairDelay<float>>();
        auto& delays                       = fixture.emplace<TensorCollector<float>>();
        expect(fixture.connect<"out", "in">(source, gcc).has_value());
        expect(fixture.connect<"out", "in">(gcc, curves).has_value());
        expect(fixture.connect<"out", "in">(gcc, peak).has_value());
        expect(fixture.connect<"out", "in">(peak, delays).has_value());
        expect(fixture.run().has_value());
        const std::string originKey{gr::tag::kAxisOrigin.key()};
        const std::string stepKey{gr::tag::kAxisStep.key()};
        const std::string nameKey{gr::tag::kAxisName.key()};
        const auto        axesTags = [&](const TensorCollector<float>& sink) { return sink._tags | std::views::filter([&](const auto& t) { return t.second.contains(stepKey); }) | std::ranges::to<std::vector>(); };
        const auto        lagAxes  = axesTags(curves);
        expect(fatal(eq(lagAxes.size(), 1UZ))) << "once: the second curve has the same lags";
        expect(eq(lagAxes[0].first, 0UZ));
        const auto origin = lagAxes[0].second.template value_or<gr::Tensor<double>>(originKey, gr::Tensor<double>{});
        const auto step   = lagAxes[0].second.template value_or<gr::Tensor<double>>(stepKey, gr::Tensor<double>{});
        expect(fatal(eq(origin.size(), 2UZ) && eq(step.size(), 2UZ)));
        expect(eq(origin[0], 0.0) && eq(step[0], 1.0)) << "pairs 0, 1, 2";
        expect(approx(origin[1], -0.012, 1e-15) && approx(step[1], 0.001, 1e-15)) << "lag -12 at 1 kHz is -12 ms";
        const auto delayAxes = axesTags(delays);
        expect(fatal(eq(delayAxes.size(), 1UZ))) << "PairDelay's own, the lag axes not forwarded";
        const auto names = delayAxes[0].second.template value_or<gr::Tensor<gr::pmt::Value>>(nameKey, gr::Tensor<gr::pmt::Value>{});
        expect(fatal(eq(names.size(), 2UZ)));
        expect(names[0].value_or(std::string{}) == std::string("pair") && names[1].value_or(std::string{}) == std::string("column"));
    };
};

const auto averaging = [] {
    "an average of the same frame twice is that frame's curve, for every weighting"_test = [] {
        for (const Weighting weighting : {Weighting::none, Weighting::phat, Weighting::scot, Weighting::roth}) {
            const std::string name{gr::meta::enumName(weighting).value_or("")};
            const auto        once  = correlationsOf(1UZ, {{"weighting", name}});
            const auto        twice = correlationsOf(2UZ, {{"weighting", name}, {"averages", gr::Size_t(2)}});
            expect(fatal(eq(once.size(), 1UZ) && eq(twice.size(), 1UZ))) << name << "two frames, one curve";
            expect(std::ranges::equal(once[0], twice[0], [](double a, double b) { return std::abs(a - b) <= 1e-12 * std::max(1.0, std::abs(a)); })) << name;
        }
    };

    "a second element at twice the gain: Roth, referred to the pair's first element, peaks at 2, SCOT at 1"_test = [] {
        constexpr std::size_t kN    = 64UZ;
        const auto            noise = gr::testing::whiteNoise(kN, 3U);
        std::vector<double>   frame(2UZ * kN);
        for (std::size_t n = 0UZ; n < kN; ++n) {
            frame[n]                   = noise[n];
            frame[kN + (n + 4UZ) % kN] = 2.0 * noise[n];
        }
        for (const auto& [weighting, height] : {std::pair{Weighting::roth, 2.0}, std::pair{Weighting::scot, 1.0}}) {
            gr::testing::GraphFixture<> fixture;
            auto&                       source = fixture.emplace<TensorSource<double>>();
            source._frames                     = {{.extents = {2UZ, kN}, .elements = frame}};
            auto& gcc                          = fixture.emplace<PairCrossCorrelation<double>>({{"max_lag", gr::Size_t(12)}, {"eps", 0.f}, {"weighting", std::string(gr::meta::enumName(weighting).value_or(""))}});
            auto& sink                         = fixture.emplace<TensorCollector<double>>();
            expect(fixture.connect<"out", "in">(source, gcc).has_value());
            expect(fixture.connect<"out", "in">(gcc, sink).has_value());
            expect(fixture.run().has_value());
            expect(fatal(eq(sink._elements.size(), 1UZ)));
            expect(approx(std::ranges::max(sink._elements[0]), height, 1e-9)) << gr::meta::enumName(weighting).value_or("") << "G_01 = 2 |X|^2 over G_00 = |X|^2 (Roth) or sqrt(G_00 G_11) = 2 |X|^2 (SCOT)";
        }
    };

    "SCOT and Roth over four frames of a circular delay peak at height 1 on each pair's delay, as PHAT does"_test = [] {
        for (const Weighting weighting : {Weighting::scot, Weighting::roth}) {
            gr::testing::GraphFixture<> fixture;
            auto&                       source = fixture.emplace<TensorSource<double>>();
            for (std::size_t f = 0UZ; f < 4UZ; ++f) {
                source._frames.push_back({.extents = {3UZ, 64UZ}, .elements = delayedRows<double>(kDelays, 64UZ)});
            }
            auto& gcc  = fixture.emplace<PairCrossCorrelation<double>>({{"max_lag", gr::Size_t(12)}, {"eps", 0.f}, {"weighting", std::string(gr::meta::enumName(weighting).value_or(""))}, {"averages", gr::Size_t(4)}});
            auto& sink = fixture.emplace<TensorCollector<double>>();
            expect(fixture.connect<"out", "in">(source, gcc).has_value());
            expect(fixture.connect<"out", "in">(gcc, sink).has_value());
            expect(fixture.run().has_value());
            expectKnownDelays<double>(sink, 12UZ, 1.0, gr::meta::enumName(weighting).value_or("")); // a circular delay keeps |X_i| = |X_j|
        }
    };
};

const auto gccPhat = [] {
    "GCC-PHAT of circularly delayed noise peaks at height 1 on each pair's delay difference"_test = []<typename T>() {
        gr::testing::GraphFixture<> fixture;
        gccPhatOf<T>(fixture);
    } | std::tuple<float, double, std::complex<float>, std::complex<double>>{};

    "max_lag 0 gives the lags +-(N/2 - 1); a band from gr:sample_rate scales the peak by the bins it keeps"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       source = fixture.emplace<TensorSource<double>>();
        source._frames                     = {{.extents = {3UZ, 64UZ}, .elements = delayedRows<double>(kDelays, 64UZ)}};
        source._sampleRate                 = 64.f;
        auto& gcc                          = fixture.emplace<PairCrossCorrelation<double>>({{"eps", 0.f}, {"f_min", 8.f}, {"f_max", 16.f}});
        auto& sink                         = fixture.emplace<TensorCollector<double>>();
        expect(fixture.connect<"out", "in">(source, gcc).has_value());
        expect(fixture.connect<"out", "in">(gcc, sink).has_value());
        expect(fixture.run().has_value());
        expect(eq(gcc.sample_rate.value, 64.f)) << "taken from the tag";
        expectKnownDelays<double>(sink, 31UZ, 18.0 / 64.0, "|f| 8..16 Hz keeps bins 8..16 and 48..56: 18 of 64");
    };

    "max_lag of N/2 or more stops the correlation"_test = [] {
        for (const gr::property_map& settings : {gr::property_map{{"max_lag", gr::Size_t(8)}}}) {
            gr::testing::GraphFixture<> fixture;
            auto&                       source = fixture.emplace<TensorSource<float>>();
            source._frames                     = {{.extents = {2UZ, 16UZ}, .elements = std::vector<float>(32UZ, 1.f)}};
            auto& gcc                          = fixture.emplace<PairCrossCorrelation<float>>(settings);
            auto& sink                         = fixture.emplace<TensorCollector<float>>();
            expect(fixture.connect<"out", "in">(source, gcc).has_value());
            expect(fixture.connect<"out", "in">(gcc, sink).has_value());
            expect(!fixture.run().has_value());
            expect(sink._elements.empty());
        }
    };
};

/** spectra [3,4] -> PairCrossSpectra on `domain`: each bin X_i conj(X_j) */
void crossSpectraOnDomain(auto& fixture, std::string_view domain) {
    using C           = std::complex<float>;
    const auto     re = gr::testing::whiteNoise(12UZ, 3U);
    const auto     im = gr::testing::whiteNoise(12UZ, 5U);
    std::vector<C> X(12UZ);
    for (std::size_t n = 0UZ; n < X.size(); ++n) {
        X[n] = C(static_cast<float>(re[n]), static_cast<float>(im[n]));
    }
    const gr::property_map placed{{"gr:compute_domain", std::string(domain) + "!"}};
    auto&                  source = fixture.template emplace<TensorSource<C>>();
    source._frames                = {{.extents = {3UZ, 4UZ}, .elements = X}};
    auto& cross                   = fixture.template emplace<PairCrossSpectra<C>>(placed);
    auto& sink                    = fixture.template emplace<TensorCollector<C>>();
    expect(fixture.template connect<"out", "in">(source, cross).has_value());
    expect(fixture.template connect<"out", "in">(cross, sink).has_value());
    const auto ran = fixture.run();
    expect(ran.has_value()) << domain << (ran.has_value() ? std::string{} : ran.error().message);
    expect(eq(sink._elements.size(), 1UZ)) << domain;
    if (sink._elements.size() != 1UZ) {
        return;
    }
    constexpr std::array<std::array<std::size_t, 2>, 3> kPairs{{{0, 1}, {0, 2}, {1, 2}}};
    for (std::size_t p = 0UZ; p < kPairs.size(); ++p) {
        for (std::size_t k = 0UZ; k < 4UZ; ++k) {
            const std::complex<double> G = std::complex<double>(X[kPairs[p][0] * 4UZ + k]) * std::conj(std::complex<double>(X[kPairs[p][1] * 4UZ + k]));
            expect(le(std::abs(std::complex<double>(sink._elements[0][p * 4UZ + k]) - G), 1e-5)) << domain << "pair" << p << "bin" << k;
        }
    }
}

const auto crossSpectraDomains = [] {
    using gr::testing::operator""_domain_test;
    using gr::testing::kAllDomains;
    using gr::testing::overGraphs;
    "GCC-PHAT of real frames finds the known delays on each served domain"_domain_test    = [](auto& fixture, std::string_view domain) { gccPhatOf<float>(fixture, domain); } | overGraphs(kAllDomains);
    "GCC-PHAT of complex frames finds the known delays on each served domain"_domain_test = [](auto& fixture, std::string_view domain) { gccPhatOf<std::complex<float>>(fixture, domain); } | overGraphs(kAllDomains);
    "cross-spectra weighted by PHAT on each served domain"_domain_test                    = [](auto& fixture, std::string_view domain) { crossSpectraOnDomain(fixture, domain); } | overGraphs(kAllDomains);
};

[[nodiscard]] std::vector<float> twoDelayedRows(std::size_t N, std::size_t delay) {
    const auto         noise = gr::testing::whiteNoise(N, 5U);
    std::vector<float> frame(2UZ * N);
    for (std::size_t n = 0UZ; n < N; ++n) {
        frame[n]                   = static_cast<float>(noise[n]);
        frame[N + (n + delay) % N] = static_cast<float>(noise[n]);
    }
    return frame;
}

const auto lagCurves = [] {
    "a lag curve's peak, its delay, and the ratio to the second local maximum"_test = [] {
        // lags -4..4; the peak at lag +2 with equal neighbours, a second local maximum 0.5 at lag -3
        constexpr std::array<double, 9> kCurve{0.1, 0.5, 0.2, 0.0, 0.1, 0.4, 1.0, 0.4, 0.0};
        std::array<double, 3>           row{};
        gr::localisation::detail::findPairDelay(kCurve.data(), kCurve.size(), 4UZ, 0.5, row.data());
        expect(le(std::abs(row[0] - 1.0), 1e-12)) << "lag 2 at 2 lags per second";
        expect(le(std::abs(row[1] - 1.0), 1e-12)) << "symmetric neighbours: the sample itself";
        expect(le(std::abs(row[2] - 2.0), 1e-12));

        gr::localisation::detail::findPairDelay(kCurve.data(), kCurve.size(), 2UZ, 0.5, row.data());
        expect(le(std::abs(row[0] - 1.0), 1e-12)) << "max_lag 2 still holds lag 2";
        expect(eq(row[2], std::numeric_limits<double>::infinity())) << "no second local maximum within +-2";
    };

    "a peak at the search's edge on a rising slope is not interpolated"_test = [] {
        // lags -4..4 rising throughout; searched within +-2 the largest is lag +2, below its outer neighbour
        constexpr std::array<double, 9> kRising{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9, 1.0};
        std::array<double, 3>           row{};
        gr::localisation::detail::findPairDelay(kRising.data(), kRising.size(), 2UZ, 1.0, row.data());
        expect(eq(row[0], 2.0)) << "lag 2, not a parabola's vertex beyond it";
        expect(eq(row[1], 0.6));
        expect(eq(row[2], std::numeric_limits<double>::infinity())) << "the curve's end samples are no local maxima";
    };
};

void delayThroughGccAndPairDelay(auto& fixture, std::string_view domain = {}) {
    constexpr std::size_t kN     = 128UZ;
    auto&                 source = fixture.template emplace<TensorSource<float>>();
    source._frames               = {{.extents = {2UZ, kN}, .elements = twoDelayedRows(kN, 5UZ)}};
    source._sampleRate           = 1000.f;
    gr::property_map placed;
    if (!domain.empty()) {
        placed.insert_or_assign("gr:compute_domain", std::string(domain) + "!");
    }
    gr::property_map gccSettings = placed;
    gccSettings.insert_or_assign("max_lag", gr::Size_t(20));
    auto& gcc  = fixture.template emplace<PairCrossCorrelation<float>>(gccSettings);
    auto& peak = fixture.template emplace<PairDelay<float>>(placed);
    auto& sink = fixture.template emplace<TensorCollector<float>>();
    expect(fixture.template connect<"out", "in">(source, gcc).has_value());
    expect(fixture.template connect<"out", "in">(gcc, peak).has_value());
    expect(fixture.template connect<"out", "in">(peak, sink).has_value());
    const auto ran = fixture.run();
    expect(ran.has_value()) << domain << (ran.has_value() ? std::string{} : ran.error().message);
    expect(eq(sink._elements.size(), 1UZ)) << domain;
    if (sink._elements.size() == 1UZ) {
        expect(le(std::abs(sink._elements[0][0] + 0.005f), 1e-6f)) << domain << "delay";
        expect(le(std::abs(sink._elements[0][1] - 1.f), 1e-4f)) << domain << "peak";
        expect(ge(sink._elements[0][2], 1e3f)) << domain << "confidence of a clean delta";
    }
}

const auto blocks = [] {
    "GCC-PHAT and PairDelay place a 5-sample delay at 1 kHz at -5 ms, in seconds from gr:sample_rate"_test = [] {
        gr::testing::GraphFixture<> fixture;
        delayThroughGccAndPairDelay(fixture);
    };

    "a direct path and its half-strength reflection 7 samples later: PairDelay takes the direct one, its confidence the two paths' ratio"_test = [] {
        constexpr std::size_t kN      = 128UZ;
        constexpr std::size_t kDirect = 5UZ;
        constexpr std::size_t kEcho   = 12UZ;
        const auto            noise   = gr::testing::whiteNoise(kN, 5U);
        std::vector<float>    frame(2UZ * kN, 0.f);
        for (std::size_t n = 0UZ; n < kN; ++n) {
            frame[n] = static_cast<float>(noise[n]);
            frame[kN + (n + kDirect) % kN] += static_cast<float>(noise[n]);
            frame[kN + (n + kEcho) % kN] += 0.5f * static_cast<float>(noise[n]);
        }
        const auto correlationAt = [&](std::size_t delay) { // the circular correlation of the two rows, computed directly
            double sum = 0.;
            for (std::size_t n = 0UZ; n < kN; ++n) {
                sum += static_cast<double>(frame[n]) * static_cast<double>(frame[kN + (n + delay) % kN]);
            }
            return sum;
        };
        gr::testing::GraphFixture<> fixture;
        auto&                       source = fixture.emplace<TensorSource<float>>();
        source._frames                     = {{.extents = {2UZ, kN}, .elements = frame}};
        source._sampleRate                 = 1000.f;
        auto& gcc                          = fixture.emplace<PairCrossCorrelation<float>>({{"max_lag", gr::Size_t(20)}, {"weighting", std::string("none")}});
        auto& peak                         = fixture.emplace<PairDelay<float>>();
        auto& sink                         = fixture.emplace<TensorCollector<float>>();
        expect(fixture.connect<"out", "in">(source, gcc).has_value());
        expect(fixture.connect<"out", "in">(gcc, peak).has_value());
        expect(fixture.connect<"out", "in">(peak, sink).has_value());
        expect(fixture.run().has_value());
        expect(fatal(eq(sink._elements.size(), 1UZ)));
        expect(le(std::abs(sink._elements[0][0] + 0.005f), 1e-4f)) << "the direct path at -5 ms, within a tenth of a sample";
        const double pathRatio = correlationAt(kDirect) / correlationAt(kEcho);
        expect(le(std::abs(static_cast<double>(sink._elements[0][2]) - pathRatio), 0.1 * pathRatio)) << std::format("confidence {} against the paths' ratio {}", sink._elements[0][2], pathRatio);
    };

    "averaging keeps the signal's sample_rate, so PairDelay still places a 5-sample delay at 1 kHz at -5 ms"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       source = fixture.emplace<TensorSource<float>>();
        source._frames                     = {{.extents = {2UZ, 128UZ}, .elements = twoDelayedRows(128UZ, 5UZ)}, {.extents = {2UZ, 128UZ}, .elements = twoDelayedRows(128UZ, 5UZ)}};
        source._sampleRate                 = 1000.f;
        auto& gcc                          = fixture.emplace<PairCrossCorrelation<float>>({{"max_lag", gr::Size_t(20)}, {"averages", gr::Size_t(2)}});
        auto& peak                         = fixture.emplace<PairDelay<float>>();
        auto& sink                         = fixture.emplace<TensorCollector<float>>();
        expect(fixture.connect<"out", "in">(source, gcc).has_value());
        expect(fixture.connect<"out", "in">(gcc, peak).has_value());
        expect(fixture.connect<"out", "in">(peak, sink).has_value());
        expect(fixture.run().has_value());
        expect(fatal(eq(sink._elements.size(), 1UZ))) << "two frames, one curve, one delay row";
        expect(le(std::abs(sink._elements[0][0] + 0.005f), 1e-6f)) << std::format("delay {} s: the rate is the signal's, not divided by averages", sink._elements[0][0]);
    };

    "an even curve length stops the peak search"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       source = fixture.emplace<TensorSource<double>>();
        source._frames                     = {{.extents = {1UZ, 4UZ}, .elements = {0.0, 1.0, 0.0, 0.0}}};
        auto& peak                         = fixture.emplace<PairDelay<double>>();
        auto& sink                         = fixture.emplace<TensorCollector<double>>();
        expect(fixture.connect<"out", "in">(source, peak).has_value());
        expect(fixture.connect<"out", "in">(peak, sink).has_value());
        expect(!fixture.run().has_value());
        expect(sink._elements.empty());
    };
};

const auto pairDelayDomains = [] {
    using gr::testing::operator""_domain_test;
    "GCC-PHAT and PairDelay find a known delay on each served domain"_domain_test = [](auto& fixture, std::string_view domain) { delayThroughGccAndPairDelay(fixture, domain); } | gr::testing::overGraphs(gr::testing::kAllDomains);
};
} // namespace

int main() {
    "pairs and bands"_test                         = pairsAndBands;
    "PairCrossSpectra and the PHAT weighting"_test = crossSpectra;
    "PairCrossCorrelation"_test                    = gccPhat;
    "PairCrossCorrelation averaging"_test          = averaging;
    "PairCrossCorrelation and PairDelay axes"_test = axes;
    "on every served domain"_test                  = crossSpectraDomains;
    "pair delay of a lag curve"_test               = lagCurves;
    "PairDelay"_test                               = blocks;
    "PairDelay on every domain"_test               = pairDelayDomains;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
