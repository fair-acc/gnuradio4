#include <boost/ut.hpp>
#include <gnuradio-4.0/fourier/STFT.hpp>
#include <gnuradio-4.0/localisation/PhaseSlopeDelay.hpp>
#include <gnuradio-4.0/test/DeviceTestHelper.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TensorSource.hpp>

const boost::ut::suite<"PhaseSlopeDelay"> phaseSlopeDelayTests = [] {
    using namespace boost::ut;
    using gr::testing::operator""_domain_test;
    "a known cross spectral phase slope gives the analytic delay on every served domain"_domain_test = [](auto& fixture, std::string_view domain) {
        using C                      = std::complex<double>;
        constexpr std::size_t kBins  = 64UZ;
        constexpr double      kDelay = 0.00025;
        std::vector<C>        spectrum(kBins);
        for (std::size_t i = 0UZ; i < kBins; ++i) {
            const double angle = 0.3 - 2.0 * std::numbers::pi * 125.0 * static_cast<double>(i) * kDelay;
            spectrum[i]        = {std::cos(angle), std::sin(angle)};
        }
        auto& source    = fixture.template emplace<gr::testing::TensorSource<C>>();
        source._frames  = {{{1UZ, kBins}, spectrum}};
        auto& estimator = fixture.template emplace<gr::localisation::PhaseSlopeDelay<std::complex<double>>>({{"sample_rate", 8000.f}, {"f_min", 500.f}, {"f_max", 3000.f}, {"gr:compute_domain", domain == "host" ? std::string(domain) : std::string(domain) + "!"}});
        auto& sink      = fixture.template emplace<gr::testing::TensorCollector<double>>();
        expect(fixture.template connect<"out", "in">(source, estimator).has_value());
        expect(fixture.template connect<"out", "in">(estimator, sink).has_value());
        expect(fixture.run().has_value());
        expect(eq(sink._elements.size(), 1UZ));
        expect(std::ranges::none_of(sink._tags, [](const auto& entry) { return entry.second.contains("delay_ambiguity_period"); })) << "a fit over many bins has no ambiguity period";
        if (sink._elements.size() == 1UZ) {
            expect(approx(sink._elements[0][0], kDelay, 1.e-10));
            expect(le(sink._elements[0][1], 1.e-6));
            expect(eq(sink._elements[0][2], 2500.0));
        }
    } | gr::testing::overGraphs(gr::testing::kAllDomains);
    "real samples: one-sided spectra of two rows 3 samples apart give -3 / 8000 s, as PairDelay's sign has it"_test = [] {
        using C                     = std::complex<float>;
        constexpr std::size_t kN    = 64UZ;
        constexpr std::size_t kBins = kN / 2UZ + 1UZ; // what STFT<float> emits
        const auto            noise = gr::testing::whiteNoise(kN, 9U);
        std::vector<C>        spectra(2UZ * kBins);
        for (std::size_t k = 0UZ; k < kBins; ++k) { // the DFT of each row, computed directly: row 1 is row 0 delayed circularly by 3
            std::complex<double> first{}, second{};
            for (std::size_t n = 0UZ; n < kN; ++n) {
                const std::complex<double> twiddle = std::polar(1.0, -2.0 * std::numbers::pi * static_cast<double>(k * n) / static_cast<double>(kN));
                first += noise[n] * twiddle;
                second += noise[(n + kN - 3UZ) % kN] * twiddle;
            }
            spectra[k]         = C(static_cast<float>(first.real()), static_cast<float>(first.imag()));
            spectra[kBins + k] = C(static_cast<float>(second.real()), static_cast<float>(second.imag()));
        }
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<gr::testing::TensorSource<C>>();
        source._frames                   = {{{2UZ, kBins}, spectra}};
        auto& cross                      = fixture.emplace<gr::localisation::PairCrossSpectra<float>>();
        auto& estimator                  = fixture.emplace<gr::localisation::PhaseSlopeDelay<float>>({{"sample_rate", 8000.f}, {"f_min", 500.f}, {"f_max", 3000.f}});
        auto& sink                       = fixture.emplace<gr::testing::TensorCollector<float>>();
        expect(fixture.connect<"out", "in">(source, cross).has_value());
        expect(fixture.connect<"out", "in">(cross, estimator).has_value());
        expect(fixture.connect<"out", "in">(estimator, sink).has_value());
        expect(fixture.run().has_value());
        expect(fatal(eq(sink._elements.size(), 1UZ)));
        expect(approx(sink._elements[0][0], -3.f / 8000.f, 1e-7f)) << "bins 125 Hz apart (8000 / 64), not 8000 / 33";
        expect(approx(sink._elements[0][2], 2500.f, 1e-3f)) << "the band 500..3000 Hz, as asked";
    };

    "a 440 Hz tone 0.2 ms later on row 1, through STFT and PairCrossSpectra, gives -0.2 ms from its one bin"_test = [] {
        constexpr float       kSampleRate   = 96000.f;
        constexpr double      kToneHz       = 440.0;
        constexpr double      kDelaySeconds = 0.2e-3;
        constexpr std::size_t kN            = 4800UZ; // 20 Hz bins: the tone sits on bin 22
        std::vector<float>    rows(2UZ * kN);
        for (std::size_t n = 0UZ; n < kN; ++n) {
            const double t = static_cast<double>(n) / static_cast<double>(kSampleRate);
            rows[n]        = static_cast<float>(std::sin(2.0 * std::numbers::pi * kToneHz * t));
            rows[kN + n]   = static_cast<float>(std::sin(2.0 * std::numbers::pi * kToneHz * (t - kDelaySeconds)));
        }
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<gr::testing::TensorSource<float>>();
        source._frames                   = {{{2UZ, kN}, rows}};
        auto& stft                       = fixture.emplace<gr::blocks::fft::STFT<float>>();
        auto& cross                      = fixture.emplace<gr::localisation::PairCrossSpectra<float>>();
        auto& estimator                  = fixture.emplace<gr::localisation::PhaseSlopeDelay<float>>({{"sample_rate", kSampleRate}, {"f_min", 435.f}, {"f_max", 445.f}});
        auto& sink                       = fixture.emplace<gr::testing::TensorCollector<float>>();
        expect(fixture.connect<"out", "in">(source, stft).has_value());
        expect(fixture.connect<"out", "in">(stft, cross).has_value());
        expect(fixture.connect<"out", "in">(cross, estimator).has_value());
        expect(fixture.connect<"out", "in">(estimator, sink).has_value());
        expect(fixture.run().has_value());
        expect(fatal(eq(sink._elements.size(), 1UZ)));
        expect(approx(sink._elements[0][0], static_cast<float>(-kDelaySeconds), 1e-8f)) << "delta phi / (2 pi 440 Hz), PairDelay's sign";
        const auto tagged = std::ranges::find_if(sink._tags, [](const auto& entry) { return entry.second.contains("delay_ambiguity_period"); });
        expect(fatal(tagged != sink._tags.end()));
        const auto periods = tagged->second.template value_or<gr::Tensor<float>>("delay_ambiguity_period", gr::Tensor<float>{});
        expect(fatal(eq(periods.size(), 1UZ)));
        expect(approx(periods[0], static_cast<float>(1.0 / kToneHz), 1e-9f)) << "one bin: the delay is known modulo 1 / 440 Hz";
    };

    "one tone exposes its phase ambiguity and zero energy is invalid"_test = [] {
        std::array<std::complex<double>, 9UZ> spectrum{};
        spectrum[4UZ]  = {std::cos(0.7), std::sin(0.7)};
        const auto one = gr::localisation::estimatePhaseSlope<double>(spectrum, 250.0, 1000.0, 1000.0);
        expect(approx(one.delaySeconds, -0.7 / (2000.0 * std::numbers::pi), 1.e-14));
        expect(eq(one.ambiguityPeriodSeconds, 0.001));
        expect(eq(one.bandwidthHz, 0.0));
        spectrum.fill({});
        const auto empty = gr::localisation::estimatePhaseSlope<double>(spectrum, 250.0, 0.0, 2000.0);
        expect(std::isnan(empty.delaySeconds) && std::isinf(empty.fitErrorRad));
    };

    "a one-bin estimate tags its ambiguity period"_test = [] {
        using C                     = std::complex<double>;
        constexpr std::size_t kBins = 8UZ;
        std::vector<C>        spectrum(kBins);
        spectrum[2UZ] = {std::cos(0.7), std::sin(0.7)};
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<gr::testing::TensorSource<C>>();
        source._frames                   = {{{1UZ, kBins}, spectrum}};
        auto& estimator                  = fixture.emplace<gr::localisation::PhaseSlopeDelay<std::complex<double>>>({{"sample_rate", 8000.f}});
        auto& sink                       = fixture.emplace<gr::testing::TensorCollector<double>>();
        expect(fixture.connect<"out", "in">(source, estimator).has_value());
        expect(fixture.connect<"out", "in">(estimator, sink).has_value());
        expect(fixture.run().has_value());
        const auto tagged = std::ranges::find_if(sink._tags, [](const auto& entry) { return entry.second.contains("delay_ambiguity_period"); });
        expect(fatal(tagged != sink._tags.end()));
        const auto periods = tagged->second.template value_or<gr::Tensor<double>>("delay_ambiguity_period", gr::Tensor<double>{});
        expect(fatal(eq(periods.size(), 1UZ)));
        expect(eq(periods[0], 0.0005)) << "bin 2 of 8 at 8 kHz is 2 kHz";
    };

    "an inverted band is refused and the previous band kept"_test = [] {
        gr::localisation::PhaseSlopeDelay<std::complex<double>> estimator;
        gr::MsgPortIn                                           errors;
        expect(estimator.msgOut.connect(errors).has_value());
        estimator.settings().init();
        std::ignore = estimator.settings().applyStagedParameters();
        expect(estimator.settings().set({{"f_min", 500.f}, {"f_max", 3000.f}}).empty());
        std::ignore = estimator.settings().activateContext();
        std::ignore = estimator.settings().applyStagedParameters();
        expect(estimator.settings().set({{"f_min", 4000.f}}).empty());
        std::ignore = estimator.settings().activateContext();
        std::ignore = estimator.settings().applyStagedParameters();
        expect(eq(estimator.f_min.value, 500.f));
        expect(eq(estimator.f_max.value, 3000.f));
        expect(eq(errors.streamReader().available(), 1UZ));
    };
};

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
