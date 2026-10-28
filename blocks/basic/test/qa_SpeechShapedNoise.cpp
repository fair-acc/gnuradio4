#include <boost/ut.hpp>
#include <gnuradio-4.0/basic/SpeechShapedNoise.hpp>
#include <gnuradio-4.0/test/DeviceTestHelper.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

const boost::ut::suite<"SpeechShapedNoise"> speechShapedNoiseTests = [] {
    using namespace boost::ut;
    using gr::testing::operator""_domain_test;

    "speech shaped noise has the requested stationary RMS on each served domain"_domain_test = [](auto& fixture, std::string_view domain) {
        constexpr std::size_t kSettling = 4096UZ;
        auto&                 clock     = fixture.template emplace<gr::testing::TagSource<std::uint8_t>>({{"n_samples_max", gr::Size_t{131072U}}, {"sample_rate", 48000.f}});
        auto&                 source    = fixture.template emplace<gr::basic::SpeechShapedNoise>({{"gr:compute_domain", domain == "host" ? std::string(domain) : std::string(domain) + "!"}, {"amplitude", 0.1f}});
        auto&                 sink      = fixture.template emplace<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        expect(fixture.template connect<"out", "clk">(clock, source).has_value());
        expect(fixture.template connect<"out", "in">(source, sink).has_value());
        expect(fixture.run().has_value());
        expect(eq(sink._samples.size(), 131072UZ));
        double sum = 0.0, squares = 0.0;
        for (std::size_t i = kSettling; i < sink._samples.size(); ++i) {
            const double value = static_cast<double>(sink._samples[i]);
            expect(std::isfinite(value));
            sum += value;
            squares += value * value;
        }
        const double count = static_cast<double>(sink._samples.size() - kSettling);
        expect(approx(sum / count, 0.0, 0.005));
        expect(approx(std::sqrt(squares / count), 0.1, 0.005));
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "starting the source again restores its seed and filter history"_test = [] {
        gr::basic::SpeechShapedNoise source;
        gr::PortOut<std::uint8_t>    clock;
        expect(clock.connect(source.clk).has_value());
        source.start();
        std::array<float, 17UZ> first{};
        for (float& sample : first) {
            sample = source._state.generateSample();
        }
        source.stop();
        source.start();
        for (float sample : first) {
            expect(eq(source._state.generateSample(), sample));
        }
        source.stop();
    };

    "chunk_size and amplitude keep the noise sequence, a change of seed restarts it"_test = [] {
        gr::basic::SpeechShapedNoise source;
        gr::PortOut<std::uint8_t>    clock;
        expect(clock.connect(source.clk).has_value());
        source.start();
        std::ignore         = source._state.generateSample();
        const auto advanced = source._state;
        source.chunk_size   = 200U;
        source.settingsChanged({}, {{"chunk_size", gr::Size_t{200U}}});
        auto expectedAdvanced = advanced;
        expect(eq(source._state.generateSample(), expectedAdvanced.generateSample())) << "chunk_size does not reset the filter";
        auto unscaled    = source._state;
        source.amplitude = 0.2f;
        source.settingsChanged({}, {{"amplitude", 0.2f}});
        expect(approx(source._state.generateSample(), 2.f * unscaled.generateSample(), 1.e-6f)) << "amplitude rescales the same sequence";
        source.seed = 7U;
        source.settingsChanged({}, {{"seed", std::uint64_t{7U}}});
        gr::basic::SpeechShapedNoise reference;
        gr::PortOut<std::uint8_t>    referenceClock;
        expect(referenceClock.connect(reference.clk).has_value());
        reference.seed      = 7U;
        reference.amplitude = 0.2f;
        reference.start();
        for (std::size_t i = 0UZ; i < 17UZ; ++i) {
            expect(eq(source._state.generateSample(), reference._state.generateSample()));
        }
        reference.stop();
        source.stop();
    };

    "a sample rate at or below twice the highest cut-off is refused with an error message"_test = [] {
        gr::basic::SpeechShapedNoise source;
        gr::PortOut<std::uint8_t>    clock;
        gr::MsgPortIn                errors;
        expect(clock.connect(source.clk).has_value());
        expect(source.msgOut.connect(errors).has_value());
        source.sample_rate = 12000.f;
        source.start();
        expect(eq(errors.streamReader().available(), 1UZ)) << "12 kHz does not exceed 2 x 6 kHz";
        source.stop();
    };

    "the shaping transfer function follows three second order Butterworth sections"_test = [] {
        gr::basic::SpeechShapedNoise source;
        gr::PortOut<std::uint8_t>    clock;
        expect(clock.connect(source.clk).has_value());
        source.start();
        constexpr std::array<double, 3UZ> kCutoffs{110.0, 2600.0, 6000.0};
        for (const double frequency : {110.0, 1000.0, 2600.0, 6000.0, 10000.0}) {
            const std::complex<double> z = std::polar(1.0, -2.0 * std::numbers::pi * frequency / 48000.0);
            std::complex<double>       response{1.0, 0.0};
            double                     expected = 1.0;
            for (std::size_t section = 0UZ; section < 3UZ; ++section) {
                const auto offset = 3UZ * section;
                response *= (source._state.b[offset] + source._state.b[offset + 1UZ] * z + source._state.b[offset + 2UZ] * z * z) / (source._state.a[offset] + source._state.a[offset + 1UZ] * z + source._state.a[offset + 2UZ] * z * z);
                const double ratio       = std::tan(std::numbers::pi * frequency / 48000.0) / std::tan(std::numbers::pi * kCutoffs[section] / 48000.0);
                const double fourthPower = ratio * ratio * ratio * ratio;
                expected *= section == 0UZ ? std::sqrt(fourthPower / (1.0 + fourthPower)) : 1.0 / std::sqrt(1.0 + fourthPower);
            }
            expect(approx(std::abs(response), expected, 1.e-7));
        }
        source.stop();
    };
};

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
