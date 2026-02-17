#include <boost/ut.hpp>

#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <print>
#include <vector>

#include <gnuradio-4.0/algorithm/signal/SignalGeneratorCore.hpp>

using namespace boost::ut;

const boost::ut::suite signalGeneratorCoreTests = [] {
    using gr::signal::SignalGeneratorCore;
    using gr::signal::SignalType;

    // --- double output (identity, no conversion) ---

    "double sine matches ToneGenerator directly"_test = [] {
        constexpr float             phase = std::numbers::pi_v<float> / 4.f;
        SignalGeneratorCore<double> core;
        core.configure(SignalType::Sin, 256.f, 2048.f, phase, 1.f, 2.f, 0);

        gr::signal::ToneGenerator<double> ref;
        ref.configure(gr::signal::ToneType::Sin, 256., 2048., static_cast<double>(phase), 1., 2.);

        for (int i = 0; i < 100; ++i) {
            expect(eq(core.generateSample(), ref.generateSample())) << std::format("mismatch at {}", i);
        }
    };

    "double noise matches NoiseGenerator directly"_test = [] {
        SignalGeneratorCore<double> core;
        core.configure(SignalType::GaussianNoise, 0.f, 0.f, 0.f, 1.f, 0.f, 42);

        gr::signal::NoiseGenerator<double> ref;
        ref.configure(gr::signal::NoiseType::Gaussian, 1., 0., 42);

        for (int i = 0; i < 100; ++i) {
            expect(eq(core.generateSample(), ref.generateSample())) << std::format("mismatch at {}", i);
        }
    };

    "double fill matches generateSample sequence"_test = [] {
        SignalGeneratorCore<double> core1;
        core1.configure(SignalType::Saw, 100.f, 1000.f, 0.f, 1.f, 0.f, 0);

        SignalGeneratorCore<double> core2;
        core2.configure(SignalType::Saw, 100.f, 1000.f, 0.f, 1.f, 0.f, 0);

        std::vector<double> filled(50);
        core1.fill(filled);

        for (std::size_t i = 0; i < filled.size(); ++i) {
            expect(eq(filled[i], core2.generateSample())) << std::format("fill mismatch at {}", i);
        }
    };

    // --- float output (double -> float conversion) ---

    "float sine converts from double precision"_test = [] {
        constexpr float            phase = std::numbers::pi_v<float> / 4.f;
        SignalGeneratorCore<float> core;
        core.configure(SignalType::Sin, 256.f, 2048.f, phase, 1.f, 0.f, 0);

        SignalGeneratorCore<double> ref;
        ref.configure(SignalType::Sin, 256.f, 2048.f, phase, 1.f, 0.f, 0);

        for (int i = 0; i < 50; ++i) {
            const float  val    = core.generateSample();
            const double refVal = ref.generateSample();
            expect(approx(static_cast<double>(val), refVal, 1e-6)) << std::format("float mismatch at {}", i);
        }
    };

    // --- integer output (truncation + clamp) ---

    "int16_t sine output matches truncated double"_test = [] {
        SignalGeneratorCore<std::int16_t> core;
        core.configure(SignalType::Sin, 256.f, 2048.f, 0.f, 100.f, 0.f, 0);

        SignalGeneratorCore<double> ref;
        ref.configure(SignalType::Sin, 256.f, 2048.f, 0.f, 100.f, 0.f, 0);

        for (int i = 0; i < 100; ++i) {
            const auto   val      = core.generateSample();
            const double refVal   = ref.generateSample();
            const auto   expected = static_cast<std::int16_t>(std::clamp(refVal, -32768., 32767.));
            expect(eq(val, expected)) << std::format("int16 mismatch at {}: got {} expected {}", i, val, expected);
        }
    };

    "int8_t clamp at overflow"_test = [] {
        SignalGeneratorCore<std::int8_t> core;
        core.configure(SignalType::Sin, 256.f, 2048.f, 0.f, 200.f, 0.f, 0);

        bool hasMax = false;
        bool hasMin = false;
        for (int i = 0; i < 100; ++i) {
            const auto val = core.generateSample();
            expect(ge(val, std::int8_t(-128))) << std::format("below min at {}", i);
            expect(le(val, std::int8_t(127))) << std::format("above max at {}", i);
            if (val == std::int8_t(127)) {
                hasMax = true;
            }
            if (val == std::int8_t(-128)) {
                hasMin = true;
            }
        }
        expect(hasMax) << "never reached +127 clamp";
        expect(hasMin) << "never reached -128 clamp";
    };

    "uint8_t clamp negative to zero"_test = [] {
        SignalGeneratorCore<std::uint8_t> core;
        core.configure(SignalType::Sin, 256.f, 2048.f, 0.f, 200.f, 0.f, 0);

        bool hasZero = false;
        bool hasMax  = false;
        for (int i = 0; i < 100; ++i) {
            const auto val = core.generateSample();
            expect(le(val, std::uint8_t(255))) << std::format("above max at {}", i);
            if (val == std::uint8_t(0)) {
                hasZero = true;
            }
            if (val == std::uint8_t(200)) {
                hasMax = true; // amplitude=200, sin peak rounds to 200
            }
        }
        expect(hasZero) << "negative sine never clamped to 0";
        expect(hasMax) << "positive sine never reached 200";
    };

    "uint16_t with offset"_test = [] {
        SignalGeneratorCore<std::uint16_t> core;
        core.configure(SignalType::Sin, 100.f, 1000.f, 0.f, 1000.f, 32768.f, 0);

        bool inRange = true;
        for (int i = 0; i < 200; ++i) {
            const auto val = core.generateSample();
            if (val < 31768 || val > 33768) {
                inRange = false;
            }
        }
        expect(inRange) << "uint16 with offset out of expected range";
    };

    "int32_t noise output"_test = [] {
        SignalGeneratorCore<std::int32_t> core;
        core.configure(SignalType::UniformNoise, 0.f, 0.f, 0.f, 1000.f, 0.f, 42);

        bool hasPositive = false;
        bool hasNegative = false;
        for (int i = 0; i < 1000; ++i) {
            const auto val = core.generateSample();
            expect(ge(val, -1000)) << std::format("noise below -1000 at {}", i);
            expect(le(val, 1000)) << std::format("noise above 1000 at {}", i);
            if (val > 0) {
                hasPositive = true;
            }
            if (val < 0) {
                hasNegative = true;
            }
        }
        expect(hasPositive) << "no positive noise samples";
        expect(hasNegative) << "no negative noise samples";
    };

    "int64_t large amplitude no UB"_test = [] {
        SignalGeneratorCore<std::int64_t> core;
        core.configure(SignalType::Sin, 100.f, 1000.f, 0.f, 1e18f, 0.f, 0);

        for (int i = 0; i < 100; ++i) {
            const auto val = core.generateSample();
            expect(ge(val, std::numeric_limits<std::int64_t>::min()));
            expect(le(val, std::numeric_limits<std::int64_t>::max()));
        }
    };

    "uint64_t clamp negative values"_test = [] {
        SignalGeneratorCore<std::uint64_t> core;
        core.configure(SignalType::Sin, 256.f, 2048.f, 0.f, 100.f, 0.f, 0);

        bool hasZero = false;
        for (int i = 0; i < 100; ++i) {
            const auto val = core.generateSample();
            if (val == 0ULL) {
                hasZero = true;
            }
        }
        expect(hasZero) << "negative sine values never clamped to 0 for uint64";
    };

    // --- complex output ---

    "complex<double> sine analytic signal"_test = [] {
        SignalGeneratorCore<std::complex<double>> core;
        core.configure(SignalType::Sin, 100.f, 1000.f, 0.f, 1.f, 0.f, 0);

        for (int i = 0; i < 50; ++i) {
            const auto s = core.generateSample();
            expect(approx(std::abs(s), 1.0, 1e-12)) << std::format("|s| at {} = {}", i, std::abs(s));
        }
    };

    "complex<float> sine analytic signal"_test = [] {
        SignalGeneratorCore<std::complex<float>> core;
        core.configure(SignalType::Sin, 100.f, 1000.f, 0.f, 2.f, 0.f, 0);

        for (int i = 0; i < 50; ++i) {
            const auto s = core.generateSample();
            expect(approx(static_cast<double>(std::abs(s)), 2.0, 1e-5)) << std::format("|s| at {} = {}", i, std::abs(s));
        }
    };

    "complex<double> cos analytic signal"_test = [] {
        SignalGeneratorCore<std::complex<double>> core;
        core.configure(SignalType::Cos, 100.f, 1000.f, 0.f, 3.f, 0.f, 0);

        for (int i = 0; i < 50; ++i) {
            const auto s = core.generateSample();
            expect(approx(std::abs(s), 3.0, 1e-12)) << std::format("cos |s| at {} = {}", i, std::abs(s));
        }
    };

    "complex<double> Gaussian Option B"_test = [] {
        SignalGeneratorCore<std::complex<double>> core;
        core.configure(SignalType::GaussianNoise, 0.f, 0.f, 0.f, 1.f, 0.f, 88);

        constexpr int N        = 100'000;
        double        powerSum = 0.0;
        for (int i = 0; i < N; ++i) {
            const auto s = core.generateSample();
            powerSum += std::norm(s);
        }
        expect(approx(powerSum / N, 1.0, 0.05)) << std::format("E[|n|^2] = {}", powerSum / N);
    };

    "complex<double> square has zero imag"_test = [] {
        SignalGeneratorCore<std::complex<double>> core;
        core.configure(SignalType::Square, 100.f, 1000.f, 0.f, 1.f, 0.f, 0);

        for (int i = 0; i < 50; ++i) {
            const auto s = core.generateSample();
            expect(eq(s.imag(), 0.0)) << std::format("square imag at {} = {}", i, s.imag());
        }
    };

    "complex fillComplex matches generateSample"_test = [] {
        SignalGeneratorCore<std::complex<double>> core1;
        core1.configure(SignalType::Sin, 100.f, 1000.f, 0.f, 1.f, 0.f, 0);

        SignalGeneratorCore<std::complex<double>> core2;
        core2.configure(SignalType::Sin, 100.f, 1000.f, 0.f, 1.f, 0.f, 0);

        std::vector<std::complex<double>> filled(30);
        core1.fill(filled);

        for (std::size_t i = 0; i < filled.size(); ++i) {
            const auto s = core2.generateSample();
            expect(eq(filled[i].real(), s.real())) << std::format("fill real mismatch at {}", i);
            expect(eq(filled[i].imag(), s.imag())) << std::format("fill imag mismatch at {}", i);
        }
    };

    // --- reset and determinism ---

    "reset restores deterministic state"_test = [] {
        SignalGeneratorCore<double> core;
        core.configure(SignalType::GaussianNoise, 0.f, 0.f, 0.f, 1.f, 0.f, 55);

        std::vector<double> first(30);
        core.fill(first);

        core.reset();

        std::vector<double> second(30);
        core.fill(second);

        for (std::size_t i = 0; i < first.size(); ++i) {
            expect(eq(first[i], second[i])) << std::format("reset mismatch at {}", i);
        }
    };

    "tone reset restores waveform"_test = [] {
        SignalGeneratorCore<double> core;
        core.configure(SignalType::Sin, 100.f, 1000.f, 0.f, 1.f, 0.f, 0);

        std::vector<double> first(20);
        core.fill(first);
        core.reset();
        std::vector<double> second(20);
        core.fill(second);

        for (std::size_t i = 0; i < first.size(); ++i) {
            expect(eq(first[i], second[i])) << std::format("tone reset mismatch at {}", i);
        }
    };

    // --- all signal types produce non-zero output ---

    "all SignalType values produce non-zero output"_test = [] {
        for (auto type : {SignalType::Const, SignalType::Sin, SignalType::Cos, SignalType::Square, SignalType::Saw, SignalType::Triangle, SignalType::FastSin, SignalType::FastCos, SignalType::UniformNoise, SignalType::TriangularNoise, SignalType::GaussianNoise}) {
            SignalGeneratorCore<double> core;
            core.configure(type, 100.f, 1000.f, 0.f, 1.f, 0.f, 42);
            bool hasNonZero = false;
            for (int i = 0; i < 100; ++i) {
                if (core.generateSample() != 0.0) {
                    hasNonZero = true;
                    break;
                }
            }
            expect(hasNonZero) << std::format("type={} all zeros", static_cast<int>(type));
        }
    };

    "all SignalType values work with int16"_test = [] {
        for (auto type : {SignalType::Const, SignalType::Sin, SignalType::Cos, SignalType::Square, SignalType::Saw, SignalType::Triangle, SignalType::FastSin, SignalType::FastCos, SignalType::UniformNoise, SignalType::TriangularNoise, SignalType::GaussianNoise}) {
            SignalGeneratorCore<std::int16_t> core;
            core.configure(type, 100.f, 1000.f, 0.f, 100.f, 0.f, 42);
            bool hasNonZero = false;
            for (int i = 0; i < 100; ++i) {
                const auto val = core.generateSample();
                expect(ge(val, std::numeric_limits<std::int16_t>::min()));
                expect(le(val, std::numeric_limits<std::int16_t>::max()));
                if (val != 0) {
                    hasNonZero = true;
                }
            }
            expect(hasNonZero) << std::format("int16 type={} all zeros", static_cast<int>(type));
        }
    };

    "all SignalType values work with complex<double>"_test = [] {
        for (auto type : {SignalType::Const, SignalType::Sin, SignalType::Cos, SignalType::Square, SignalType::Saw, SignalType::Triangle, SignalType::FastSin, SignalType::FastCos, SignalType::UniformNoise, SignalType::TriangularNoise, SignalType::GaussianNoise}) {
            SignalGeneratorCore<std::complex<double>> core;
            core.configure(type, 100.f, 1000.f, 0.f, 1.f, 0.f, 42);
            bool hasNonZero = false;
            for (int i = 0; i < 100; ++i) {
                const auto s = core.generateSample();
                if (s.real() != 0.0 || s.imag() != 0.0) {
                    hasNonZero = true;
                    break;
                }
            }
            expect(hasNonZero) << std::format("complex type={} all zeros", static_cast<int>(type));
        }
    };
};

int main() { /* not needed for UT */ }
