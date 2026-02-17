#include <boost/ut.hpp>

#include <cmath>
#include <complex>
#include <map>
#include <numbers>
#include <print>
#include <string>
#include <vector>

#include <gnuradio-4.0/algorithm/signal/ToneGenerator.hpp>

using namespace boost::ut;

const boost::ut::suite toneGeneratorTests = [] {
    using gr::signal::ToneGenerator;
    using gr::signal::ToneType;

    "numerical equivalence with existing SignalGenerator test vectors"_test = [] {
        // exact same parameters and expected values as qa_sources.cpp "SignalGenerator test"
        // sample_rate=2048, frequency=256, amplitude=1, offset=2, phase=pi/4
        constexpr std::size_t N      = 16;
        constexpr double      offset = 2.;

        struct WaveformCase {
            ToneType            type;
            std::vector<double> expected; // at amplitude=1, offset=0
        };

        // clang-format off
        const std::vector<WaveformCase> cases{
            {ToneType::Const,    {1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.}},
            {ToneType::Sin,      {0.707106, 1., 0.707106, 0., -0.707106, -1., -0.707106, 0., 0.707106, 1., 0.707106, 0., -0.707106, -1., -0.707106, 0.}},
            {ToneType::Cos,      {0.707106, 0., -0.707106, -1., -0.7071067, 0., 0.707106, 1., 0.707106, 0., -0.707106, -1., -0.707106, 0., 0.707106, 1.}},
            {ToneType::Square,   {1., 1., 1., -1., -1., -1., -1., 1., 1., 1., 1., -1., -1., -1., -1., 1.}},
            {ToneType::Saw,      {0.25, 0.5, 0.75, -1., -0.75, -0.5, -0.25, 0., 0.25, 0.5, 0.75, -1., -0.75, -0.5, -0.25, 0.}},
            {ToneType::Triangle, {0.5, 1., 0.5, 0., -0.5, -1., -0.5, 0., 0.5, 1., 0.5, 0., -0.5, -1., -0.5, 0.}},
        };
        // clang-format on

        for (const auto& [type, expected] : cases) {
            ToneGenerator<double> gen;
            gen.configure(type, 256., 2048., std::numbers::pi / 4., 1., offset);

            for (std::size_t i = 0; i < N; ++i) {
                const double val = gen.generateSample();
                const double exp = expected[i] + offset;
                expect(approx(exp, val, 1e-5)) << std::format("type={} i={} expected={} got={}", static_cast<int>(type), i, exp, val);
            }
        }
    };

    "continuity across multiple fill calls"_test = [] {
        ToneGenerator<double> gen;
        gen.configure(ToneType::Sin, 100., 1000., 0., 1., 0.);

        std::vector<double> block1(50);
        std::vector<double> block2(50);
        gen.fill(block1);
        gen.fill(block2);

        // generate reference in one shot
        ToneGenerator<double> ref;
        ref.configure(ToneType::Sin, 100., 1000., 0., 1., 0.);
        std::vector<double> full(100);
        ref.fill(full);

        for (std::size_t i = 0; i < 50; ++i) {
            expect(eq(block1[i], full[i])) << std::format("block1 mismatch at {}", i);
        }
        for (std::size_t i = 0; i < 50; ++i) {
            expect(eq(block2[i], full[50 + i])) << std::format("block2 mismatch at {}", i);
        }
    };

    "reset restarts waveform"_test = [] {
        ToneGenerator<double> gen;
        gen.configure(ToneType::Sin, 100., 1000., 0., 1., 0.);

        std::vector<double> first(10);
        gen.fill(first);
        gen.reset();
        std::vector<double> afterReset(10);
        gen.fill(afterReset);

        for (std::size_t i = 0; i < first.size(); ++i) {
            expect(eq(first[i], afterReset[i])) << std::format("reset mismatch at {}", i);
        }
    };

    "float precision"_test = [] {
        ToneGenerator<float> gen;
        gen.configure(ToneType::Sin, 256.f, 2048.f, std::numbers::pi_v<float> / 4.f, 1.f, 0.f);

        const float val = gen.generateSample();
        expect(approx(static_cast<double>(val), 0.707106, 1e-4)) << std::format("float sin(pi/4) = {}", val);
    };

    "fillComplex Sin produces analytic signal"_test = [] {
        ToneGenerator<double> gen;
        gen.configure(ToneType::Sin, 100., 1000., 0., 1., 0.);

        constexpr std::size_t             N = 10;
        std::vector<std::complex<double>> complexOut(N);
        gen.fillComplex(complexOut);

        // reference: real part should match scalar generateSample
        ToneGenerator<double> ref;
        ref.configure(ToneType::Sin, 100., 1000., 0., 1., 0.);

        for (std::size_t i = 0; i < N; ++i) {
            const double realRef = ref.generateSample();
            expect(approx(complexOut[i].real(), realRef, 1e-12)) << std::format("complex real mismatch at {}", i);
        }

        // magnitude should be ~amplitude for all samples (analytic signal property)
        for (std::size_t i = 0; i < N; ++i) {
            expect(approx(std::abs(complexOut[i]), 1.0, 1e-12)) << std::format("complex magnitude at {} = {}", i, std::abs(complexOut[i]));
        }
    };

    "fillComplex Cos produces analytic signal"_test = [] {
        ToneGenerator<double> gen;
        gen.configure(ToneType::Cos, 100., 1000., 0., 2., 0.);

        constexpr std::size_t             N = 10;
        std::vector<std::complex<double>> complexOut(N);
        gen.fillComplex(complexOut);

        ToneGenerator<double> ref;
        ref.configure(ToneType::Cos, 100., 1000., 0., 2., 0.);

        for (std::size_t i = 0; i < N; ++i) {
            const double realRef = ref.generateSample();
            expect(approx(complexOut[i].real(), realRef, 1e-12)) << std::format("cos complex real mismatch at {}", i);
            expect(approx(std::abs(complexOut[i]), 2.0, 1e-12)) << std::format("cos complex magnitude at {}", i);
        }
    };

    "fillComplex non-sinusoidal has zero imaginary"_test = [] {
        for (auto type : {ToneType::Const, ToneType::Square, ToneType::Saw, ToneType::Triangle}) {
            ToneGenerator<double> gen;
            gen.configure(type, 100., 1000., 0., 1., 0.);

            std::vector<std::complex<double>> out(20);
            gen.fillComplex(out);

            ToneGenerator<double> ref;
            ref.configure(type, 100., 1000., 0., 1., 0.);

            for (std::size_t i = 0; i < out.size(); ++i) {
                expect(eq(out[i].imag(), 0.0)) << std::format("type={} i={} imag={}", static_cast<int>(type), i, out[i].imag());
                expect(approx(out[i].real(), ref.generateSample(), 1e-12)) << std::format("type={} i={} real mismatch", static_cast<int>(type), i);
            }
        }
    };

    "FastSin short-term precision matches Sin"_test = [] {
        ToneGenerator<double> fast;
        fast.configure(ToneType::FastSin, 256., 2048., std::numbers::pi / 4., 1., 2.);

        ToneGenerator<double> ref;
        ref.configure(ToneType::Sin, 256., 2048., std::numbers::pi / 4., 1., 2.);

        for (std::size_t i = 0; i < 200; ++i) {
            const double fastVal = fast.generateSample();
            const double refVal  = ref.generateSample();
            expect(approx(fastVal, refVal, 1e-12)) << std::format("FastSin vs Sin at {}: fast={} ref={}", i, fastVal, refVal);
        }
    };

    "FastCos short-term precision matches Cos"_test = [] {
        ToneGenerator<double> fast;
        fast.configure(ToneType::FastCos, 256., 2048., std::numbers::pi / 4., 1., 2.);

        ToneGenerator<double> ref;
        ref.configure(ToneType::Cos, 256., 2048., std::numbers::pi / 4., 1., 2.);

        for (std::size_t i = 0; i < 200; ++i) {
            const double fastVal = fast.generateSample();
            const double refVal  = ref.generateSample();
            expect(approx(fastVal, refVal, 1e-12)) << std::format("FastCos vs Cos at {}: fast={} ref={}", i, fastVal, refVal);
        }
    };

    "FastSin long-term drift remains bounded"_test = [] {
        ToneGenerator<double> fast;
        fast.configure(ToneType::FastSin, 440., 48000., 0., 1., 0.);

        ToneGenerator<double> ref;
        ref.configure(ToneType::Sin, 440., 48000., 0., 1., 0.);

        double maxError = 0.;
        for (std::size_t i = 0; i < 100'000; ++i) {
            const double fastVal = fast.generateSample();
            const double refVal  = ref.generateSample();
            maxError             = std::max(maxError, std::abs(fastVal - refVal));
        }
        expect(lt(maxError, 1e-8)) << std::format("FastSin max error after 100k samples: {:.2e}", maxError);
    };

    "FastSin fillComplex produces analytic signal"_test = [] {
        ToneGenerator<double> gen;
        gen.configure(ToneType::FastSin, 100., 1000., 0., 1., 0.);

        constexpr std::size_t             N = 100;
        std::vector<std::complex<double>> out(N);
        gen.fillComplex(out);

        for (std::size_t i = 0; i < N; ++i) {
            expect(approx(std::abs(out[i]), 1.0, 1e-12)) << std::format("FastSin magnitude at {} = {}", i, std::abs(out[i]));
        }
    };

    "FastCos fillComplex produces analytic signal"_test = [] {
        ToneGenerator<double> gen;
        gen.configure(ToneType::FastCos, 100., 1000., 0., 2., 0.);

        constexpr std::size_t             N = 100;
        std::vector<std::complex<double>> out(N);
        gen.fillComplex(out);

        ToneGenerator<double> ref;
        ref.configure(ToneType::FastCos, 100., 1000., 0., 2., 0.);

        for (std::size_t i = 0; i < N; ++i) {
            const double realRef = ref.generateSample();
            expect(approx(out[i].real(), realRef, 1e-12)) << std::format("FastCos complex real mismatch at {}", i);
            expect(approx(std::abs(out[i]), 2.0, 1e-12)) << std::format("FastCos complex magnitude at {}", i);
        }
    };

    "FastSin reset restarts waveform"_test = [] {
        ToneGenerator<double> gen;
        gen.configure(ToneType::FastSin, 100., 1000., 0., 1., 0.);

        std::vector<double> first(20);
        gen.fill(first);
        gen.reset();
        std::vector<double> afterReset(20);
        gen.fill(afterReset);

        for (std::size_t i = 0; i < first.size(); ++i) {
            expect(eq(first[i], afterReset[i])) << std::format("FastSin reset mismatch at {}", i);
        }
    };

    "FastSin float precision"_test = [] {
        ToneGenerator<float> gen;
        gen.configure(ToneType::FastSin, 256.f, 2048.f, std::numbers::pi_v<float> / 4.f, 1.f, 0.f);

        const float val = gen.generateSample();
        expect(approx(static_cast<double>(val), 0.707106, 1e-4)) << std::format("float FastSin(pi/4) = {}", val);
    };

    "all waveform types produce non-zero output"_test = [] {
        for (auto type : {ToneType::Const, ToneType::Sin, ToneType::Cos, ToneType::Square, ToneType::Saw, ToneType::Triangle, ToneType::FastSin, ToneType::FastCos}) {
            ToneGenerator<double> gen;
            gen.configure(type, 100., 1000., 0., 1., 0.);
            bool hasNonZero = false;
            for (int i = 0; i < 100; ++i) {
                if (gen.generateSample() != 0.0) {
                    hasNonZero = true;
                    break;
                }
            }
            expect(hasNonZero) << std::format("type={} produced all zeros", static_cast<int>(type));
        }
    };
};

int main() { /* not needed for UT */ }
