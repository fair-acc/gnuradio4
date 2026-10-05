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

    constexpr double kTwoPi = 2. * std::numbers::pi_v<double>;

    "frequency change keeps the phase continuous"_test = [] {
        constexpr double fs = 1000., f1 = 2., f2 = 7., amplitude = 1.5, phase = 0.3;
        constexpr int    N = 36, M = 200;
        for (auto type : {ToneType::Sin, ToneType::FastSin}) {
            ToneGenerator<double> gen;
            gen.configure(type, f1, fs, phase, amplitude, 0.);
            double last = 0.;
            for (int n = 0; n < N; ++n) {
                last = gen.generateSample();
            }
            gen.configure(type, f2, fs, phase, amplitude, 0.);
            const double maxStep = kTwoPi * std::max(f1, f2) * amplitude / fs;
            for (int k = 0; k < M; ++k) {
                const double value    = gen.generateSample();
                const double expected = amplitude * std::sin(kTwoPi * (f1 * N + f2 * k) / fs + phase);
                expect(approx(value, expected, 1e-9)) << std::format("type={} k={}", static_cast<int>(type), k);
                if (k == 0) {
                    expect(lt(std::abs(value - last), maxStep)) << std::format("type={} jump at the change", static_cast<int>(type));
                }
            }
        }
    };

    "phase change steps the phase by the difference"_test = [] {
        constexpr double fs = 1000., f = 5., phase1 = 0.2, phase2 = 1.1;
        constexpr int    N = 123, M = 100;
        for (auto type : {ToneType::Sin, ToneType::FastSin}) {
            ToneGenerator<double> gen;
            gen.configure(type, f, fs, phase1, 1., 0.);
            for (int n = 0; n < N; ++n) {
                std::ignore = gen.generateSample();
            }
            gen.configure(type, f, fs, phase2, 1., 0.);
            for (int k = 0; k < M; ++k) {
                const double expected = std::sin(kTwoPi * f * (N + k) / fs + phase2);
                expect(approx(gen.generateSample(), expected, 1e-9)) << std::format("type={} k={}", static_cast<int>(type), k);
            }
        }
    };

    "float generator keeps a sample-accurate phase over 1e9 samples"_test = [] {
        constexpr float       fs = 1e6f, f = 1e3f;
        constexpr std::size_t kSamples = 1'000'000'000UZ, kChunk = 1UZ << 16U;

        ToneGenerator<float> gen;
        gen.configure(ToneType::FastSin, f, fs, 0.f, 1.f, 0.f);
        std::vector<float> chunk(kChunk);
        for (std::size_t n = 0UZ; n < kSamples; n += kChunk) {
            gen.fill(std::span(chunk).first(std::min(kChunk, kSamples - n)));
        }
        const auto expectedAt = [&](std::size_t n) { return std::fmod(kTwoPi * static_cast<double>(f) * static_cast<double>(n) / static_cast<double>(fs), kTwoPi); };

        const double fastValue = static_cast<double>(gen.generateSample());
        expect(lt(std::abs(fastValue - std::sin(expectedAt(kSamples))), 1e-3)) << "FastSin phase after 1e9 samples";

        gen.configure(ToneType::Sin, f, fs, 0.f, 1.f, 0.f);
        const double sinValue = static_cast<double>(gen.generateSample());
        expect(lt(std::abs(sinValue - std::sin(expectedAt(kSamples + 1UZ))), 1e-3)) << "Sin phase after 1e9 samples";

        gen.configure(ToneType::Cos, f, fs, 0.f, 1.f, 0.f);
        const double cosValue = static_cast<double>(gen.generateSample());
        expect(lt(std::abs(cosValue - std::cos(expectedAt(kSamples + 2UZ))), 1e-3)) << "Cos phase after 1e9 samples";
    };

    "unchanged settings reproduce the analytic waveform"_test = [] {
        constexpr double fs = 48000., f = 440., phase = 0.3, amplitude = 0.8, offset = 0.1;
        constexpr int    N = 10'000;

        const auto reference = [&](ToneType type, int n) {
            const double cycle = f * n / fs + phase / kTwoPi;
            const double frac  = cycle - std::floor(cycle);
            switch (type) {
            case ToneType::Sin:
            case ToneType::FastSin: return amplitude * std::sin(kTwoPi * cycle) + offset;
            case ToneType::Cos:
            case ToneType::FastCos: return amplitude * std::cos(kTwoPi * cycle) + offset;
            case ToneType::Square: return (frac < 0.5 ? amplitude : -amplitude) + offset;
            case ToneType::Saw: return amplitude * (2. * (cycle - std::floor(cycle + 0.5))) + offset;
            case ToneType::Triangle: return amplitude * (4. * std::abs(cycle - std::floor(cycle + 0.75) + 0.25) - 1.) + offset;
            case ToneType::Const: return amplitude + offset;
            }
            return 0.;
        };

        for (auto type : {ToneType::Sin, ToneType::Cos, ToneType::FastSin, ToneType::FastCos, ToneType::Square, ToneType::Saw, ToneType::Triangle}) {
            ToneGenerator<double> gen;
            gen.configure(type, f, fs, phase, amplitude, offset);
            ToneGenerator<float> genFloat;
            genFloat.configure(type, static_cast<float>(f), static_cast<float>(fs), static_cast<float>(phase), static_cast<float>(amplitude), static_cast<float>(offset));
            std::vector<double> filled(N);
            gen.fill(filled);
            for (int n = 0; n < N; ++n) {
                const double expected = reference(type, n);
                expect(approx(filled[static_cast<std::size_t>(n)], expected, 1e-9)) << std::format("double type={} n={}", static_cast<int>(type), n);
                expect(approx(static_cast<double>(genFloat.generateSample()), expected, 1e-4)) << std::format("float type={} n={}", static_cast<int>(type), n);
            }
        }
    };
};

int main() { /* not needed for UT */ }
