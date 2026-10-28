#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <memory>
#include <numeric>
#include <print>
#include <utility>
#include <vector>

#include <gnuradio-4.0/algorithm/signal/NoiseGenerator.hpp>

using namespace boost::ut;

const boost::ut::suite noiseGeneratorWrapperTests = [] {
    using gr::signal::NoiseGenerator;
    using gr::signal::NoiseType;

    const auto checkAssignment = [](auto assign) {
        constexpr std::array<std::uint64_t, 4> kState{1, 2, 3, 4};
        NoiseGenerator<double>                 source;
        source.configure(NoiseType::Gaussian, 2.0, 1.0, 42);
        std::ranges::copy(kState, source._rng._state);
        source._gauss._spare    = 0.25;
        source._gauss._hasSpare = true;

        NoiseGenerator<double> destination;
        destination.configure(NoiseType::Uniform, 3.0, 5.0, 99);
        assign(destination, source);
        expect(std::addressof(destination._gauss._rng) == std::addressof(destination._rng));
        expect(std::ranges::equal(destination._rng._state, kState));
        assign(destination, destination);
        expect(eq(destination.generateSample(), 1.5));
        expect(!destination._gauss._hasSpare);
        expect(std::ranges::equal(destination._rng._state, kState));

        source.reset(77);
        expect(std::ranges::equal(destination._rng._state, kState));
        destination._type                    = NoiseType::Uniform;
        constexpr double kFirstUniformSample = 2.0 * (2.0 * 20480.0 / 9007199254740992.0 - 1.0) + 1.0;
        expect(eq(destination.generateSample(), kFirstUniformSample));
        constexpr std::array<std::uint64_t, 4> kAdvancedState{7, 0, 262146, 211106232532992};
        expect(std::ranges::equal(destination._rng._state, kAdvancedState));

        assign(destination, destination);
        expect(std::addressof(destination._gauss._rng) == std::addressof(destination._rng));
        expect(std::ranges::equal(destination._rng._state, kAdvancedState));

        destination._gauss._hasSpare = true;
        destination._gauss._spare    = 123.0;
        assign(destination, source);
        expect(!destination._gauss._hasSpare);
        expect(destination._type == NoiseType::Gaussian);
        expect(eq(destination._amplitude, 2.0) && eq(destination._offset, 1.0));
        expect(std::addressof(destination._gauss._rng) == std::addressof(destination._rng));
    };

    "copy assignment preserves Gaussian spare and owns its random engine"_test = [checkAssignment] { checkAssignment([](auto& destination, auto& source) { destination = source; }); };

    "move assignment preserves Gaussian spare and owns its random engine"_test = [checkAssignment] { checkAssignment([](auto& destination, auto& source) { destination = std::move(source); }); };

    "Uniform: range and mean"_test = [] {
        NoiseGenerator<double> gen;
        gen.configure(NoiseType::Uniform, 1.0, 0.0, 42);

        constexpr int N   = 100'000;
        double        sum = 0.0;
        for (int i = 0; i < N; ++i) {
            const double v = gen.generateSample();
            expect(ge(v, -1.0)) << std::format("uniform sample {} below -1: {}", i, v);
            expect(lt(v, 1.0)) << std::format("uniform sample {} above +1: {}", i, v);
            sum += v;
        }
        expect(approx(sum / N, 0.0, 0.02)) << std::format("uniform mean = {}", sum / N);
    };

    "Triangular: range and mean"_test = [] {
        NoiseGenerator<double> gen;
        gen.configure(NoiseType::Triangular, 1.0, 0.0, 77);

        constexpr int N   = 100'000;
        double        sum = 0.0;
        for (int i = 0; i < N; ++i) {
            const double v = gen.generateSample();
            expect(ge(v, -1.0)) << std::format("triangular sample {} below -1: {}", i, v);
            expect(lt(v, 1.0)) << std::format("triangular sample {} above +1: {}", i, v);
            sum += v;
        }
        expect(approx(sum / N, 0.0, 0.02)) << std::format("triangular mean = {}", sum / N);
    };

    "Gaussian: mean and variance"_test = [] {
        NoiseGenerator<double> gen;
        gen.configure(NoiseType::Gaussian, 1.0, 0.0, 123);

        constexpr int N   = 100'000;
        double        sum = 0.0;
        double        sq  = 0.0;
        for (int i = 0; i < N; ++i) {
            const double v = gen.generateSample();
            sum += v;
            sq += v * v;
        }
        const double mean     = sum / N;
        const double variance = sq / N - mean * mean;
        expect(approx(mean, 0.0, 0.02)) << std::format("gaussian mean = {}", mean);
        expect(approx(variance, 1.0, 0.05)) << std::format("gaussian variance = {}", variance);
    };

    "amplitude and offset applied correctly"_test = [] {
        for (auto type : {NoiseType::Uniform, NoiseType::Triangular, NoiseType::Gaussian}) {
            NoiseGenerator<double> gen;
            gen.configure(type, 3.0, 5.0, 42);

            constexpr int N   = 10'000;
            double        sum = 0.0;
            for (int i = 0; i < N; ++i) {
                sum += gen.generateSample();
            }
            const double mean = sum / N;
            // mean should be near offset (5.0) since base noise is zero-mean
            expect(approx(mean, 5.0, 0.3)) << std::format("type={} mean with A=3 O=5: {}", static_cast<int>(type), mean);
        }
    };

    "fill produces same sequence as generateSample"_test = [] {
        for (auto type : {NoiseType::Uniform, NoiseType::Triangular, NoiseType::Gaussian}) {
            NoiseGenerator<double> gen1;
            gen1.configure(type, 2.0, 1.0, 99);

            NoiseGenerator<double> gen2;
            gen2.configure(type, 2.0, 1.0, 99);

            constexpr std::size_t N = 50;
            std::vector<double>   filled(N);
            gen1.fill(filled);

            for (std::size_t i = 0; i < N; ++i) {
                const double sample = gen2.generateSample();
                expect(eq(filled[i], sample)) << std::format("type={} fill mismatch at {}", static_cast<int>(type), i);
            }
        }
    };

    "reset restores deterministic sequence"_test = [] {
        NoiseGenerator<double> gen;
        gen.configure(NoiseType::Gaussian, 1.0, 0.0, 55);

        std::vector<double> first(20);
        gen.fill(first);

        gen.reset(55);

        std::vector<double> second(20);
        gen.fill(second);

        for (std::size_t i = 0; i < first.size(); ++i) {
            expect(eq(first[i], second[i])) << std::format("reset mismatch at {}", i);
        }
    };

    "complex Uniform: independent real+imag, range check"_test = [] {
        NoiseGenerator<double> gen;
        gen.configure(NoiseType::Uniform, 1.0, 0.0, 42);

        constexpr int N       = 50'000;
        double        realSum = 0.0;
        double        imagSum = 0.0;
        for (int i = 0; i < N; ++i) {
            const auto s = gen.generateComplexSample();
            expect(ge(s.real(), -1.0) && lt(s.real(), 1.0)) << std::format("real out of range at {}", i);
            expect(ge(s.imag(), -1.0) && lt(s.imag(), 1.0)) << std::format("imag out of range at {}", i);
            realSum += s.real();
            imagSum += s.imag();
        }
        expect(approx(realSum / N, 0.0, 0.02)) << std::format("complex uniform real mean = {}", realSum / N);
        expect(approx(imagSum / N, 0.0, 0.02)) << std::format("complex uniform imag mean = {}", imagSum / N);
    };

    "complex Gaussian Option B: E[|n|^2] ~ 1"_test = [] {
        NoiseGenerator<double> gen;
        gen.configure(NoiseType::Gaussian, 1.0, 0.0, 88);

        constexpr int N        = 100'000;
        double        powerSum = 0.0;
        double        realSq   = 0.0;
        double        imagSq   = 0.0;
        for (int i = 0; i < N; ++i) {
            const auto s = gen.generateComplexSample();
            powerSum += std::norm(s);
            realSq += s.real() * s.real();
            imagSq += s.imag() * s.imag();
        }
        expect(approx(powerSum / N, 1.0, 0.05)) << std::format("E[|n|^2] = {}", powerSum / N);
        expect(approx(realSq / N, 0.5, 0.05)) << std::format("Var(real) = {}", realSq / N);
        expect(approx(imagSq / N, 0.5, 0.05)) << std::format("Var(imag) = {}", imagSq / N);
    };

    "complex Gaussian with amplitude: E[|n|^2] ~ A^2"_test = [] {
        NoiseGenerator<double> gen;
        gen.configure(NoiseType::Gaussian, 3.0, 0.0, 88);

        constexpr int N        = 100'000;
        double        powerSum = 0.0;
        for (int i = 0; i < N; ++i) {
            const auto s = gen.generateComplexSample();
            powerSum += std::norm(s);
        }
        expect(approx(powerSum / N, 9.0, 0.5)) << std::format("E[|n|^2] with A=3 = {}", powerSum / N);
    };

    "fillComplex produces same sequence as generateComplexSample"_test = [] {
        for (auto type : {NoiseType::Uniform, NoiseType::Triangular, NoiseType::Gaussian}) {
            NoiseGenerator<double> gen1;
            gen1.configure(type, 2.0, 1.0, 99);

            NoiseGenerator<double> gen2;
            gen2.configure(type, 2.0, 1.0, 99);

            constexpr std::size_t             N = 30;
            std::vector<std::complex<double>> filled(N);
            gen1.fillComplex(filled);

            for (std::size_t i = 0; i < N; ++i) {
                const auto sample = gen2.generateComplexSample();
                expect(eq(filled[i].real(), sample.real())) << std::format("type={} fillComplex real mismatch at {}", static_cast<int>(type), i);
                expect(eq(filled[i].imag(), sample.imag())) << std::format("type={} fillComplex imag mismatch at {}", static_cast<int>(type), i);
            }
        }
    };

    "float precision"_test = [] {
        NoiseGenerator<float> gen;
        gen.configure(NoiseType::Gaussian, 1.0f, 0.0f, 42);

        constexpr int N   = 10'000;
        double        sum = 0.0;
        for (int i = 0; i < N; ++i) {
            sum += static_cast<double>(gen.generateSample());
        }
        expect(approx(sum / N, 0.0, 0.05)) << std::format("float Gaussian mean = {}", sum / N);
    };

    "all noise types produce non-zero output"_test = [] {
        for (auto type : {NoiseType::Uniform, NoiseType::Triangular, NoiseType::Gaussian}) {
            NoiseGenerator<double> gen;
            gen.configure(type, 1.0, 0.0, 42);
            bool hasNonZero = false;
            for (int i = 0; i < 100; ++i) {
                if (gen.generateSample() != 0.0) {
                    hasNonZero = true;
                    break;
                }
            }
            expect(hasNonZero) << std::format("type={} all zeros", static_cast<int>(type));
        }
    };
};

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
