#include <boost/ut.hpp>

#include <cmath>
#include <complex>
#include <numeric>
#include <print>
#include <vector>

#include <gnuradio-4.0/algorithm/rng/GaussianNoise.hpp>
#include <gnuradio-4.0/algorithm/rng/Xoshiro256pp.hpp>

using namespace boost::ut;

const boost::ut::suite noiseGeneratorTests = [] {
    using gr::rng::GaussianNoise;
    using gr::rng::Xoshiro256pp;

    "real Gaussian N(0,1) mean and variance"_test = [] {
        Xoshiro256pp          rng(42);
        GaussianNoise<double> gauss(rng);
        constexpr int         N   = 100'000;
        double                sum = 0.0;
        double                sq  = 0.0;
        for (int i = 0; i < N; ++i) {
            const double v = gauss();
            sum += v;
            sq += v * v;
        }
        const double mean     = sum / N;
        const double variance = sq / N - mean * mean;
        expect(approx(mean, 0.0, 0.02)) << std::format("mean {:.6f} not near 0", mean);
        expect(approx(variance, 1.0, 0.05)) << std::format("variance {:.6f} not near 1", variance);
    };

    "real Gaussian<float> mean and variance"_test = [] {
        Xoshiro256pp         rng(77);
        GaussianNoise<float> gauss(rng);
        constexpr int        N   = 100'000;
        double               sum = 0.0;
        double               sq  = 0.0;
        for (int i = 0; i < N; ++i) {
            const double v = static_cast<double>(gauss());
            sum += v;
            sq += v * v;
        }
        const double mean     = sum / N;
        const double variance = sq / N - mean * mean;
        expect(approx(mean, 0.0, 0.02)) << std::format("float Gaussian mean {:.6f}", mean);
        expect(approx(variance, 1.0, 0.05)) << std::format("float Gaussian variance {:.6f}", variance);
    };

    "Gaussian determinism"_test = [] {
        Xoshiro256pp          rng1(123);
        Xoshiro256pp          rng2(123);
        GaussianNoise<double> g1(rng1);
        GaussianNoise<double> g2(rng2);
        for (int i = 0; i < 1000; ++i) {
            expect(eq(g1(), g2())) << std::format("mismatch at {}", i);
        }
    };

    "complex Gaussian Option B: E[|n|^2] ~ 1"_test = [] {
        Xoshiro256pp          rng(55);
        GaussianNoise<double> gauss(rng);
        constexpr int         N         = 100'000;
        double                powerSum  = 0.0;
        double                realSqSum = 0.0;
        double                imagSqSum = 0.0;
        for (int i = 0; i < N; ++i) {
            const auto sample = gauss.complexSample();
            powerSum += std::norm(sample); // |n|^2
            realSqSum += sample.real() * sample.real();
            imagSqSum += sample.imag() * sample.imag();
        }
        const double meanPower = powerSum / N;
        const double realVar   = realSqSum / N;
        const double imagVar   = imagSqSum / N;
        expect(approx(meanPower, 1.0, 0.05)) << std::format("E[|n|^2] = {:.6f}, expected ~1", meanPower);
        expect(approx(realVar, 0.5, 0.05)) << std::format("Var(real) = {:.6f}, expected ~0.5", realVar);
        expect(approx(imagVar, 0.5, 0.05)) << std::format("Var(imag) = {:.6f}, expected ~0.5", imagVar);
    };

    "complex Gaussian<float> Option B"_test = [] {
        Xoshiro256pp         rng(88);
        GaussianNoise<float> gauss(rng);
        constexpr int        N        = 100'000;
        double               powerSum = 0.0;
        for (int i = 0; i < N; ++i) {
            const auto sample = gauss.complexSample();
            powerSum += static_cast<double>(std::norm(sample));
        }
        const double meanPower = powerSum / N;
        expect(approx(meanPower, 1.0, 0.05)) << std::format("float complex E[|n|^2] = {:.6f}", meanPower);
    };

    "Gaussian samples are not all identical"_test = [] {
        Xoshiro256pp          rng(0);
        GaussianNoise<double> gauss(rng);
        const double          first     = gauss();
        bool                  different = false;
        for (int i = 0; i < 100; ++i) {
            if (gauss() != first) {
                different = true;
                break;
            }
        }
        expect(different) << "all Gaussian samples identical";
    };

    "uniform noise range and mean"_test = [] {
        Xoshiro256pp  rng(31);
        constexpr int N      = 100'000;
        double        sum    = 0.0;
        double        minVal = 0.0;
        double        maxVal = 0.0;
        for (int i = 0; i < N; ++i) {
            const double v = rng.uniformM11<double>();
            sum += v;
            minVal = std::min(minVal, v);
            maxVal = std::max(maxVal, v);
        }
        expect(ge(minVal, -1.0)) << "uniform below -1";
        expect(lt(maxVal, 1.0)) << "uniform above +1";
        expect(approx(sum / N, 0.0, 0.01)) << std::format("uniform mean {:.6f}", sum / N);
    };

    "triangular noise range and mean"_test = [] {
        Xoshiro256pp  rng(31);
        constexpr int N      = 100'000;
        double        sum    = 0.0;
        double        minVal = 0.0;
        double        maxVal = 0.0;
        for (int i = 0; i < N; ++i) {
            const double v = rng.triangularM11<double>();
            sum += v;
            minVal = std::min(minVal, v);
            maxVal = std::max(maxVal, v);
        }
        expect(ge(minVal, -1.0)) << "triangular below -1";
        expect(lt(maxVal, 1.0)) << "triangular above +1";
        expect(approx(sum / N, 0.0, 0.01)) << std::format("triangular mean {:.6f}", sum / N);
    };
};

int main() { /* not needed for UT */ }
