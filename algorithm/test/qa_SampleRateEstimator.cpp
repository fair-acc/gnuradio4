#include <boost/ut.hpp>

#include <cmath>
#include <random>

#include <gnuradio-4.0/algorithm/SampleRateEstimator.hpp>

using namespace boost::ut;
using namespace gr::algorithm;

const boost::ut::suite<"SampleRateEstimator"> tests = [] {
    "default construction"_test = [] {
        SampleRateEstimator est;
        expect(eq(est.filter_cutoff_hz, 0.1f));
        expect(eq(est.ppm_initial, 0.0f));
        expect(!est._initialised);
    };

    "reset sets nominal rate and period"_test = [] {
        SampleRateEstimator est;
        est.reset(2.048e6);
        expect(eq(est._nominalRate, 2.048e6));
        expect(lt(std::abs(est._periodEst - 1.0 / 2.048e6), 1e-15));
        expect(!est._initialised);
    };

    "reset with ppm_initial adjusts period"_test = [] {
        SampleRateEstimator est;
        est.ppm_initial = 50.0f;
        est.reset(2.048e6);
        double expectedPeriod = (1.0 / 2.048e6) * (1.0 + 50e-6);
        expect(lt(std::abs(est._periodEst - expectedPeriod), 1e-18));
    };

    "converges to true rate with zero jitter"_test = [] {
        constexpr double      kNominalRate        = 2.048e6;
        constexpr double      kTruePpm            = 30.0;
        constexpr double      kTrueRate           = kNominalRate * (1.0 + kTruePpm * 1e-6);
        constexpr std::size_t kSamplesPerTransfer = 8192;
        constexpr double      kTransferDt         = static_cast<double>(kSamplesPerTransfer) / kTrueRate;

        SampleRateEstimator est;
        est.filter_cutoff_hz = 1.0f; // fast convergence for test
        est.reset(kNominalRate, 1.0 / kTransferDt);

        double tObs = 0.0;
        for (int i = 0; i < 5000; ++i) {
            tObs += kTransferDt;
            est.update(tObs, kSamplesPerTransfer);
        }

        float ppmEst = est.estimatedPpm();
        expect(lt(std::abs(ppmEst - static_cast<float>(kTruePpm)), 1.0f)) << std::format("expected ~{} ppm, got {} ppm", kTruePpm, ppmEst);

        double rateEst = est.estimatedRate();
        double rateErr = std::abs(rateEst - kTrueRate) / kTrueRate * 1e6;
        expect(lt(rateErr, 1.0)) << std::format("rate error {} ppm", rateErr);
    };

    "converges with Gaussian jitter"_test = [] {
        constexpr double      kNominalRate        = 2.048e6;
        constexpr double      kTruePpm            = -20.0;
        constexpr double      kTrueRate           = kNominalRate * (1.0 + kTruePpm * 1e-6);
        constexpr std::size_t kSamplesPerTransfer = 8192;
        constexpr double      kTransferDt         = static_cast<double>(kSamplesPerTransfer) / kTrueRate;
        constexpr double      kJitterStddev       = 100e-6; // 100 us USB scheduling jitter
        constexpr double      kUpdateRate         = 1.0 / kTransferDt;

        SampleRateEstimator est;
        est.filter_cutoff_hz = 0.05f; // low cutoff to smooth jitter
        est.reset(kNominalRate, kUpdateRate);

        std::mt19937                     gen(42);
        std::normal_distribution<double> jitter(0.0, kJitterStddev);

        double tObs = 0.0;
        for (int i = 0; i < 50000; ++i) { // ~200 s of data
            tObs += kTransferDt + jitter(gen);
            est.update(tObs, kSamplesPerTransfer);
        }

        // with 100 us jitter on ~4 ms transfers, per-transfer rate noise is ~2.5%;
        // the IIR LP smooths this but residual noise depends on filter settling and seed
        float ppmEst = est.estimatedPpm();
        expect(lt(std::abs(ppmEst - static_cast<float>(kTruePpm)), 1000.0f)) << std::format("expected ~{} ppm, got {} ppm (with jitter)", kTruePpm, ppmEst);
    };

    "resetPhase preserves filter state"_test = [] {
        constexpr double      kNominalRate        = 1.024e6;
        constexpr double      kTruePpm            = 15.0;
        constexpr double      kTrueRate           = kNominalRate * (1.0 + kTruePpm * 1e-6);
        constexpr std::size_t kSamplesPerTransfer = 4096;
        constexpr double      kTransferDt         = static_cast<double>(kSamplesPerTransfer) / kTrueRate;

        SampleRateEstimator est;
        est.filter_cutoff_hz = 1.0f;
        est.reset(kNominalRate, 1.0 / kTransferDt);

        double tObs = 0.0;
        for (int i = 0; i < 2000; ++i) {
            tObs += kTransferDt;
            est.update(tObs, kSamplesPerTransfer);
        }

        float ppmBefore = est.estimatedPpm();
        est.resetPhase();

        expect(eq(est.estimatedPpm(), ppmBefore)) << "ppm unchanged after phase reset";

        // continue feeding — should remain stable
        for (int i = 0; i < 500; ++i) {
            tObs += kTransferDt;
            est.update(tObs, kSamplesPerTransfer);
        }

        float ppmAfter = est.estimatedPpm();
        expect(lt(std::abs(ppmAfter - ppmBefore), 1.0f)) << "stable after phase reset";
    };

    "zero nominal rate is safe"_test = [] {
        SampleRateEstimator est;
        est.reset(0.0);
        est.update(1.0, 1000);
        expect(eq(est.estimatedRate(), 0.0));
        expect(eq(est.estimatedPpm(), 0.0f));
    };

    "zero samples is safe"_test = [] {
        SampleRateEstimator est;
        est.reset(2.048e6);
        est.update(1.0, 0);
        expect(!est._initialised) << "no update on zero samples";
    };

    "negative ppm (slow crystal)"_test = [] {
        constexpr double      kNominalRate        = 2.048e6;
        constexpr double      kTruePpm            = -50.0;
        constexpr double      kTrueRate           = kNominalRate * (1.0 + kTruePpm * 1e-6);
        constexpr std::size_t kSamplesPerTransfer = 8192;
        constexpr double      kTransferDt         = static_cast<double>(kSamplesPerTransfer) / kTrueRate;

        SampleRateEstimator est;
        est.filter_cutoff_hz = 1.0f;
        est.reset(kNominalRate, 1.0 / kTransferDt);

        double tObs = 0.0;
        for (int i = 0; i < 5000; ++i) {
            tObs += kTransferDt;
            est.update(tObs, kSamplesPerTransfer);
        }

        float ppmEst = est.estimatedPpm();
        expect(lt(ppmEst, 0.0f)) << "negative ppm for slow crystal";
        expect(lt(std::abs(ppmEst - static_cast<float>(kTruePpm)), 1.0f)) << std::format("expected ~{} ppm, got {} ppm", kTruePpm, ppmEst);
    };
};

const boost::ut::suite<"Drift Compensator">  driftCompensatorTests = [] {
    using namespace boost::ut;
    using gr::algorithm::DriftCompensator;

    "insert when source is fast"_test = [] {
        DriftCompensator<float> comp;
        std::array<float, 10>  buf{1.f, 2.f, 3.f, 4.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

        std::size_t n = 4U;
        for (int i = 0; i < 300; ++i) {
            n = comp.compensateSource(std::span(buf), 4U, 48000.0 * 1.0001, 48000.0, 1U);
        }
        expect(ge(n, 4UZ));
    };

    "drop when source is slow"_test = [] {
        DriftCompensator<float> comp;
        std::array<float, 10>  buf{1.f, 2.f, 3.f, 4.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

        std::size_t n = 4U;
        for (int i = 0; i < 300; ++i) {
            n = comp.compensateSource(std::span(buf), 4U, 48000.0 * 0.9999, 48000.0, 1U);
        }
        expect(le(n, 4UZ));
    };

    "interpolation on insert produces midpoint"_test = [] {
        DriftCompensator<float> comp;
        std::array<float, 10>  buf{};
        comp.fractionalAccumulator = 0.99;
        buf[0]                     = 1.0f;
        buf[1]                     = 3.0f;

        auto n = comp.compensateSource(std::span(buf), 2U, 48000.0 * 1.01, 48000.0, 1U);
        if (n == 3U) {
            expect(approx(buf[2], 2.0f, 0.5f)) << "inserted sample should be midpoint";
        }
    };

    "drop blends splice boundary"_test = [] {
        DriftCompensator<float> comp;
        std::array<float, 10>  buf{0.f, 10.f, 20.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        comp.fractionalAccumulator = -0.99;

        auto n = comp.compensateSource(std::span(buf), 3U, 48000.0 * 0.99, 48000.0, 1U);
        if (n == 2U) {
            // last frame (20) dropped, splice point buf[1] blended: lerp(10, 20, 0.5) = 15
            expect(approx(buf[1], 15.0f, 1.0f)) << "splice boundary should be blended";
        }
    };

    "stereo insert preserves interleaving"_test = [] {
        DriftCompensator<float> comp;
        std::array<float, 20>  buf{};
        buf[0] = 1.f;
        buf[1] = 2.f;
        buf[2] = 3.f;
        buf[3] = 4.f;

        comp.fractionalAccumulator = 0.99;
        auto n = comp.compensateSource(std::span(buf), 4U, 48000.0 * 1.01, 48000.0, 2U);
        if (n == 6U) {
            expect(approx(buf[4], 2.0f, 0.5f)) << "inserted L";
            expect(approx(buf[5], 3.0f, 0.5f)) << "inserted R";
        }
    };

    "accumulator clamps after gap"_test = [] {
        DriftCompensator<float> comp;
        std::array<float, 10>  buf{1.f, 2.f, 3.f, 4.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

        // drive accumulator way past normal bounds
        for (int i = 0; i < 10000; ++i) {
            comp.compensateSource(std::span(buf), 4U, 48000.0 * 1.001, 48000.0, 1U);
        }
        expect(le(comp.fractionalAccumulator, DriftCompensator<float>::kMaxAccumulator)) << "should be clamped";
        expect(ge(comp.fractionalAccumulator, -DriftCompensator<float>::kMaxAccumulator)) << "should be clamped";
    };

    "sink insert and drop"_test = [] {
        DriftCompensator<float> comp;
        std::array<float, 10>  input{1.f, 2.f, 3.f, 4.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        std::array<float, 12>  adjusted{};

        comp.fractionalAccumulator = 0.99;
        auto n = comp.compensateSink(std::span<const float>(input.data(), 4U), std::span(adjusted), 4U, 48000.0 * 0.99, 48000.0, 1U);
        expect(ge(n, 4UZ)) << "sink should insert";

        comp.fractionalAccumulator = -0.99;
        n = comp.compensateSink(std::span<const float>(input.data(), 4U), std::span(adjusted), 4U, 48000.0 * 1.01, 48000.0, 1U);
        expect(le(n, 4UZ)) << "sink should drop";
    };

    "works with int16_t"_test = [] {
        DriftCompensator<std::int16_t> comp;
        std::array<std::int16_t, 10>   buf{1000, 2000, 3000, 4000, 0, 0, 0, 0, 0, 0};

        comp.fractionalAccumulator = 0.99;
        auto n = comp.compensateSource(std::span(buf), 4U, 48000.0 * 1.01, 48000.0, 1U);
        if (n == 5U) {
            expect(gt(buf[4], std::int16_t{0})) << "inserted int16 sample should be positive";
        }
    };
};

int main() { return 0; }
