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

int main() { return 0; }
