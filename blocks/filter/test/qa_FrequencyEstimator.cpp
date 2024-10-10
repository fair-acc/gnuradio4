// FrequencyEstimatorTests.cpp
#include <boost/ut.hpp>
#include <cmath>
#include <random>
#include <vector>

#include <gnuradio-4.0/filter/FrequencyEstimator.hpp>

const boost::ut::suite FrequencyEstimatorTests = [] {
    using namespace boost::ut;
    using namespace gr::filter;

    // test frequencies
    constexpr static std::array testFrequencies{49.9f,                                          //
        50.0f, 50.001f, 50.002f, 50.003f, 50.004f, 50.005f, 50.006f, 50.007f, 50.008f, 50.009f, //
        50.01f, 50.02f, 50.03f, 50.04f, 50.05f, 50.06f, 50.07f, 50.08f, 50.09f,                 //
        50.1f, 50.2f, 50.3f, 50.4f, 50.5f, 50.6f, 50.7f, 50.8f, 50.9f, 51.0f};

    "Frequency Estimator General Tests"_test = [] {
        constexpr float       sample_rate = 1000.0f; // sampling frequency 1 kHz
        constexpr std::size_t numSamples  = 128;     // number of samples for generating test signal
        constexpr float       noiseAmp    = 0.01f;   // 1% noise level

        FrequencyEstimator<float, EstimatorMethod::Fast> estimator;
        estimator.sample_rate  = sample_rate;
        estimator.n_periods    = 3U;
        estimator.f_min        = 45.f;
        estimator.f_expected   = 50.f;
        estimator.f_max        = 55.f;
        estimator.min_fft_size = 128U;
        estimator.reset();

        auto generateTestSignal = [](float trueFreq, float fs, float noise) {
            static int                            n = 0;
            static std::mt19937                   gen(42); // fixed seed for unit-test reproducibility
            std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

            return std::sin(2.f * std::numbers::pi_v<float> * trueFreq * float(n++) / fs) + noise * dist(gen);
        };

        std::vector<float> frequencyTrue;
        std::vector<float> frequencyEstimates;
        for (float trueFreq : testFrequencies) {
            // Generate test signal
            estimator.reset();

            // process the samples and collect frequency estimates

            for (std::size_t i = 0; i < numSamples; i++) {
                float sample       = generateTestSignal(trueFreq, sample_rate, noiseAmp);
                float freqEstimate = estimator.processOne(sample);
                frequencyTrue.push_back(trueFreq);
                frequencyEstimates.push_back(freqEstimate);
            }

            // Get the last frequency estimate
            float           estimatedFreq = frequencyEstimates.back();
            constexpr float tolerance     = 0.03f;
            expect(approx(estimatedFreq, trueFreq, tolerance)) << fmt::format("estimated {:.6f} Hz vs true frequency {:.6f} Hz - diff: {:.6f} Hz", estimatedFreq, trueFreq, estimatedFreq - trueFreq);
        }
        fmt::println("frequencies:\n");
        for (std::size_t i = 0UZ; i < frequencyEstimates.size(); i++) {
            fmt::println("{:.4f}, {:.4f}", frequencyTrue[i], frequencyEstimates[i]);
        }
    };
};

int main() {
    return 0; // Not needed for unit tests
}
