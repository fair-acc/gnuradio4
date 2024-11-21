#include <boost/ut.hpp>
#include <cmath>
#include <random>
#include <vector>

#include <gnuradio-4.0/filter/FrequencyEstimator.hpp>

namespace {
auto generateTestSignal = [](float trueFreq, float fs, float noise, size_t numSamples) {
    std::vector<float>                    samples(numSamples);
    std::mt19937                          gen(42); // Fixed seed for unit-test reproducibility
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    static float                          phase = 0.f; // ensures that sine-wave doesn't have non-physical discontinuities
    for (size_t i = 0; i < numSamples; ++i) {
        phase += 2.f * std::numbers::pi_v<float> * trueFreq / fs;
        samples[i] = std::sin(phase) + noise * dist(gen);
    }
    return samples;
};

template<typename EstimatorType, typename ProcessFunc>
void testFrequencyEstimator(EstimatorType& estimator, ProcessFunc processFunc, const std::vector<float>& testFrequencies, float sample_rate, size_t numSamples, float noiseAmp, float tolerance) {
    using namespace boost::ut;

    std::vector<float> frequencyTrue;
    std::vector<float> frequencyEstimates;
    std::vector<float> deviations;

    for (float trueFreq : testFrequencies) {
        estimator.reset();

        auto samples = generateTestSignal(trueFreq, sample_rate, noiseAmp, numSamples);
        processFunc(samples, frequencyTrue, frequencyEstimates, trueFreq);
        if (frequencyEstimates.empty()) {
            continue;
        }

        float estimatedFreq = frequencyEstimates.back();
        float deviation     = std::abs(estimatedFreq - trueFreq);
        deviations.push_back(deviation);

        expect(approx(estimatedFreq, trueFreq, tolerance)) << fmt::format("Estimator {}: Estimated {:.6f} Hz vs true frequency {:.6f} Hz - diff: {:.6f} Hz", gr::meta::type_name<EstimatorType>(), estimatedFreq, trueFreq, estimatedFreq - trueFreq);
    }

    if (!deviations.empty()) {
        float maxDeviation = *std::ranges::max_element(deviations);
        if (maxDeviation > tolerance) {
            fmt::println("Frequency estimates for {} exceed tolerance of {:.6f} Hz (max deviation: {:.6f} Hz):\ntrue [Hz], estimated [Hz], deviation [Hz]", gr::meta::type_name<EstimatorType>(), tolerance, maxDeviation);
            for (std::size_t i = 0; i < frequencyEstimates.size(); ++i) {
                fmt::println("{:.4f}, {:.4f}, {:.6f}", frequencyTrue[i], frequencyEstimates[i], std::abs(frequencyEstimates[i] - frequencyTrue[i]));
            }
        } else {
            fmt::println("{:100} - max deviation: {:.6f} Hz vs. tolerance: {:.6f} Hz", gr::meta::type_name<EstimatorType>(), maxDeviation, tolerance);
        }
    } else {
        fmt::println("No frequency estimates were generated for {}.", gr::meta::type_name<EstimatorType>());
    }
}

} // namespace

const boost::ut::suite<"FrequencyEstimatorTests"> FrequencyEstimatorTests = [] {
    using namespace boost::ut;
    using namespace gr::filter;

    constexpr static std::array<float, 30> testFrequencies{49.9f,                               //
        50.0f, 50.001f, 50.002f, 50.003f, 50.004f, 50.005f, 50.006f, 50.007f, 50.008f, 50.009f, //
        50.01f, 50.02f, 50.03f, 50.04f, 50.05f, 50.06f, 50.07f, 50.08f, 50.09f,                 //
        50.1f, 50.2f, 50.3f, 50.4f, 50.5f, 50.6f, 50.7f, 50.8f, 50.9f, 51.0f};

    "Frequency Estimator - Time Domain"_test = [] {
        constexpr float       sample_rate = 1000.0f; // sampling frequency 1 kHz
        constexpr std::size_t numSamples  = 128UZ;   // number of samples
        constexpr float       noiseAmp    = 0.01f;   // 1% noise level
        constexpr float       tolerance   = 0.03f;

        FrequencyEstimatorTimeDomain<float> estimator;
        estimator.sample_rate = sample_rate;
        estimator.n_periods   = 3U;
        estimator.f_min       = 45.f;
        estimator.f_expected  = 50.f;
        estimator.f_max       = 55.f;
        estimator.reset();

        // processing function using processOne
        auto processFunc = [&estimator](const std::vector<float>& samples, std::vector<float>& frequencyTrue, std::vector<float>& frequencyEstimates, float trueFreq) {
            for (const auto& sample : samples) {
                float freqEstimate = estimator.processOne(sample);
                frequencyTrue.push_back(trueFreq);
                frequencyEstimates.push_back(freqEstimate);
            }
        };

        testFrequencyEstimator(estimator, processFunc, std::vector<float>(testFrequencies.begin(), testFrequencies.end()), sample_rate, numSamples, noiseAmp, tolerance);
    };

    skip / "Frequency Estimator - Time Domain Decimating"_test = [] {
        constexpr float       sample_rate = 1000.0f; // sampling frequency 1 kHz
        constexpr std::size_t numSamples  = 1280UZ;  // number of samples (multiple of chunk size)
        constexpr float       noiseAmp    = 0.01f;   // 1% noise level
        constexpr float       tolerance   = 0.03f;

        FrequencyEstimatorTimeDomainDecimating<float> estimator;
        estimator.sample_rate = sample_rate;
        estimator.n_periods   = 3U;
        estimator.f_min       = 45.f;
        estimator.f_expected  = 50.f;
        estimator.f_max       = 55.f;
        estimator.reset();

        // processing function using processBulk
        auto processFunc = [&estimator](const std::vector<float>& samples, std::vector<float>& frequencyTrue, std::vector<float>& frequencyEstimates, float trueFreq) {
            std::size_t chunkSize = estimator.input_chunk_size;
            std::size_t numChunks = samples.size() / chunkSize;

            for (std::size_t i = 0; i < numChunks; ++i) {
                std::span<const float> inputChunk(samples.data() + i * chunkSize, chunkSize);
                float                  outputFrequency;
                std::span<float>       outputChunk(&outputFrequency, 1);
                expect(estimator.processBulk(inputChunk, outputChunk) == work::Status::OK);
                frequencyTrue.push_back(trueFreq);
                frequencyEstimates.push_back(outputFrequency);
            }
        };

        testFrequencyEstimator(estimator, processFunc, std::vector<float>(testFrequencies.begin(), testFrequencies.end()), sample_rate, numSamples, noiseAmp, tolerance);
    };

    "Frequency Estimator - Frequency Domain"_test = [] {
        constexpr float       sample_rate = 1000.0f; // sampling frequency 1 kHz
        constexpr std::size_t numSamples  = 4100UZ;  // number of samples
        constexpr float       noiseAmp    = 0.01f;   // 1% noise level
        constexpr float       tolerance   = 1.0f;

        FrequencyEstimatorFrequencyDomain<float> estimator;
        estimator.sample_rate  = sample_rate;
        estimator.f_min        = 45.f;
        estimator.f_expected   = 50.f;
        estimator.f_max        = 55.f;
        estimator.min_fft_size = 4096U;
        estimator.reset();

        // Processing function using processOne
        auto processFunc = [&estimator](const std::vector<float>& samples, std::vector<float>& frequencyTrue, std::vector<float>& frequencyEstimates, float trueFreq) {
            size_t i = 0;
            for (const auto& sample : samples) {
                float freqEstimate = estimator.processOne(sample);
                if (i > estimator.min_fft_size) {
                    frequencyTrue.push_back(trueFreq);
                    frequencyEstimates.push_back(freqEstimate);
                }
                ++i;
            }
        };

        testFrequencyEstimator(estimator, processFunc, std::vector<float>(testFrequencies.begin(), testFrequencies.end()), sample_rate, numSamples, noiseAmp, tolerance);
    };

    skip / "Frequency Estimator - Frequency Domain Decimating"_test = [] {
        constexpr float       sample_rate = 1000.0f; // sampling frequency 1 kHz
        constexpr std::size_t numSamples  = 40960UZ; // number of samples (multiple of chunk size)
        constexpr float       noiseAmp    = 0.01f;   // 1% noise level
        constexpr float       tolerance   = 1.0f;

        FrequencyEstimatorFrequencyDomainDecimating<float> estimator;
        estimator.sample_rate      = sample_rate;
        estimator.f_min            = 45.f;
        estimator.f_expected       = 50.f;
        estimator.f_max            = 55.f;
        estimator.min_fft_size     = 4096U;
        estimator.input_chunk_size = 4096U;
        estimator.reset();

        auto processFunc = [&estimator](const std::vector<float>& samples, std::vector<float>& frequencyTrue, std::vector<float>& frequencyEstimates, float trueFreq) {
            std::size_t chunkSize = estimator.input_chunk_size;
            std::size_t numChunks = samples.size() / chunkSize;

            for (std::size_t i = 0; i < numChunks; ++i) {
                std::span<const float> inputChunk(samples.data() + i * chunkSize, chunkSize);
                float                  outputFrequency;
                std::span<float>       outputChunk(&outputFrequency, 1);
                expect(estimator.processBulk(inputChunk, outputChunk) == work::Status::OK);
                frequencyTrue.push_back(trueFreq);
                frequencyEstimates.push_back(outputFrequency);
            }
        };

        testFrequencyEstimator(estimator, processFunc, std::vector<float>(testFrequencies.begin(), testFrequencies.end()), sample_rate, numSamples, noiseAmp, tolerance);
    };
};

int main() { return 0; /* not needed for UT */ }
