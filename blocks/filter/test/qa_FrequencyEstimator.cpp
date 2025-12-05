#include <boost/ut.hpp>
#include <cmath>
#include <limits>
#include <numeric>
#include <print>
#include <random>
#include <span>
#include <vector>

#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/filter/FrequencyEstimator.hpp>

namespace {
auto generateTestSignal = [](float trueFreq, float fs, float noise, std::size_t numSamples) {
    std::vector<float>                    samples(numSamples);
    std::mt19937                          gen(42); // Fixed seed for unit-test reproducibility
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    float                                 phase = 0.f; // Local phase - resets for each signal generation
    for (std::size_t i = 0UZ; i < numSamples; ++i) {
        phase += 2.f * std::numbers::pi_v<float> * trueFreq / fs;
        samples[i] = std::sin(phase) + noise * dist(gen);
    }
    return samples;
};

template<typename EstimatorType, typename ProcessFunc>
void testFrequencyEstimator(EstimatorType& estimator, ProcessFunc processFunc, std::span<const float> testFrequencies, float sample_rate, std::size_t numSamples, float noiseAmp, float tolerance) {
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

        expect(approx(estimatedFreq, trueFreq, tolerance)) << std::format("Estimator {}: Estimated {:.6f} Hz vs true frequency {:.6f} Hz - diff: {:.6f} Hz", gr::meta::type_name<EstimatorType>(), estimatedFreq, trueFreq, estimatedFreq - trueFreq);
    }

    if (!deviations.empty()) {
        float maxDeviation = *std::ranges::max_element(deviations);
        if (maxDeviation > tolerance) {
            std::println("Frequency estimates for {} exceed tolerance of {:.6f} Hz (max deviation: {:.6f} Hz):\ntrue [Hz], estimated [Hz], deviation [Hz]", gr::meta::type_name<EstimatorType>(), tolerance, maxDeviation);
            for (std::size_t i = 0UZ; i < frequencyEstimates.size(); ++i) {
                std::println("{:.4f}, {:.4f}, {:.6f}", frequencyTrue[i], frequencyEstimates[i], std::abs(frequencyEstimates[i] - frequencyTrue[i]));
            }
        } else {
            std::println("{:100} - max deviation: {:.6f} Hz vs. tolerance: {:.6f} Hz", gr::meta::type_name<EstimatorType>(), maxDeviation, tolerance);
        }
    } else {
        std::println("No frequency estimates were generated for {}.", gr::meta::type_name<EstimatorType>());
    }
}

} // namespace

const boost::ut::suite<"FrequencyEstimatorTests"> FrequencyEstimatorTests = [] {
    using namespace boost::ut;
    using namespace gr::filter;

    constexpr static std::array<float, 30> testFrequencies{
        49.9f,                                                                                  //
        50.0f, 50.001f, 50.002f, 50.003f, 50.004f, 50.005f, 50.006f, 50.007f, 50.008f, 50.009f, // sub-mHz resolution
        50.01f, 50.02f, 50.03f, 50.04f, 50.05f, 50.06f, 50.07f, 50.08f, 50.09f,                 // 10 mHz steps
        50.1f, 50.2f, 50.3f, 50.4f, 50.5f, 50.6f, 50.7f, 50.8f, 50.9f, 51.0f                    // 100 mHz steps
    };

    "Frequency Estimator - Time Domain"_test = [] {
        constexpr float       sample_rate = 1000.0f; // 1 kHz sampling - 20x oversampling for 50 Hz
        constexpr std::size_t numSamples  = 128UZ;   // ~6.4 periods at 50 Hz, sufficient for 3-period averaging
        constexpr float       noiseAmp    = 0.01f;   // 1% noise - typical for clean power systems
        constexpr float       tolerance   = 0.03f;   // 30 mHz tolerance - meets grid monitoring requirements

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

        testFrequencyEstimator(estimator, processFunc, testFrequencies, sample_rate, numSamples, noiseAmp, tolerance);
    };

    "Frequency Estimator - Time Domain Decimating"_test = [] {
        constexpr float       sample_rate = 1000.0f; // 1 kHz sampling
        constexpr std::size_t numSamples  = 1280UZ;  // Multiple of expected chunk size (128 * 10)
        constexpr float       noiseAmp    = 0.01f;   // 1% noise level
        constexpr float       tolerance   = 0.03f;   // 30 mHz tolerance

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

            for (std::size_t i = 0UZ; i < numChunks; ++i) {
                std::span<const float> inputChunk(samples.data() + i * chunkSize, chunkSize);
                float                  outputFrequency;
                std::span<float>       outputChunk(&outputFrequency, 1);
                expect(estimator.processBulk(inputChunk, outputChunk) == work::Status::OK);
                frequencyTrue.push_back(trueFreq);
                frequencyEstimates.push_back(outputFrequency);
            }
        };

        testFrequencyEstimator(estimator, processFunc, testFrequencies, sample_rate, numSamples, noiseAmp, tolerance);
    };

    "Frequency Estimator - Frequency Domain"_test = [] {
        constexpr float       sample_rate = 1000.0f; // 1 kHz sampling
        constexpr std::size_t numSamples  = 4100UZ;  // Slightly more than 4096 FFT size for settling
        constexpr float       noiseAmp    = 0.01f;   // 1% noise level
        // Note: FFT resolution is fs/N = 1000/4096 ≈ 0.244 Hz; Gaussian interpolation improves this
        // but 1 Hz tolerance is conservative for the 4096-point FFT
        constexpr float tolerance = 1.0f;

        FrequencyEstimatorFrequencyDomain<float> estimator;
        estimator.sample_rate  = sample_rate;
        estimator.f_min        = 45.f;
        estimator.f_expected   = 50.f;
        estimator.f_max        = 55.f;
        estimator.min_fft_size = 4096U;
        estimator.reset();

        // Processing function using processOne
        auto processFunc = [&estimator](const std::vector<float>& samples, std::vector<float>& frequencyTrue, std::vector<float>& frequencyEstimates, float trueFreq) {
            std::size_t i = 0UZ;
            for (const auto& sample : samples) {
                float freqEstimate = estimator.processOne(sample);
                if (i > estimator.min_fft_size) {
                    frequencyTrue.push_back(trueFreq);
                    frequencyEstimates.push_back(freqEstimate);
                }
                ++i;
            }
        };

        testFrequencyEstimator(estimator, processFunc, testFrequencies, sample_rate, numSamples, noiseAmp, tolerance);
    };

    "Frequency Estimator - Frequency Domain Decimating"_test = [] {
        constexpr float       sample_rate = 1000.0f; // 1 kHz sampling
        constexpr std::size_t numSamples  = 40960UZ; // 10x FFT size for multiple chunks
        constexpr float       noiseAmp    = 0.01f;   // 1% noise level
        constexpr float       tolerance   = 1.0f;    // 1 Hz tolerance (limited by FFT bin resolution)

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

            for (std::size_t i = 0UZ; i < numChunks; ++i) {
                std::span<const float> inputChunk(samples.data() + i * chunkSize, chunkSize);
                float                  outputFrequency;
                std::span<float>       outputChunk(&outputFrequency, 1);
                expect(estimator.processBulk(inputChunk, outputChunk) == work::Status::OK);
                frequencyTrue.push_back(trueFreq);
                frequencyEstimates.push_back(outputFrequency);
            }
        };

        testFrequencyEstimator(estimator, processFunc, testFrequencies, sample_rate, numSamples, noiseAmp, tolerance);
    };
};

template<typename T = float>
auto generateIQTestSignals(float freq, float fs, T ampRatio, T phaseShift, T dcOffset, T noise, std::size_t numSamples) {
    std::vector<T>                    ref(numSamples), resp(numSamples);
    std::mt19937                      gen(42);
    std::uniform_real_distribution<T> dist(T(-0.5), T(0.5));

    const T omega = T(2) * std::numbers::pi_v<T> * static_cast<T>(freq) / static_cast<T>(fs);
    for (std::size_t i = 0UZ; i < numSamples; ++i) {
        const T t = static_cast<T>(i);
        ref[i]    = std::sin(omega * t) + dcOffset + noise * dist(gen);
        resp[i]   = ampRatio * std::sin(omega * t + phaseShift) + dcOffset + noise * dist(gen);
    }
    return std::make_pair(ref, resp);
}

const boost::ut::suite<"IQDemodulator"> iqDemodulatorTests = [] {
    using namespace boost::ut;
    using namespace gr::filter;

    // test parameters: {carrier_freq, sample_rate} - must satisfy Nyquist: freq < fs/2
    struct TestCase {
        float freq;
        float fs;
    };

    "IQDemodulator basic amplitude/phase extraction"_test = []<typename T> {
        "frequency sweep"_test =
            [](TestCase tc) {
                "phase unit"_test = [tc](PhaseUnit phaseUnit) {
                    const float           fs         = tc.fs;
                    const float           freq       = tc.freq;
                    constexpr T           ampRatio   = T(0.8);  // response 80% of reference
                    constexpr T           phaseShift = T(0.5);  // ~28.6 degrees leading
                    constexpr T           dcOffset   = T(0.1);  // 10% DC offset
                    constexpr T           noise      = T(0.01); // 1% noise
                    constexpr std::size_t chunkSize  = 1024U;
                    constexpr std::size_t numChunks  = 100U;
                    constexpr std::size_t numSamples = chunkSize * numChunks;

                    auto [refSignal, respSignal] = generateIQTestSignals<T>(freq, fs, ampRatio, phaseShift, dcOffset, noise, numSamples);

                    IQDemodulator<T, Resampling<1024U, 1U, false>> demod;
                    demod.sample_rate      = fs;
                    demod.f_high_pass      = 100.f;
                    demod.f_low_pass       = 10000.f;
                    demod.phase_unit       = phaseUnit;
                    demod.invert_phase     = false;
                    demod.input_chunk_size = chunkSize;
                    demod.start();

                    std::vector<T> ampOut(numChunks);
                    std::vector<T> phaseOut(numChunks);
                    std::vector<T> freqOut(numChunks);

                    auto status = demod.processBulk(std::span{refSignal}, std::span{respSignal}, std::span{ampOut}, std::span{phaseOut}, std::span{freqOut});
                    expect(status == work::Status::OK);

                    constexpr std::size_t settleChunks = 20UZ; // skip first ~20 samples to allow HP and LP to settle
                    T                     ampSum       = T{0};
                    T                     phaseSum     = T{0};
                    T                     freqSum      = T{0};
                    std::size_t           count        = 0UZ;
                    for (std::size_t i = settleChunks; i < numChunks; ++i) {
                        ampSum += ampOut[i];
                        phaseSum += phaseOut[i];
                        freqSum += freqOut[i];
                        ++count;
                    }
                    const T ampMean   = ampSum / static_cast<T>(count);
                    const T phaseMean = phaseSum / static_cast<T>(count);
                    const T freqMean  = freqSum / static_cast<T>(count);

                    // tolerances: amplitude within 5%, phase within 0.1 rad (~6 deg), frequency within 5%
                    expect(approx(ampMean, ampRatio, T(0.05))) << std::format("amplitude deviation: {} vs. expected {}", ampMean, ampRatio);
                    if (phaseUnit == PhaseUnit::Degrees) {
                        const T expectedDeg = phaseShift * T(180) / std::numbers::pi_v<T>;
                        expect(approx(phaseMean, expectedDeg, T(3.0))) << std::format("phase deviation: {}° vs. expected {}°", phaseMean, expectedDeg);
                    } else {
                        expect(approx(phaseMean, phaseShift, T(0.1))) << std::format("phase deviation: {} rad vs. expected {} rad", phaseMean, phaseShift);
                    }
                    expect(approx(freqMean, static_cast<T>(freq), T(0.05) * static_cast<T>(freq))) << std::format("frequency deviation: {} vs. expected {}", freqMean, freq);
                } | std::vector{PhaseUnit::Radians, PhaseUnit::Degrees};
            } |
            std::vector<TestCase>{
                {100e3f, 1e6f}, // 100 kHz carrier at 1 MHz sample rate (10x oversampling)
                {5e6f, 62.5e6f} // 5 MHz carrier at 62.5 MHz sample rate (12.5x oversampling)
            };
    } | std::tuple<float, double>{};

    "IQDemodulator phase inversion"_test = [] {
        constexpr float       fs         = 1e6f;
        constexpr float       freq       = 150e3f;
        constexpr float       phaseShift = 0.3f;
        constexpr std::size_t chunkSize  = 256U;
        constexpr std::size_t numChunks  = 300U;
        constexpr std::size_t numSamples = chunkSize * numChunks;

        auto [refSignal, respSignal] = [=] {
            std::vector<float> ref(numSamples), resp(numSamples);
            const float        omega = 2.f * std::numbers::pi_v<float> * freq / fs;
            for (std::size_t i = 0UZ; i < numSamples; ++i) {
                const float t = static_cast<float>(i);
                ref[i]        = std::sin(omega * t);
                resp[i]       = std::sin(omega * t + phaseShift);
            }
            return std::make_pair(ref, resp);
        }();

        // test with invert_phase = false
        IQDemodulator<float, Resampling<256U, 1U, false>> demodNormal;
        demodNormal.sample_rate      = fs;
        demodNormal.f_high_pass      = 50.f;
        demodNormal.f_low_pass       = 15000.f;
        demodNormal.invert_phase     = false;
        demodNormal.input_chunk_size = chunkSize;
        demodNormal.start();

        std::vector<float> amp1(numChunks), phase1(numChunks), freq1(numChunks);
        std::ignore = demodNormal.processBulk(refSignal, respSignal, amp1, phase1, freq1);

        // test with invert_phase = true
        IQDemodulator<float, Resampling<256U, 1U, false>> demodInverted;
        demodInverted.sample_rate      = fs;
        demodInverted.f_high_pass      = 50.f;
        demodInverted.f_low_pass       = 15000.f;
        demodInverted.invert_phase     = true;
        demodInverted.input_chunk_size = chunkSize;
        demodInverted.start();

        std::vector<float> amp2(numChunks), phase2(numChunks), freq2(numChunks);
        std::ignore = demodInverted.processBulk(refSignal, respSignal, amp2, phase2, freq2);

        // after settling, phases should be negatives of each other
        const float phaseMean1 = std::accumulate(phase1.begin() + 100, phase1.end(), 0.f) / static_cast<float>(numChunks - 100);
        const float phaseMean2 = std::accumulate(phase2.begin() + 100, phase2.end(), 0.f) / static_cast<float>(numChunks - 100);
        expect(std::abs(phaseMean1 + phaseMean2) < 0.05f) << "inverted phases should sum to ~0:" << phaseMean1 << "+" << phaseMean2;
    };

    "IQDemodulator settings change resets filters"_test = [] {
        constexpr float       fs         = 1e6f;
        constexpr float       freq       = 100e3f;
        constexpr std::size_t chunkSize  = 256U;
        constexpr std::size_t numChunks  = 50U;
        constexpr std::size_t numSamples = chunkSize * numChunks;

        std::vector<float> ref(numSamples), resp(numSamples);
        const float        omega = 2.f * std::numbers::pi_v<float> * freq / fs;
        for (std::size_t i = 0UZ; i < numSamples; ++i) {
            ref[i]  = std::sin(omega * static_cast<float>(i));
            resp[i] = std::sin(omega * static_cast<float>(i) + 0.2f);
        }

        IQDemodulator<float, Resampling<256U, 1U, false>> demod;
        demod.sample_rate      = fs;
        demod.f_high_pass      = 100.f;
        demod.f_low_pass       = 10000.f;
        demod.input_chunk_size = chunkSize;
        demod.start();

        std::vector<float> amp1(numChunks), phase1(numChunks), freq1(numChunks);
        std::ignore = demod.processBulk(ref, resp, amp1, phase1, freq1);

        // change LP cutoff - should trigger filter reset
        property_map oldSettings, newSettings;
        newSettings["f_low_pass"] = 5000.f;
        demod.f_low_pass          = 5000.f;
        demod.settingsChanged(oldSettings, newSettings);

        // process again - filter states should have been reset
        std::vector<float> amp2(numChunks), phase2(numChunks), freq2(numChunks);
        std::ignore = demod.processBulk(ref, resp, amp2, phase2, freq2);

        // just verify no crash and outputs are valid (not NaN/inf)
        expect(std::isfinite(amp2.back()));
        expect(std::isfinite(phase2.back()));
        expect(std::isfinite(freq2.back()));
    };

    "IQDemodulator derivative methods"_test = [] {
        constexpr float       fs         = 1e6f;
        constexpr float       freq       = 100e3f;
        constexpr std::size_t chunkSize  = 512U;
        constexpr std::size_t numChunks  = 100U;
        constexpr std::size_t numSamples = chunkSize * numChunks;

        auto [refSignal, respSignal] = generateIQTestSignals<float>(freq, fs, 0.8f, 0.3f, 0.f, 0.01f, numSamples);

        for (auto method : {DerivativeMethod::SymmetricDifference, DerivativeMethod::SavitzkyGolay5, DerivativeMethod::SavitzkyGolay7}) {
            IQDemodulator<float, Resampling<512U, 1U, false>> demod;
            demod.sample_rate       = fs;
            demod.f_high_pass       = 50.f;
            demod.f_low_pass        = 20000.f;
            demod.derivative_method = method;
            demod.input_chunk_size  = chunkSize;
            demod.start();

            std::vector<float> ampOut(numChunks), phaseOut(numChunks), freqOut(numChunks);
            auto               status = demod.processBulk(refSignal, respSignal, ampOut, phaseOut, freqOut);
            expect(status == work::Status::OK);

            // verify outputs are finite after settling
            const float ampMean = std::accumulate(ampOut.begin() + 30, ampOut.end(), 0.f) / static_cast<float>(numChunks - 30);
            expect(std::isfinite(ampMean) && ampMean > 0.f) << "derivative method " << static_cast<int>(method) << " amplitude:" << ampMean;
        }
    };

    "IQDemodulator zero/near-zero amplitude inputs"_test = [] {
        // simulates ADC failure mode or disconnected signal
        constexpr float       fs         = 1e6f;
        constexpr std::size_t chunkSize  = 256U;
        constexpr std::size_t numChunks  = 50U;
        constexpr std::size_t numSamples = chunkSize * numChunks;

        // generate zero-amplitude signals
        std::vector<float> ref(numSamples, 0.f);
        std::vector<float> resp(numSamples, 0.f);

        IQDemodulator<float, Resampling<256U, 1U, false>> demod;
        demod.sample_rate      = fs;
        demod.f_high_pass      = 100.f;
        demod.f_low_pass       = 10000.f;
        demod.input_chunk_size = chunkSize;
        demod.start();

        std::vector<float> ampOut(numChunks), phaseOut(numChunks), freqOut(numChunks);
        auto               status = demod.processBulk(ref, resp, ampOut, phaseOut, freqOut);
        expect(status == work::Status::OK);

        // all outputs should be zero (no signal) and finite (no NaN/inf)
        for (std::size_t i = 10; i < numChunks; ++i) {
            expect(std::isfinite(ampOut[i])) << "amplitude should be finite at chunk" << i;
            expect(std::isfinite(phaseOut[i])) << "phase should be finite at chunk" << i;
            expect(std::isfinite(freqOut[i])) << "frequency should be finite at chunk" << i;
            expect(ampOut[i] == 0.f) << "amplitude should be zero for zero input";
        }
    };

    "IQDemodulator DC-only input (no carrier)"_test = [] {
        // DC offset without any AC component
        // HP filter blocks DC on both ref and resp, leaving only numerical noise
        // ratio of two small noise values is unstable, so we check for bounded output
        constexpr float       fs         = 1e6f;
        constexpr float       dcLevel    = 0.5f;
        constexpr std::size_t chunkSize  = 256U;
        constexpr std::size_t numChunks  = 100U;
        constexpr std::size_t numSamples = chunkSize * numChunks;

        std::vector<float> ref(numSamples, dcLevel);
        std::vector<float> resp(numSamples, dcLevel * 0.8f);

        IQDemodulator<float, Resampling<256U, 1U, false>> demod;
        demod.sample_rate      = fs;
        demod.f_high_pass      = 100.f;
        demod.f_low_pass       = 10000.f;
        demod.input_chunk_size = chunkSize;
        demod.start();

        std::vector<float> ampOut(numChunks), phaseOut(numChunks), freqOut(numChunks);
        auto               status = demod.processBulk(ref, resp, ampOut, phaseOut, freqOut);
        expect(status == work::Status::OK);

        // after HP filter settles, outputs should be finite (HP blocks DC, leaving noise)
        for (std::size_t i = 50; i < numChunks; ++i) {
            expect(std::isfinite(ampOut[i])) << "amplitude should be finite for DC input";
            expect(std::isfinite(phaseOut[i])) << "phase should be finite for DC input";
            expect(std::isfinite(freqOut[i])) << "frequency should be finite for DC input";
        }
    };

    "IQDemodulator frequency at HP cutoff boundary"_test = [] {
        // carrier frequency equal to HP cutoff
        // HP attenuates both ref and resp equally, so amplitude RATIO should be preserved
        constexpr float       fs         = 1e6f;
        constexpr float       f_hp       = 1000.f;
        constexpr float       freq       = 1000.f; // exactly at HP cutoff
        constexpr float       ampRatio   = 0.8f;
        constexpr std::size_t chunkSize  = 1024U;
        constexpr std::size_t numChunks  = 200U;
        constexpr std::size_t numSamples = chunkSize * numChunks;

        auto [refSignal, respSignal] = generateIQTestSignals<float>(freq, fs, ampRatio, 0.3f, 0.f, 0.f, numSamples);

        IQDemodulator<float, Resampling<1024U, 1U, false>> demod;
        demod.sample_rate      = fs;
        demod.f_high_pass      = f_hp;
        demod.f_low_pass       = 10000.f;
        demod.input_chunk_size = chunkSize;
        demod.start();

        std::vector<float> ampOut(numChunks), phaseOut(numChunks), freqOut(numChunks);
        auto               status = demod.processBulk(refSignal, respSignal, ampOut, phaseOut, freqOut);
        expect(status == work::Status::OK);

        // HP attenuates both signals equally, so ratio should be preserved
        const float ampMean = std::accumulate(ampOut.begin() + 100, ampOut.end(), 0.f) / static_cast<float>(numChunks - 100);
        expect(std::isfinite(ampMean)) << "amplitude should be finite at HP boundary";
        // ratio should still be close to input ampRatio (both attenuated equally)
        expect(approx(ampMean, ampRatio, 0.15f)) << "amplitude ratio at HP cutoff:" << ampMean << "vs expected" << ampRatio;
    };

    "IQDemodulator frequency near Nyquist"_test = [] {
        // carrier close to Nyquist limit
        // frequency estimation has fundamental limitation: arcsin(x) only valid for |x| <= 1
        // at high frequencies, sin(ω) → 1, limiting frequency estimate to fs/4
        constexpr float       fs         = 1e6f;
        constexpr float       freq       = 200e3f; // 40% of Nyquist - more reasonable test
        constexpr std::size_t chunkSize  = 512U;
        constexpr std::size_t numChunks  = 200U;
        constexpr std::size_t numSamples = chunkSize * numChunks;

        auto [refSignal, respSignal] = generateIQTestSignals<float>(freq, fs, 0.9f, 0.4f, 0.f, 0.f, numSamples);

        IQDemodulator<float, Resampling<512U, 1U, false>> demod;
        demod.sample_rate      = fs;
        demod.f_high_pass      = 100.f;
        demod.f_low_pass       = 50000.f;
        demod.input_chunk_size = chunkSize;
        demod.start();

        std::vector<float> ampOut(numChunks), phaseOut(numChunks), freqOut(numChunks);
        auto               status = demod.processBulk(refSignal, respSignal, ampOut, phaseOut, freqOut);
        expect(status == work::Status::OK);

        const float ampMean   = std::accumulate(ampOut.begin() + 50, ampOut.end(), 0.f) / static_cast<float>(numChunks - 50);
        const float phaseMean = std::accumulate(phaseOut.begin() + 50, phaseOut.end(), 0.f) / static_cast<float>(numChunks - 50);
        const float freqMean  = std::accumulate(freqOut.begin() + 50, freqOut.end(), 0.f) / static_cast<float>(numChunks - 50);

        expect(approx(ampMean, 0.9f, 0.05f)) << "amplitude at 40% Nyquist:" << ampMean;
        expect(approx(phaseMean, 0.4f, 0.1f)) << "phase at 40% Nyquist:" << phaseMean;
        expect(approx(freqMean, freq, 0.1f * freq)) << "frequency at 40% Nyquist:" << freqMean << "vs" << freq;
    };

    "IQDemodulator frequency sweep 0.1-5 MHz"_test = [](DerivativeMethod derivative_method) {
        constexpr float       fs           = 62.5e6f; // 62.5 MHz sample rate
        constexpr float       f_start      = 0.1e6f;  // 0.1 MHz start
        constexpr float       f_end        = 5.0e6f;  // 5 MHz end
        constexpr float       sweepTime    = 0.5f;    // 0.5 second sweep
        constexpr float       ampRatio     = 0.85f;   // response amplitude
        constexpr float       phaseShift   = 0.4f;    // phase shift (radians)
        constexpr float       noise        = 0.01f;   // 1% noise
        constexpr std::size_t chunkSize    = 1024U;
        constexpr std::size_t numSamples   = static_cast<std::size_t>(fs * sweepTime);
        constexpr std::size_t numChunks    = numSamples / chunkSize;
        constexpr std::size_t settleChunks = 500UZ; // ~8 ms settling

        std::mt19937                          gen(42);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

        // generate linear frequency sweep with noise
        std::vector<float> refSignal(numSamples), respSignal(numSamples);
        std::vector<float> trueFreq(numChunks); // true frequency at each chunk

        float phase = 0.f;
        for (std::size_t i = 0UZ; i < numSamples; ++i) {
            const float t     = static_cast<float>(i) / fs;
            const float freq  = f_start + (f_end - f_start) * t / sweepTime; // linear chirp
            const float omega = 2.f * std::numbers::pi_v<float> * freq;

            phase += omega / fs; // instantaneous phase increment
            if (phase > 2.f * std::numbers::pi_v<float>) {
                phase -= 2.f * std::numbers::pi_v<float>;
            }

            refSignal[i]  = std::sin(phase) + noise * dist(gen);
            respSignal[i] = ampRatio * std::sin(phase + phaseShift) + noise * dist(gen);

            // record true frequency at chunk boundaries
            if (i % chunkSize == 0 && i / chunkSize < numChunks) {
                trueFreq[i / chunkSize] = freq;
            }
        }

        IQDemodulator<float, Resampling<1024U, 1U, false>> demod;
        demod.sample_rate       = fs;
        demod.f_high_pass       = 1000.f;  // low HP for 0.1 MHz carrier
        demod.f_low_pass        = 10000.f; // 10 kHz LP bandwidth
        demod.input_chunk_size  = chunkSize;
        demod.derivative_method = derivative_method;
        demod.start();

        std::vector<float> ampOut(numChunks), phaseOut(numChunks), freqOut(numChunks);
        auto               status = demod.processBulk(refSignal, respSignal, ampOut, phaseOut, freqOut);
        expect(status == work::Status::OK);

        // compute statistics after settling
        float       ampSum = 0.f, ampMin = std::numeric_limits<float>::max(), ampMax = std::numeric_limits<float>::lowest();
        float       phaseSum = 0.f, phaseMin = std::numeric_limits<float>::max(), phaseMax = std::numeric_limits<float>::lowest();
        float       freqErrMax = 0.f, freqErrSum = 0.f;
        std::size_t count = 0UZ;

        for (std::size_t i = settleChunks; i < numChunks; ++i) {
            ampSum += ampOut[i];
            ampMin = std::min(ampMin, ampOut[i]);
            ampMax = std::max(ampMax, ampOut[i]);

            phaseSum += phaseOut[i];
            phaseMin = std::min(phaseMin, phaseOut[i]);
            phaseMax = std::max(phaseMax, phaseOut[i]);

            // frequency error (relative to true instantaneous frequency)
            const float freqErr = std::abs(freqOut[i] - trueFreq[i]);
            freqErrSum += freqErr;
            freqErrMax = std::max(freqErrMax, freqErr);
            ++count;
        }

        const float ampMean     = ampSum / static_cast<float>(count);
        const float phaseMean   = phaseSum / static_cast<float>(count);
        const float freqErrMean = freqErrSum / static_cast<float>(count);

        // prepare time and reference vectors for plotting
        std::vector<float> timeMs(numChunks);
        std::vector<float> ampMeasured(numChunks), ampExpected(numChunks);
        std::vector<float> phaseMeasured(numChunks), phaseExpected(numChunks);
        std::vector<float> freqMeasured(numChunks), freqReference(numChunks);

        for (std::size_t i = 0; i < numChunks; ++i) {
            timeMs[i]        = static_cast<float>(i * chunkSize) / fs * 1e3f; // time in ms
            ampMeasured[i]   = ampOut[i];
            ampExpected[i]   = ampRatio;
            phaseMeasured[i] = phaseOut[i];
            phaseExpected[i] = phaseShift;
            freqMeasured[i]  = freqOut[i] / 1e6f;  // MHz
            freqReference[i] = trueFreq[i] / 1e6f; // MHz
        }

        std::println("\n=== IQDemodulator Frequency Sweep Statistics (0.1-5 MHz, 62.5 MHz fs) derivative_method: {} ===", demod.derivative_method);
        // plots
        {
            auto chart = gr::graphs::ImChart<130UZ, 16UZ>({{0., timeMs.back()}, {0., 5.5}});
            chart.draw(timeMs, freqMeasured, "f_meas");
            chart.draw(timeMs, freqReference, "f_ref [MHz]");
            chart.draw();
        }
        {
            auto chart = gr::graphs::ImChart<130UZ, 16UZ>({{0., timeMs.back()}, {static_cast<double>(ampMin) * 0.95, static_cast<double>(ampMax) * 1.05}});
            chart.draw(timeMs, ampMeasured, "A_meas");
            chart.draw(timeMs, ampExpected, "A_ref");
            chart.draw();
        }

        {
            auto chart = gr::graphs::ImChart<130UZ, 16UZ>({{0., timeMs.back()}, {static_cast<double>(phaseMin) * 0.95, static_cast<double>(phaseMax) * 1.05}});
            chart.draw(timeMs, phaseMeasured, "meas");
            chart.draw(timeMs, phaseExpected, "phase-ref [rad]");
            chart.draw();
        }

        // print statistics
        std::println("Amplitude:  mean={:.4f}, min={:.4f}, max={:.4f}, expected={:.4f}", ampMean, ampMin, ampMax, ampRatio);
        std::println("Phase:      mean={:.4f} rad, min={:.4f}, max={:.4f}, expected={:.4f} rad", phaseMean, phaseMin, phaseMax, phaseShift);
        std::println("Freq error: mean={:.1f} Hz, max={:.1f} Hz", freqErrMean, freqErrMax);
        std::println("Phase lag:  {:.4f} rad ({:.2f}°)", phaseMean - phaseShift, (phaseMean - phaseShift) * 180.f / std::numbers::pi_v<float>);
        std::println("============================================================================\n");

        // amplitude should track input ratio within 10%
        expect(approx(ampMean, ampRatio, 0.1f)) << "sweep amplitude mean:" << ampMean;
        expect(ampMin > 0.6f * ampRatio) << "sweep amplitude min:" << ampMin;
        expect(ampMax < 1.4f * ampRatio) << "sweep amplitude max:" << ampMax;

        // phase should track within 0.2 rad (~11°)
        expect(approx(phaseMean, phaseShift, 0.2f)) << "sweep phase mean:" << phaseMean;

        // frequency tracking: all methods should achieve similar accuracy after gain correction
        const float f_center = (f_start + f_end) / 2.f;
        expect(freqErrMean < 0.02f * f_center) << "sweep freq mean error:" << freqErrMean;
        expect(freqErrMax < 0.05f * f_center) << "sweep freq max error:" << freqErrMax;
    } | std::vector<DerivativeMethod>{DerivativeMethod::SymmetricDifference, DerivativeMethod::SavitzkyGolay5, DerivativeMethod::SavitzkyGolay7};

    "IQDemodulator static decimation"_test = [] {
        constexpr float       fs         = 1e6f;
        constexpr float       freq       = 100e3f;
        constexpr std::size_t chunkSize  = 512U;
        constexpr std::size_t numChunks  = 100U;
        constexpr std::size_t numSamples = chunkSize * numChunks;

        auto [refSignal, respSignal] = generateIQTestSignals<float>(freq, fs, 0.8f, 0.5f, 0.1f, 0.01f, numSamples);

        // static decimation: Resampling<512U, 1U, true> - compile-time fixed
        IQDemodulator<float, Resampling<512U, 1U, true>> demod;
        demod.sample_rate = fs;
        demod.f_high_pass = 100.f;
        demod.f_low_pass  = 10000.f;
        // input_chunk_size should be set from NTTP, but we verify it's correct
        expect(demod.input_chunk_size == 512U) << "static decimation chunk size should be 512";
        demod.start();

        std::vector<float> ampOut(numChunks), phaseOut(numChunks), freqOut(numChunks);
        auto               status = demod.processBulk(refSignal, respSignal, ampOut, phaseOut, freqOut);
        expect(status == work::Status::OK);

        // verify results are valid
        const float ampMean   = std::accumulate(ampOut.begin() + 20, ampOut.end(), 0.f) / static_cast<float>(numChunks - 20);
        const float phaseMean = std::accumulate(phaseOut.begin() + 20, phaseOut.end(), 0.f) / static_cast<float>(numChunks - 20);
        const float freqMean  = std::accumulate(freqOut.begin() + 20, freqOut.end(), 0.f) / static_cast<float>(numChunks - 20);

        expect(approx(ampMean, 0.8f, 0.05f)) << "static decimation amplitude:" << ampMean;
        expect(approx(phaseMean, 0.5f, 0.1f)) << "static decimation phase:" << phaseMean;
        expect(approx(freqMean, freq, 0.05f * freq)) << "static decimation frequency:" << freqMean;
    };

    "IQDemodulator fixed derivative NTTP"_test = [] {
        constexpr float       fs         = 1e6f;
        constexpr float       freq       = 100e3f;
        constexpr std::size_t chunkSize  = 256U;
        constexpr std::size_t numChunks  = 100U;
        constexpr std::size_t numSamples = chunkSize * numChunks;

        auto [refSignal, respSignal] = generateIQTestSignals<float>(freq, fs, 0.8f, 0.3f, 0.f, 0.01f, numSamples);

        // use IQDemodulatorFixed alias with compile-time fixed SavitzkyGolay5
        IQDemodulatorFixed<float, DerivativeMethod::SavitzkyGolay5> demod;
        demod.sample_rate      = fs;
        demod.f_high_pass      = 100.f;
        demod.f_low_pass       = 10000.f;
        demod.input_chunk_size = chunkSize;
        demod.start();

        // verify derivative_method is set correctly from NTTP
        expect(demod.derivative_method == DerivativeMethod::SavitzkyGolay5) << "derivative method should be SG5 from NTTP";

        std::vector<float> ampOut(numChunks), phaseOut(numChunks), freqOut(numChunks);
        auto               status = demod.processBulk(refSignal, respSignal, ampOut, phaseOut, freqOut);
        expect(status == work::Status::OK);

        const float ampMean = std::accumulate(ampOut.begin() + 30, ampOut.end(), 0.f) / static_cast<float>(numChunks - 30);
        expect(approx(ampMean, 0.8f, 0.05f)) << "fixed derivative amplitude:" << ampMean;

        // verify that changing derivative_method at runtime throws
        property_map oldSettings, newSettings;
        newSettings["derivative_method"] = static_cast<int>(DerivativeMethod::SymmetricDifference);
        expect(throws([&] { demod.settingsChanged(oldSettings, newSettings); })) << "changing fixed derivative_method should throw";
    };
};

int main() { return 0; /* not needed for UT */ }
