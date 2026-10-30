#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <concepts>
#include <limits>
#include <numbers>
#include <random>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/analog/AMModulation.hpp>
#include <gnuradio-4.0/analog/AutomaticGainControl.hpp>
#include <gnuradio-4.0/analog/FMEmphasis.hpp>
#include <gnuradio-4.0/analog/FMModulation.hpp>
#include <gnuradio-4.0/analog/FMStereoDecoder.hpp>
#include <gnuradio-4.0/analog/PMModulation.hpp>
#include <gnuradio-4.0/analog/PhaseLockedLoop.hpp>
#include <gnuradio-4.0/analog/PowerSquelch.hpp>
#include <gnuradio-4.0/test/DeviceTestHelper.hpp>
#include <gnuradio-4.0/testing/StreamProbes.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace {

constexpr double kSampleRate         = 240'000.;
constexpr double kDeviation          = 75'000.;
constexpr double kUnitInputPhaseStep = 2. * std::numbers::pi * kDeviation / kSampleRate;
constexpr float  kTimeConstant       = 50e-6f;
constexpr float  kLongerTimeConstant = 75e-6f;

constexpr float      kInfinity = std::numeric_limits<float>::infinity();
constexpr float      kNaN      = std::numeric_limits<float>::quiet_NaN();
constexpr std::array kInvalidForPositive{0.f, -1.f, kInfinity, kNaN};
constexpr std::array kInvalidForNonNegative{-1.f, kInfinity, kNaN};
constexpr std::array kInvalidForFinite{kInfinity, -kInfinity, kNaN};

constexpr float       kSquelchTimeConstant    = 1e-4f;
constexpr std::size_t kSquelchSettlingSamples = static_cast<std::size_t>(100. * kSampleRate * static_cast<double>(kSquelchTimeConstant));

constexpr double      kCarrierOffset  = 300.;
constexpr double      kCarrierPhase   = 0.7;
constexpr std::size_t kLockingSamples = 120'000UZ;

constexpr double      kPilotFrequency  = 19'000.;
constexpr std::size_t kAnalysisSamples = static_cast<std::size_t>(kSampleRate / 50.);

constexpr std::size_t kMessagePeriod    = 64UZ;
constexpr double      kMessageFrequency = kSampleRate / static_cast<double>(kMessagePeriod);

template<typename T>
constexpr T kTolerance = std::same_as<T, float> ? T(1e-5) : T(1e-12);

template<typename T>
constexpr T kLockedPhaseTolerance = std::same_as<T, float> ? T(1e-5) : T(1e-11);

[[nodiscard]] constexpr double phaseStep(double frequencyOffset) { return 2. * std::numbers::pi * frequencyOffset / kSampleRate; }

[[nodiscard]] double lowPassPole(float timeConstant) { return std::exp(-1. / (kSampleRate * static_cast<double>(timeConstant))); }

[[nodiscard]] std::vector<float> sineMessage(double amplitude) {
    std::vector<float> message(kMessagePeriod);
    for (std::size_t n = 0UZ; n < message.size(); ++n) {
        message[n] = static_cast<float>(amplitude * std::sin(2. * std::numbers::pi * static_cast<double>(n) / static_cast<double>(kMessagePeriod)));
    }
    return message;
}

[[nodiscard]] double tone(double frequency, double amplitude, std::size_t n) { return amplitude * std::sin(phaseStep(frequency) * static_cast<double>(n)); }

[[nodiscard]] double stereoMultiplex(double leftAudio, double rightAudio, double pilotAmplitude, std::size_t n) {
    const double pilotPhase = 0.5 + phaseStep(kPilotFrequency) * static_cast<double>(n);
    return (leftAudio + rightAudio) / 2. + (leftAudio - rightAudio) / 2. * std::sin(2. * pilotPhase) + pilotAmplitude * std::sin(pilotPhase);
}

template<typename T>
[[nodiscard]] double toneAmplitude(const std::vector<T>& samples, double frequency) {
    std::complex<double> sum{};
    for (std::size_t n = 0UZ; n < samples.size(); ++n) {
        sum += static_cast<double>(samples[n]) * std::polar(1., -phaseStep(frequency) * static_cast<double>(n));
    }
    return 2. * std::abs(sum) / static_cast<double>(samples.size());
}

template<typename T>
[[nodiscard]] std::complex<T> phasor(double phase, double amplitude = 1.) {
    return std::polar(static_cast<T>(amplitude), static_cast<T>(std::remainder(phase, 2. * std::numbers::pi)));
}

[[nodiscard]] double amplitudeForPower(double decibel) { return std::pow(10., decibel / 20.); }

template<typename T>
[[nodiscard]] T distance(std::complex<T> actual, std::complex<T> expected) {
    return std::abs(actual - expected);
}

template<typename TBlock>
using InputType = typename decltype(TBlock::in)::value_type;

template<typename TBlock>
using OutputTypes = typename gr::traits::block::stream_output_port_types<TBlock>::template apply<std::tuple>;

template<typename TBlock>
[[nodiscard]] auto processSample(TBlock& block, InputType<TBlock> sample) {
    if constexpr (requires { block.processOne(sample); }) {
        return block.processOne(sample);
    } else {
        const std::span<const InputType<TBlock>> input{&sample, 1UZ};
        OutputTypes<TBlock>                      outputs{};
        std::apply(
            [&block, &input](auto&... output) {
                const auto processViews = [&block, &input](auto... view) { return block.processBulk(input, view...); };
                boost::ut::expect(processViews(std::span{&output, 1UZ}...) == gr::work::Status::OK);
            },
            outputs);
        if constexpr (std::tuple_size_v<OutputTypes<TBlock>> == 1UZ) {
            return std::get<0>(outputs);
        } else {
            return outputs;
        }
    }
}

template<typename TBlock>
void startOutsideScheduler(TBlock& block) {
    block.settings().init();
    std::ignore = block.settings().applyStagedParameters();
    if constexpr (requires { block.start(); }) {
        block.start();
    }
}

template<typename TBlock>
void applySettings(TBlock& block, const gr::property_map& settings) {
    boost::ut::expect(block.settings().set(settings).empty());
    boost::ut::expect(block.settings().activateContext() != std::nullopt);
    std::ignore = block.settings().applyStagedParameters();
}

template<typename TBlock>
void expectRejected(const gr::property_map& update) {
    using namespace boost::ut;
    using TInput = InputType<TBlock>;
    gr::MsgPortIn errors;
    TBlock        block;
    TBlock        unchanged;
    startOutsideScheduler(block);
    startOutsideScheduler(unchanged);
    expect(block.msgOut.connect(errors).has_value()) << fatal;
    const gr::property_map settingsBefore = block.settings().get();
    applySettings(block, update);
    expect(eq(errors.streamReader().available(), 1UZ)) << "one error reported";
    expect(block.settings().get() == settingsBefore) << "previous settings kept";
    for (const TInput sample : {TInput{1}, TInput{-1}}) {
        expect(processSample(block, sample) == processSample(unchanged, sample)) << "previous processing kept";
    }
}

template<typename T>
[[nodiscard]] std::size_t countPassed(gr::analog::PowerSquelch<T>& block, double powerDecibel, std::size_t nSamples) {
    std::size_t passed = 0UZ;
    for (std::size_t n = 0UZ; n < nSamples; ++n) {
        const std::complex<T> sample = phasor<T>(0.3 * static_cast<double>(n), amplitudeForPower(powerDecibel));
        const std::complex<T> output = processSample(block, sample);
        boost::ut::expect(output == sample || output == std::complex<T>{}) << "passed or muted";
        passed += output == sample ? 1UZ : 0UZ;
    }
    return passed;
}

template<typename T, typename TMultiplex>
[[nodiscard]] std::pair<std::vector<T>, std::vector<T>> decodeStereo(gr::analog::FMStereoDecoder<T>& block, TMultiplex multiplex, std::size_t firstSample, std::size_t nSamples) {
    std::pair<std::vector<T>, std::vector<T>> channels{std::vector<T>(nSamples), std::vector<T>(nSamples)};
    for (std::size_t k = 0UZ; k < nSamples; ++k) {
        std::tie(channels.first[k], channels.second[k]) = processSample(block, static_cast<T>(multiplex(firstSample + k)));
    }
    return channels;
}

constexpr auto kTypes = std::tuple<float, double>{};

} // namespace

const auto fmDemodulatorTests = [] {
    using namespace boost::ut;
    using gr::analog::FMDemodulator;

    "constant offset"_test = []<typename T> {
        for (const double offset : {-0.45 * kSampleRate, -kDeviation, 0., kDeviation / 3., kDeviation, 0.45 * kSampleRate}) {
            FMDemodulator<T> block;
            startOutsideScheduler(block);
            for (std::size_t n = 0UZ; n < 64UZ; ++n) {
                const T expected = n == 0UZ ? T{0} : static_cast<T>(offset / kDeviation);
                expect(approx(processSample(block, phasor<T>(0.7 + phaseStep(offset) * static_cast<double>(n))), expected, kTolerance<T>)) << "offset=" << offset << " n=" << n;
            }
        }
    } | kTypes;

    "sinusoidal FM"_test = []<typename T> {
        FMDemodulator<T> block;
        startOutsideScheduler(block);
        const auto phase = [](std::size_t n) { return 2. * std::sin(2. * std::numbers::pi * static_cast<double>(n) / 97.); };
        for (std::size_t n = 0UZ; n < 1024UZ; ++n) {
            const double amplitude = std::pow(10., 3. * std::sin(0.01 * static_cast<double>(n)));
            const T      expected  = n == 0UZ ? T{0} : static_cast<T>((phase(n) - phase(n - 1UZ)) * kSampleRate / (2. * std::numbers::pi * kDeviation));
            expect(approx(processSample(block, phasor<T>(phase(n), amplitude)), expected, kTolerance<T>)) << "n=" << n;
        }
    } | kTypes;

    "radian output"_test = []<typename T> {
        const float      deviationForRadianOutput = static_cast<float>(1'000. / (2. * std::numbers::pi));
        FMDemodulator<T> block({{"sample_rate", 1'000.f}, {"frequency_deviation", deviationForRadianOutput}});
        startOutsideScheduler(block);
        std::ignore = processSample(block, {T{1}, T{0}});
        expect(approx(processSample(block, {T{0}, T{1}}), std::numbers::pi_v<T> / T{2}, T(1e-6))) << "quarter turn forward";
        expect(approx(processSample(block, {T{1}, T{0}}), -std::numbers::pi_v<T> / T{2}, T(1e-6))) << "quarter turn back";
    } | kTypes;

    "zero samples"_test = []<typename T> {
        FMDemodulator<T> block;
        startOutsideScheduler(block);
        const std::complex<T> thirdQuadrant{T{-1}, T{-1}};
        expect(eq(processSample(block, thirdQuadrant), T{0})) << "first sample after start";
        expect(eq(processSample(block, {}), T{0})) << "zero sample";
        expect(eq(processSample(block, thirdQuadrant), T{0})) << "sample after a zero sample";
        expect(eq(processSample(block, thirdQuadrant), T{0})) << "no phase step";
        block.reset();
        expect(eq(processSample(block, thirdQuadrant), T{0})) << "first sample after reset";
    } | kTypes;

    "settings change"_test = []<typename T> {
        constexpr double kOffset = 30'000.;
        FMDemodulator<T> block;
        startOutsideScheduler(block);
        for (std::size_t n = 0UZ; n < 8UZ; ++n) {
            std::ignore = processSample(block, phasor<T>(phaseStep(kOffset) * static_cast<double>(n)));
        }
        applySettings(block, {{"frequency_deviation", 15'000.f}});
        expect(approx(processSample(block, phasor<T>(phaseStep(kOffset) * 8.)), static_cast<T>(kOffset / 15'000.), kTolerance<T>)) << "new gain and phase history";
    } | kTypes;

    "sample_rate tag"_test = [] {
        using namespace gr::testing;
        constexpr std::size_t kTagIndex = 512UZ;

        std::vector<std::complex<float>> tone(8UZ);
        for (std::size_t n = 0UZ; n < tone.size(); ++n) {
            tone[n] = phasor<float>(phaseStep(30'000.) * static_cast<double>(n));
        }

        gr::Graph flow;
        auto&     source      = flow.emplaceBlock<TagSource<std::complex<float>>>({{"n_samples_max", gr::Size_t{1024U}}, {"values", tone}, {"sample_rate", 240'000.f}, {"mark_tag", false}});
        auto&     demodulator = flow.emplaceBlock<FMDemodulator<float>>();
        auto&     sink        = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});
        source._tags.emplace_back(kTagIndex, gr::property_map{{gr::tag::kSampleRate.key(), 480'000.f}});

        expect(flow.connect<"out", "in">(source, demodulator).has_value()) << fatal;
        expect(flow.connect<"out", "in">(demodulator, sink).has_value()) << fatal;
        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value()) << fatal;
        expect(sched.runAndWait().has_value());

        expect(eq(sink._samples.size(), 1024UZ)) << fatal;
        for (std::size_t n = 1UZ; n < sink._samples.size(); ++n) {
            const float expected = n < kTagIndex ? 30'000.f / 75'000.f : 60'000.f / 75'000.f;
            expect(approx(sink._samples[n], expected, 1e-5f)) << "n=" << n;
        }
        expect(std::ranges::any_of(sink._tags, [](const auto& receivedTag) { return receivedTag.index == kTagIndex && receivedTag.map.contains(gr::tag::kSampleRate.key()); })) << "tag forwarded";
    };

    "invalid settings"_test = [] {
        expectRejected<FMDemodulator<float>>({{"sample_rate", 0.f}});
        for (const float invalid : kInvalidForPositive) {
            expectRejected<FMDemodulator<float>>({{"sample_rate", 480'000.f}, {"frequency_deviation", invalid}});
        }
    };
};

const auto fmModulatorTests = [] {
    using namespace boost::ut;
    using gr::analog::FMModulator;

    "constant input"_test = []<typename T> {
        for (const double input : {-1., -0.25, 0., 0.5, 1.}) {
            FMModulator<T> block;
            startOutsideScheduler(block);
            for (std::size_t n = 0UZ; n < 64UZ; ++n) {
                const std::complex<T> sample = processSample(block, static_cast<T>(input));
                expect(approx(distance(sample, phasor<T>(kUnitInputPhaseStep * input * static_cast<double>(n + 1UZ))), T{0}, kTolerance<T>)) << "input=" << input << " n=" << n;
            }
        }
    } | kTypes;

    "phase accuracy"_test = []<typename T> {
        constexpr std::size_t kPowerOfTwoCount = 1UZ << 20UZ;
        constexpr T           kInput           = T(0.1234);
        FMModulator<T>        block;
        startOutsideScheduler(block);
        std::complex<T> last{};
        for (std::size_t n = 0UZ; n < kPowerOfTwoCount; ++n) {
            last = processSample(block, kInput);
        }
        const double expectedPhase = kUnitInputPhaseStep * static_cast<double>(kInput) * static_cast<double>(kPowerOfTwoCount);
        expect(approx(distance(last, phasor<T>(expectedPhase)), T{0}, T(1e-6))) << "phase drift";
    } | kTypes;

    "radian input"_test = []<typename T> {
        constexpr double kPhaseStep              = 0.5;
        const float      deviationForRadianInput = static_cast<float>(1'000. / (2. * std::numbers::pi));
        FMModulator<T>   block({{"sample_rate", 1'000.f}, {"frequency_deviation", deviationForRadianInput}});
        startOutsideScheduler(block);
        for (std::size_t n = 0UZ; n < 8UZ; ++n) {
            expect(approx(distance(processSample(block, T(kPhaseStep)), phasor<T>(kPhaseStep * static_cast<double>(n + 1UZ))), T{0}, T(1e-5))) << "n=" << n;
        }
    } | kTypes;

    "non-finite input"_test = []<typename T> {
        FMModulator<T> block;
        startOutsideScheduler(block);
        const std::complex<T> before = processSample(block, T{1});
        for (const T invalid : {std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::infinity(), -std::numeric_limits<T>::infinity()}) {
            expect(approx(distance(processSample(block, invalid), before), T{0}, kTolerance<T>)) << "phase held for " << invalid;
        }
        expect(approx(distance(processSample(block, T{1}), phasor<T>(2. * kUnitInputPhaseStep)), T{0}, kTolerance<T>)) << "continues from the held phase";
    } | kTypes;

    "settings change"_test = []<typename T> {
        FMModulator<T> block;
        startOutsideScheduler(block);
        std::ignore = processSample(block, T{1});
        applySettings(block, {{"frequency_deviation", 30'000.f}});
        expect(approx(distance(processSample(block, T{1}), phasor<T>(kUnitInputPhaseStep + phaseStep(30'000.))), T{0}, kTolerance<T>)) << "new step and continuous phase";
    } | kTypes;

    "reset"_test = []<typename T> {
        FMModulator<T> block;
        startOutsideScheduler(block);
        std::ignore = processSample(block, T{1});
        block.reset();
        expect(approx(distance(processSample(block, T{0}), phasor<T>(0.)), T{0}, kTolerance<T>));
    } | kTypes;

    "invalid settings"_test = [] {
        expectRejected<FMModulator<float>>({{"sample_rate", 0.f}});
        for (const float invalid : kInvalidForPositive) {
            expectRejected<FMModulator<float>>({{"sample_rate", 480'000.f}, {"frequency_deviation", invalid}});
        }
    };
};

const auto deemphasisTests = [] {
    using namespace boost::ut;
    using gr::analog::FMDeemphasis;

    "step response"_test = []<typename T> {
        FMDeemphasis<T> block;
        startOutsideScheduler(block);
        const double pole = lowPassPole(kTimeConstant);
        for (std::size_t n = 0UZ; n < 64UZ; ++n) {
            expect(approx(processSample(block, T{1}), static_cast<T>(1. - std::pow(pole, static_cast<double>(n + 1UZ))), kTolerance<T>)) << "n=" << n;
        }
    } | kTypes;

    "zero time constant"_test = []<typename T> {
        FMDeemphasis<T> block({{"time_constant", 0.f}});
        startOutsideScheduler(block);
        for (const T sample : {T{1}, T(-0.3), T(0.7), T{0}}) {
            expect(approx(processSample(block, sample), sample, kTolerance<T>));
        }
    } | kTypes;

    "non-finite input"_test = []<typename T> {
        FMDeemphasis<T> block;
        startOutsideScheduler(block);
        const T before = processSample(block, T{1});
        for (const T invalid : {std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::infinity(), -std::numeric_limits<T>::infinity()}) {
            expect(eq(processSample(block, invalid), before)) << "output held for " << invalid;
        }
        expect(approx(processSample(block, T{1}), static_cast<T>(1. - std::pow(lowPassPole(kTimeConstant), 2.)), kTolerance<T>)) << "continues from the held output";
    } | kTypes;

    "settings change"_test = []<typename T> {
        FMDeemphasis<T> block;
        startOutsideScheduler(block);
        for (std::size_t n = 0UZ; n < 8UZ; ++n) {
            std::ignore = processSample(block, T{1});
        }
        applySettings(block, {{"time_constant", kLongerTimeConstant}});
        const double remainingAfterEightSamples = std::pow(lowPassPole(kTimeConstant), 8.);
        expect(approx(processSample(block, T{1}), static_cast<T>(1. - remainingAfterEightSamples * lowPassPole(kLongerTimeConstant)), kTolerance<T>)) << "new time constant and output history";
    } | kTypes;

    "reset"_test = []<typename T> {
        FMDeemphasis<T> block;
        startOutsideScheduler(block);
        std::ignore = processSample(block, T{1});
        block.reset();
        expect(approx(processSample(block, T{1}), static_cast<T>(1. - lowPassPole(kTimeConstant)), kTolerance<T>));
    } | kTypes;

    "invalid settings"_test = [] {
        expectRejected<FMDeemphasis<float>>({{"sample_rate", 0.f}});
        for (const float invalid : kInvalidForNonNegative) {
            expectRejected<FMDeemphasis<float>>({{"sample_rate", 480'000.f}, {"time_constant", invalid}});
        }
    };
};

const auto preemphasisTests = [] {
    using namespace boost::ut;
    using gr::analog::FMPreemphasis;

    "step response"_test = []<typename T> {
        FMPreemphasis<T> block;
        startOutsideScheduler(block);
        expect(approx(processSample(block, T{1}), static_cast<T>(1. / (1. - lowPassPole(kTimeConstant))), kTolerance<T>)) << "first sample";
        for (std::size_t n = 1UZ; n < 64UZ; ++n) {
            expect(approx(processSample(block, T{1}), T{1}, kTolerance<T>)) << "unit gain at DC, n=" << n;
        }
    } | kTypes;

    "Nyquist gain"_test = []<typename T> {
        FMPreemphasis<T> block;
        startOutsideScheduler(block);
        const double pole        = lowPassPole(kTimeConstant);
        const double nyquistGain = (1. + pole) / (1. - pole);
        std::ignore              = processSample(block, T{1});
        for (std::size_t n = 1UZ; n < 64UZ; ++n) {
            const T alternating = n % 2UZ == 0UZ ? T{1} : T{-1};
            expect(approx(processSample(block, alternating), static_cast<T>(nyquistGain) * alternating, kTolerance<T>)) << "n=" << n;
        }
    } | kTypes;

    "zero time constant"_test = []<typename T> {
        FMPreemphasis<T> block({{"time_constant", 0.f}});
        startOutsideScheduler(block);
        for (const T sample : {T{1}, T(-0.3), T(0.7), T{0}}) {
            expect(eq(processSample(block, sample), sample));
        }
    } | kTypes;

    "settings change"_test = []<typename T> {
        FMPreemphasis<T> block;
        startOutsideScheduler(block);
        std::ignore = processSample(block, T{1});
        applySettings(block, {{"time_constant", kLongerTimeConstant}});
        const double newPole = lowPassPole(kLongerTimeConstant);
        expect(approx(processSample(block, T{0}), static_cast<T>(-newPole / (1. - newPole)), kTolerance<T>)) << "new time constant and input history";
    } | kTypes;

    "reset"_test = []<typename T> {
        FMPreemphasis<T> block;
        startOutsideScheduler(block);
        std::ignore = processSample(block, T{1});
        block.reset();
        expect(approx(processSample(block, T{1}), static_cast<T>(1. / (1. - lowPassPole(kTimeConstant))), kTolerance<T>));
    } | kTypes;

    "invalid settings"_test = [] {
        expectRejected<FMPreemphasis<float>>({{"sample_rate", 0.f}});
        for (const float invalid : kInvalidForNonNegative) {
            expectRejected<FMPreemphasis<float>>({{"sample_rate", 480'000.f}, {"time_constant", invalid}});
        }
    };
};

const auto fmStereoDecoderTests = [] {
    using namespace boost::ut;
    using gr::analog::FMStereoDecoder;

    const auto twoTones = [](std::size_t n) { return stereoMultiplex(tone(1'000., 0.4, n), tone(2'500., 0.3, n), 0.09, n); };

    "stereo separation"_test = [twoTones]<typename T> {
        FMStereoDecoder<T> block;
        startOutsideScheduler(block);
        std::ignore              = decodeStereo(block, twoTones, 0UZ, kLockingSamples);
        const auto [left, right] = decodeStereo(block, twoTones, kLockingSamples, kAnalysisSamples);
        expect(approx(toneAmplitude(left, 1'000.), 0.4, 1e-4)) << "left tone in left";
        expect(lt(toneAmplitude(left, 2'500.), 1e-4)) << "right tone not in left";
        expect(approx(toneAmplitude(right, 2'500.), 0.3, 1e-4)) << "right tone in right";
        expect(lt(toneAmplitude(right, 1'000.), 1e-4)) << "left tone not in right";
    } | kTypes;

    "mono without pilot"_test = []<typename T> {
        const auto         monoTone = [](std::size_t n) { return tone(1'000., 0.4, n); };
        FMStereoDecoder<T> block;
        startOutsideScheduler(block);
        const auto [left, right] = decodeStereo(block, monoTone, 0UZ, kLockingSamples);
        std::size_t notMono      = 0UZ;
        for (std::size_t n = 0UZ; n < kLockingSamples; ++n) {
            notMono += left[n] == static_cast<T>(monoTone(n)) && right[n] == left[n] ? 0UZ : 1UZ;
        }
        expect(eq(notMono, 0UZ)) << "left = right = input";
    } | kTypes;

    "stereo blend"_test = []<typename T> {
        const auto         leftOnlyWeakPilot = [](std::size_t n) { return stereoMultiplex(tone(1'000., 0.4, n), 0., 0.03, n); };
        FMStereoDecoder<T> block;
        startOutsideScheduler(block);
        std::ignore              = decodeStereo(block, leftOnlyWeakPilot, 0UZ, kLockingSamples);
        const auto [left, right] = decodeStereo(block, leftOnlyWeakPilot, kLockingSamples, kAnalysisSamples);
        expect(approx(toneAmplitude(left, 1'000.), 0.2 + 0.2 / 2., 1e-4)) << "mono part plus half the difference";
        expect(approx(toneAmplitude(right, 1'000.), 0.2 - 0.2 / 2., 1e-4)) << "mono part minus half the difference";
    } | kTypes;

    "mono until locked"_test = []<typename T> {
        constexpr std::size_t kUnlockedSamples = 2'400UZ;
        const auto            invertedPilot    = [](std::size_t n) { return stereoMultiplex(tone(1'000., 0.4, n), tone(2'500., 0.3, n), -0.09, n); };
        FMStereoDecoder<T>    block;
        startOutsideScheduler(block);
        const auto [left, right] = decodeStereo(block, invertedPilot, 0UZ, kUnlockedSamples);
        expect(left == right) << "mono while the loop is still half a turn away from the pilot";
    } | kTypes;

    "non-finite input"_test = [twoTones]<typename T> {
        using Limits = std::numeric_limits<T>;
        FMStereoDecoder<T> block;
        startOutsideScheduler(block);
        std::ignore   = decodeStereo(block, twoTones, 0UZ, kLockingSamples);
        std::size_t n = kLockingSamples;
        for (const T invalid : {Limits::quiet_NaN(), Limits::infinity(), -Limits::infinity()}) {
            std::ignore = processSample(block, invalid);
            ++n;
        }
        const auto [left, right] = decodeStereo(block, twoTones, n, kAnalysisSamples);
        expect(approx(toneAmplitude(left, 1'000.), 0.4, 1e-4)) << "still locked";
        expect(lt(toneAmplitude(left, 2'500.), 1e-4)) << "still separated";
    } | kTypes;

    "settings change"_test = [twoTones]<typename T> {
        FMStereoDecoder<T> block;
        startOutsideScheduler(block);
        std::ignore = decodeStereo(block, twoTones, 0UZ, kLockingSamples);
        applySettings(block, {{"pilot_threshold", 1.f}});
        const auto [left, right] = decodeStereo(block, twoTones, kLockingSamples, 64UZ);
        expect(left == right) << "mono once the pilot is below the threshold";
    } | kTypes;

    "reset"_test = [twoTones]<typename T> {
        FMStereoDecoder<T> block;
        startOutsideScheduler(block);
        std::ignore = decodeStereo(block, twoTones, 0UZ, kLockingSamples);
        block.reset();
        const auto [left, right] = decodeStereo(block, twoTones, kLockingSamples, 1UZ);
        expect(left == right) << "mono until the pilot is found again";
    } | kTypes;

    "invalid settings"_test = [] {
        expectRejected<FMStereoDecoder<float>>({{"sample_rate", 0.f}});
        for (const float invalid : kInvalidForPositive) {
            expectRejected<FMStereoDecoder<float>>({{"sample_rate", 480'000.f}, {"pilot_threshold", invalid}});
        }
        expectRejected<FMStereoDecoder<float>>({{"sample_rate", 48'000.f}});
    };
};

const auto amModulatorTests = [] {
    using namespace boost::ut;
    using gr::analog::AMModulator;

    "constant input"_test = []<typename T> {
        for (const float modulationIndex : {0.25f, 1.f}) {
            AMModulator<T> block({{"modulation_index", modulationIndex}});
            startOutsideScheduler(block);
            for (const T input : {T{-1}, T(-0.25), T{0}, T(0.5), T{1}}) {
                const std::complex<T> expected{T{1} + static_cast<T>(modulationIndex) * input, T{0}};
                expect(approx(distance(block.processOne(input), expected), T{0}, kTolerance<T>)) << "modulation_index=" << modulationIndex << " input=" << input;
            }
        }
    } | kTypes;

    "settings change"_test = []<typename T> {
        AMModulator<T> block;
        startOutsideScheduler(block);
        applySettings(block, {{"modulation_index", 0.25f}});
        expect(approx(distance(block.processOne(T{1}), std::complex<T>{T(1.25), T{0}}), T{0}, kTolerance<T>));
    } | kTypes;

    "invalid settings"_test = [] {
        for (const float invalid : kInvalidForPositive) {
            expectRejected<AMModulator<float>>({{"modulation_index", invalid}});
        }
    };
};

const auto amDemodulatorTests = [] {
    using namespace boost::ut;
    using gr::analog::AMDemodulator;

    "constant carrier"_test = []<typename T> {
        for (const double amplitude : {1e-3, 1., 1e3}) {
            for (const double offset : {0., 30'000.}) {
                AMDemodulator<T> block;
                startOutsideScheduler(block);
                for (std::size_t n = 0UZ; n < 64UZ; ++n) {
                    expect(approx(processSample(block, phasor<T>(0.7 + phaseStep(offset) * static_cast<double>(n), amplitude)), T{0}, kTolerance<T>)) << "amplitude=" << amplitude << " offset=" << offset << " n=" << n;
                }
            }
        }
    } | kTypes;

    "envelope step"_test = []<typename T> {
        constexpr float  kModulationIndex         = 0.5f;
        constexpr float  kFastCarrierTimeConstant = 1e-4f;
        constexpr double kMessageStep             = 0.5;
        constexpr double kCarrierAmplitude        = 2.;
        constexpr double kStepEnvelope            = 1. + static_cast<double>(kModulationIndex) * kMessageStep;
        AMDemodulator<T> block({{"modulation_index", kModulationIndex}, {"carrier_time_constant", kFastCarrierTimeConstant}});
        startOutsideScheduler(block);
        for (std::size_t n = 0UZ; n < 8UZ; ++n) {
            std::ignore = processSample(block, phasor<T>(0.3 * static_cast<double>(n), kCarrierAmplitude));
        }
        const double pole = lowPassPole(kFastCarrierTimeConstant);
        for (std::size_t k = 0UZ; k < 64UZ; ++k) {
            const double relativeCarrierLevel = 1. + (kStepEnvelope - 1.) * (1. - std::pow(pole, static_cast<double>(k)));
            const double expected             = (kStepEnvelope / relativeCarrierLevel - 1.) / static_cast<double>(kModulationIndex);
            expect(approx(processSample(block, phasor<T>(0.3 * static_cast<double>(8UZ + k), kCarrierAmplitude * kStepEnvelope)), static_cast<T>(expected), kTolerance<T>)) << "k=" << k;
        }
    } | kTypes;

    "zero carrier"_test = []<typename T> {
        AMDemodulator<T> block;
        startOutsideScheduler(block);
        expect(eq(processSample(block, {}), T{0})) << "zero input before any carrier";
        expect(eq(processSample(block, phasor<T>(0., 3.)), T{0})) << "first carrier sample sets the level";
        expect(approx(processSample(block, {}), T{-1}, kTolerance<T>)) << "carrier loss is full negative modulation";
        block.reset();
        expect(eq(processSample(block, phasor<T>(0., 5.)), T{0})) << "first sample after reset";
    } | kTypes;

    "non-finite input"_test = []<typename T> {
        using Limits = std::numeric_limits<T>;
        AMDemodulator<T> block;
        startOutsideScheduler(block);
        std::ignore = processSample(block, phasor<T>(0., 1.));
        for (const std::complex<T> invalid : {std::complex<T>{Limits::quiet_NaN(), T{0}}, std::complex<T>{Limits::infinity(), T{0}}, std::complex<T>{T{0}, -Limits::infinity()}}) {
            expect(eq(processSample(block, invalid), T{0})) << "invalid sample gives 0";
        }
        expect(eq(processSample(block, phasor<T>(0., 1.)), T{0})) << "carrier level unchanged";
    } | kTypes;

    "muted input"_test = []<typename T> {
        AMDemodulator<T> block({{"carrier_time_constant", 1e-4f}});
        startOutsideScheduler(block);
        for (std::size_t n = 0UZ; n < 64UZ; ++n) {
            std::ignore = processSample(block, phasor<T>(0.3 * static_cast<double>(n)));
        }
        for (std::size_t n = 0UZ; n < 10'000UZ; ++n) {
            std::ignore = processSample(block, {});
        }
        expect(approx(processSample(block, phasor<T>(0.)), T{0}, kTolerance<T>)) << "carrier level held while the input is zero";
    } | kTypes;

    "settings change"_test = []<typename T> {
        AMDemodulator<T> block;
        startOutsideScheduler(block);
        std::ignore = processSample(block, phasor<T>(0., 1.));
        applySettings(block, {{"modulation_index", 0.25f}});
        expect(approx(processSample(block, phasor<T>(0., 1.5)), T{2}, kTolerance<T>)) << "new modulation index and kept carrier level";
    } | kTypes;

    "invalid settings"_test = [] {
        expectRejected<AMDemodulator<float>>({{"sample_rate", 0.f}});
        for (const float invalid : kInvalidForPositive) {
            expectRejected<AMDemodulator<float>>({{"sample_rate", 480'000.f}, {"modulation_index", invalid}});
            expectRejected<AMDemodulator<float>>({{"sample_rate", 480'000.f}, {"carrier_time_constant", invalid}});
        }
    };
};

const auto pmModulatorTests = [] {
    using namespace boost::ut;
    using gr::analog::PMModulator;

    "constant input"_test = []<typename T> {
        for (const float phaseDeviation : {0.5f, 1.f}) {
            PMModulator<T> block({{"phase_deviation", phaseDeviation}});
            startOutsideScheduler(block);
            for (const T input : {T{-4}, T{-1}, T(-0.25), T{0}, T(0.5), T{1}, T{4}}) {
                const double expectedPhase = static_cast<double>(phaseDeviation) * static_cast<double>(input);
                expect(approx(distance(block.processOne(input), phasor<T>(expectedPhase)), T{0}, kTolerance<T>)) << "phase_deviation=" << phaseDeviation << " input=" << input;
            }
        }
    } | kTypes;

    "settings change"_test = []<typename T> {
        PMModulator<T> block;
        startOutsideScheduler(block);
        applySettings(block, {{"phase_deviation", 0.25f}});
        expect(approx(distance(block.processOne(T{2}), phasor<T>(0.5)), T{0}, kTolerance<T>));
    } | kTypes;

    "invalid settings"_test = [] {
        for (const float invalid : kInvalidForPositive) {
            expectRejected<PMModulator<float>>({{"phase_deviation", invalid}});
        }
    };
};

const auto pmDemodulatorTests = [] {
    using namespace boost::ut;
    using gr::analog::PMDemodulator;

    "constant phase"_test = []<typename T> {
        for (const float phaseDeviation : {0.5f, 1.f}) {
            PMDemodulator<T> block({{"phase_deviation", phaseDeviation}});
            startOutsideScheduler(block);
            for (const double phase : {-3., -1., 0., 0.5, 3.}) {
                for (const double amplitude : {1e-3, 1., 1e3}) {
                    const T expected = static_cast<T>(phase / static_cast<double>(phaseDeviation));
                    expect(approx(block.processOne(phasor<T>(phase, amplitude)), expected, kTolerance<T>)) << "phase_deviation=" << phaseDeviation << " phase=" << phase << " amplitude=" << amplitude;
                }
            }
        }
    } | kTypes;

    "zero samples"_test = []<typename T> {
        PMDemodulator<T> block;
        startOutsideScheduler(block);
        for (const std::complex<T> zero : {std::complex<T>{T{0}, T{0}}, std::complex<T>{-T{0}, T{0}}, std::complex<T>{-T{0}, -T{0}}, std::complex<T>{T{0}, -T{0}}}) {
            expect(eq(block.processOne(zero), T{0})) << "signed zeros give 0";
        }
    } | kTypes;

    "settings change"_test = []<typename T> {
        PMDemodulator<T> block;
        startOutsideScheduler(block);
        applySettings(block, {{"phase_deviation", 0.25f}});
        expect(approx(block.processOne(phasor<T>(0.5)), T{2}, kTolerance<T>));
    } | kTypes;

    "invalid settings"_test = [] {
        for (const float invalid : kInvalidForPositive) {
            expectRejected<PMDemodulator<float>>({{"phase_deviation", invalid}});
        }
    };
};

const auto phaseLockedLoopTests = [] {
    using namespace boost::ut;
    using gr::analog::PhaseLockedLoop;

    "carrier offset"_test = []<typename T> {
        for (const double amplitude : {1e-3, 1e3}) {
            PhaseLockedLoop<T> block;
            startOutsideScheduler(block);
            const auto received = [amplitude](std::size_t n) { return phasor<T>(kCarrierPhase + phaseStep(kCarrierOffset) * static_cast<double>(n), amplitude); };
            for (std::size_t n = 0UZ; n < kLockingSamples; ++n) {
                std::ignore = processSample(block, received(n));
            }
            for (std::size_t n = kLockingSamples; n < kLockingSamples + 64UZ; ++n) {
                const auto [tracked, carrier, frequency] = processSample(block, received(n));
                expect(approx(distance(tracked, std::complex<T>{static_cast<T>(amplitude), T{0}}) / static_cast<T>(amplitude), T{0}, kLockedPhaseTolerance<T>)) << "carrier at 0 Hz with zero phase, amplitude=" << amplitude;
                expect(approx(distance(carrier * tracked, received(n)) / static_cast<T>(amplitude), T{0}, kTolerance<T>)) << "carrier times out gives the input";
                expect(approx(frequency, static_cast<T>(kCarrierOffset), T(1e-3))) << "loop frequency";
            }
        }
    } | kTypes;

    "frequency limits"_test = []<typename T> {
        PhaseLockedLoop<T> block({{"min_frequency", 800.f}, {"max_frequency", 1'000.f}});
        startOutsideScheduler(block);
        T frequency{};
        for (std::size_t n = 0UZ; n < kLockingSamples; ++n) {
            frequency = std::get<2>(processSample(block, phasor<T>(phaseStep(1'050.) * static_cast<double>(n))));
        }
        expect(approx(frequency, T{1'000}, T(1e-3))) << "held at max_frequency for a carrier above it";
    } | kTypes;

    "zero and non-finite input"_test = []<typename T> {
        using Limits = std::numeric_limits<T>;
        PhaseLockedLoop<T> block({{"min_frequency", 100.f}, {"max_frequency", 300.f}});
        startOutsideScheduler(block);
        std::size_t n = 0UZ;
        for (const std::complex<T> unusable : {std::complex<T>{}, std::complex<T>{Limits::quiet_NaN(), T{0}}, std::complex<T>{Limits::infinity(), T{0}}, std::complex<T>{T{0}, -Limits::infinity()}, std::complex<T>{}}) {
            const auto outputs = processSample(block, unusable);
            expect(approx(distance(std::get<1>(outputs), phasor<T>(phaseStep(200.) * static_cast<double>(n))), T{0}, kTolerance<T>)) << "carrier runs on at the start frequency, n=" << n;
            expect(approx(std::get<2>(outputs), T{200}, T(1e-3))) << "frequency unchanged, n=" << n;
            ++n;
        }
    } | kTypes;

    "settings change"_test = []<typename T> {
        PhaseLockedLoop<T> block;
        startOutsideScheduler(block);
        applySettings(block, {{"min_frequency", 500.f}, {"max_frequency", 600.f}});
        expect(approx(std::get<2>(processSample(block, phasor<T>(0.))), T{500}, T(1e-3))) << "loop frequency moved into the new range";
    } | kTypes;

    "sample_rate change"_test = []<typename T> {
        PhaseLockedLoop<T> block;
        startOutsideScheduler(block);
        for (std::size_t n = 0UZ; n < kLockingSamples; ++n) {
            std::ignore = processSample(block, phasor<T>(kCarrierPhase + phaseStep(kCarrierOffset) * static_cast<double>(n)));
        }
        applySettings(block, {{"sample_rate", 480'000.f}});
        expect(approx(std::get<2>(processSample(block, phasor<T>(0.))), static_cast<T>(kCarrierOffset), T(1e-3))) << "loop frequency kept in Hz";
    } | kTypes;

    "range beyond sample_rate / 2"_test = []<typename T> {
        PhaseLockedLoop<T> block({{"sample_rate", 1'000.f}, {"min_frequency", 200.f}});
        startOutsideScheduler(block);
        expect(approx(std::get<2>(processSample(block, phasor<T>(0.))), T{350}, T(1e-3))) << "range from 200 Hz to 1 kHz limited to 500 Hz, starting in the middle";
    } | kTypes;

    "reset"_test = []<typename T> {
        PhaseLockedLoop<T> block;
        startOutsideScheduler(block);
        for (std::size_t n = 0UZ; n < 1'000UZ; ++n) {
            std::ignore = processSample(block, phasor<T>(kCarrierPhase + phaseStep(kCarrierOffset) * static_cast<double>(n)));
        }
        block.reset();
        const auto [tracked, carrier, frequency] = processSample(block, phasor<T>(0.));
        expect(approx(distance(carrier, std::complex<T>{T{1}, T{0}}), T{0}, kTolerance<T>)) << "zero phase";
        expect(approx(frequency, T{0}, T(1e-3))) << "middle of the frequency range";
        expect(approx(distance(tracked, std::complex<T>{T{1}, T{0}}), T{0}, kTolerance<T>)) << "input passes unrotated";
    } | kTypes;

    "invalid settings"_test = [] {
        expectRejected<PhaseLockedLoop<float>>({{"sample_rate", 0.f}});
        for (const float invalid : kInvalidForPositive) {
            expectRejected<PhaseLockedLoop<float>>({{"sample_rate", 480'000.f}, {"loop_bandwidth", invalid}});
        }
        for (const float invalid : kInvalidForFinite) {
            expectRejected<PhaseLockedLoop<float>>({{"sample_rate", 480'000.f}, {"min_frequency", invalid}});
            expectRejected<PhaseLockedLoop<float>>({{"sample_rate", 480'000.f}, {"max_frequency", invalid}});
        }
        expectRejected<PhaseLockedLoop<float>>({{"min_frequency", 2'000.f}});
    };
};

const auto automaticGainControlTests = [] {
    using namespace boost::ut;
    using gr::analog::AutomaticGainControl;

    "constant level"_test = []<typename T> {
        for (const float targetAmplitude : {0.5f, 1.f}) {
            for (const double amplitude : {1e-2, 1., 1e3}) {
                AutomaticGainControl<T> block({{"target_amplitude", targetAmplitude}});
                startOutsideScheduler(block);
                for (std::size_t n = 0UZ; n < 64UZ; ++n) {
                    const double phase = 0.3 * static_cast<double>(n);
                    expect(approx(distance(processSample(block, phasor<T>(phase, amplitude)), phasor<T>(phase, static_cast<double>(targetAmplitude))), T{0}, kTolerance<T>)) << "target_amplitude=" << targetAmplitude << " amplitude=" << amplitude << " n=" << n;
                }
            }
        }
    } | kTypes;

    "attack and decay"_test = []<typename T> {
        constexpr float         kAttackTime    = 1e-4f;
        constexpr float         kDecayTime     = 1e-3f;
        constexpr double        kLoudAmplitude = 10.;
        constexpr double        kLoudPower     = kLoudAmplitude * kLoudAmplitude;
        constexpr double        kQuietPower    = 1.;
        AutomaticGainControl<T> block({{"attack_time", kAttackTime}, {"decay_time", kDecayTime}});
        startOutsideScheduler(block);
        std::ignore = processSample(block, phasor<T>(0.));

        const double attackPole = lowPassPole(kAttackTime);
        for (std::size_t k = 0UZ; k < 64UZ; ++k) {
            const double meanPower = kLoudPower + (kQuietPower - kLoudPower) * std::pow(attackPole, static_cast<double>(k + 1UZ));
            expect(approx(std::abs(processSample(block, phasor<T>(0.3 * static_cast<double>(k), kLoudAmplitude))), static_cast<T>(kLoudAmplitude / std::sqrt(meanPower)), kTolerance<T>)) << "attack k=" << k;
        }

        const double decayPole                 = lowPassPole(kDecayTime);
        const double shortTermPowerAfterAttack = kLoudPower + (kQuietPower - kLoudPower) * std::pow(attackPole, 64.);
        const double longTermPowerAfterAttack  = kLoudPower + (kQuietPower - kLoudPower) * std::pow(decayPole, 64.);
        for (std::size_t k = 0UZ; k < 64UZ; ++k) {
            const double shortTermPower = kQuietPower + (shortTermPowerAfterAttack - kQuietPower) * std::pow(attackPole, static_cast<double>(k + 1UZ));
            const double longTermPower  = kQuietPower + (longTermPowerAfterAttack - kQuietPower) * std::pow(decayPole, static_cast<double>(k + 1UZ));
            expect(approx(std::abs(processSample(block, phasor<T>(0.3 * static_cast<double>(k)))), static_cast<T>(1. / std::sqrt(std::max(shortTermPower, longTermPower))), kTolerance<T>)) << "decay k=" << k;
        }
    } | kTypes;

    "QAM16 level"_test = []<typename T> {
        static constexpr std::array kLevels{-3., -1., 1., 3.};
        constexpr std::size_t       kSymbols   = 40'000UZ;
        const double                levelScale = 1. / std::sqrt(10.);
        std::mt19937                generator{42U};
        const auto                  level = [&generator, levelScale] { return kLevels[generator() % kLevels.size()] * levelScale; };
        AutomaticGainControl<T>     block({{"sample_rate", 1'000.f}, {"attack_time", 0.1f}, {"decay_time", 1.f}});
        startOutsideScheduler(block);
        double settledPower = 0.;
        for (std::size_t n = 0UZ; n < kSymbols; ++n) {
            const double          inPhase    = level();
            const double          quadrature = level();
            const std::complex<T> output     = processSample(block, {static_cast<T>(inPhase), static_cast<T>(quadrature)});
            settledPower += n >= kSymbols / 2UZ ? static_cast<double>(std::norm(output)) : 0.;
        }
        expect(approx(std::sqrt(settledPower / static_cast<double>(kSymbols / 2UZ)), 1., 0.02)) << "unit RMS for an attack_time of 100 symbols";
    } | kTypes;

    "max gain"_test = []<typename T> {
        AutomaticGainControl<T> block({{"max_gain", 40.f}});
        startOutsideScheduler(block);
        expect(approx(std::abs(processSample(block, phasor<T>(0., 1e-4))), T(1e-2), kTolerance<T>)) << "a weak signal gets at most max_gain";
    } | kTypes;

    "zero input"_test = []<typename T> {
        AutomaticGainControl<T> block;
        startOutsideScheduler(block);
        expect(processSample(block, {}) == std::complex<T>{}) << "zero stays zero";
        expect(approx(std::abs(processSample(block, phasor<T>(0., 0.2))), T{1}, kTolerance<T>)) << "first non-zero sample sets the mean power";
    } | kTypes;

    "non-finite input"_test = []<typename T> {
        using Limits = std::numeric_limits<T>;
        AutomaticGainControl<T> block;
        startOutsideScheduler(block);
        std::ignore = processSample(block, phasor<T>(0., 2.));
        for (const T invalid : {Limits::quiet_NaN(), Limits::infinity(), -Limits::infinity()}) {
            std::ignore = processSample(block, {invalid, T{0}});
        }
        expect(approx(std::abs(processSample(block, phasor<T>(0., 2.))), T{1}, kTolerance<T>)) << "mean power unchanged";
    } | kTypes;

    "settings change"_test = []<typename T> {
        AutomaticGainControl<T> block;
        startOutsideScheduler(block);
        std::ignore = processSample(block, phasor<T>(0., 2.));
        applySettings(block, {{"target_amplitude", 0.25f}});
        expect(approx(std::abs(processSample(block, phasor<T>(0., 2.))), T(0.25), kTolerance<T>)) << "new target and kept mean power";
    } | kTypes;

    "reset"_test = []<typename T> {
        AutomaticGainControl<T> block;
        startOutsideScheduler(block);
        std::ignore = processSample(block, phasor<T>(0., 2.));
        block.reset();
        expect(approx(std::abs(processSample(block, phasor<T>(0., 0.1))), T{1}, kTolerance<T>)) << "mean power restarts from the next sample";
    } | kTypes;

    "invalid settings"_test = [] {
        expectRejected<AutomaticGainControl<float>>({{"sample_rate", 0.f}});
        for (const float invalid : kInvalidForPositive) {
            expectRejected<AutomaticGainControl<float>>({{"sample_rate", 480'000.f}, {"target_amplitude", invalid}});
            expectRejected<AutomaticGainControl<float>>({{"sample_rate", 480'000.f}, {"attack_time", invalid}});
            expectRejected<AutomaticGainControl<float>>({{"sample_rate", 480'000.f}, {"decay_time", invalid}});
        }
        for (const float invalid : kInvalidForFinite) {
            expectRejected<AutomaticGainControl<float>>({{"sample_rate", 480'000.f}, {"max_gain", invalid}});
        }
    };
};

const auto powerSquelchTests = [] {
    using namespace boost::ut;
    using gr::analog::PowerSquelch;

    "threshold"_test = []<typename T> {
        PowerSquelch<T> below({{"threshold", -30.f}, {"time_constant", kSquelchTimeConstant}});
        startOutsideScheduler(below);
        expect(eq(countPassed(below, -33., kSquelchSettlingSamples), 0UZ)) << "stays closed below threshold";

        PowerSquelch<T> above({{"threshold", -30.f}, {"time_constant", kSquelchTimeConstant}});
        startOutsideScheduler(above);
        expect(eq(countPassed(above, -20., 1UZ), 0UZ)) << "closed while the mean power rises";
        std::ignore = countPassed(above, -20., kSquelchSettlingSamples);
        expect(eq(countPassed(above, -20., 64UZ), 64UZ)) << "open above threshold";
    } | kTypes;

    "hysteresis"_test = []<typename T> {
        PowerSquelch<T> block({{"threshold", -30.f}, {"hysteresis", 6.f}, {"time_constant", kSquelchTimeConstant}});
        startOutsideScheduler(block);
        std::ignore = countPassed(block, -20., kSquelchSettlingSamples);
        expect(eq(countPassed(block, -33., kSquelchSettlingSamples), kSquelchSettlingSamples)) << "stays open down to threshold - hysteresis";
        std::ignore = countPassed(block, -40., kSquelchSettlingSamples);
        expect(eq(countPassed(block, -40., 64UZ), 0UZ)) << "closed below threshold - hysteresis";
        expect(eq(countPassed(block, -33., kSquelchSettlingSamples), 0UZ)) << "stays closed up to threshold";
    } | kTypes;

    "non-finite input"_test = []<typename T> {
        PowerSquelch<T> block({{"threshold", -30.f}, {"time_constant", kSquelchTimeConstant}});
        startOutsideScheduler(block);
        for (const T invalid : {std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::infinity(), -std::numeric_limits<T>::infinity()}) {
            expect(processSample(block, {invalid, T{0}}) == std::complex<T>{}) << "stays closed";
        }
        std::ignore = countPassed(block, -20., kSquelchSettlingSamples);
        expect(eq(countPassed(block, -20., 64UZ), 64UZ)) << "opens normally afterwards";
    } | kTypes;

    "settings change"_test = []<typename T> {
        PowerSquelch<T> block({{"threshold", -30.f}, {"time_constant", kSquelchTimeConstant}});
        startOutsideScheduler(block);
        std::ignore = countPassed(block, -20., kSquelchSettlingSamples);
        applySettings(block, {{"threshold", -10.f}});
        expect(eq(countPassed(block, -20., 64UZ), 0UZ)) << "closes on a raised threshold";
    } | kTypes;

    "reset"_test = []<typename T> {
        PowerSquelch<T> block({{"threshold", -30.f}, {"time_constant", kSquelchTimeConstant}});
        startOutsideScheduler(block);
        std::ignore = countPassed(block, -20., kSquelchSettlingSamples);
        block.reset();
        expect(eq(countPassed(block, -20., 1UZ), 0UZ)) << "closed after reset";
    } | kTypes;

    "invalid settings"_test = [] {
        expectRejected<PowerSquelch<float>>({{"sample_rate", 0.f}});
        for (const float invalid : kInvalidForPositive) {
            expectRejected<PowerSquelch<float>>({{"sample_rate", 480'000.f}, {"time_constant", invalid}});
        }
        for (const float invalid : kInvalidForFinite) {
            expectRejected<PowerSquelch<float>>({{"sample_rate", 480'000.f}, {"threshold", invalid}});
        }
        for (const float invalid : kInvalidForNonNegative) {
            expectRejected<PowerSquelch<float>>({{"sample_rate", 480'000.f}, {"hysteresis", invalid}});
        }
    };
};

const auto commonTests = [] {
    using namespace boost::ut;
    using namespace gr::analog;

    "emphasis inverse"_test = []<typename T> {
        for (const float timeConstant : {0.f, kTimeConstant, kLongerTimeConstant}) {
            FMPreemphasis<T> preemphasis({{"time_constant", timeConstant}});
            FMDeemphasis<T>  deemphasis({{"time_constant", timeConstant}});
            startOutsideScheduler(preemphasis);
            startOutsideScheduler(deemphasis);
            for (std::size_t n = 0UZ; n < 256UZ; ++n) {
                const T nyquistComponent = n % 2UZ == 0UZ ? T(0.5) : T(-0.5);
                const T message          = static_cast<T>(std::sin(0.3 * static_cast<double>(n))) + nyquistComponent;
                expect(approx(processSample(deemphasis, processSample(preemphasis, message)), message, kTolerance<T>)) << "time_constant=" << timeConstant << " n=" << n;
            }
        }
    } | kTypes;

    "FM round trip"_test = [] {
        using namespace gr::testing;
        const std::vector<float> message = sineMessage(0.6);

        gr::Graph flow;
        auto&     source      = flow.emplaceBlock<TagSource<float>>({{"n_samples_max", gr::Size_t{4096U}}, {"values", message}, {"sample_rate", 240'000.f}, {"mark_tag", false}});
        auto&     preemphasis = flow.emplaceBlock<FMPreemphasis<float>>();
        auto&     modulator   = flow.emplaceBlock<FMModulator<float>>({{"frequency_deviation", 30'000.f}});
        auto&     demodulator = flow.emplaceBlock<FMDemodulator<float>>({{"frequency_deviation", 30'000.f}});
        auto&     deemphasis  = flow.emplaceBlock<FMDeemphasis<float>>();
        auto&     sink        = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});

        expect(flow.connect<"out", "in">(source, preemphasis).has_value()) << fatal;
        expect(flow.connect<"out", "in">(preemphasis, modulator).has_value()) << fatal;
        expect(flow.connect<"out", "in">(modulator, demodulator).has_value()) << fatal;
        expect(flow.connect<"out", "in">(demodulator, deemphasis).has_value()) << fatal;
        expect(flow.connect<"out", "in">(deemphasis, sink).has_value()) << fatal;
        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value()) << fatal;
        expect(sched.runAndWait().has_value());

        expect(eq(sink._samples.size(), 4096UZ)) << fatal;
        for (std::size_t n = 0UZ; n < sink._samples.size(); ++n) {
            expect(approx(sink._samples[n], message[n % message.size()], 1e-5f)) << "n=" << n;
        }
    };

    "AM round trip"_test = [] {
        using namespace gr::testing;
        constexpr float          kModulationIndex     = 0.5f;
        constexpr float          kCarrierTimeConstant = 0.1f;
        const std::vector<float> message              = sineMessage(0.6);

        gr::Graph flow;
        auto&     source      = flow.emplaceBlock<TagSource<float>>({{"n_samples_max", gr::Size_t{4096U}}, {"values", message}, {"sample_rate", 240'000.f}, {"mark_tag", false}});
        auto&     modulator   = flow.emplaceBlock<AMModulator<float>>({{"modulation_index", kModulationIndex}});
        auto&     demodulator = flow.emplaceBlock<AMDemodulator<float>>({{"modulation_index", kModulationIndex}, {"carrier_time_constant", kCarrierTimeConstant}});
        auto&     sink        = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});

        expect(flow.connect<"out", "in">(source, modulator).has_value()) << fatal;
        expect(flow.connect<"out", "in">(modulator, demodulator).has_value()) << fatal;
        expect(flow.connect<"out", "in">(demodulator, sink).has_value()) << fatal;
        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value()) << fatal;
        expect(sched.runAndWait().has_value());

        const float carrierRipple = static_cast<float>(1. / (2. * std::numbers::pi * kMessageFrequency * static_cast<double>(kCarrierTimeConstant)));
        expect(eq(sink._samples.size(), 4096UZ)) << fatal;
        for (std::size_t n = 0UZ; n < sink._samples.size(); ++n) {
            expect(approx(sink._samples[n], message[n % message.size()], 2.f * carrierRipple)) << "n=" << n;
        }
    };

    "PM round trip"_test = [] {
        using namespace gr::testing;
        constexpr float          kPhaseDeviation = 2.f;
        const std::vector<float> message         = sineMessage(0.6);

        gr::Graph flow;
        auto&     source      = flow.emplaceBlock<TagSource<float>>({{"n_samples_max", gr::Size_t{4096U}}, {"values", message}, {"sample_rate", 240'000.f}, {"mark_tag", false}});
        auto&     modulator   = flow.emplaceBlock<PMModulator<float>>({{"phase_deviation", kPhaseDeviation}});
        auto&     demodulator = flow.emplaceBlock<PMDemodulator<float>>({{"phase_deviation", kPhaseDeviation}});
        auto&     sink        = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});

        expect(flow.connect<"out", "in">(source, modulator).has_value()) << fatal;
        expect(flow.connect<"out", "in">(modulator, demodulator).has_value()) << fatal;
        expect(flow.connect<"out", "in">(demodulator, sink).has_value()) << fatal;
        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value()) << fatal;
        expect(sched.runAndWait().has_value());

        expect(eq(sink._samples.size(), 4096UZ)) << fatal;
        for (std::size_t n = 0UZ; n < sink._samples.size(); ++n) {
            expect(approx(sink._samples[n], message[n % message.size()], 1e-5f)) << "n=" << n;
        }
    };

    "PM on an offset carrier"_test = []<typename T> {
        constexpr double   kMessageAmplitude = 0.6;
        constexpr float    kLoopBandwidth    = 100.f;
        PhaseLockedLoop<T> carrierTracker({{"loop_bandwidth", kLoopBandwidth}});
        PMDemodulator<T>   demodulator;
        startOutsideScheduler(carrierTracker);
        startOutsideScheduler(demodulator);
        const auto   message         = [](std::size_t n) { return kMessageAmplitude * std::sin(2. * std::numbers::pi * static_cast<double>(n) / static_cast<double>(kMessagePeriod)); };
        const double loopFeedthrough = static_cast<double>(kLoopBandwidth) / kMessageFrequency;
        for (std::size_t n = 0UZ; n < kLockingSamples + 4'096UZ; ++n) {
            const std::complex<T> received    = phasor<T>(kCarrierPhase + phaseStep(kCarrierOffset) * static_cast<double>(n) + message(n));
            const T               demodulated = demodulator.processOne(std::get<0>(processSample(carrierTracker, received)));
            if (n >= kLockingSamples) {
                expect(approx(demodulated, static_cast<T>(message(n)), static_cast<T>(loopFeedthrough * kMessageAmplitude))) << "n=" << n;
            }
        }
    } | kTypes;

    "PLL in a graph"_test = [] {
        using namespace gr::testing;
        std::vector<std::complex<float>> tone(static_cast<std::size_t>(kSampleRate / kCarrierOffset));
        for (std::size_t n = 0UZ; n < tone.size(); ++n) {
            tone[n] = phasor<float>(phaseStep(kCarrierOffset) * static_cast<double>(n));
        }

        gr::Graph flow;
        auto&     source        = flow.emplaceBlock<TagSource<std::complex<float>>>({{"n_samples_max", static_cast<gr::Size_t>(kLockingSamples)}, {"values", tone}, {"sample_rate", 240'000.f}, {"mark_tag", false}});
        auto&     tracker       = flow.emplaceBlock<PhaseLockedLoop<float>>();
        auto&     trackedSink   = flow.emplaceBlock<TagSink<std::complex<float>, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});
        auto&     carrierSink   = flow.emplaceBlock<TagSink<std::complex<float>, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});
        auto&     frequencySink = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});

        expect(flow.connect<"out", "in">(source, tracker).has_value()) << fatal;
        expect(flow.connect<"out", "in">(tracker, trackedSink).has_value()) << fatal;
        expect(flow.connect<"carrier", "in">(tracker, carrierSink).has_value()) << fatal;
        expect(flow.connect<"frequency", "in">(tracker, frequencySink).has_value()) << fatal;
        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value()) << fatal;
        expect(sched.runAndWait().has_value());

        expect(eq(trackedSink._samples.size(), kLockingSamples)) << fatal;
        expect(eq(carrierSink._samples.size(), kLockingSamples)) << fatal;
        expect(eq(frequencySink._samples.size(), kLockingSamples)) << fatal;
        expect(approx(distance(trackedSink._samples[kLockingSamples - 1UZ], std::complex<float>{1.f, 0.f}), 0.f, 1e-5f)) << "carrier removed";
        expect(approx(distance(carrierSink._samples[kLockingSamples - 1UZ], tone[(kLockingSamples - 1UZ) % tone.size()]), 0.f, 1e-5f)) << "carrier regenerated";
        expect(approx(frequencySink._samples[kLockingSamples - 1UZ], static_cast<float>(kCarrierOffset), 1e-3f)) << "frequency output";
    };
};

const auto domainTests = [] {
    using namespace boost::ut;
    using gr::analog::AMModulator;
    using gr::analog::PMDemodulator;
    using gr::analog::PMModulator;
    using gr::testing::operator""_domain_test;

    "AM and PM"_domain_test = [](auto& fixture, std::string_view domain) {
        constexpr float    kModulationIndex = 0.5f;
        constexpr float    kPhaseDeviation  = 2.f;
        const std::string  requiredDomain   = std::string(domain) + "!";
        std::vector<float> message(4'096UZ);
        for (std::size_t n = 0UZ; n < message.size(); ++n) {
            message[n] = std::sin(0.01f * static_cast<float>(n));
        }

        auto& source        = gr::testing::sourceOf(fixture, message);
        auto& amModulator   = fixture.template emplace<AMModulator<float>>({{"modulation_index", kModulationIndex}, {"gr:compute_domain", requiredDomain}});
        auto& pmModulator   = fixture.template emplace<PMModulator<float>>({{"phase_deviation", kPhaseDeviation}, {"gr:compute_domain", requiredDomain}});
        auto& pmDemodulator = fixture.template emplace<PMDemodulator<float>>({{"phase_deviation", kPhaseDeviation}, {"gr:compute_domain", requiredDomain}});
        auto& amSink        = fixture.template emplace<gr::testing::ProbeSink<std::complex<float>>>({{"log_samples", true}});
        auto& pmSink        = fixture.template emplace<gr::testing::ProbeSink<float>>({{"log_samples", true}});
        expect(fixture.template connect<"out", "in">(source, amModulator).has_value()) << fatal;
        expect(fixture.template connect<"out", "in">(amModulator, amSink).has_value()) << fatal;
        expect(fixture.template connect<"out", "in">(source, pmModulator).has_value()) << fatal;
        expect(fixture.template connect<"out", "in">(pmModulator, pmDemodulator).has_value()) << fatal;
        expect(fixture.template connect<"out", "in">(pmDemodulator, pmSink).has_value()) << fatal;
        const auto ran = fixture.run();
        expect(ran.has_value()) << domain << (ran.has_value() ? std::string{} : ran.error().message);

        expect(eq(amSink._samples.size(), message.size())) << fatal;
        expect(eq(pmSink._samples.size(), message.size())) << fatal;
        const auto worstError = [&message](auto error) { return std::ranges::max(std::views::iota(0UZ, message.size()) | std::views::transform(error)); };
        expect(lt(worstError([&](std::size_t n) { return distance(amSink._samples[n], std::complex<float>{1.f + kModulationIndex * message[n], 0.f}); }), kTolerance<float>)) << domain << ": AM gives 1 + m x";
        expect(lt(worstError([&](std::size_t n) { return std::abs(pmSink._samples[n] - message[n]); }), kTolerance<float>)) << domain << ": PM there and back gives x";
    } | gr::testing::overGraphs(gr::testing::kAllDomains);
};

int main() {
    using namespace boost::ut;
    "FMDemodulator"_test          = fmDemodulatorTests;
    "FMModulator"_test            = fmModulatorTests;
    "FMDeemphasis"_test           = deemphasisTests;
    "FMPreemphasis"_test          = preemphasisTests;
    "FMStereoDecoder"_test        = fmStereoDecoderTests;
    "AMModulator"_test            = amModulatorTests;
    "AMDemodulator"_test          = amDemodulatorTests;
    "PMModulator"_test            = pmModulatorTests;
    "PMDemodulator"_test          = pmDemodulatorTests;
    "PhaseLockedLoop"_test        = phaseLockedLoopTests;
    "AutomaticGainControl"_test   = automaticGainControlTests;
    "PowerSquelch"_test           = powerSquelchTests;
    "analog"_test                 = commonTests;
    "analog on every domain"_test = domainTests;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
