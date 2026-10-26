#ifndef GNURADIO_TESTING_ARRAY_SCENE_HPP
#define GNURADIO_TESTING_ARRAY_SCENE_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <functional>
#include <numbers>
#include <numeric>
#include <optional>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/algorithm/ArrayGeometry.hpp>
#include <gnuradio-4.0/testing/ExampleRun.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/testing/TensorSource.hpp>
#if __has_include(<gnuradio-4.0/audio/AudioBlocks.hpp>) && !defined(__EMSCRIPTEN__)
#include <gnuradio-4.0/audio/AudioBlocks.hpp>
#include <gnuradio-4.0/filter/time_domain_filter.hpp>
#endif

namespace gr::testing {

inline constexpr double kMicrophoneSnrDb = 20.0;

[[nodiscard]] inline std::string syntheticInputDescription() { return std::format("synthesised audio, additive sensor noise at {:.0f} dB input SNR", kMicrophoneSnrDb); }

/** clean samples and noisy sums must fit float; added sensor noise has clean-mixture/noise power ratio 10^(kMicrophoneSnrDb/10), within float rounding. */
inline void addMicrophoneNoise(std::span<float> samples, std::size_t element) {
    if (samples.empty()) {
        return;
    }
    constexpr std::uint32_t kSeedStride = 0x9e3779b9U;
    const auto              noise       = whiteNoise(samples.size(), kSeedStride * (static_cast<std::uint32_t>(element) + 1U));
    const double            signalPower = std::inner_product(samples.begin(), samples.end(), samples.begin(), 0.0, std::plus{}, [](double first, double second) { return first * second; });
    const double            noisePower  = std::inner_product(noise.begin(), noise.end(), noise.begin(), 0.0);
    const double            scale       = std::sqrt(signalPower / (noisePower * std::pow(10.0, kMicrophoneSnrDb / 10.0)));
    for (std::size_t n = 0UZ; n < samples.size(); ++n) {
        samples[n] += static_cast<float>(scale * noise[n]);
    }
}

/** four elements, a 2x2 grid of 4 m pitch in the x-y plane, waves at 1000 m/s sampled at 1 kHz (1 m is one sample): source 1 at azimuth 0 reaches
 *  the column at larger x four samples early, source 2 at azimuth pi/2 and half the amplitude (-6 dB) the row at larger y; both independent white noise */
inline constexpr std::size_t                                             kSquareArrayElements = 4UZ;
inline constexpr std::array<std::array<double, 3>, kSquareArrayElements> kSquareArray{{{0, 0, 0}, {4, 0, 0}, {0, 4, 0}, {4, 4, 0}}};

[[nodiscard]] inline gr::property_map squareArrayGeometry() { return {{"element_positions", gr::algorithm::geometry::positionsOf(kSquareArray)}, {"propagation_speed", 1000.0}, {"field", std::string("far")}}; }

/** element m's `nSamples`; both sources are read 8 samples ahead so every shift stays inside the noise */
[[nodiscard]] inline std::vector<float> squareArraySamples(std::size_t m, std::size_t nSamples) {
    const auto         s1     = whiteNoise(nSamples + 16UZ, 21U);
    const auto         s2     = whiteNoise(nSamples + 16UZ, 42U);
    const auto         early1 = static_cast<std::size_t>(kSquareArray[m][0]);
    const auto         early2 = static_cast<std::size_t>(kSquareArray[m][1]);
    std::vector<float> x(nSamples);
    for (std::size_t n = 0UZ; n < nSamples; ++n) {
        x[n] = static_cast<float>(s1[n + 8UZ + early1] + 0.5 * s2[n + 8UZ + early2]);
    }
    return x;
}

/** positive delay reads earlier source samples */
struct Arrival {
    double        delaySamples;
    double        gain;
    std::uint32_t seed;
};

inline constexpr std::size_t kMaxArrivalDelaySamples = 1024UZ;

/** `samples` read at the fractional `position` by linear interpolation, which colours noise and biases wideband delay estimates; position + 1 < size */
[[nodiscard]] inline double interpolateAt(std::span<const double> samples, double position) noexcept {
    const auto   first      = static_cast<std::size_t>(position);
    const double fractional = position - static_cast<double>(first);
    return (1.0 - fractional) * samples[first] + fractional * samples[first + 1UZ];
}

/** |delaySamples| < kMaxArrivalDelaySamples - 2, gain finite */
[[nodiscard]] inline std::vector<float> delayedNoises(std::span<const Arrival> arrivals, std::size_t nSamples) {
    constexpr std::size_t kMargin = kMaxArrivalDelaySamples;
    std::vector<double>   heard(nSamples, 0.0);
    for (const Arrival& arrival : arrivals) {
        assert(std::abs(arrival.delaySamples) < static_cast<double>(kMargin - 2UZ));
        const auto noise = whiteNoise(nSamples + 2UZ * kMargin, arrival.seed);
        for (std::size_t n = 0UZ; n < nSamples; ++n) {
            heard[n] += arrival.gain * interpolateAt(noise, static_cast<double>(n + kMargin) - arrival.delaySamples);
        }
    }
    return {heard.begin(), heard.end()};
}

inline constexpr double kSweepBearingLimitDegrees = 80.0;

/** bearing of the synthetic stereo scene's source at a sample of the repeating scene: a linear sweep from -kSweepBearingLimitDegrees to +kSweepBearingLimitDegrees, right positive; sceneSamples > 0. */
[[nodiscard]] inline double sweepBearingRadians(std::size_t sampleIndex, std::size_t sceneSamples) noexcept {
    assert(sceneSamples > 0UZ);
    const double progress = static_cast<double>(sampleIndex % sceneSamples) / static_cast<double>(sceneSamples);
    return (2.0 * progress - 1.0) * kSweepBearingLimitDegrees * std::numbers::pi / 180.0;
}

inline void addElementSource(gr::Graph& graph, auto& frames, std::size_t m, std::vector<float> samples, double rate) {
    auto& source = graph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"values", std::move(samples)}, {"n_samples_max", gr::Size_t(0)}, {"mark_tag", false}});
    source._tags = {OwningTag(0UZ, {{std::string(gr::tag::SAMPLE_RATE.key()), static_cast<float>(rate)}})};
    std::ignore  = graph.connect(source, "out", frames, std::format("in#{}", m));
}

inline constexpr float kRumbleCutoffHz = 150.f;

#if __has_include(<gnuradio-4.0/audio/AudioBlocks.hpp>) && !defined(__EMSCRIPTEN__)
inline constexpr bool kLiveAudio = true;

/** announced on stdout; nullopt when no device matches or it has fewer than two input channels */
[[nodiscard]] inline std::optional<ChannelArray> microphoneArray(std::string_view device) {
    const auto selected = gr::audio::detail::selectedDeviceInfo({.device = std::string(device)}, true);
    if (!selected || selected->numChannels < 2U) {
        return std::nullopt;
    }
    const ChannelArray array = channelArrayOf(selected->numChannels);
    std::println("array: {} of {}, {:g} m {} (device '{}' has {} input channels; --layout/--detectors override)", array.layout, array.elements, array.spacing, array.spacingRole, selected->name, selected->numChannels);
    return array;
}

/** refused when the device has fewer than M input channels */
[[nodiscard]] inline std::expected<void, std::string> connectMicrophones(gr::Graph& graph, auto& frames, std::string_view device, std::size_t M, double rate) {
    const auto selected = gr::audio::detail::selectedDeviceInfo({.device = std::string(device)}, true);
    if (!selected) {
        return std::unexpected(selected.error().message);
    }
    if (selected->numChannels < M) {
        return std::unexpected(std::format("device '{}' has {} input channels, the array needs {}; choose another --layout or --device", selected->name, selected->numChannels, M));
    }
    std::println("input: true audio, sound card '{}' ({} input channels, selector {})", selected->name, selected->numChannels, device.empty() ? std::string_view("default") : device);
    auto&                  microphones = graph.emplaceBlock<gr::audio::AudioSourceMultiplexed<float>>({{"device", std::string(device)}, {"n_outputs", gr::Size_t(M)}, {"req_sample_rate", static_cast<float>(rate)}});
    const gr::property_map highPass{{"filter_type", std::string("IIR")}, {"filter_response", std::string("HIGHPASS")}, {"filter_order", gr::Size_t(4)}, {"f_high", kRumbleCutoffHz}, {"sample_rate", static_cast<float>(rate)}};
    for (std::size_t m = 0UZ; m < M; ++m) {
        auto& rumbleFree = graph.emplaceBlock<gr::filter::BasicFilter<float>>(highPass);
        std::ignore      = graph.connect(microphones, std::format("out#{}", m), rumbleFree, "in");
        std::ignore      = graph.connect(rumbleFree, "out", frames, std::format("in#{}", m));
    }
    return {};
}
#else
inline constexpr bool kLiveAudio = false;

[[nodiscard]] inline std::optional<ChannelArray> microphoneArray(std::string_view) { return std::nullopt; }

[[nodiscard]] inline std::expected<void, std::string> connectMicrophones(gr::Graph&, auto&, std::string_view, std::size_t, double) { return {}; }
#endif

} // namespace gr::testing

#endif // GNURADIO_TESTING_ARRAY_SCENE_HPP
